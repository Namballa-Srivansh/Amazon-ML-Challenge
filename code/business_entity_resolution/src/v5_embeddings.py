"""
v5 — embedding-juror
====================
Adds semantic similarity features from a multilingual sentence-transformer
on top of v3's character/string features, then retrains the classifier.

Model: paraphrase-multilingual-MiniLM-L12-v2 (MIT licensed, ~118M params,
well under the 8B cap, handles French out-of-the-box with zero French
training examples — this is the fix for the France-generalization risk
flagged in HANDOFF.md).

New features added to the v3 set:
  - name_embed_cosine
  - addr_embed_cosine

Everything else (entity-level split, hard negatives, threshold tuning via
macro F0.5) follows the same discipline as v3 to stay comparable.

Outputs (MODELS_DIR):
  - v5_classifier.pkl
  - v5_vec_name.pkl / v5_vec_addr.pkl   (re-fit on train split, same as v3)
  - v5_threshold.txt
  - EMBED_MODEL_NAME is hard-coded below and re-loaded by name at
    inference time — the sentence-transformer itself is not pickled.
"""

import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import os, re, time, pickle
import numpy as np
import pandas as pd
import jellyfish
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sentence_transformers import SentenceTransformer

REPO_ROOT   = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
                os.path.abspath(__file__)))))
TRAIN_DIR   = os.path.join(REPO_ROOT, "dataset", "mini_train")
OUTPUT_DIR  = os.path.join(REPO_ROOT, "output")
MODELS_DIR  = os.path.join(REPO_ROOT, "models")

EMBED_MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"
EMBED_BATCH_SIZE = 256

FEATURES = [
    "name_jw", "name_lev", "name_tfidf_cosine",
    "addr_jw", "addr_lev", "addr_tfidf_cosine",
    "num_overlap", "name_x_addr", "lookalike_flag",
    "name_embed_cosine", "addr_embed_cosine",
]

def safe_jw(s1, s2):
    if not s1 and not s2: return 1.0
    if not s1 or not s2:  return 0.0
    return jellyfish.jaro_winkler_similarity(s1, s2)

def safe_lev(s1, s2):
    if not s1 and not s2: return 1.0
    if not s1 or not s2:  return 0.0
    d = jellyfish.levenshtein_distance(s1, s2)
    m = max(len(s1), len(s2))
    return 1.0 - (d / m) if m > 0 else 0.0

def num_overlap(s1, s2):
    t1 = set(re.findall(r'\d+', str(s1)))
    t2 = set(re.findall(r'\d+', str(s2)))
    if not t1 or not t2: return 0.0
    return len(t1 & t2) / len(t1 | t2)

def cos_rows(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Row-wise cosine similarity between two equally-shaped dense arrays."""
    num = (a * b).sum(axis=1)
    den = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
    den[den == 0] = 1e-9
    return num / den

def main():
    print("=" * 60)
    print("v5 — embedding-juror (multilingual sentence-transformer)")
    print("=" * 60)
    t0 = time.time()

    # 1. Load data + v2 candidate pairs (same as v3)
    print("\n[1/7] Loading data and candidate pairs...")
    s1  = pd.read_csv(os.path.join(TRAIN_DIR, "train_source1.tsv"), sep="\t", dtype=str).fillna("")
    s23 = pd.concat([
        pd.read_csv(os.path.join(TRAIN_DIR, "train_source2.tsv"), sep="\t", dtype=str).fillna(""),
        pd.read_csv(os.path.join(TRAIN_DIR, "train_source3.tsv"), sep="\t", dtype=str).fillna(""),
    ], ignore_index=True)
    gt = pd.read_csv(os.path.join(TRAIN_DIR, "train_ground_truth.tsv"), sep="\t", dtype=str).fillna("")
    cands_df = pd.read_csv(os.path.join(OUTPUT_DIR, "candidate_pairs.tsv"), sep="\t", dtype=str).fillna("")

    gt_map = {}
    for row in gt.itertuples(index=False):
        gt_map[row.source1_entity_id] = set(row.matched_entity_ids.split(",")) - {""}

    pairs = []
    for row in cands_df.itertuples(index=False):
        s1_id, c_ids = row.source1_entity_id, set(row.candidate_entity_ids.split(",")) - {""}
        true_set = gt_map.get(s1_id, set())
        for cid in c_ids:
            pairs.append({"s1_id": s1_id, "cid": cid, "label": int(cid in true_set)})
    df_pairs = pd.DataFrame(pairs)
    print(f"  {len(df_pairs):,} total pairs, {df_pairs['label'].mean():.2%} positive")

    # 2. Same entity-level split as v3 (comparable results)
    print("\n[2/7] Entity-level train/val split (80/20, matches v3)...")
    unique_s1 = df_pairs["s1_id"].unique()
    train_s1, val_s1 = train_test_split(unique_s1, test_size=0.2, random_state=42)
    train_df = df_pairs[df_pairs["s1_id"].isin(set(train_s1))].copy()
    val_df   = df_pairs[df_pairs["s1_id"].isin(set(val_s1))].copy()

    # 3. Join text
    print("\n[3/7] Joining text...")
    s1_dict  = s1.set_index("entity_id")[["business_name", "business_address"]].to_dict("index")
    s23_dict = s23.set_index("entity_id")[["business_name", "business_address"]].to_dict("index")

    def get_texts(df):
        s1n = [str(s1_dict.get(i, {}).get("business_name",  "")).lower().strip() for i in df["s1_id"]]
        s1a = [str(s1_dict.get(i, {}).get("business_address","")).lower().strip() for i in df["s1_id"]]
        cn  = [str(s23_dict.get(i, {}).get("business_name",  "")).lower().strip() for i in df["cid"]]
        ca  = [str(s23_dict.get(i, {}).get("business_address","")).lower().strip() for i in df["cid"]]
        return s1n, s1a, cn, ca

    tr_s1n, tr_s1a, tr_cn, tr_ca = get_texts(train_df)
    va_s1n, va_s1a, va_cn, va_ca = get_texts(val_df)

    # 4. Fit TF-IDF on train only (same discipline as v3)
    print("\n[4/7] Fitting TF-IDF vectorizers on TRAIN split only...")
    vec_name = TfidfVectorizer(analyzer="char_wb", ngram_range=(2,4), max_features=10_000, sublinear_tf=True)
    vec_addr = TfidfVectorizer(analyzer="char_wb", ngram_range=(2,4), max_features=10_000, sublinear_tf=True)
    vec_name.fit(list(set(tr_s1n + tr_cn)))
    vec_addr.fit(list(set(tr_s1a + tr_ca)))

    # 5. Load embedding model, encode every unique string ONCE (cache by text)
    print(f"\n[5/7] Loading {EMBED_MODEL_NAME} and encoding unique strings...")
    embedder = SentenceTransformer(EMBED_MODEL_NAME)

    all_names = sorted(set(tr_s1n + tr_cn + va_s1n + va_cn))
    all_addrs = sorted(set(tr_s1a + tr_ca + va_s1a + va_ca))
    print(f"  Unique names: {len(all_names):,}  |  Unique addresses: {len(all_addrs):,}")

    name_vecs = embedder.encode(all_names, batch_size=EMBED_BATCH_SIZE, show_progress_bar=True,
                                 convert_to_numpy=True, normalize_embeddings=False)
    addr_vecs = embedder.encode(all_addrs, batch_size=EMBED_BATCH_SIZE, show_progress_bar=True,
                                 convert_to_numpy=True, normalize_embeddings=False)
    name2vec = dict(zip(all_names, name_vecs))
    addr2vec = dict(zip(all_addrs, addr_vecs))

    def embed_features(df, s1n, s1a, cn, ca):
        v1n = np.stack([name2vec[t] for t in s1n])
        v2n = np.stack([name2vec[t] for t in cn])
        v1a = np.stack([addr2vec[t] for t in s1a])
        v2a = np.stack([addr2vec[t] for t in ca])
        df = df.copy()
        df["name_embed_cosine"] = cos_rows(v1n, v2n)
        df["addr_embed_cosine"] = cos_rows(v1a, v2a)
        return df

    # 6. Full feature build (v3 features + embedding features)
    print("\n[6/7] Building full feature set...")
    def add_all_features(df, s1n, s1a, cn, ca):
        df = df.copy()
        df["name_jw"]  = [safe_jw(a, b)  for a, b in zip(s1n, cn)]
        df["name_lev"] = [safe_lev(a, b) for a, b in zip(s1n, cn)]
        df["addr_jw"]  = [safe_jw(a, b)  for a, b in zip(s1a, ca)]
        df["addr_lev"] = [safe_lev(a, b) for a, b in zip(s1a, ca)]
        df["num_overlap"] = [num_overlap(a, b) for a, b in zip(s1a, ca)]

        s1n_v = vec_name.transform(s1n);  cn_v  = vec_name.transform(cn)
        s1a_v = vec_addr.transform(s1a);  ca_v  = vec_addr.transform(ca)
        df["name_tfidf_cosine"] = np.array(s1n_v.multiply(cn_v).sum(axis=1)).flatten()
        df["addr_tfidf_cosine"] = np.array(s1a_v.multiply(ca_v).sum(axis=1)).flatten()

        df["name_x_addr"]    = df["name_tfidf_cosine"] * df["addr_tfidf_cosine"]
        df["lookalike_flag"] = ((df["name_jw"] > 0.90) & (df["addr_jw"] < 0.50)).astype(int)
        df = embed_features(df, s1n, s1a, cn, ca)
        return df

    train_df = add_all_features(train_df, tr_s1n, tr_s1a, tr_cn, tr_ca)
    val_df   = add_all_features(val_df,   va_s1n, va_s1a, va_cn, va_ca)

    # 7. Train + threshold tune
    print("\n[7/7] Training Logistic Regression (11 features) + threshold sweep...")
    X_train, y_train = train_df[FEATURES], train_df["label"]
    X_val,   y_val   = val_df[FEATURES],   val_df["label"]

    model = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
    model.fit(X_train, y_train)

    print("\n  Model Coefficients:")
    for f, coef in zip(FEATURES, model.coef_[0]):
        print(f"    {f}: {coef:+.4f}")

    val_df = val_df.copy()
    val_df["prob"] = model.predict_proba(X_val)[:, 1]

    def score_at_threshold(thresh):
        pred_matches = {}
        for row in val_df.itertuples(index=False):
            pred_matches.setdefault(row.s1_id, set())
            if row.prob >= thresh:
                pred_matches[row.s1_id].add(row.cid)
        scores = []
        for s1_id in val_s1:
            gt_set   = gt_map.get(s1_id, set())
            pred_set = pred_matches.get(s1_id, set())
            if not gt_set and not pred_set:
                scores.append(1.0); continue
            tp = len(gt_set & pred_set)
            p  = tp / len(pred_set) if pred_set else 0.0
            r  = tp / len(gt_set)   if gt_set   else 0.0
            f05 = 1.25 * p * r / (0.25 * p + r) if (p + r) > 0 else 0.0
            scores.append(f05)
        return sum(scores) / len(scores) if scores else 0.0

    best_t, best_f05 = 0.5, -1.0
    for t in np.arange(0.10, 0.96, 0.05):
        f05 = score_at_threshold(t)
        if f05 > best_f05:
            best_f05, best_t = f05, t

    print(f"\n  Best Threshold: {best_t:.2f}")
    print(f"  Best Val F0.5:  {best_f05:.4f}  (compare against v3's saved value in HANDOFF.md)")

    pickle.dump(model,    open(os.path.join(MODELS_DIR, "v5_classifier.pkl"), "wb"))
    pickle.dump(vec_name, open(os.path.join(MODELS_DIR, "v5_vec_name.pkl"),   "wb"))
    pickle.dump(vec_addr, open(os.path.join(MODELS_DIR, "v5_vec_addr.pkl"),   "wb"))
    with open(os.path.join(MODELS_DIR, "v5_threshold.txt"), "w") as f:
        f.write(str(best_t))

    print(f"\n[Done] v5 finished in {time.time()-t0:.1f}s")
    print("  Saved: v5_classifier.pkl, v5_vec_name.pkl, v5_vec_addr.pkl, v5_threshold.txt")
    print(f"  NOTE: embedder is reloaded by name ('{EMBED_MODEL_NAME}') at inference time,")
    print("        not pickled — keep that string in sync with generate_final_submission.py")

if __name__ == "__main__":
    main()
