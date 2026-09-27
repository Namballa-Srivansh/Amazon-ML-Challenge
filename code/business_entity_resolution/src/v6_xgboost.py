"""
v6 — phonetic-blocking (part 2 of 2) · XGBoost classifier upgrade
==================================================================
Run v6_blocking_metaphone.py FIRST (it overwrites candidate_pairs.tsv
with the recall-improved candidate set). This script then retrains the
classifier as XGBoost instead of Logistic Regression, using the same
11 features as v5 (string distance + TF-IDF cosine + embedding cosine),
so it can capture the nonlinear interactions LR can't (e.g. name_x_addr
and lookalike_flag combining differently depending on country).

Everything else -- entity-level split, hard negatives from blocking,
train-only vectorizer fitting, macro-F0.5 threshold sweep -- follows
the same discipline as v3/v5.

Outputs (MODELS_DIR):
  - v6_classifier.pkl   (XGBClassifier)
  - v6_vec_name.pkl / v6_vec_addr.pkl
  - v6_threshold.txt
"""

import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import os, re, time, pickle
import numpy as np
import pandas as pd
import jellyfish
from xgboost import XGBClassifier
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
FEATURE_CHUNK_SIZE = 200_000   # rows per chunk in add_all_features_chunked -- keeps
                                 # TF-IDF transform() and embedding np.stack() calls
                                 # bounded instead of building 13.7M-row sparse/dense
                                 # arrays in one shot (that's what caused the MemoryError
                                 # in vec_addr.transform(ca)). Lower this further (e.g.
                                 # 50_000) if you still see memory pressure.

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

def cos_rows(a, b):
    num = (a * b).sum(axis=1)
    den = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
    den[den == 0] = 1e-9
    return num / den

def macro_f05(pred_matches, gt_map, entity_ids):
    scores = []
    for s1_id in entity_ids:
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

def main():
    print("=" * 60)
    print("v6 — XGBoost classifier upgrade")
    print("=" * 60)
    t0 = time.time()

    print("\n[1/6] Loading data + (metaphone-improved) candidate pairs...")
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
    print(f"  {len(df_pairs):,} pairs, {df_pairs['label'].mean():.2%} positive "
          f"(imbalance is expected -- XGBoost handles it via scale_pos_weight)")

    print("\n[2/6] Entity-level split (80/20, same random_state=42 as v3/v5)...")
    unique_s1 = df_pairs["s1_id"].unique()
    train_s1, val_s1 = train_test_split(unique_s1, test_size=0.2, random_state=42)
    train_df = df_pairs[df_pairs["s1_id"].isin(set(train_s1))].copy()
    val_df   = df_pairs[df_pairs["s1_id"].isin(set(val_s1))].copy()

    s1_dict  = s1.set_index("entity_id")[["business_name", "business_address"]].to_dict("index")
    s23_dict = s23.set_index("entity_id")[["business_name", "business_address"]].to_dict("index")

    def get_texts(df):
        s1n = [str(s1_dict.get(i, {}).get("business_name",  "")).lower().strip() for i in df["s1_id"]]
        s1a = [str(s1_dict.get(i, {}).get("business_address","")).lower().strip() for i in df["s1_id"]]
        cn  = [str(s23_dict.get(i, {}).get("business_name",  "")).lower().strip() for i in df["cid"]]
        ca  = [str(s23_dict.get(i, {}).get("business_address","")).lower().strip() for i in df["cid"]]
        return s1n, s1a, cn, ca

    print("\n[3/6] Fitting TF-IDF (train only) and loading embedder...")
    tr_s1n, tr_s1a, tr_cn, tr_ca = get_texts(train_df)
    va_s1n, va_s1a, va_cn, va_ca = get_texts(val_df)

    vec_name = TfidfVectorizer(analyzer="char_wb", ngram_range=(2,4), max_features=10_000, sublinear_tf=True)
    vec_addr = TfidfVectorizer(analyzer="char_wb", ngram_range=(2,4), max_features=10_000, sublinear_tf=True)
    vec_name.fit(list(set(tr_s1n + tr_cn)))
    vec_addr.fit(list(set(tr_s1a + tr_ca)))

    embedder = SentenceTransformer(EMBED_MODEL_NAME)
    all_names = sorted(set(tr_s1n + tr_cn + va_s1n + va_cn))
    all_addrs = sorted(set(tr_s1a + tr_ca + va_s1a + va_ca))
    name_vecs = embedder.encode(all_names, batch_size=EMBED_BATCH_SIZE, show_progress_bar=True, convert_to_numpy=True)
    addr_vecs = embedder.encode(all_addrs, batch_size=EMBED_BATCH_SIZE, show_progress_bar=True, convert_to_numpy=True)
    name2vec, addr2vec = dict(zip(all_names, name_vecs)), dict(zip(all_addrs, addr_vecs))

    print("\n[4/6] Building features (chunked to bound memory)...")

    def add_all_features_chunked(df, s1n, s1a, cn, ca, label: str):
        """Same feature logic as before, but processes FEATURE_CHUNK_SIZE rows
        at a time so vec_*.transform() and the embedding np.stack() calls only
        ever hold one chunk's sparse/dense arrays in memory, not all 13.7M rows'
        worth at once. Only the small feature columns (not the sparse TF-IDF
        matrices or embedding vectors themselves) are kept between chunks."""
        n = len(df)
        s1_ids  = df["s1_id"].to_numpy()
        cids    = df["cid"].to_numpy()
        labels  = df["label"].to_numpy()
        n_chunks = (n + FEATURE_CHUNK_SIZE - 1) // FEATURE_CHUNK_SIZE
        parts = []
        t_start = time.time()

        for i, start in enumerate(range(0, n, FEATURE_CHUNK_SIZE)):
            end = min(start + FEATURE_CHUNK_SIZE, n)
            c_s1n, c_s1a = s1n[start:end], s1a[start:end]
            c_cn,  c_ca  = cn[start:end],  ca[start:end]

            chunk = pd.DataFrame({
                "s1_id": s1_ids[start:end],
                "cid":   cids[start:end],
                "label": labels[start:end],
            })
            chunk["name_jw"]  = [safe_jw(a, b)  for a, b in zip(c_s1n, c_cn)]
            chunk["name_lev"] = [safe_lev(a, b) for a, b in zip(c_s1n, c_cn)]
            chunk["addr_jw"]  = [safe_jw(a, b)  for a, b in zip(c_s1a, c_ca)]
            chunk["addr_lev"] = [safe_lev(a, b) for a, b in zip(c_s1a, c_ca)]
            chunk["num_overlap"] = [num_overlap(a, b) for a, b in zip(c_s1a, c_ca)]

            # TF-IDF transform + cosine on THIS CHUNK ONLY -- this is the line
            # that OOM'd at full scale (vec_addr.transform(ca) over 13.7M rows)
            s1n_v = vec_name.transform(c_s1n); cn_v = vec_name.transform(c_cn)
            s1a_v = vec_addr.transform(c_s1a); ca_v = vec_addr.transform(c_ca)
            chunk["name_tfidf_cosine"] = np.array(s1n_v.multiply(cn_v).sum(axis=1)).flatten()
            chunk["addr_tfidf_cosine"] = np.array(s1a_v.multiply(ca_v).sum(axis=1)).flatten()
            chunk["name_x_addr"]    = chunk["name_tfidf_cosine"] * chunk["addr_tfidf_cosine"]
            chunk["lookalike_flag"] = ((chunk["name_jw"] > 0.90) & (chunk["addr_jw"] < 0.50)).astype(int)

            # Embedding cosine, also chunk-bounded (np.stack over 200k x 384-dim
            # float32 is ~300MB per array instead of ~21GB at 13.7M rows)
            v1n = np.stack([name2vec[t] for t in c_s1n]); v2n = np.stack([name2vec[t] for t in c_cn])
            v1a = np.stack([addr2vec[t] for t in c_s1a]); v2a = np.stack([addr2vec[t] for t in c_ca])
            chunk["name_embed_cosine"] = cos_rows(v1n, v2n)
            chunk["addr_embed_cosine"] = cos_rows(v1a, v2a)

            parts.append(chunk)
            del s1n_v, cn_v, s1a_v, ca_v, v1n, v2n, v1a, v2a
            if (i + 1) % 5 == 0 or (i + 1) == n_chunks:
                print(f"    [{label}] chunk {i+1}/{n_chunks} "
                      f"({end:,}/{n:,} rows, {time.time()-t_start:.0f}s elapsed)")

        return pd.concat(parts, ignore_index=True)

    train_df = add_all_features_chunked(train_df, tr_s1n, tr_s1a, tr_cn, tr_ca, "train")
    val_df   = add_all_features_chunked(val_df,   va_s1n, va_s1a, va_cn, va_ca, "val")

    print("\n[5/6] Training XGBoost...")
    X_train, y_train = train_df[FEATURES], train_df["label"]
    X_val,   y_val   = val_df[FEATURES],   val_df["label"]

    n_pos, n_neg = y_train.sum(), len(y_train) - y_train.sum()
    scale_pos_weight = (n_neg / n_pos) if n_pos > 0 else 1.0
    print(f"  scale_pos_weight = {scale_pos_weight:.2f} ({n_pos:,} pos / {n_neg:,} neg)")

    model = XGBClassifier(
        n_estimators=300, max_depth=5, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        eval_metric="aucpr", random_state=42, n_jobs=-1,
    )
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

    print("\n  Feature importances:")
    for f, imp in sorted(zip(FEATURES, model.feature_importances_), key=lambda x: -x[1]):
        print(f"    {f}: {imp:.4f}")

    print("\n[6/6] Threshold sweep (macro F0.5)...")
    val_df = val_df.copy()
    val_df["prob"] = model.predict_proba(X_val)[:, 1]

    def pred_matches_at(thresh):
        d = {}
        for row in val_df.itertuples(index=False):
            d.setdefault(row.s1_id, set())
            if row.prob >= thresh:
                d[row.s1_id].add(row.cid)
        return d

    best_t, best_f05 = 0.5, -1.0
    for t in np.arange(0.05, 0.97, 0.02):
        f05 = macro_f05(pred_matches_at(t), gt_map, val_s1)
        if f05 > best_f05:
            best_f05, best_t = f05, t

    print(f"  Best Threshold: {best_t:.2f}")
    print(f"  Best Val F0.5:  {best_f05:.4f}  (compare against v3=0.9332 and v5's logged score)")

    # Write debug_scores.tsv -- v7_qwen_jury.py (and v4_calibration.py's own
    # fallback path) read this to know which pairs fall in the ambiguous band.
    debug_out = val_df[["s1_id", "cid", "prob", "label"]].copy()
    debug_out.to_csv(os.path.join(OUTPUT_DIR, "debug_scores.tsv"), sep="\t", index=False)
    print(f"  Saved: debug_scores.tsv ({len(debug_out):,} val rows, for v4/v7 to consume)")

    pickle.dump(model,    open(os.path.join(MODELS_DIR, "v6_classifier.pkl"), "wb"))
    pickle.dump(vec_name, open(os.path.join(MODELS_DIR, "v6_vec_name.pkl"),   "wb"))
    pickle.dump(vec_addr, open(os.path.join(MODELS_DIR, "v6_vec_addr.pkl"),   "wb"))
    with open(os.path.join(MODELS_DIR, "v6_threshold.txt"), "w") as f:
        f.write(str(best_t))

    print(f"\n[Done] v6 finished in {time.time()-t0:.1f}s")
    print("  Saved: v6_classifier.pkl, v6_vec_name.pkl, v6_vec_addr.pkl, v6_threshold.txt, debug_scores.tsv")
    print("  NOTE: run `python v4_calibration.py --model-prefix v6` next to calibrate this")
    print("        model's scores and define the ambiguous band for v7 -- XGBoost's")
    print("        predict_proba is not guaranteed as well-calibrated as LR's.")

if __name__ == "__main__":
    main()