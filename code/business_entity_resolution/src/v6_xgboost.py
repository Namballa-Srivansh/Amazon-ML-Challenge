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

    print("\n[4/6] Building features...")
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

        v1n = np.stack([name2vec[t] for t in s1n]); v2n = np.stack([name2vec[t] for t in cn])
        v1a = np.stack([addr2vec[t] for t in s1a]); v2a = np.stack([addr2vec[t] for t in ca])
        df["name_embed_cosine"] = cos_rows(v1n, v2n)
        df["addr_embed_cosine"] = cos_rows(v1a, v2a)
        return df

    train_df = add_all_features(train_df, tr_s1n, tr_s1a, tr_cn, tr_ca)
    val_df   = add_all_features(val_df,   va_s1n, va_s1a, va_cn, va_ca)

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

    pickle.dump(model,    open(os.path.join(MODELS_DIR, "v6_classifier.pkl"), "wb"))
    pickle.dump(vec_name, open(os.path.join(MODELS_DIR, "v6_vec_name.pkl"),   "wb"))
    pickle.dump(vec_addr, open(os.path.join(MODELS_DIR, "v6_vec_addr.pkl"),   "wb"))
    with open(os.path.join(MODELS_DIR, "v6_threshold.txt"), "w") as f:
        f.write(str(best_t))

    print(f"\n[Done] v6 finished in {time.time()-t0:.1f}s")
    print("  Saved: v6_classifier.pkl, v6_vec_name.pkl, v6_vec_addr.pkl, v6_threshold.txt")
    print("  NOTE: run v4_calibration.py's logic again against this model before v7 if")
    print("        you want a freshly-calibrated ambiguous band for XGBoost's scores --")
    print("        XGBoost's predict_proba is not guaranteed as well-calibrated as LR's.")

if __name__ == "__main__":
    main()
