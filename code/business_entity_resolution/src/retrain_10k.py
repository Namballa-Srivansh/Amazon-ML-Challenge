"""
retrain_10k.py — Retrain TF-IDF vectorizers with max_features=10,000
=====================================================================
Retrains ONLY the vectorizers and Logistic Regression classifier with
a smaller vocabulary (10k instead of 50k) so the matrices fit cleanly
into the NVIDIA RTX 4060 VRAM for GPU-accelerated inference.

Original v3 model files are untouched (kept as backup).
New files: v3_vec_name_10k.pkl, v3_vec_addr_10k.pkl, v3_classifier_10k.pkl,
           v3_threshold_10k.txt
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

REPO_ROOT   = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
TRAIN_DIR   = os.path.join(REPO_ROOT, "dataset", "mini_train")
OUTPUT_DIR  = os.path.join(REPO_ROOT, "output")
MODELS_DIR  = os.path.join(REPO_ROOT, "models")

FEATURES = [
    "name_jw", "name_lev", "name_tfidf_cosine",
    "addr_jw", "addr_lev", "addr_tfidf_cosine",
    "num_overlap", "name_x_addr", "lookalike_flag"
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

def main():
    print("=" * 60)
    print("Retrain TF-IDF (10k features) + Logistic Regression")
    print("=" * 60)
    t0 = time.time()

    print("\n[1/5] Loading data & candidate pairs...")
    s1  = pd.read_csv(os.path.join(TRAIN_DIR, "train_source1.tsv"), sep="\t", dtype=str).fillna("")
    s23 = pd.concat([
        pd.read_csv(os.path.join(TRAIN_DIR, "train_source2.tsv"), sep="\t", dtype=str).fillna(""),
        pd.read_csv(os.path.join(TRAIN_DIR, "train_source3.tsv"), sep="\t", dtype=str).fillna(""),
    ], ignore_index=True)
    gt = pd.read_csv(os.path.join(TRAIN_DIR, "train_ground_truth.tsv"), sep="\t", dtype=str).fillna("")
    cands_df = pd.read_csv(os.path.join(OUTPUT_DIR, "candidate_pairs.tsv"), sep="\t", dtype=str).fillna("")

    print("[2/5] Building labeled pairs...")
    gt_map = {}
    for row in gt.itertuples(index=False):
        gt_map[row.source1_entity_id] = set(row.matched_entity_ids.split(",")) - {""}

    pairs = []
    for row in cands_df.itertuples(index=False):
        s1_id = row.source1_entity_id
        c_ids = set(row.candidate_entity_ids.split(",")) - {""}
        true_set = gt_map.get(s1_id, set())
        for cid in c_ids:
            pairs.append({"s1_id": s1_id, "cid": cid, "label": int(cid in true_set)})

    df_pairs = pd.DataFrame(pairs)
    print(f"  Total pairs: {len(df_pairs):,}  |  Positive rate: {df_pairs['label'].mean():.2%}")

    print("[3/5] Entity-level train/val split...")
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

    tr_s1n, tr_s1a, tr_cn, tr_ca = get_texts(train_df)
    va_s1n, va_s1a, va_cn, va_ca = get_texts(val_df)

    print("[4/5] Fitting 10k TF-IDF vectorizers & computing features...")
    vec_name = TfidfVectorizer(analyzer="char_wb", ngram_range=(2,4), max_features=10_000, sublinear_tf=True)
    vec_addr = TfidfVectorizer(analyzer="char_wb", ngram_range=(2,4), max_features=10_000, sublinear_tf=True)
    vec_name.fit(list(set(tr_s1n + tr_cn)))
    vec_addr.fit(list(set(tr_s1a + tr_ca)))
    print(f"  Name vocab: {len(vec_name.vocabulary_):,}  |  Addr vocab: {len(vec_addr.vocabulary_):,}")

    def add_features(df, s1n, s1a, cn, ca):
        df["name_jw"]  = [safe_jw(a, b)  for a, b in zip(s1n, cn)]
        df["name_lev"] = [safe_lev(a, b) for a, b in zip(s1n, cn)]
        df["addr_jw"]  = [safe_jw(a, b)  for a, b in zip(s1a, ca)]
        df["addr_lev"] = [safe_lev(a, b) for a, b in zip(s1a, ca)]
        df["num_overlap"] = [num_overlap(a, b) for a, b in zip(s1a, ca)]
        s1n_v = vec_name.transform(s1n); cn_v = vec_name.transform(cn)
        s1a_v = vec_addr.transform(s1a); ca_v = vec_addr.transform(ca)
        df["name_tfidf_cosine"] = np.array(s1n_v.multiply(cn_v).sum(axis=1)).flatten()
        df["addr_tfidf_cosine"] = np.array(s1a_v.multiply(ca_v).sum(axis=1)).flatten()
        df["name_x_addr"]    = df["name_tfidf_cosine"] * df["addr_tfidf_cosine"]
        df["lookalike_flag"] = ((df["name_jw"] > 0.90) & (df["addr_jw"] < 0.50)).astype(int)
        return df

    train_df = add_features(train_df, tr_s1n, tr_s1a, tr_cn, tr_ca)
    val_df   = add_features(val_df,   va_s1n, va_s1a, va_cn, va_ca)

    model = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
    model.fit(train_df[FEATURES], train_df["label"])

    val_probs = model.predict_proba(val_df[FEATURES])[:, 1]
    val_df["prob"] = val_probs

    # Threshold tuning
    def score_at_threshold(thresh):
        pred_matches = {}
        for row in val_df.itertuples(index=False):
            if row.s1_id not in pred_matches:
                pred_matches[row.s1_id] = set()
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

    best_t, best_f05 = 0.5, 0.0
    for t in np.arange(0.10, 0.96, 0.05):
        f05 = score_at_threshold(t)
        if f05 > best_f05:
            best_f05, best_t = f05, t

    print(f"\n[5/5] Results:")
    print(f"  Best Threshold : {best_t:.2f}")
    print(f"  Best Val F0.5  : {best_f05:.4f}")

    # Save as GPU-friendly models (originals untouched)
    pickle.dump(model,    open(os.path.join(MODELS_DIR, "v3_classifier_10k.pkl"),  "wb"))
    pickle.dump(vec_name, open(os.path.join(MODELS_DIR, "v3_vec_name_10k.pkl"),    "wb"))
    pickle.dump(vec_addr, open(os.path.join(MODELS_DIR, "v3_vec_addr_10k.pkl"),    "wb"))
    with open(os.path.join(MODELS_DIR, "v3_threshold_10k.txt"), "w") as f:
        f.write(str(best_t))

    print(f"\n  Saved: v3_classifier_10k.pkl, v3_vec_name_10k.pkl, v3_vec_addr_10k.pkl")
    print(f"  Original v3 models are untouched.")
    print(f"  Total time: {time.time()-t0:.1f}s")

if __name__ == "__main__":
    main()
