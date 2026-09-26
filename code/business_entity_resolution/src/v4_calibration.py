"""
v4 — calibration
================
Applies Platt scaling (manual sigmoid fit on the classifier's decision
function) to the v3 Logistic Regression classifier, then re-tunes the
merge threshold on a *fresh* held-out split and defines the ambiguous
band [low_thresh, high_thresh] that v7's Qwen jury will operate on.

Why manual Platt scaling instead of CalibratedClassifierCV(cv="prefit"):
  - cv="prefit" was removed/changed across recent sklearn versions.
  - A manual sigmoid fit (LogisticRegression on the 1-D decision score)
    is exactly what Platt scaling is, has zero version risk, and is
    trivial to serialize as two floats (a, b).

Split discipline (avoids re-using v3's val set for two different jobs):
  1. Reproduce v3's exact train_s1 / val_s1 split (same random_state=42)
     so we never touch the entities the classifier was trained on.
  2. Split val_s1 again (50/50, random_state=7) into:
       - calib_s1: fit the Platt sigmoid (a, b)
       - eval_s1:  re-tune the threshold + define the ambiguous band
     This keeps calibration-fitting and threshold-picking on disjoint data.

Outputs (all in MODELS_DIR):
  - v4_platt.pkl        -> {"a": float, "b": float}
  - v4_threshold.txt    -> single best point threshold (post-calibration)
  - v4_band_low.txt     -> lower bound of ambiguous band
  - v4_band_high.txt    -> upper bound of ambiguous band
  - v4_reliability.tsv  -> bucketed predicted-prob vs actual positive rate
"""

import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import os, re, time, pickle
import numpy as np
import pandas as pd
import jellyfish
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# ---------------------------------------------------------------------------
# Paths (same convention as v3 / generate_v3_submission)
# ---------------------------------------------------------------------------
REPO_ROOT   = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
                os.path.abspath(__file__)))))
TRAIN_DIR   = os.path.join(REPO_ROOT, "dataset", "mini_train")
OUTPUT_DIR  = os.path.join(REPO_ROOT, "output")
MODELS_DIR  = os.path.join(REPO_ROOT, "models")

# Cap the ambiguous band so v7's Qwen jury stays within budget
# (VERSIONS.md target: < 20% of candidates)
MAX_BAND_FRACTION = 0.20

FEATURES = [
    "name_jw", "name_lev", "name_tfidf_cosine",
    "addr_jw", "addr_lev", "addr_tfidf_cosine",
    "num_overlap", "name_x_addr", "lookalike_flag"
]

# ---------------------------------------------------------------------------
# Feature helpers (identical to v3, kept local so this file is standalone)
# ---------------------------------------------------------------------------
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

def add_features(df, s1n, s1a, cn, ca, vec_name, vec_addr):
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
    return df

def macro_f05(pred_matches: dict, gt_map: dict, entity_ids) -> float:
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

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 60)
    print("v4 — calibration (Platt scaling + ambiguous band)")
    print("=" * 60)
    t0 = time.time()

    for fname in ["v3_classifier.pkl", "v3_vec_name.pkl", "v3_vec_addr.pkl"]:
        if not os.path.exists(os.path.join(MODELS_DIR, fname)):
            print(f"ERROR: {fname} missing. Run v3_classifier.py first.")
            return

    model    = pickle.load(open(os.path.join(MODELS_DIR, "v3_classifier.pkl"), "rb"))
    vec_name = pickle.load(open(os.path.join(MODELS_DIR, "v3_vec_name.pkl"),   "rb"))
    vec_addr = pickle.load(open(os.path.join(MODELS_DIR, "v3_vec_addr.pkl"),   "rb"))

    # 1. Reload data + rebuild the same labeled pairs v3 built
    print("\n[1/5] Reloading data and candidate pairs...")
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

    # 2. Reproduce v3's exact split, then split val further into calib/eval
    print("\n[2/5] Reproducing v3 split, carving out calib/eval from v3's val set...")
    unique_s1 = df_pairs["s1_id"].unique()
    train_s1, val_s1 = train_test_split(unique_s1, test_size=0.2, random_state=42)
    calib_s1, eval_s1 = train_test_split(val_s1, test_size=0.5, random_state=7)

    calib_df = df_pairs[df_pairs["s1_id"].isin(set(calib_s1))].copy()
    eval_df  = df_pairs[df_pairs["s1_id"].isin(set(eval_s1))].copy()
    print(f"  calib pairs: {len(calib_df):,}  |  eval pairs: {len(eval_df):,}")

    # 3. Build features (transform only — vectorizers are frozen from v3)
    print("\n[3/5] Building features for calib/eval...")
    s1_dict  = s1.set_index("entity_id")[["business_name", "business_address"]].to_dict("index")
    s23_dict = s23.set_index("entity_id")[["business_name", "business_address"]].to_dict("index")

    def get_texts(df):
        s1n = [str(s1_dict.get(i, {}).get("business_name",  "")).lower().strip() for i in df["s1_id"]]
        s1a = [str(s1_dict.get(i, {}).get("business_address","")).lower().strip() for i in df["s1_id"]]
        cn  = [str(s23_dict.get(i, {}).get("business_name",  "")).lower().strip() for i in df["cid"]]
        ca  = [str(s23_dict.get(i, {}).get("business_address","")).lower().strip() for i in df["cid"]]
        return s1n, s1a, cn, ca

    c_s1n, c_s1a, c_cn, c_ca = get_texts(calib_df)
    e_s1n, e_s1a, e_cn, e_ca = get_texts(eval_df)
    calib_df = add_features(calib_df, c_s1n, c_s1a, c_cn, c_ca, vec_name, vec_addr)
    eval_df  = add_features(eval_df,  e_s1n, e_s1a, e_cn, e_ca, vec_name, vec_addr)

    # 4. Manual Platt scaling: fit sigmoid(a * decision_score + b) on calib
    print("\n[4/5] Fitting Platt sigmoid on calib set...")
    z_calib = model.decision_function(calib_df[FEATURES])
    y_calib = calib_df["label"].values
    platt = LogisticRegression()
    platt.fit(z_calib.reshape(-1, 1), y_calib)
    a, b = float(platt.coef_[0][0]), float(platt.intercept_[0])
    print(f"  Platt params: a={a:.4f}, b={b:.4f}")

    def calibrated_prob(feat_df):
        z = model.decision_function(feat_df[FEATURES])
        return 1.0 / (1.0 + np.exp(-(a * z + b)))

    # 5. Threshold sweep + ambiguous band on EVAL (disjoint from calib)
    print("\n[5/5] Re-tuning threshold and defining ambiguous band on eval set...")
    eval_df = eval_df.copy()
    eval_df["prob"] = calibrated_prob(eval_df)

    def pred_matches_at(thresh):
        d = {}
        for row in eval_df.itertuples(index=False):
            d.setdefault(row.s1_id, set())
            if row.prob >= thresh:
                d[row.s1_id].add(row.cid)
        return d

    best_t, best_f05 = 0.5, -1.0
    for t in np.arange(0.05, 0.96, 0.02):
        f05 = macro_f05(pred_matches_at(t), gt_map, eval_s1)
        if f05 > best_f05:
            best_f05, best_t = f05, t
    print(f"  Best calibrated threshold: {best_t:.2f}  |  Eval F0.5: {best_f05:.4f}")

    # Ambiguous band: probability bins where pair-level accuracy is weak.
    # (accuracy here = fraction of pairs in that bin whose label matches
    #  the point decision at `best_t` — low agreement = genuinely unsure)
    print("\n  Scanning probability bins for ambiguous band...")
    bins = np.arange(0.0, 1.01, 0.05)
    eval_df["bin"] = pd.cut(eval_df["prob"], bins, include_lowest=True)
    reliability = (eval_df.groupby("bin", observed=True)
                   .agg(n=("label", "size"), pos_rate=("label", "mean"))
                   .reset_index())
    reliability.to_csv(os.path.join(MODELS_DIR, "v4_reliability.tsv"), sep="\t", index=False)
    print(reliability.to_string(index=False))

    # A bin is "ambiguous" if its positive rate isn't close to 0 or 1
    # (i.e. the calibrated score genuinely doesn't separate match/no-match)
    reliability["low_edge"]  = reliability["bin"].apply(lambda b: b.left)
    reliability["high_edge"] = reliability["bin"].apply(lambda b: b.right)
    ambiguous = reliability[(reliability["pos_rate"] > 0.15) & (reliability["pos_rate"] < 0.85)
                             & (reliability["n"] > 0)]

    if ambiguous.empty:
        # Classifier is already well-separated — fall back to a narrow
        # band around the threshold so v7 still has *something* to check.
        low_thresh, high_thresh = max(0.0, best_t - 0.10), min(1.0, best_t + 0.10)
    else:
        low_thresh  = float(ambiguous["low_edge"].min())
        high_thresh = float(ambiguous["high_edge"].max())

    # Enforce the <20% budget cap by shrinking symmetrically around best_t
    band_frac = ((eval_df["prob"] >= low_thresh) & (eval_df["prob"] <= high_thresh)).mean()
    while band_frac > MAX_BAND_FRACTION and (high_thresh - low_thresh) > 0.02:
        low_thresh  = min(low_thresh + 0.02, best_t)
        high_thresh = max(high_thresh - 0.02, best_t)
        band_frac = ((eval_df["prob"] >= low_thresh) & (eval_df["prob"] <= high_thresh)).mean()

    print(f"\n  Ambiguous band: [{low_thresh:.2f}, {high_thresh:.2f}]  "
          f"({band_frac:.1%} of eval candidates)")

    # Save everything
    pickle.dump({"a": a, "b": b}, open(os.path.join(MODELS_DIR, "v4_platt.pkl"), "wb"))
    with open(os.path.join(MODELS_DIR, "v4_threshold.txt"), "w") as f:
        f.write(str(best_t))
    with open(os.path.join(MODELS_DIR, "v4_band_low.txt"), "w") as f:
        f.write(str(low_thresh))
    with open(os.path.join(MODELS_DIR, "v4_band_high.txt"), "w") as f:
        f.write(str(high_thresh))

    print(f"\n[Done] v4 calibration finished in {time.time()-t0:.1f}s")
    print("  Saved: v4_platt.pkl, v4_threshold.txt, v4_band_low.txt, "
          "v4_band_high.txt, v4_reliability.tsv")

if __name__ == "__main__":
    main()
