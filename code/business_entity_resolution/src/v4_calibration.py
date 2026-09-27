"""
v4 — calibration
================
Applies Platt scaling (manual sigmoid fit on the classifier's raw score)
to a trained classifier, then re-tunes the merge threshold on a *fresh*
held-out split and defines the ambiguous band [low_thresh, high_thresh]
that v7's Qwen jury will operate on.

Model-agnostic (v3 LR or v6 XGBoost): pass --model-prefix to pick which
saved model to calibrate. Default is "v6" since that's the current best
classifier -- pass --model-prefix v3 if you want to calibrate the older
Logistic Regression model instead.

  python v4_calibration.py                  # calibrates v6 (XGBoost, 11 features)
  python v4_calibration.py --model-prefix v3  # calibrates v3 (Logistic Regression, 9 features)

IMPORTANT: you do NOT need to retrain the classifier to run this script.
It only loads the already-saved {prefix}_classifier.pkl and its
vectorizers and fits a small calibration layer on top -- XGBoost training
is untouched.

Raw score handling (this is the actual v3->v6 compatibility fix):
  - Logistic Regression exposes decision_function() (an unbounded linear
    margin) -- that's what v3's original version of this script used.
  - XGBClassifier does NOT implement decision_function() at all; calling
    it throws AttributeError. XGBoost only exposes predict_proba().
  - raw_score() below picks whichever the loaded model actually supports,
    so the exact same Platt-scaling logic works for either model. Platt
    scaling recalibrating an already-probabilistic predict_proba() output
    is standard practice (tree ensembles are frequently overconfident near
    0/1), not a hack.

Feature-set handling: v3 used 9 features, v5/v6 added 2 embedding cosine
features (11 total). This script detects which set the loaded model
expects via model.n_features_in_ and builds the matching features,
including loading the sentence-transformer embedder ONLY if the 11-feature
set is needed (skips that cost entirely when calibrating v3).

Why manual Platt scaling instead of CalibratedClassifierCV(cv="prefit"):
  - cv="prefit" was removed/changed across recent sklearn versions.
  - A manual sigmoid fit (LogisticRegression on the 1-D raw score) is
    exactly what Platt scaling is, has zero version risk, and is trivial
    to serialize as two floats (a, b).

Split discipline (avoids re-using the classifier's val set for two jobs):
  1. Reproduce the classifier's exact train_s1 / val_s1 split (same
     random_state=42 used by v3/v5/v6) so we never touch entities the
     classifier was trained on.
  2. Split val_s1 again (50/50, random_state=7) into:
       - calib_s1: fit the Platt sigmoid (a, b)
       - eval_s1:  re-tune the threshold + define the ambiguous band
     This keeps calibration-fitting and threshold-picking on disjoint data.

Outputs (all in MODELS_DIR, suffixed by --model-prefix so v3 and v6
calibration artifacts never clobber each other):
  - v4_platt_{prefix}.pkl        -> {"a": float, "b": float}
  - v4_threshold_{prefix}.txt    -> single best point threshold (post-calibration)
  - v4_band_low_{prefix}.txt     -> lower bound of ambiguous band
  - v4_band_high_{prefix}.txt    -> upper bound of ambiguous band
  - v4_reliability_{prefix}.tsv  -> bucketed predicted-prob vs actual positive rate

v7_qwen_jury.py defaults to reading the "v6" versions of these files.
"""

import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import os, re, time, pickle, argparse
import numpy as np
import pandas as pd
import jellyfish
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# ---------------------------------------------------------------------------
# Paths (same convention as v3/v5/v6)
# ---------------------------------------------------------------------------
REPO_ROOT   = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
                os.path.abspath(__file__)))))
TRAIN_DIR   = os.path.join(REPO_ROOT, "dataset", "mini_train")
OUTPUT_DIR  = os.path.join(REPO_ROOT, "output")
MODELS_DIR  = os.path.join(REPO_ROOT, "models")

EMBED_MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"
EMBED_BATCH_SIZE = 256

# Cap the ambiguous band so v7's Qwen jury stays within budget
# (VERSIONS.md target: < 20% of candidates)
MAX_BAND_FRACTION = 0.20

FEATURES_9 = [
    "name_jw", "name_lev", "name_tfidf_cosine",
    "addr_jw", "addr_lev", "addr_tfidf_cosine",
    "num_overlap", "name_x_addr", "lookalike_flag",
]
FEATURES_11 = FEATURES_9 + ["name_embed_cosine", "addr_embed_cosine"]

# ---------------------------------------------------------------------------
# Feature helpers
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

def cos_rows(a, b):
    num = (a * b).sum(axis=1)
    den = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
    den[den == 0] = 1e-9
    return num / den

def add_features(df, s1n, s1a, cn, ca, vec_name, vec_addr,
                  name2vec=None, addr2vec=None):
    """Builds FEATURES_9 always; additionally builds the two embedding
    cosine columns if name2vec/addr2vec caches are provided (i.e. the
    loaded model needs FEATURES_11)."""
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

    if name2vec is not None:
        v1n = np.stack([name2vec[t] for t in s1n]); v2n = np.stack([name2vec[t] for t in cn])
        v1a = np.stack([addr2vec[t] for t in s1a]); v2a = np.stack([addr2vec[t] for t in ca])
        df["name_embed_cosine"] = cos_rows(v1n, v2n)
        df["addr_embed_cosine"] = cos_rows(v1a, v2a)
    return df

def raw_score(model, X):
    """LR exposes decision_function(); XGBClassifier does not and only
    has predict_proba(). Use whichever the loaded model actually supports
    so the same Platt-scaling code works for both."""
    if hasattr(model, "decision_function"):
        return model.decision_function(X)
    return model.predict_proba(X)[:, 1]

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
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-prefix", default="v6",
                     help="Which saved model to calibrate: 'v6' (XGBoost, "
                          "default) or 'v3' (Logistic Regression).")
    args = ap.parse_args()
    prefix = args.model_prefix

    print("=" * 60)
    print(f"v4 — calibration (Platt scaling + ambiguous band) — model: {prefix}")
    print("=" * 60)
    t0 = time.time()

    required = [f"{prefix}_classifier.pkl", f"{prefix}_vec_name.pkl", f"{prefix}_vec_addr.pkl"]
    for fname in required:
        if not os.path.exists(os.path.join(MODELS_DIR, fname)):
            print(f"ERROR: {fname} missing. Run {prefix}_classifier.py / {prefix}_xgboost.py first.")
            return

    model    = pickle.load(open(os.path.join(MODELS_DIR, f"{prefix}_classifier.pkl"), "rb"))
    vec_name = pickle.load(open(os.path.join(MODELS_DIR, f"{prefix}_vec_name.pkl"),   "rb"))
    vec_addr = pickle.load(open(os.path.join(MODELS_DIR, f"{prefix}_vec_addr.pkl"),   "rb"))

    n_feat = model.n_features_in_
    if n_feat == 9:
        FEATURES, needs_embed = FEATURES_9, False
    elif n_feat == 11:
        FEATURES, needs_embed = FEATURES_11, True
    else:
        print(f"ERROR: loaded model expects {n_feat} features -- this script only "
              f"knows the 9-feature (v3) and 11-feature (v5/v6) sets. Update "
              f"FEATURES_9/FEATURES_11 above if you've changed the feature set.")
        return
    print(f"  Model expects {n_feat} features (using {'FEATURES_11' if needs_embed else 'FEATURES_9'})"
          f" -- {'has' if hasattr(model, 'decision_function') else 'no'} decision_function()")

    embedder = None
    if needs_embed:
        print(f"  Loading {EMBED_MODEL_NAME} (needed for embedding features)...")
        from sentence_transformers import SentenceTransformer
        embedder = SentenceTransformer(EMBED_MODEL_NAME)

    # 1. Reload data + rebuild the same labeled pairs the classifier was built from
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

    # 2. Reproduce the exact train/val split, then split val further into calib/eval
    print("\n[2/5] Reproducing train/val split, carving out calib/eval from the val set...")
    unique_s1 = df_pairs["s1_id"].unique()
    train_s1, val_s1 = train_test_split(unique_s1, test_size=0.2, random_state=42)
    calib_s1, eval_s1 = train_test_split(val_s1, test_size=0.5, random_state=7)

    calib_df = df_pairs[df_pairs["s1_id"].isin(set(calib_s1))].copy()
    eval_df  = df_pairs[df_pairs["s1_id"].isin(set(eval_s1))].copy()
    print(f"  calib pairs: {len(calib_df):,}  |  eval pairs: {len(eval_df):,}")

    # 3. Build features (transform only — vectorizers are frozen from training)
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

    name2vec = addr2vec = None
    if needs_embed:
        all_names = sorted(set(c_s1n + c_cn + e_s1n + e_cn))
        all_addrs = sorted(set(c_s1a + c_ca + e_s1a + e_ca))
        name_vecs = embedder.encode(all_names, batch_size=EMBED_BATCH_SIZE, show_progress_bar=True, convert_to_numpy=True)
        addr_vecs = embedder.encode(all_addrs, batch_size=EMBED_BATCH_SIZE, show_progress_bar=True, convert_to_numpy=True)
        name2vec, addr2vec = dict(zip(all_names, name_vecs)), dict(zip(all_addrs, addr_vecs))

    calib_df = add_features(calib_df, c_s1n, c_s1a, c_cn, c_ca, vec_name, vec_addr, name2vec, addr2vec)
    eval_df  = add_features(eval_df,  e_s1n, e_s1a, e_cn, e_ca, vec_name, vec_addr, name2vec, addr2vec)

    # 4. Manual Platt scaling: fit sigmoid(a * raw_score + b) on calib
    print("\n[4/5] Fitting Platt sigmoid on calib set...")
    z_calib = raw_score(model, calib_df[FEATURES])
    y_calib = calib_df["label"].values
    platt = LogisticRegression()
    platt.fit(z_calib.reshape(-1, 1), y_calib)
    a, b = float(platt.coef_[0][0]), float(platt.intercept_[0])
    print(f"  Platt params: a={a:.4f}, b={b:.4f}")

    def calibrated_prob(feat_df):
        z = raw_score(model, feat_df[FEATURES])
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

    # Ambiguous band: probability bins where pair-level positive rate is weak.
    print("\n  Scanning probability bins for ambiguous band...")
    bins = np.arange(0.0, 1.01, 0.05)
    eval_df["bin"] = pd.cut(eval_df["prob"], bins, include_lowest=True)
    reliability = (eval_df.groupby("bin", observed=True)
                   .agg(n=("label", "size"), pos_rate=("label", "mean"))
                   .reset_index())
    reliability.to_csv(os.path.join(MODELS_DIR, f"v4_reliability_{prefix}.tsv"), sep="\t", index=False)
    print(reliability.to_string(index=False))

    reliability["low_edge"]  = reliability["bin"].apply(lambda b: b.left)
    reliability["high_edge"] = reliability["bin"].apply(lambda b: b.right)
    ambiguous = reliability[(reliability["pos_rate"] > 0.15) & (reliability["pos_rate"] < 0.85)
                             & (reliability["n"] > 0)]

    if ambiguous.empty:
        low_thresh, high_thresh = max(0.0, best_t - 0.10), min(1.0, best_t + 0.10)
    else:
        low_thresh  = float(ambiguous["low_edge"].min())
        high_thresh = float(ambiguous["high_edge"].max())

    band_frac = ((eval_df["prob"] >= low_thresh) & (eval_df["prob"] <= high_thresh)).mean()
    while band_frac > MAX_BAND_FRACTION and (high_thresh - low_thresh) > 0.02:
        low_thresh  = min(low_thresh + 0.02, best_t)
        high_thresh = max(high_thresh - 0.02, best_t)
        band_frac = ((eval_df["prob"] >= low_thresh) & (eval_df["prob"] <= high_thresh)).mean()

    print(f"\n  Ambiguous band: [{low_thresh:.2f}, {high_thresh:.2f}]  "
          f"({band_frac:.1%} of eval candidates)")

    pickle.dump({"a": a, "b": b}, open(os.path.join(MODELS_DIR, f"v4_platt_{prefix}.pkl"), "wb"))
    with open(os.path.join(MODELS_DIR, f"v4_threshold_{prefix}.txt"), "w") as f:
        f.write(str(best_t))
    with open(os.path.join(MODELS_DIR, f"v4_band_low_{prefix}.txt"), "w") as f:
        f.write(str(low_thresh))
    with open(os.path.join(MODELS_DIR, f"v4_band_high_{prefix}.txt"), "w") as f:
        f.write(str(high_thresh))

    print(f"\n[Done] v4 calibration ({prefix}) finished in {time.time()-t0:.1f}s")
    print(f"  Saved: v4_platt_{prefix}.pkl, v4_threshold_{prefix}.txt, "
          f"v4_band_low_{prefix}.txt, v4_band_high_{prefix}.txt, v4_reliability_{prefix}.tsv")

if __name__ == "__main__":
    main()