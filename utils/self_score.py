"""
self_score.py — Fast F0.5 self-scorer on training data
=======================================================
Runs the EXACT SAME exact-match blocking as v1_baseline but using
fast pandas merge (not row-by-row loops), then computes macro-averaged
F0.5 against train_ground_truth.tsv and prints a detailed breakdown.

Usage:
    python utils/self_score.py

Output printed to console:
  - Overall macro F0.5
  - Precision, Recall breakdown
  - Singleton accuracy
  - False positive / false negative counts
"""

import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import os
import pandas as pd
import numpy as np

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_ROOT   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STUDENT_RES = os.path.join(REPO_ROOT, "6ab10eb3b23ba_student_resource", "student_resource")
TRAIN_DIR   = os.path.join(STUDENT_RES, "dataset", "train")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def sep(char="=", n=60): print(char * n)

def normalise(series: pd.Series) -> pd.Series:
    """Lowercase + strip — same normalisation as v1_baseline."""
    return series.str.lower().str.strip()

def f05(precision: float, recall: float) -> float:
    """F0.5 = (1 + 0.25) * P * R / (0.25 * P + R)"""
    if precision + recall == 0:
        return 0.0
    return 1.25 * precision * recall / (0.25 * precision + recall)

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
def load():
    sep()
    print("Loading training data...")
    s1 = pd.read_csv(os.path.join(TRAIN_DIR, "train_source1.tsv"), sep="\t", dtype=str).fillna("")
    s2 = pd.read_csv(os.path.join(TRAIN_DIR, "train_source2.tsv"), sep="\t", dtype=str).fillna("")
    s3 = pd.read_csv(os.path.join(TRAIN_DIR, "train_source3.tsv"), sep="\t", dtype=str).fillna("")
    gt = pd.read_csv(os.path.join(TRAIN_DIR, "train_ground_truth.tsv"), sep="\t", dtype=str).fillna("")
    print(f"  S1: {len(s1):,}  S2: {len(s2):,}  S3: {len(s3):,}  GT: {len(gt):,}")
    return s1, s2, s3, gt

# ---------------------------------------------------------------------------
# Exact-match blocking (vectorised)
# ---------------------------------------------------------------------------
def exact_match_candidates(s1, s2, s3) -> pd.DataFrame:
    """
    Returns DataFrame with columns [source1_entity_id, candidate_entity_id]
    matched on (norm_name, country).
    Uses pandas merge — fast even on 2M+ rows.
    """
    print("\nBuilding exact-match candidates (vectorised)...")

    # Normalise
    for df in [s1, s2, s3]:
        df["_key"] = normalise(df["business_name"]) + "|" + df["country"].str.strip()

    s23 = pd.concat([
        s2[["entity_id", "_key"]].rename(columns={"entity_id": "candidate_entity_id"}),
        s3[["entity_id", "_key"]].rename(columns={"entity_id": "candidate_entity_id"}),
    ], ignore_index=True)

    candidates = s1[["entity_id", "_key"]].rename(
        columns={"entity_id": "source1_entity_id"}
    ).merge(s23, on="_key", how="left").drop(columns="_key")

    # Count stats
    matched_s1 = candidates.dropna(subset=["candidate_entity_id"])
    print(f"  Total candidate pairs : {len(matched_s1):,}")
    print(f"  S1 entities with ≥1 candidate: "
          f"{matched_s1['source1_entity_id'].nunique():,} / {len(s1):,}")
    return candidates

# ---------------------------------------------------------------------------
# Self-score against ground truth
# ---------------------------------------------------------------------------
def score(candidates: pd.DataFrame, gt: pd.DataFrame):
    print("\nScoring against ground truth...")

    # Build prediction dict: s1_id -> set of predicted matches
    pred = (
        candidates.dropna(subset=["candidate_entity_id"])
        .groupby("source1_entity_id")["candidate_entity_id"]
        .apply(set)
        .to_dict()
    )

    # Build ground truth dict: s1_id -> set of true matches
    def parse_gt(x):
        return set(x.split(",")) - {""} if x.strip() else set()

    gt_dict = {
        row["source1_entity_id"]: parse_gt(row["matched_entity_ids"])
        for _, row in gt.iterrows()
    }

    # Per-entity F0.5
    f_scores, precisions, recalls = [], [], []
    singletons_correct = singletons_wrong = 0
    total_tp = total_fp = total_fn = 0

    for s1_id, true_set in gt_dict.items():
        pred_set = pred.get(s1_id, set())

        # Singleton handling
        if not true_set:
            if not pred_set:
                f_scores.append(1.0)
                singletons_correct += 1
            else:
                f_scores.append(0.0)
                singletons_wrong += 1
            precisions.append(1.0 if not pred_set else 0.0)
            recalls.append(1.0)
            continue

        tp = len(true_set & pred_set)
        fp = len(pred_set - true_set)
        fn = len(true_set - pred_set)
        total_tp += tp; total_fp += fp; total_fn += fn

        p = tp / len(pred_set) if pred_set else 0.0
        r = tp / len(true_set)
        precisions.append(p)
        recalls.append(r)
        f_scores.append(f05(p, r))

    # Aggregate
    macro_f05  = np.mean(f_scores)
    macro_p    = np.mean(precisions)
    macro_r    = np.mean(recalls)
    micro_p    = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    micro_r    = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0

    sep()
    print("RESULTS — v1 Exact-Match Baseline")
    sep()
    print(f"  Macro F0.5           : {macro_f05:.4f}  ← leaderboard metric")
    print(f"  Macro Precision      : {macro_p:.4f}")
    print(f"  Macro Recall         : {macro_r:.4f}")
    sep("-")
    print(f"  Micro Precision      : {micro_p:.4f}  (global TP/predicted)")
    print(f"  Micro Recall         : {micro_r:.4f}  (global TP/actual)")
    sep("-")
    print(f"  Total TP             : {total_tp:,}")
    print(f"  Total FP (false merge): {total_fp:,}")
    print(f"  Total FN (missed)    : {total_fn:,}")
    sep("-")
    n_singletons = singletons_correct + singletons_wrong
    print(f"  Singletons correct   : {singletons_correct:,} / {n_singletons:,} "
          f"({singletons_correct/n_singletons:.1%})" if n_singletons else "  No singletons found")
    print(f"  Singletons wrong     : {singletons_wrong:,}  (each scores 0.0)")
    sep()
    print("Interpretation:")
    if macro_r < 0.5:
        print("  ⚠ Low recall — exact match misses most true matches (expected for v1).")
        print("    v2 blocking (TF-IDF + fuzzy) will fix this.")
    if macro_p > 0.9:
        print("  ✓ High precision — exact matches are almost always correct.")
    print(f"\n  This is the v1 BASELINE score. Target for v3+: F0.5 > 0.7")

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    s1, s2, s3, gt = load()
    candidates = exact_match_candidates(s1, s2, s3)
    score(candidates, gt)

if __name__ == "__main__":
    main()
