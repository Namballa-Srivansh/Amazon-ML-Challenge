"""
v1 — baseline
=============
Dumb exact-match pipeline.
Goal: validate file format and get a SCORED status on the leaderboard.
No real ML. Score doesn't matter yet.

Usage:
    python code/business_entity_resolution/src/v1_baseline.py

Output:
    output/matching_results.tsv
    output/candidate_pairs.tsv
"""
from __future__ import annotations   # Python 3.8 compat for dict[str, ...] hints

import os
import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import pandas as pd

# ---------------------------------------------------------------------------
# Paths — pointing at student_resource (the official Amazon dataset folder)
# ---------------------------------------------------------------------------
REPO_ROOT    = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
                 os.path.abspath(__file__)))))          # c:\Projects\amazon-ml-challenge
STUDENT_RES  = os.path.join(REPO_ROOT, "6ab10eb3b23ba_student_resource", "student_resource")

TRAIN_DIR    = os.path.join(STUDENT_RES, "dataset", "train")
TEST_DIR     = os.path.join(STUDENT_RES, "dataset", "test")
OUTPUT_DIR   = os.path.join(STUDENT_RES, "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
def load_sources(data_dir: str, split: str):
    """Load source1/2/3 TSVs for a given split (train or test)."""
    dfs = {}
    for src in [1, 2, 3]:
        path = os.path.join(data_dir, f"{split}_source{src}.tsv")
        df = pd.read_csv(path, sep="\t", dtype=str).fillna("")
        dfs[src] = df
        print(f"  Loaded {split}_source{src}.tsv — {len(df):,} rows")
    return dfs


def load_ground_truth(train_dir: str) -> pd.DataFrame:
    path = os.path.join(train_dir, "train_ground_truth.tsv")
    gt = pd.read_csv(path, sep="\t", dtype=str).fillna("")
    print(f"  Loaded train_ground_truth.tsv — {len(gt):,} rows")
    return gt


# ---------------------------------------------------------------------------
# Normalise (very lightweight for v1 — just lowercase + strip)
# ---------------------------------------------------------------------------
def normalise(text: str) -> str:
    return text.lower().strip()


# ---------------------------------------------------------------------------
# Blocking — exact match on (normalised business_name, country)
# ---------------------------------------------------------------------------
def build_candidates(s1: pd.DataFrame,
                     s2: pd.DataFrame,
                     s3: pd.DataFrame) -> dict[str, list[str]]:
    """
    For every S1 entity, find S2/S3 records whose normalised business_name
    AND country match exactly. Skips records with empty business names.

    Returns: {source1_entity_id: [candidate_entity_id, ...]}
    """
    # Build lookup: (norm_name, country) -> list of entity_ids
    # Use itertuples for speed (avoids per-row Series overhead of iterrows)
    lookup: dict[tuple, list[str]] = {}
    for src_df in [s2, s3]:
        for row in src_df.itertuples(index=False):
            norm = normalise(row.business_name)
            if not norm:          # skip records with empty business names
                continue
            key = (norm, row.country.strip())
            lookup.setdefault(key, []).append(row.entity_id)

    candidates: dict[str, list[str]] = {}
    for row in s1.itertuples(index=False):
        eid  = row.entity_id
        norm = normalise(row.business_name)
        if not norm:              # empty name → singleton (no candidates)
            candidates[eid] = []
            continue
        key  = (norm, row.country.strip())
        hits = lookup.get(key, [])
        # Deduplicate (shouldn't happen, but be safe)
        candidates[eid] = list(dict.fromkeys(hits))

    matched    = sum(1 for v in candidates.values() if v)
    total      = len(candidates)
    print(f"  Blocking: {matched}/{total} S1 entities have ≥1 candidate")
    return candidates


# ---------------------------------------------------------------------------
# Write outputs
# ---------------------------------------------------------------------------
def write_tsv(data: dict[str, list[str]], path: str,
              id_col: str, match_col: str) -> None:
    rows = []
    for s1_id, matches in data.items():
        rows.append({
            id_col:    s1_id,
            match_col: ",".join(matches),   # empty string when no matches
        })
    df = pd.DataFrame(rows, columns=[id_col, match_col])
    df.to_csv(path, sep="\t", index=False)
    print(f"  Wrote {path}  ({len(df):,} rows)")


# ---------------------------------------------------------------------------
# Self-score on training data (optional sanity check)
# ---------------------------------------------------------------------------
def self_score(candidates: dict[str, list[str]],
               gt: pd.DataFrame) -> float:
    """
    Compute macro-averaged F0.5 on the training ground truth.
    F0.5 = (1 + 0.5^2) * P * R / (0.5^2 * P + R)
         = 1.25 * P * R / (0.25 * P + R)
    """
    beta_sq = 0.25   # beta = 0.5
    scores  = []

    for row in gt.itertuples(index=False):
        s1_id     = row.source1_entity_id
        gt_str    = getattr(row, "matched_entity_ids", "")
        gt_set    = set(gt_str.split(",")) - {""} if gt_str else set()
        pred_set  = set(candidates.get(s1_id, []))

        if not gt_set and not pred_set:
            scores.append(1.0)
            continue

        tp = len(gt_set & pred_set)
        p  = tp / len(pred_set) if pred_set else 0.0
        r  = tp / len(gt_set)   if gt_set   else 0.0

        if p + r == 0:
            scores.append(0.0)
        else:
            f = (1 + beta_sq) * p * r / (beta_sq * p + r)
            scores.append(f)

    macro = sum(scores) / len(scores) if scores else 0.0
    return macro


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 60)
    print("v1 — baseline (exact-match blocking)")
    print("=" * 60)

    # --- Training data self-check ---
    print("\n[Train] Loading training data...")
    train_dfs = load_sources(TRAIN_DIR, "train")
    gt        = load_ground_truth(TRAIN_DIR)

    print("\n[Train] Building candidates (exact-match)...")
    train_candidates = build_candidates(
        train_dfs[1], train_dfs[2], train_dfs[3])

    print("\n[Train] Self-scoring on ground truth...")
    train_f05 = self_score(train_candidates, gt)
    print(f"  Train F0.5 (macro) = {train_f05:.4f}")

    # --- Test data ---
    print("\n[Test] Loading test data...")
    test_dfs = load_sources(TEST_DIR, "test")

    print("\n[Test] Building candidates (exact-match)...")
    test_candidates = build_candidates(
        test_dfs[1], test_dfs[2], test_dfs[3])

    # --- Write outputs ---
    print("\n[Output] Writing submission files...")
    write_tsv(
        test_candidates,
        os.path.join(OUTPUT_DIR, "matching_results.tsv"),
        id_col="source1_entity_id",
        match_col="matched_entity_ids",
    )
    write_tsv(
        test_candidates,
        os.path.join(OUTPUT_DIR, "candidate_pairs.tsv"),
        id_col="source1_entity_id",
        match_col="candidate_entity_ids",
    )

    print("\n[Done] Run Amazon's official validator next (from student_resource/ dir):")
    print("  cd 6ab10eb3b23ba_student_resource/student_resource")
    print("  python utils/validate_submission.py \\")
    print("      --matching output/matching_results.tsv \\")
    print("      --candidate output/candidate_pairs.tsv \\")
    print("      --test-dir dataset/test")
    print(f"\n  Train self-score F0.5 = {train_f05:.4f}  (v1 baseline)")


if __name__ == "__main__":
    main()
