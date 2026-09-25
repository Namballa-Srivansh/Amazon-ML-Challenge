"""
explore_data.py — quick data exploration script
================================================
Run this once after the dataset lands to understand:
  - Row counts per source
  - Column dtypes and null rates
  - Country distribution
  - Ground truth match rate (singletons vs matched)
  - Sample rows for sanity check

Usage:
    python utils/explore_data.py
"""

import sys, io
# Force UTF-8 output — needed on Windows where default stdout is cp1252
# and Indian/French business names cause UnicodeEncodeError
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import os
import pandas as pd

REPO_ROOT   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # c:\Projects\amazon-ml-challenge
STUDENT_RES = os.path.join(REPO_ROOT, "6ab10eb3b23ba_student_resource", "student_resource")
TRAIN_DIR   = os.path.join(STUDENT_RES, "dataset", "train")
TEST_DIR    = os.path.join(STUDENT_RES, "dataset", "test")


def sep():
    print("-" * 60)


def explore_source(path: str, label: str) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t", dtype=str).fillna("")
    sep()
    print(f"{label}  —  {len(df):,} rows")
    print(f"  Columns : {list(df.columns)}")
    print(f"  Countries: {df['country'].value_counts().to_dict()}")
    null_rates = (df == "").mean().round(3)
    print(f"  Empty rates:\n{null_rates.to_string()}")
    print(f"\n  Sample (first 3 rows):")
    print(df.head(3).to_string(index=False))
    return df


def explore_ground_truth(path: str, s1: pd.DataFrame):
    gt = pd.read_csv(path, sep="\t", dtype=str).fillna("")
    sep()
    print(f"Ground truth  —  {len(gt):,} rows")
    gt["n_matches"] = gt["matched_entity_ids"].apply(
        lambda x: len(x.split(",")) if x.strip() else 0)
    singletons = (gt["n_matches"] == 0).sum()
    print(f"  Singletons (no match) : {singletons:,} ({singletons/len(gt):.1%})")
    print(f"  Has ≥1 match          : {len(gt)-singletons:,} ({(len(gt)-singletons)/len(gt):.1%})")
    print(f"  Max matches for one S1: {gt['n_matches'].max()}")
    print(f"  Avg matches (non-zero): {gt[gt['n_matches']>0]['n_matches'].mean():.2f}")
    print(f"\n  Match count distribution:")
    print(gt["n_matches"].value_counts().sort_index().head(10).to_string())


def main():
    print("=" * 60)
    print("Data Exploration — Amazon ML Challenge 2026")
    print("=" * 60)

    # Training sources
    s1_train = explore_source(os.path.join(TRAIN_DIR, "train_source1.tsv"), "Train Source 1")
    s2_train = explore_source(os.path.join(TRAIN_DIR, "train_source2.tsv"), "Train Source 2")
    s3_train = explore_source(os.path.join(TRAIN_DIR, "train_source3.tsv"), "Train Source 3")
    explore_ground_truth(os.path.join(TRAIN_DIR, "train_ground_truth.tsv"), s1_train)

    # Test sources
    sep()
    print("TEST SET")
    explore_source(os.path.join(TEST_DIR, "test_source1.tsv"), "Test Source 1")
    explore_source(os.path.join(TEST_DIR, "test_source2.tsv"), "Test Source 2")
    explore_source(os.path.join(TEST_DIR, "test_source3.tsv"), "Test Source 3")

    sep()
    print("Done. Key things to note:")
    print("  1. Country distribution in test — does France appear?")
    print("  2. Null rates — are any fields missing at high rates?")
    print("  3. Max/avg matches — how many candidates should K cover?")


if __name__ == "__main__":
    main()
