"""
create_mini_train.py — Fast mini_train creator from real train data
===================================================================
Samples 100,000 S1 entities + all their GT matches from S2/S3 + random negatives.
Designed to run in ~2 minutes on a laptop with the full train data.
"""

import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import os, time
import pandas as pd
import numpy as np

REPO_ROOT  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN_DIR  = os.path.join(REPO_ROOT, "dataset", "train")
MINI_DIR   = os.path.join(REPO_ROOT, "dataset", "mini_train")
SAMPLE_N   = 100_000
SEED       = 42

os.makedirs(MINI_DIR, exist_ok=True)

def main():
    t0 = time.time()
    print("Loading full train data (this takes ~30s)...")

    s1 = pd.read_csv(os.path.join(TRAIN_DIR, "train_source1.tsv"), sep="\t", dtype=str).fillna("")
    gt = pd.read_csv(os.path.join(TRAIN_DIR, "train_ground_truth.tsv"), sep="\t", dtype=str).fillna("")
    s2 = pd.read_csv(os.path.join(TRAIN_DIR, "train_source2.tsv"), sep="\t", dtype=str).fillna("")
    s3 = pd.read_csv(os.path.join(TRAIN_DIR, "train_source3.tsv"), sep="\t", dtype=str).fillna("")

    print(f"  S1: {len(s1):,} | S2: {len(s2):,} | S3: {len(s3):,} | GT: {len(gt):,}")

    # Sample S1
    sample_n = min(SAMPLE_N, len(s1))
    s1_sample = s1.sample(n=sample_n, random_state=SEED).reset_index(drop=True)
    s1_ids = set(s1_sample["entity_id"])

    # Get GT for sampled S1s
    gt_sample = gt[gt["source1_entity_id"].isin(s1_ids)].reset_index(drop=True)

    # Collect all S2/S3 IDs referenced in ground truth
    needed_ids = set()
    for row in gt_sample.itertuples():
        if row.matched_entity_ids:
            for mid in row.matched_entity_ids.split(","):
                if mid: needed_ids.add(mid)

    needed_s2 = needed_ids & set(s2["entity_id"])
    needed_s3 = needed_ids & set(s3["entity_id"])

    # Include true matches + random noise records
    noise_s2 = s2[~s2["entity_id"].isin(needed_s2)].sample(n=min(50000, len(s2)), random_state=SEED)
    noise_s3 = s3[~s3["entity_id"].isin(needed_s3)].sample(n=min(50000, len(s3)), random_state=SEED)

    s2_sample = pd.concat([s2[s2["entity_id"].isin(needed_s2)], noise_s2]).drop_duplicates("entity_id").reset_index(drop=True)
    s3_sample = pd.concat([s3[s3["entity_id"].isin(needed_s3)], noise_s3]).drop_duplicates("entity_id").reset_index(drop=True)

    # Save
    s1_sample.to_csv(os.path.join(MINI_DIR, "train_source1.tsv"), sep="\t", index=False)
    s2_sample.to_csv(os.path.join(MINI_DIR, "train_source2.tsv"), sep="\t", index=False)
    s3_sample.to_csv(os.path.join(MINI_DIR, "train_source3.tsv"), sep="\t", index=False)
    gt_sample.to_csv(os.path.join(MINI_DIR, "train_ground_truth.tsv"), sep="\t", index=False)

    print(f"\nSaved mini_train:")
    print(f"  S1:  {len(s1_sample):,}")
    print(f"  S2:  {len(s2_sample):,}  ({len(needed_s2):,} true matches + noise)")
    print(f"  S3:  {len(s3_sample):,}  ({len(needed_s3):,} true matches + noise)")
    print(f"  GT:  {len(gt_sample):,}")
    print(f"  Time: {time.time()-t0:.0f}s")

if __name__ == "__main__":
    main()
