"""
create_mini_dataset.py
======================
Creates a 5% miniature training set for rapid local development.
It samples S1, then guarantees that all true matches for those S1 entities
are included from S2 and S3, along with some random noise.
"""

import os
import pandas as pd
import numpy as np

# Paths
REPO_ROOT   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STUDENT_RES = os.path.join(REPO_ROOT, "6ab10eb3b23ba_student_resource", "student_resource")
TRAIN_DIR   = os.path.join(STUDENT_RES, "dataset", "train")
MINI_DIR    = os.path.join(STUDENT_RES, "dataset", "mini_train")

os.makedirs(MINI_DIR, exist_ok=True)

N_S1_SAMPLES = 100_000   # Take 100k out of 2.2M (approx 5%)

def main():
    print(f"Creating mini training set ({N_S1_SAMPLES:,} S1 entities)...")
    
    # 1. Sample S1
    print("Loading S1...")
    s1 = pd.read_csv(os.path.join(TRAIN_DIR, "train_source1.tsv"), sep="\t", dtype=str)
    s1_mini = s1.sample(n=N_S1_SAMPLES, random_state=42)
    s1_mini.to_csv(os.path.join(MINI_DIR, "train_source1.tsv"), sep="\t", index=False)
    
    # 2. Get Ground Truth for those S1s
    print("Filtering Ground Truth...")
    gt = pd.read_csv(os.path.join(TRAIN_DIR, "train_ground_truth.tsv"), sep="\t", dtype=str).fillna("")
    gt_mini = gt[gt["source1_entity_id"].isin(s1_mini["entity_id"])]
    gt_mini.to_csv(os.path.join(MINI_DIR, "train_ground_truth.tsv"), sep="\t", index=False)
    
    # Extract all S2/S3 IDs that are true matches for our sample
    required_s23_ids = set()
    for _, row in gt_mini.iterrows():
        matches = [m.strip() for m in row["matched_entity_ids"].split(",") if m.strip()]
        required_s23_ids.update(matches)
        
    print(f"Required S2/S3 true matches: {len(required_s23_ids):,}")
    
    # 3. Filter S2 and S3 (keep required matches + some random noise)
    for source in ["source2", "source3"]:
        print(f"Processing {source}...")
        df = pd.read_csv(os.path.join(TRAIN_DIR, f"train_{source}.tsv"), sep="\t", dtype=str)
        
        # Split into required and others
        mask_required = df["entity_id"].isin(required_s23_ids)
        df_required = df[mask_required]
        df_others = df[~mask_required]
        
        # Sample random noise (approx 150k per source)
        df_noise = df_others.sample(n=min(150_000, len(df_others)), random_state=42)
        
        # Combine
        df_mini = pd.concat([df_required, df_noise]).sample(frac=1, random_state=42) # shuffle
        df_mini.to_csv(os.path.join(MINI_DIR, f"train_{source}.tsv"), sep="\t", index=False)
        print(f"  Saved {len(df_mini):,} rows for {source}")

    print(f"\n✅ Mini dataset created at: {MINI_DIR}")

if __name__ == "__main__":
    main()
