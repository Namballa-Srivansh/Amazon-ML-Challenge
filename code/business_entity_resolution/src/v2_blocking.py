"""
v2 — blocking (Chunked S1, Colab-safe, Fast)
============================================
Key optimisations vs previous version:
  - S23 TF-IDF matrix pre-computed ONCE per country (not per chunk)
  - S23 entity_id → row index map pre-built ONCE per country
  - Chunk size tuned to stay within 12GB Colab RAM
  - No row-by-row reindex (was the 1000s bottleneck)
"""

import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import os, re, time, pickle
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import jellyfish

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_ROOT   = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
                os.path.abspath(__file__)))))
TRAIN_DIR   = os.path.join(REPO_ROOT, "dataset", "mini_train")
OUTPUT_DIR  = os.path.join(REPO_ROOT, "output")
MODELS_DIR  = os.path.join(REPO_ROOT, "models")

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

K        = 50
S1_CHUNK = 5_000
MAX_PAIRS_PER_KEY = 500_000

# ---------------------------------------------------------------------------
# Text helpers
# ---------------------------------------------------------------------------
_PUNCT_RE = re.compile(r'[^\w\s]')
_SPACE_RE = re.compile(r'\s+')

def norm(text):
    return _SPACE_RE.sub(' ', _PUNCT_RE.sub(' ', str(text).lower().strip())).strip()

def get_keys(nn, na):
    nw, aw = nn.split(), na.split()
    keys = set()
    if len(nw) >= 2:
        acr = "".join(w[0] for w in nw if w)
        if len(acr) >= 2: keys.add(f"acr:{acr}")
    for w in nw + aw:
        if len(w) >= 3: keys.add(f"w:{w}")
        if len(w) >= 5: keys.add(f"s:{jellyfish.soundex(w)}")
    return list(keys)

def add_cols(df):
    df = df.copy()
    df["nn"] = df["business_name"].apply(norm)
    df["na"] = df["business_address"].fillna("").apply(norm)
    df["tfidf_text"] = df["nn"] + " " + df["na"]
    return df

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("="*60)
    print("v2 — Blocking (Chunked S1, Fast)")
    print("="*60)
    t_start = time.time()

    # 1. Load
    print("\n[1/3] Loading mini_train...")
    s1  = add_cols(pd.read_csv(os.path.join(TRAIN_DIR,"train_source1.tsv"), sep="\t", dtype=str).fillna(""))
    s23 = add_cols(pd.concat([
        pd.read_csv(os.path.join(TRAIN_DIR,"train_source2.tsv"), sep="\t", dtype=str).fillna(""),
        pd.read_csv(os.path.join(TRAIN_DIR,"train_source3.tsv"), sep="\t", dtype=str).fillna(""),
    ], ignore_index=True))
    print(f"  S1: {len(s1):,}  |  S23: {len(s23):,}")

    # 2. Fit vectorizer
    print("\n[2/3] Fitting TF-IDF vectorizer...")
    vec = TfidfVectorizer(analyzer="word", ngram_range=(1,2), max_features=100_000, sublinear_tf=True)
    vec.fit(pd.concat([s1["tfidf_text"], s23["tfidf_text"]]))
    pickle.dump(vec, open(os.path.join(MODELS_DIR,"tfidf_vectorizer.pkl"),"wb"))
    print("  Saved models/tfidf_vectorizer.pkl")

    # 3. Per-country chunked blocking
    print("\n[3/3] Chunked blocking per country...")
    all_cands = {}

    for country in s1["country"].unique():
        c_s1  = s1[s1["country"]  == country].reset_index(drop=True)
        c_s23 = s23[s23["country"] == country].reset_index(drop=True)
        if c_s1.empty or c_s23.empty:
            continue

        t_c = time.time()
        print(f"\n  [{country}] S1={len(c_s1):,}, S23={len(c_s23):,}")

        # Pre-compute S23 keys ONCE for this country
        c_s23["bkeys"] = c_s23.apply(lambda r: get_keys(r["nn"], r["na"]), axis=1)
        s23_key_df = (c_s23[["entity_id","bkeys"]].explode("bkeys")
                        .dropna(subset=["bkeys"])
                        .rename(columns={"bkeys":"bkey"}))
        s23_key_df = s23_key_df[s23_key_df["bkey"].str.len() > 3]

        # Pre-compute S23 TF-IDF matrix ONCE for this country
        print(f"    Pre-computing S23 TF-IDF ({len(c_s23):,} rows)...")
        s23_vecs = vec.transform(c_s23["tfidf_text"])         # sparse matrix
        s23_id2idx = {eid: i for i, eid in enumerate(c_s23["entity_id"])}

        n_chunks = (len(c_s1) + S1_CHUNK - 1) // S1_CHUNK

        for ci, start in enumerate(range(0, len(c_s1), S1_CHUNK)):
            chunk = c_s1.iloc[start:start+S1_CHUNK].copy()
            t1 = time.time()

            # Build S1 keys for this chunk
            chunk["bkeys"] = chunk.apply(lambda r: get_keys(r["nn"], r["na"]), axis=1)
            s1_exp = (chunk[["entity_id","bkeys"]].explode("bkeys")
                        .dropna(subset=["bkeys"])
                        .rename(columns={"bkeys":"bkey"}))
            s1_exp = s1_exp[s1_exp["bkey"].str.len() > 3]

            if s1_exp.empty:
                continue

            valid_s1_keys = set(s1_exp["bkey"].unique())
            s23_rel = s23_key_df[s23_key_df["bkey"].isin(valid_s1_keys)]

            if s23_rel.empty:
                continue

            # Hot-key cap
            c1_cnt  = s1_exp["bkey"].value_counts()
            c23_cnt = s23_rel["bkey"].value_counts()
            safe = set(k for k, n1 in c1_cnt.items()
                        if n1 * c23_cnt.get(k, 0) <= MAX_PAIRS_PER_KEY)

            pairs = (s1_exp[s1_exp["bkey"].isin(safe)]
                     .merge(s23_rel[s23_rel["bkey"].isin(safe)]
                            .rename(columns={"entity_id":"cid"}), on="bkey")
                     [["entity_id","cid"]].drop_duplicates())

            if pairs.empty:
                continue

            # TF-IDF cosine — use pre-built S23 matrix, build S1 matrix on-the-fly
            s1id2row = {eid: i for i, eid in enumerate(chunk["entity_id"])}
            s1_vecs_full = vec.transform(chunk["tfidf_text"])   # small: 5k rows

            s1_rows  = [s1id2row[eid] for eid in pairs["entity_id"]]
            s23_rows = [s23_id2idx[cid] for cid in pairs["cid"]]

            v1  = s1_vecs_full[s1_rows]
            v23 = s23_vecs[s23_rows]
            pairs["score"] = np.array(v1.multiply(v23).sum(axis=1)).flatten()

            # Keep top-K per S1 entity
            pairs = (pairs.sort_values(["entity_id","score"], ascending=[True,False])
                          .groupby("entity_id").head(K))

            for row in pairs.itertuples(index=False):
                all_cands.setdefault(row.entity_id, {})[row.cid] = row.score

            n_m = sum(1 for eid in chunk["entity_id"] if eid in all_cands)
            print(f"    chunk {ci+1}/{n_chunks} | {len(pairs):,} pairs | {n_m:,} matched | {time.time()-t1:.1f}s")

        print(f"  [{country}] done in {time.time()-t_c:.1f}s")

    # Score recall
    print("\nScoring blocking recall...")
    gt = pd.read_csv(os.path.join(TRAIN_DIR,"train_ground_truth.tsv"), sep="\t", dtype=str).fillna("")
    found = total = 0
    for row in gt.itertuples(index=False):
        gt_ids = set(row.matched_entity_ids.split(",")) - {""}
        if not gt_ids: continue
        total += len(gt_ids)
        found += len(gt_ids & set(all_cands.get(row.source1_entity_id, {}).keys()))
    recall = found/total if total else 0
    print(f"\n  Recall: {found:,}/{total:,} = {recall:.4f} ({recall:.1%})")
    print("  GATE PASSED" if recall >= 0.95 else "  GATE FAILED")

    # Save
    print("\nSaving candidate_pairs.tsv...")
    rows = [{"source1_entity_id": sid,
             "candidate_entity_ids": ",".join(all_cands.get(sid,{}).keys())}
            for sid in s1["entity_id"]]
    pd.DataFrame(rows).to_csv(os.path.join(OUTPUT_DIR,"candidate_pairs.tsv"), sep="\t", index=False)
    print(f"  Saved {len(rows):,} rows.")
    print(f"\n[Done] {time.time()-t_start:.1f}s total")

if __name__ == "__main__":
    main()
