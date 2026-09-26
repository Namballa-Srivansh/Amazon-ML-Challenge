"""
v4_fast_inference.py — The Sparse Dot-Product Breakthrough
==========================================================
Bypasses Pandas merges and SciPy row extractions entirely.
Uses C++ accelerated linear algebra (`matrix.dot()`) to score
3.8 million candidates instantly.
"""

import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import os, re, time, pickle
import numpy as np
import pandas as pd
import jellyfish
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_ROOT   = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
TEST_DIR    = os.path.join(REPO_ROOT, "dataset", "test")
OUTPUT_DIR  = os.path.join(REPO_ROOT, "output")
MODELS_DIR  = os.path.join(REPO_ROOT, "models")

K             = 50
S1_CHUNK_SIZE = 5_000

# ---------------------------------------------------------------------------
# Text helpers
# ---------------------------------------------------------------------------
_PUNCT_RE = re.compile(r'[^\w\s]')
_SPACE_RE = re.compile(r'\s+')

def norm(text):
    return _SPACE_RE.sub(' ', _PUNCT_RE.sub(' ', str(text).lower().strip())).strip()

def get_keys(nn, na):
    nw, aw = nn.split(), na.split()
    keys = []
    if len(nw) >= 2:
        acr = "".join(w[0] for w in nw if w)
        if len(acr) >= 2: keys.append(f"acr:{acr}")
    for w in nw + aw:
        if len(w) >= 3: keys.append(f"w:{w}")
        if len(w) >= 5: keys.append(f"s:{jellyfish.soundex(w)}")
    return list(set(keys))

def safe_lev(s1, s2):
    if not s1 and not s2: return 1.0
    if not s1 or not s2:  return 0.0
    d = jellyfish.levenshtein_distance(s1, s2)
    m = max(len(s1), len(s2))
    return 1.0 - (d / m) if m > 0 else 0.0

def safe_jw(s1, s2):
    if not s1 and not s2: return 1.0
    if not s1 or not s2:  return 0.0
    return jellyfish.jaro_winkler_similarity(s1, s2)

def num_overlap(s1, s2):
    t1 = set(re.findall(r'\d+', str(s1)))
    t2 = set(re.findall(r'\d+', str(s2)))
    if not t1 or not t2: return 0.0
    return len(t1 & t2) / len(t1 | t2)

FEATURES = [
    "name_jw", "name_lev", "name_tfidf_cosine",
    "addr_jw", "addr_lev", "addr_tfidf_cosine",
    "num_overlap", "name_x_addr", "lookalike_flag"
]

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 60)
    print("v4 — SPARSE DOT-PRODUCT INFERENCE (Ultra-Fast)")
    print("=" * 60)

    model     = pickle.load(open(os.path.join(MODELS_DIR, "v3_classifier.pkl"), "rb"))
    vec_name  = pickle.load(open(os.path.join(MODELS_DIR, "v3_vec_name.pkl"),   "rb"))
    vec_addr  = pickle.load(open(os.path.join(MODELS_DIR, "v3_vec_addr.pkl"),   "rb"))
    threshold = float(open(os.path.join(MODELS_DIR, "v3_threshold.txt")).read().strip())
    print(f"Loaded model | threshold = {threshold:.2f}")

    print("\nLoading dataset...")
    s1  = pd.read_csv(os.path.join(TEST_DIR, "test_source1.tsv"), sep="\t", dtype=str).fillna("")
    s23 = pd.concat([
        pd.read_csv(os.path.join(TEST_DIR, "test_source2.tsv"), sep="\t", dtype=str).fillna(""),
        pd.read_csv(os.path.join(TEST_DIR, "test_source3.tsv"), sep="\t", dtype=str).fillna(""),
    ], ignore_index=True)
    
    s1["nn"] = s1["business_name"].apply(norm)
    s1["na"] = s1["business_address"].fillna("").apply(norm)
    s23["nn"] = s23["business_name"].apply(norm)
    s23["na"] = s23["business_address"].fillna("").apply(norm)

    all_matches = []

    for country in s1["country"].unique():
        c_s1  = s1[s1["country"] == country].reset_index(drop=True)
        c_s23 = s23[s23["country"] == country].reset_index(drop=True)
        if c_s1.empty: continue

        print(f"\n[{country}] S1={len(c_s1):,}, S23={len(c_s23):,}")

        # 1. Pre-compute Matrices (The Magic)
        print("  -> Pre-computing Linear Algebra Matrices...")
        t_pre = time.time()
        
        c_s1["bkeys"] = c_s1.apply(lambda r: get_keys(r["nn"], r["na"]), axis=1)
        c_s23["bkeys"] = c_s23.apply(lambda r: get_keys(r["nn"], r["na"]), axis=1)

        key_vec = CountVectorizer(analyzer=lambda x: x, min_df=2)
        s23_key_mat = key_vec.fit_transform(c_s23["bkeys"])
        s1_key_mat  = key_vec.transform(c_s1["bkeys"])
        
        # --- CRITICAL FIX: The 25GB RAM Explosion (Hot-Key Cap) ---
        # Stop-words like "inc" or "llc" match millions of rows, creating a 25GB dot-product matrix.
        # We must filter these "hot columns" out of the matrix before multiplying!
        c1_counts = np.array(s1_key_mat.sum(axis=0)).flatten()
        c23_counts = np.array(s23_key_mat.sum(axis=0)).flatten()
        pair_counts = c1_counts * c23_counts
        
        valid_cols = np.where(pair_counts <= 150_000)[0]
        s1_key_mat = s1_key_mat[:, valid_cols]
        s23_key_mat = s23_key_mat[:, valid_cols]
        print(f"  -> Hot-Key Filter: Dropped {len(pair_counts) - len(valid_cols):,} massive stop-word keys to save RAM.")

        s23_name_mat = vec_name.transform(c_s23["business_name"].fillna("").str.lower())
        s23_addr_mat = vec_addr.transform(c_s23["business_address"].fillna("").str.lower())
        s1_name_mat  = vec_name.transform(c_s1["business_name"].fillna("").str.lower())
        s1_addr_mat  = vec_addr.transform(c_s1["business_address"].fillna("").str.lower())
        
        s23_names = c_s23["business_name"].fillna("").str.lower().str.strip().values
        s23_addrs = c_s23["business_address"].fillna("").str.lower().str.strip().values
        s23_eids  = c_s23["entity_id"].values
        
        print(f"  -> Matrices built in {time.time()-t_pre:.1f}s. Starting dot-product chunks...")

        n_chunks = (len(c_s1) + S1_CHUNK_SIZE - 1) // S1_CHUNK_SIZE
        
        for i, start in enumerate(range(0, len(c_s1), S1_CHUNK_SIZE)):
            end = min(start + S1_CHUNK_SIZE, len(c_s1))
            chunk = c_s1.iloc[start:end]
            t_chunk = time.time()
            
            # 2. Ultra-Fast Matrix Math
            ch_k = s1_key_mat[start:end]
            ch_n = s1_name_mat[start:end]
            ch_a = s1_addr_mat[start:end]
            
            # Dot products (Calculates all 5000 x 3.8M combinations in ~1 second)
            shared_keys = ch_k.dot(s23_key_mat.T)
            name_sim    = ch_n.dot(s23_name_mat.T)
            addr_sim    = ch_a.dot(s23_addr_mat.T)
            
            # Combine similarities and filter by shared keys instantly
            coarse_sim = name_sim + addr_sim
            valid_sim  = coarse_sim.multiply(shared_keys > 0)
            
            # Extract Top 50 candidates
            pairs_data = []
            for row_idx in range(valid_sim.shape[0]):
                row = valid_sim.getrow(row_idx)
                if row.nnz == 0: continue
                
                data = row.data
                indices = row.indices
                if len(data) > K:
                    top_k = np.argpartition(data, -K)[-K:]
                    indices = indices[top_k]
                    data = data[top_k]
                
                s1_eid = chunk.iloc[row_idx]["entity_id"]
                s1_nn = chunk.iloc[row_idx]["business_name"].lower().strip() if pd.notna(chunk.iloc[row_idx]["business_name"]) else ""
                s1_na = chunk.iloc[row_idx]["business_address"].lower().strip() if pd.notna(chunk.iloc[row_idx]["business_address"]) else ""
                
                for j, cid_idx in enumerate(indices):
                    pairs_data.append({
                        "entity_id": s1_eid,
                        "cid": s23_eids[cid_idx],
                        "s1_nn": s1_nn,
                        "s1_na": s1_na,
                        "c_nn": s23_names[cid_idx],
                        "c_na": s23_addrs[cid_idx],
                        "coarse_score": data[j]
                    })
            
            if not pairs_data: continue
            df_pairs = pd.DataFrame(pairs_data)
            
            # 3. Final String Math on the Top 50 (Very few rows now!)
            s1n, s1a = df_pairs["s1_nn"], df_pairs["s1_na"]
            cn, ca   = df_pairs["c_nn"], df_pairs["c_na"]
            
            df_pairs["name_jw"]  = [safe_jw(a, b)  for a, b in zip(s1n, cn)]
            df_pairs["name_lev"] = [safe_lev(a, b) for a, b in zip(s1n, cn)]
            df_pairs["addr_jw"]  = [safe_jw(a, b)  for a, b in zip(s1a, ca)]
            df_pairs["addr_lev"] = [safe_lev(a, b) for a, b in zip(s1a, ca)]
            df_pairs["num_overlap"] = [num_overlap(a, b) for a, b in zip(s1a, ca)]
            
            df_pairs["name_tfidf_cosine"] = df_pairs.apply(lambda r: name_sim[df_pairs.index.get_loc(r.name) // K if K > 1 else 0, 0] * 0, axis=1) # dummy, re-calc quickly
            
            # Actually, extract exact tf-idf from the dot products
            # To avoid slow lookups, we just use the coarse_score as a proxy or re-extract
            # Let's just re-extract exactly for the final model
            n_scores, a_scores = [], []
            s1id2row = {eid: idx for idx, eid in enumerate(chunk["entity_id"])}
            s23id2idx = {eid: idx for idx, eid in enumerate(s23_eids)}
            
            r_s1 = [s1id2row[eid] for eid in df_pairs["entity_id"]]
            r_s23 = [s23id2idx[cid] for cid in df_pairs["cid"]]
            
            df_pairs["name_tfidf_cosine"] = np.array(ch_n[r_s1].multiply(s23_name_mat[r_s23]).sum(axis=1)).flatten()
            df_pairs["addr_tfidf_cosine"] = np.array(ch_a[r_s1].multiply(s23_addr_mat[r_s23]).sum(axis=1)).flatten()

            df_pairs["name_x_addr"]    = df_pairs["name_tfidf_cosine"] * df_pairs["addr_tfidf_cosine"]
            df_pairs["lookalike_flag"] = ((df_pairs["name_jw"] > 0.90) & (df_pairs["addr_jw"] < 0.50)).astype(int)
            
            # Predict
            df_pairs["prob"] = model.predict_proba(df_pairs[FEATURES])[:, 1]
            df_m = df_pairs[df_pairs["prob"] >= threshold][["entity_id", "cid"]]
            all_matches.append(df_m)
            
            print(f"  chunk {i+1}/{n_chunks}  rows {start:,}–{end:,}  -> {len(df_m):,} matches  ({time.time()-t_chunk:.1f}s)")

    print("\nFormatting for leaderboard...")
    final_matches = pd.concat([df for df in all_matches if not df.empty], ignore_index=True) if any(not df.empty for df in all_matches) else pd.DataFrame(columns=["entity_id", "cid"])

    res = (final_matches.groupby("entity_id")["cid"].apply(lambda x: ",".join(x.unique())).reset_index())
    res.columns = ["source1_entity_id", "matched_entity_ids"]

    out_df = (pd.DataFrame({"source1_entity_id": s1["entity_id"]})
              .merge(res, on="source1_entity_id", how="left").fillna(""))

    out_path = os.path.join(OUTPUT_DIR, "matching_results_v4.tsv")
    out_df.to_csv(out_path, sep="\t", index=False)

    matched = (out_df["matched_entity_ids"] != "").sum()
    print(f"\nSUCCESS! {out_path}")
    print(f"  Entities matched: {matched:,}  ({matched/len(out_df):.1%})")

if __name__ == "__main__":
    main()
