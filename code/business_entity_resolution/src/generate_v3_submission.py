"""
generate_v3_submission.py — Full Test Set Inference (Memory-Safe + V2 Chunking)
================================================================================
Fixes applied (2026-09-25):
  - 100% matches V2 chunking logic: limits to Top 50 candidates BEFORE heavy math
  - Removed massive `s23_dict` completely (fixes MemoryError on 8GB RAM machines)
  - Uses fast numpy indexing to fetch raw strings for the top 50 candidates
"""

import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import os, re, time, pickle
import numpy as np
import pandas as pd
import jellyfish
from sklearn.feature_extraction.text import TfidfVectorizer

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_ROOT   = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
                os.path.abspath(__file__)))))
TEST_DIR    = os.path.join(REPO_ROOT, "dataset", "test")
OUTPUT_DIR  = os.path.join(REPO_ROOT, "output")
MODELS_DIR  = os.path.join(REPO_ROOT, "models")

os.makedirs(OUTPUT_DIR, exist_ok=True)

MAX_PAIRS_PER_KEY = 150_000
S1_CHUNK_SIZE     = 5_000   # Lowered to 5k to match V2 exactly
K                 = 50      # Keep top 50 candidates before heavy features

# ---------------------------------------------------------------------------
# Text helpers
# ---------------------------------------------------------------------------
_PUNCT_RE = re.compile(r'[^\w\s]')
_SPACE_RE = re.compile(r'\s+')

def norm(text):
    return _SPACE_RE.sub(' ', _PUNCT_RE.sub(' ', str(text).lower().strip())).strip()

def get_keys(nn, na):
    nw = nn.split()
    aw = na.split()
    keys = set()
    if len(nw) >= 2:
        acr = "".join(w[0] for w in nw if w)
        if len(acr) >= 2: keys.add(f"acr:{acr}")
    for w in nw + aw:
        if len(w) >= 3: keys.add(f"w:{w}")
        if len(w) >= 5: keys.add(f"s:{jellyfish.soundex(w)}")
    return list(keys)

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
# Chunk Processing
# ---------------------------------------------------------------------------
def process_chunk(chunk: pd.DataFrame, s23_key_df: pd.DataFrame,
                  s23_names: np.ndarray, s23_addrs: np.ndarray,
                  s23_name_vecs, s23_addr_vecs, s23_id2idx: dict,
                  model, vec_name, vec_addr, threshold: float):
    
    EMPTY = pd.DataFrame(columns=["entity_id", "cid"])
    if chunk.empty: return EMPTY

    chunk = chunk.copy()
    chunk["bkeys"] = chunk.apply(lambda r: get_keys(r["nn"], r["na"]), axis=1)
    s1_exp = (chunk[["entity_id", "bkeys"]].explode("bkeys")
              .dropna(subset=["bkeys"]).rename(columns={"bkeys": "bkey"}))
    s1_exp = s1_exp[s1_exp["bkey"].str.len() > 3]

    if s1_exp.empty: return EMPTY

    valid_s1_keys = set(s1_exp["bkey"].unique())
    s23_rel = s23_key_df[s23_key_df["bkey"].isin(valid_s1_keys)]

    if s23_rel.empty: return EMPTY

    c1  = s1_exp["bkey"].value_counts()
    c23 = s23_rel["bkey"].value_counts()
    safe = set(k for k, n1 in c1.items() if n1 * c23.get(k, 0) <= MAX_PAIRS_PER_KEY)

    pairs = (s1_exp[s1_exp["bkey"].isin(safe)]
             .merge(s23_rel[s23_rel["bkey"].isin(safe)].rename(columns={"entity_id": "cid"}), on="bkey")
             [["entity_id", "cid"]].drop_duplicates())

    if pairs.empty: return EMPTY

    # --- 1. Fast Coarse Ranking (TF-IDF ONLY) to prevent MemoryError ---
    s1n_full = chunk["business_name"].fillna("").str.lower()
    s1a_full = chunk["business_address"].fillna("").str.lower()
    s1n_v = vec_name.transform(s1n_full)
    s1a_v = vec_addr.transform(s1a_full)
    
    s1_id2idx = {eid: i for i, eid in enumerate(chunk["entity_id"])}
    
    pairs["s1_idx"] = pairs["entity_id"].map(s1_id2idx)
    pairs["s23_idx"] = pairs["cid"].map(s23_id2idx)
    
    # CRITICAL FIX: Chunk the sparse matrix multiplication to prevent Hard OOM crashes!
    M_CHUNK = 500_000
    n_scores, a_scores = [], []
    
    for st in range(0, len(pairs), M_CHUNK):
        ch = pairs.iloc[st:st+M_CHUNK]
        
        v1n = s1n_v[ch["s1_idx"].values]
        v23n = s23_name_vecs[ch["s23_idx"].values]
        n_scores.extend(np.array(v1n.multiply(v23n).sum(axis=1)).flatten())
        
        v1a = s1a_v[ch["s1_idx"].values]
        v23a = s23_addr_vecs[ch["s23_idx"].values]
        a_scores.extend(np.array(v1a.multiply(v23a).sum(axis=1)).flatten())
        
    pairs["name_tfidf_cosine"] = n_scores
    pairs["addr_tfidf_cosine"] = a_scores
    
    # Coarse score is just sum of the two cosines
    pairs["coarse_score"] = pairs["name_tfidf_cosine"] + pairs["addr_tfidf_cosine"]
    
    # Filter to Top K=50 candidates per S1 entity
    pairs = pairs.sort_values(["entity_id", "coarse_score"], ascending=[True, False]).groupby("entity_id").head(K)

    # --- 2. Heavy Math on Top 50 Candidates Only ---
    s1_rows_k  = [s1_id2idx[eid] for eid in pairs["entity_id"]]
    s23_rows_k = [s23_id2idx[cid] for cid in pairs["cid"]]

    s1n = s1n_full.values[s1_rows_k]
    s1a = s1a_full.values[s1_rows_k]
    cn  = s23_names[s23_rows_k]
    ca  = s23_addrs[s23_rows_k]

    pairs["name_jw"]  = [safe_jw(a, b)  for a, b in zip(s1n, cn)]
    pairs["name_lev"] = [safe_lev(a, b) for a, b in zip(s1n, cn)]
    pairs["addr_jw"]  = [safe_jw(a, b)  for a, b in zip(s1a, ca)]
    pairs["addr_lev"] = [safe_lev(a, b) for a, b in zip(s1a, ca)]
    pairs["num_overlap"] = [num_overlap(a, b) for a, b in zip(s1a, ca)]
    
    pairs["name_x_addr"]    = pairs["name_tfidf_cosine"] * pairs["addr_tfidf_cosine"]
    pairs["lookalike_flag"] = ((pairs["name_jw"] > 0.90) & (pairs["addr_jw"] < 0.50)).astype(int)

    # --- Predict ---
    pairs["prob"] = model.predict_proba(pairs[FEATURES])[:, 1]
    return pairs[pairs["prob"] >= threshold][["entity_id", "cid"]]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 60)
    print("v3 — Full Test Set Inference (Memory-Safe + V2 Chunking)")
    print("=" * 60)

    for fname in ["v3_classifier.pkl", "v3_vec_name.pkl", "v3_vec_addr.pkl", "v3_threshold.txt"]:
        if not os.path.exists(os.path.join(MODELS_DIR, fname)):
            print(f"ERROR: {fname} missing. Run v3_classifier.py first.")
            return

    model     = pickle.load(open(os.path.join(MODELS_DIR, "v3_classifier.pkl"), "rb"))
    vec_name  = pickle.load(open(os.path.join(MODELS_DIR, "v3_vec_name.pkl"),   "rb"))
    vec_addr  = pickle.load(open(os.path.join(MODELS_DIR, "v3_vec_addr.pkl"),   "rb"))
    threshold = float(open(os.path.join(MODELS_DIR, "v3_threshold.txt")).read().strip())
    print(f"  Loaded model | threshold = {threshold:.2f}")

    print("\nLoading Test Dataset...")
    t0 = time.time()
    s1  = pd.read_csv(os.path.join(TEST_DIR, "test_source1.tsv"), sep="\t", dtype=str).fillna("")
    s23 = pd.concat([
        pd.read_csv(os.path.join(TEST_DIR, "test_source2.tsv"), sep="\t", dtype=str).fillna(""),
        pd.read_csv(os.path.join(TEST_DIR, "test_source3.tsv"), sep="\t", dtype=str).fillna(""),
    ], ignore_index=True)
    
    s1["nn"] = s1["business_name"].apply(norm)
    s1["na"] = s1["business_address"].fillna("").apply(norm)
    s23["nn"] = s23["business_name"].apply(norm)
    s23["na"] = s23["business_address"].fillna("").apply(norm)
    
    print(f"  S1: {len(s1):,}  |  S23: {len(s23):,}  ({time.time()-t0:.0f}s)")

    all_matches = []

    for country in s1["country"].unique():
        c_s1  = s1[s1["country"] == country].reset_index(drop=True)
        c_s23 = s23[s23["country"] == country].reset_index(drop=True)
        if c_s1.empty: continue

        print(f"\n[{country}] S1={len(c_s1):,}, S23={len(c_s23):,}")

        print(f"  -> Building S23 keys & matrices...")
        t_pre = time.time()
        
        # Keep raw strings as fast numpy arrays (avoids MemoryError of dict)
        s23_names = c_s23["business_name"].fillna("").str.lower().str.strip().values
        s23_addrs = c_s23["business_address"].fillna("").str.lower().str.strip().values
        
        c_s23_keys = c_s23.copy()
        c_s23_keys["bkeys"] = c_s23_keys.apply(lambda r: get_keys(r["nn"], r["na"]), axis=1)
        s23_key_df = (c_s23_keys[["entity_id", "bkeys"]].explode("bkeys")
                        .dropna(subset=["bkeys"]).rename(columns={"bkeys": "bkey"}))
        s23_key_df = s23_key_df[s23_key_df["bkey"].str.len() > 3]
        
        s23_name_vecs = vec_name.transform(s23_names)
        s23_addr_vecs = vec_addr.transform(s23_addrs)
        s23_id2idx = {eid: i for i, eid in enumerate(c_s23["entity_id"])}

        print(f"  -> Matrices built in {time.time()-t_pre:.1f}s. Starting chunks...")

        n_chunks = (len(c_s1) + S1_CHUNK_SIZE - 1) // S1_CHUNK_SIZE
        for i, start in enumerate(range(0, len(c_s1), S1_CHUNK_SIZE)):
            end   = min(start + S1_CHUNK_SIZE, len(c_s1))
            chunk = c_s1.iloc[start:end]
            
            t1 = time.time()
            df_m = process_chunk(chunk, s23_key_df, s23_names, s23_addrs,
                                 s23_name_vecs, s23_addr_vecs, s23_id2idx,
                                 model, vec_name, vec_addr, threshold)
            all_matches.append(df_m)
            print(f"  chunk {i+1}/{n_chunks}  rows {start:,}–{end:,}  -> {len(df_m):,} matches  ({time.time()-t1:.1f}s)")

    print("\nFormatting for leaderboard...")
    final_matches = pd.concat([df for df in all_matches if not df.empty], ignore_index=True) if any(not df.empty for df in all_matches) else pd.DataFrame(columns=["entity_id", "cid"])

    res = (final_matches.groupby("entity_id")["cid"].apply(lambda x: ",".join(x.unique())).reset_index())
    res.columns = ["source1_entity_id", "matched_entity_ids"]

    out_df = (pd.DataFrame({"source1_entity_id": s1["entity_id"]})
              .merge(res, on="source1_entity_id", how="left").fillna(""))

    out_path = os.path.join(OUTPUT_DIR, "matching_results.tsv")
    out_df.to_csv(out_path, sep="\t", index=False)

    matched = (out_df["matched_entity_ids"] != "").sum()
    print(f"\nSUCCESS! {out_path}")
    print(f"  Total S1 rows:    {len(out_df):,}")
    print(f"  Entities matched: {matched:,}  ({matched/len(out_df):.1%})")
    print(f"  Total time:       {time.time()-t0:.0f}s")

if __name__ == "__main__":
    main()
