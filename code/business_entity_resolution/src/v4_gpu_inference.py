"""
v4_gpu_inference.py — Tiled GPU Sparse Dot-Product Inference (V4)
=============================================================
Computes EXACT TF-IDF cosine similarity on NVIDIA GPU.
Uses safe Sparse-Dense GPU multiplication to eliminate cuSPARSE OOMs.
Filters candidates with < 0.10 coarse similarity directly on GPU to avoid CPU bottleneck.
"""

import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import os, re, time, pickle, warnings
warnings.filterwarnings("ignore", message="CUDA path could not be detected")

import numpy as np
import pandas as pd
import jellyfish
import cupy as cp
import cupyx.scipy.sparse as cpx

REPO_ROOT   = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
TEST_DIR    = os.path.join(REPO_ROOT, "dataset", "test")
OUTPUT_DIR  = os.path.join(REPO_ROOT, "output")
MODELS_DIR  = os.path.join(REPO_ROOT, "models")

os.makedirs(OUTPUT_DIR, exist_ok=True)

K             = 50
S1_CHUNK_SIZE = 5_000
S23_GPU_TILE  = 50_000   # 50k rows takes ~3 GB peak VRAM

FEATURES = [
    "name_jw", "name_lev", "name_tfidf_cosine",
    "addr_jw", "addr_lev", "addr_tfidf_cosine",
    "num_overlap", "name_x_addr", "lookalike_flag"
]

def safe_lev(s1, s2):
    if not s1 and not s2: return 1.0
    if not s1 or not s2:  return 0.0
    d = jellyfish.levenshtein_distance(s1, s2)
    return 1.0 - (d / max(len(s1), len(s2)))

def safe_jw(s1, s2):
    if not s1 and not s2: return 1.0
    if not s1 or not s2:  return 0.0
    return jellyfish.jaro_winkler_similarity(s1, s2)

def num_overlap(s1, s2):
    t1 = set(re.findall(r'\d+', str(s1)))
    t2 = set(re.findall(r'\d+', str(s2)))
    if not t1 or not t2: return 0.0
    return len(t1 & t2) / len(t1 | t2)

def free_gpu():
    cp.get_default_memory_pool().free_all_blocks()
    cp.get_default_pinned_memory_pool().free_all_blocks()

def to_gpu_safe(cpu_csr):
    return cpx.csr_matrix((
        cp.array(cpu_csr.data, dtype=cp.float32),
        cp.array(cpu_csr.indices, dtype=cp.int32),
        cp.array(cpu_csr.indptr, dtype=cp.int32)
    ), shape=cpu_csr.shape)


def process_country(c_s1, c_s23, model, vec_name, vec_addr, threshold, country):
    print(f"\n[{country}] S1={len(c_s1):,}, S23={len(c_s23):,}")
    c_s1  = c_s1.reset_index(drop=True)
    c_s23 = c_s23.reset_index(drop=True)

    print(f"  -> Building CPU TF-IDF matrices...")
    t_pre = time.time()

    c_s1_names = c_s1["business_name"].fillna("").str.lower().str.strip()
    c_s1_addrs = c_s1["business_address"].fillna("").str.lower().str.strip()
    c_s23_names = c_s23["business_name"].fillna("").str.lower().str.strip()
    c_s23_addrs = c_s23["business_address"].fillna("").str.lower().str.strip()

    s1_name_cpu = vec_name.transform(c_s1_names)
    s1_addr_cpu = vec_addr.transform(c_s1_addrs)
    s23_name_cpu = vec_name.transform(c_s23_names)
    s23_addr_cpu = vec_addr.transform(c_s23_addrs)

    s23_names_arr = c_s23_names.values
    s23_addrs_arr = c_s23_addrs.values
    s23_eids = c_s23["entity_id"].values

    n_s23 = len(c_s23)
    n_tiles = (n_s23 + S23_GPU_TILE - 1) // S23_GPU_TILE
    print(f"  -> TF-IDF matrices built in {time.time()-t_pre:.1f}s. S23 tiles: {n_tiles}. Starting GPU inference...", flush=True)

    all_matches = []
    n_s1_chunks = (len(c_s1) + S1_CHUNK_SIZE - 1) // S1_CHUNK_SIZE

    for i, s1_start in enumerate(range(0, len(c_s1), S1_CHUNK_SIZE)):
        s1_end = min(s1_start + S1_CHUNK_SIZE, len(c_s1))
        chunk  = c_s1.iloc[s1_start:s1_end]
        t_chunk = time.time()

        ch_n_gpu = to_gpu_safe(s1_name_cpu[s1_start:s1_end])
        ch_a_gpu = to_gpu_safe(s1_addr_cpu[s1_start:s1_end])

        topk_scores  = np.full((s1_end - s1_start, K), -np.inf)
        topk_indices = np.full((s1_end - s1_start, K), -1, dtype=np.int64)

        for t_start in range(0, n_s23, S23_GPU_TILE):
            t_end = min(t_start + S23_GPU_TILE, n_s23)

            # Name Dot Product (Sparse @ Dense -> Dense)
            tile_n_dense_T = to_gpu_safe(s23_name_cpu[t_start:t_end]).toarray().T
            coarse = ch_n_gpu.dot(tile_n_dense_T)
            del tile_n_dense_T
            free_gpu()
            
            # Addr Dot Product (Sparse @ Dense -> Dense)
            tile_a_dense_T = to_gpu_safe(s23_addr_cpu[t_start:t_end]).toarray().T
            addr_s = ch_a_gpu.dot(tile_a_dense_T)
            del tile_a_dense_T
            free_gpu()
            
            # Combine
            coarse += addr_s
            del addr_s
            
            # Filter dense junk directly on GPU (Drops >98% of candidates instantly)
            coarse = cp.where(coarse > 0.25, coarse, 0)
            
            # Convert back to Sparse
            coarse_sparse = cpx.csr_matrix(coarse)
            del coarse
            
            coarse_cpu = coarse_sparse.get()
            del coarse_sparse
            free_gpu()

            # Fast direct CSR slice (2x faster than .getrow())
            for row_idx in range(coarse_cpu.shape[0]):
                start = coarse_cpu.indptr[row_idx]
                end   = coarse_cpu.indptr[row_idx+1]
                if start == end: continue
                
                data = coarse_cpu.data[start:end]
                indices = coarse_cpu.indices[start:end]
                
                if len(data) > K:
                    best_k = np.argpartition(data, -K)[-K:]
                    data, indices = data[best_k], indices[best_k]
                global_idx = indices + t_start

                combined_scores  = np.concatenate([topk_scores[row_idx],  data])
                combined_indices = np.concatenate([topk_indices[row_idx], global_idx])
                best = np.argpartition(combined_scores, -K)[-K:]
                topk_scores[row_idx]  = combined_scores[best]
                topk_indices[row_idx] = combined_indices[best]

        del ch_n_gpu, ch_a_gpu
        free_gpu()

        pairs_data = []
        for row_idx in range(s1_end - s1_start):
            valid_mask = topk_indices[row_idx] >= 0
            idxs = topk_indices[row_idx][valid_mask]
            if len(idxs) == 0: continue

            s1_eid = chunk.iloc[row_idx]["entity_id"]
            s1_nn  = c_s1_names.iloc[s1_start + row_idx]
            s1_na  = c_s1_addrs.iloc[s1_start + row_idx]

            for cid_idx in idxs:
                pairs_data.append({
                    "entity_id": s1_eid, "cid": s23_eids[cid_idx],
                    "s1_nn": s1_nn, "s1_na": s1_na,
                    "c_nn": s23_names_arr[cid_idx], "c_na": s23_addrs_arr[cid_idx]
                })

        if not pairs_data: 
            print(f"  chunk {i+1}/{n_s1_chunks}  -> 0 matches (no pairs generated)", flush=True)
            continue
            
        df_pairs = pd.DataFrame(pairs_data)

        s1n, s1a = df_pairs["s1_nn"], df_pairs["s1_na"]
        cn, ca   = df_pairs["c_nn"],  df_pairs["c_na"]
        df_pairs["name_jw"]  = [safe_jw(a, b)  for a, b in zip(s1n, cn)]
        df_pairs["name_lev"] = [safe_lev(a, b) for a, b in zip(s1n, cn)]
        df_pairs["addr_jw"]  = [safe_jw(a, b)  for a, b in zip(s1a, ca)]
        df_pairs["addr_lev"] = [safe_lev(a, b) for a, b in zip(s1a, ca)]
        df_pairs["num_overlap"] = [num_overlap(a, b) for a, b in zip(s1a, ca)]

        s1id2row  = {eid: idx for idx, eid in enumerate(chunk["entity_id"])}
        s23id2idx = {eid: idx for idx, eid in enumerate(s23_eids)}
        r_s1  = [s1id2row[eid]  for eid in df_pairs["entity_id"]]
        r_s23 = [s23id2idx[cid] for cid in df_pairs["cid"]]
        
        df_pairs["name_tfidf_cosine"] = np.array(s1_name_cpu[s1_start:s1_end][r_s1].multiply(s23_name_cpu[r_s23]).sum(axis=1)).flatten()
        df_pairs["addr_tfidf_cosine"] = np.array(s1_addr_cpu[s1_start:s1_end][r_s1].multiply(s23_addr_cpu[r_s23]).sum(axis=1)).flatten()

        df_pairs["name_x_addr"]    = df_pairs["name_tfidf_cosine"] * df_pairs["addr_tfidf_cosine"]
        df_pairs["lookalike_flag"] = ((df_pairs["name_jw"] > 0.90) & (df_pairs["addr_jw"] < 0.50)).astype(int)

        df_pairs["prob"] = model.predict_proba(df_pairs[FEATURES])[:, 1]
        df_m = df_pairs[df_pairs["prob"] >= threshold][["entity_id", "cid"]]
        all_matches.append(df_m)

        print(f"  chunk {i+1}/{n_s1_chunks}  rows {s1_start:,}–{s1_end:,}  -> {len(df_m):,} matches  ({time.time()-t_chunk:.1f}s)", flush=True)

    return all_matches


def main():
    print("=" * 60)
    print("v4 — TILED GPU SPARSE DOT-PRODUCT INFERENCE (FIXED OOM)")
    print(f"CUDA Devices: {cp.cuda.runtime.getDeviceCount()}")
    print("=" * 60)

    model     = pickle.load(open(os.path.join(MODELS_DIR, "v3_classifier_10k.pkl"),  "rb"))
    vec_name  = pickle.load(open(os.path.join(MODELS_DIR, "v3_vec_name_10k.pkl"),    "rb"))
    vec_addr  = pickle.load(open(os.path.join(MODELS_DIR, "v3_vec_addr_10k.pkl"),    "rb"))
    threshold = float(open(os.path.join(MODELS_DIR, "v3_threshold_10k.txt")).read().strip())
    print(f"Loaded model | threshold = {threshold:.2f}")

    print("\nLoading dataset...")
    t0 = time.time()
    s1  = pd.read_csv(os.path.join(TEST_DIR, "test_source1.tsv"), sep="\t", dtype=str).fillna("")
    s23 = pd.concat([
        pd.read_csv(os.path.join(TEST_DIR, "test_source2.tsv"), sep="\t", dtype=str).fillna(""),
        pd.read_csv(os.path.join(TEST_DIR, "test_source3.tsv"), sep="\t", dtype=str).fillna(""),
    ], ignore_index=True)
    print(f"  S1: {len(s1):,}  |  S23: {len(s23):,}  ({time.time()-t0:.0f}s)")

    all_matches = []
    for country in s1["country"].unique():
        c_s1  = s1[s1["country"] == country]
        c_s23 = s23[s23["country"] == country]
        if c_s1.empty: continue
        matches = process_country(c_s1, c_s23, model, vec_name, vec_addr, threshold, country)
        all_matches.extend(matches)
        free_gpu()

    print("\nFormatting output...", flush=True)
    final = pd.concat([m for m in all_matches if not m.empty], ignore_index=True) if all_matches else pd.DataFrame(columns=["entity_id","cid"])
    res = final.groupby("entity_id")["cid"].apply(lambda x: ",".join(x.unique())).reset_index()
    res.columns = ["source1_entity_id", "matched_entity_ids"]

    out_df = pd.DataFrame({"source1_entity_id": s1["entity_id"]}).merge(res, on="source1_entity_id", how="left").fillna("")
    out_path = os.path.join(OUTPUT_DIR, "matching_results_gpu.tsv")
    out_df.to_csv(out_path, sep="\t", index=False)

    matched = (out_df["matched_entity_ids"] != "").sum()
    print(f"\nSUCCESS! {out_path}", flush=True)
    print(f"  Total S1 rows:    {len(out_df):,}")
    print(f"  Entities matched: {matched:,}  ({matched/len(out_df):.1%})")
    print(f"  Total time:       {time.time()-t0:.0f}s")

if __name__ == "__main__":
    main()
