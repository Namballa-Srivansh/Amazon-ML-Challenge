"""
generate_debug_scores.py
=========================
Lightweight standalone script that regenerates output/debug_scores.tsv
from an ALREADY-TRAINED classifier -- no retraining involved. Use this
whenever debug_scores.tsv is missing/stale but you don't want to pay for
a full v6_xgboost.py re-run just to get it.

What it does:
  1. Loads the saved {prefix}_classifier.pkl + vectorizers (default prefix: v6)
  2. Reproduces the EXACT same train_s1/val_s1 split (random_state=42) that
     the classifier itself used
  3. Builds features for the VAL split ONLY (skips train entirely -- that's
     most of the time savings vs a full retrain)
  4. Runs predict_proba() (no .fit() anywhere in this script)
  5. Writes output/debug_scores.tsv with columns: s1_id, cid, prob, label

Uses the same FEATURE_CHUNK_SIZE-chunked feature building as v6_xgboost.py
to avoid the earlier MemoryError, but since it only processes the ~20%
val split (not the full 13.7M pairs), it should finish in a small
fraction of the full pipeline's time.

Usage:
    python generate_debug_scores.py                  # uses v6_classifier.pkl (default)
    python generate_debug_scores.py --model-prefix v3   # uses v3_classifier.pkl instead
"""

import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import os, re, time, pickle, argparse
import numpy as np
import pandas as pd
import jellyfish
from sklearn.model_selection import train_test_split

REPO_ROOT   = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
                os.path.abspath(__file__)))))
TRAIN_DIR   = os.path.join(REPO_ROOT, "dataset", "mini_train")
OUTPUT_DIR  = os.path.join(REPO_ROOT, "output")
MODELS_DIR  = os.path.join(REPO_ROOT, "models")

EMBED_MODEL_NAME    = "paraphrase-multilingual-MiniLM-L12-v2"
EMBED_BATCH_SIZE    = 256
FEATURE_CHUNK_SIZE  = 200_000   # same reasoning as v6_xgboost.py's chunking

FEATURES_9  = [
    "name_jw", "name_lev", "name_tfidf_cosine",
    "addr_jw", "addr_lev", "addr_tfidf_cosine",
    "num_overlap", "name_x_addr", "lookalike_flag",
]
FEATURES_11 = FEATURES_9 + ["name_embed_cosine", "addr_embed_cosine"]

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

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-prefix", default="v6",
                     help="Which saved model to score with (default: v6).")
    args = ap.parse_args()
    prefix = args.model_prefix

    print("=" * 60)
    print(f"generate_debug_scores.py — scoring val split with {prefix}_classifier.pkl")
    print("=" * 60)
    t0 = time.time()

    required = [f"{prefix}_classifier.pkl", f"{prefix}_vec_name.pkl", f"{prefix}_vec_addr.pkl"]
    for fname in required:
        if not os.path.exists(os.path.join(MODELS_DIR, fname)):
            print(f"ERROR: {fname} missing.")
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
        print(f"ERROR: model expects {n_feat} features -- update FEATURES_9/FEATURES_11 above.")
        return
    print(f"  Model expects {n_feat} features ({'with' if needs_embed else 'without'} embeddings)")

    embedder = None
    if needs_embed:
        print(f"  Loading {EMBED_MODEL_NAME}...")
        from sentence_transformers import SentenceTransformer
        embedder = SentenceTransformer(EMBED_MODEL_NAME)

    print("\n[1/4] Loading data + candidate pairs (needed to rebuild the exact val split)...")
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
    print(f"  {len(df_pairs):,} total pairs")

    print("\n[2/4] Reproducing the exact train/val split (random_state=42) -- val ONLY is kept...")
    unique_s1 = df_pairs["s1_id"].unique()
    _, val_s1 = train_test_split(unique_s1, test_size=0.2, random_state=42)
    val_df = df_pairs[df_pairs["s1_id"].isin(set(val_s1))].copy().reset_index(drop=True)
    print(f"  val pairs: {len(val_df):,}  (train pairs skipped entirely -- that's the time savings)")

    s1_dict  = s1.set_index("entity_id")[["business_name", "business_address"]].to_dict("index")
    s23_dict = s23.set_index("entity_id")[["business_name", "business_address"]].to_dict("index")

    va_s1n = [str(s1_dict.get(i, {}).get("business_name",  "")).lower().strip() for i in val_df["s1_id"]]
    va_s1a = [str(s1_dict.get(i, {}).get("business_address","")).lower().strip() for i in val_df["s1_id"]]
    va_cn  = [str(s23_dict.get(i, {}).get("business_name",  "")).lower().strip() for i in val_df["cid"]]
    va_ca  = [str(s23_dict.get(i, {}).get("business_address","")).lower().strip() for i in val_df["cid"]]

    name2vec, addr2vec = None, None
    if needs_embed:
        print("\n[3/4] Encoding unique val strings (val-only, much smaller than full 13.7M pairs)...")
        all_names = sorted(set(va_s1n + va_cn))
        all_addrs = sorted(set(va_s1a + va_ca))
        print(f"  Unique names: {len(all_names):,}  |  Unique addresses: {len(all_addrs):,}")
        name_vecs = embedder.encode(all_names, batch_size=EMBED_BATCH_SIZE, show_progress_bar=True, convert_to_numpy=True)
        addr_vecs = embedder.encode(all_addrs, batch_size=EMBED_BATCH_SIZE, show_progress_bar=True, convert_to_numpy=True)
        name2vec, addr2vec = dict(zip(all_names, name_vecs)), dict(zip(all_addrs, addr_vecs))
    else:
        print("\n[3/4] Model doesn't use embeddings -- skipping encoder entirely.")

    print("\n[4/4] Building features (chunked) and scoring...")
    n = len(val_df)
    s1_ids = val_df["s1_id"].to_numpy(); cids = val_df["cid"].to_numpy(); labels = val_df["label"].to_numpy()
    n_chunks = (n + FEATURE_CHUNK_SIZE - 1) // FEATURE_CHUNK_SIZE
    parts = []
    t_feat = time.time()

    for i, start in enumerate(range(0, n, FEATURE_CHUNK_SIZE)):
        end = min(start + FEATURE_CHUNK_SIZE, n)
        c_s1n, c_s1a = va_s1n[start:end], va_s1a[start:end]
        c_cn,  c_ca  = va_cn[start:end],  va_ca[start:end]

        chunk = pd.DataFrame({"s1_id": s1_ids[start:end], "cid": cids[start:end], "label": labels[start:end]})
        chunk["name_jw"]  = [safe_jw(a, b)  for a, b in zip(c_s1n, c_cn)]
        chunk["name_lev"] = [safe_lev(a, b) for a, b in zip(c_s1n, c_cn)]
        chunk["addr_jw"]  = [safe_jw(a, b)  for a, b in zip(c_s1a, c_ca)]
        chunk["addr_lev"] = [safe_lev(a, b) for a, b in zip(c_s1a, c_ca)]
        chunk["num_overlap"] = [num_overlap(a, b) for a, b in zip(c_s1a, c_ca)]

        s1n_v = vec_name.transform(c_s1n); cn_v = vec_name.transform(c_cn)
        s1a_v = vec_addr.transform(c_s1a); ca_v = vec_addr.transform(c_ca)
        chunk["name_tfidf_cosine"] = np.array(s1n_v.multiply(cn_v).sum(axis=1)).flatten()
        chunk["addr_tfidf_cosine"] = np.array(s1a_v.multiply(ca_v).sum(axis=1)).flatten()
        chunk["name_x_addr"]    = chunk["name_tfidf_cosine"] * chunk["addr_tfidf_cosine"]
        chunk["lookalike_flag"] = ((chunk["name_jw"] > 0.90) & (chunk["addr_jw"] < 0.50)).astype(int)

        if needs_embed:
            v1n = np.stack([name2vec[t] for t in c_s1n]); v2n = np.stack([name2vec[t] for t in c_cn])
            v1a = np.stack([addr2vec[t] for t in c_s1a]); v2a = np.stack([addr2vec[t] for t in c_ca])
            chunk["name_embed_cosine"] = cos_rows(v1n, v2n)
            chunk["addr_embed_cosine"] = cos_rows(v1a, v2a)
            del v1n, v2n, v1a, v2a

        chunk["prob"] = model.predict_proba(chunk[FEATURES])[:, 1]
        parts.append(chunk[["s1_id", "cid", "prob", "label"]])
        del s1n_v, cn_v, s1a_v, ca_v
        if (i + 1) % 5 == 0 or (i + 1) == n_chunks:
            print(f"    chunk {i+1}/{n_chunks} ({end:,}/{n:,} rows, {time.time()-t_feat:.0f}s elapsed)")

    debug_out = pd.concat(parts, ignore_index=True)
    debug_out.to_csv(os.path.join(OUTPUT_DIR, "debug_scores.tsv"), sep="\t", index=False)

    print(f"\n[Done] Wrote debug_scores.tsv ({len(debug_out):,} rows) in {time.time()-t0:.1f}s")
    print(f"  ({prefix}_classifier.pkl was only used for predict_proba() -- nothing was retrained)")

if __name__ == "__main__":
    main()
