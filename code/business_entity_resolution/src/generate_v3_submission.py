"""
generate_v3_submission.py — Full Test Set Inference
====================================================
Generates matching_results.tsv for leaderboard submission using the
Logistic Regression model trained in v3_classifier.py.

Runs country-by-country, with S1 processed in chunks of 25k to stay
within 8GB RAM. Uses single-threaded processing for Colab compatibility.

Fixes applied (audit 2026-09-25):
  - CRITICAL: empty-string Levenshtein bug fixed (was returning 1.0, now 0.0)
  - Loads saved vectorizers (vec_name, vec_addr) from models/ — no refitting
  - Loads saved threshold from models/v3_threshold.txt — no hardcoding
  - ProcessPoolExecutor removed — uses single-threaded loop (Colab-safe)
  - Empty DataFrame always has correct columns ["entity_id", "cid"]
  - os.makedirs(OUTPUT_DIR) added
  - c_s23 dict built once per country, not per S1 chunk
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
STUDENT_RES = os.path.join(REPO_ROOT, "6ab10eb3b23ba_student_resource", "student_resource")
TEST_DIR    = os.path.join(STUDENT_RES, "dataset", "test")
OUTPUT_DIR  = os.path.join(STUDENT_RES, "output")
MODELS_DIR  = os.path.join(REPO_ROOT, "models")

os.makedirs(OUTPUT_DIR, exist_ok=True)

MAX_PAIRS_PER_KEY = 250_000
S1_CHUNK_SIZE     = 25_000

# ---------------------------------------------------------------------------
# Text helpers (same as v2_blocking.py)
# ---------------------------------------------------------------------------
_PUNCT_RE = re.compile(r'[^\w\s]')
_SPACE_RE = re.compile(r'\s+')

def normalise_name(text):
    return _SPACE_RE.sub(' ', _PUNCT_RE.sub(' ', str(text).lower().strip())).strip()

def get_keys(norm_name: str, norm_addr: str) -> list:
    name_words = norm_name.split()
    addr_words = norm_addr.split()
    keys = set()
    if len(name_words) >= 2:
        acr = "".join(w[0] for w in name_words if w)
        if len(acr) >= 2:
            keys.add(f"acr:{acr}")
    for w in name_words + addr_words:
        if len(w) >= 2:
            keys.add(f"w:{w}")
            keys.add(f"s:{jellyfish.soundex(w)}")
    return list(keys)

# ---------------------------------------------------------------------------
# FIX: correct Levenshtein — returns 0.0 when either string is empty
# ---------------------------------------------------------------------------
def safe_lev(s1, s2):
    if not s1 and not s2: return 1.0
    if not s1 or not s2:  return 0.0   # was returning 1.0 — CRITICAL BUG
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
# Per-country chunk processing
# ---------------------------------------------------------------------------
def process_chunk(s1_chunk: pd.DataFrame, s23: pd.DataFrame,
                  s23_dict: dict, model, vec_name, vec_addr, threshold: float):
    """
    Process one chunk of S1 against the full country S23.
    Returns DataFrame with columns ["entity_id", "cid"].
    """
    EMPTY = pd.DataFrame(columns=["entity_id", "cid"])

    if s1_chunk.empty or s23.empty:
        return EMPTY

    # --- Blocking ---
    s1_chunk = s1_chunk.copy()
    s1_chunk["norm_name"] = s1_chunk["business_name"].apply(normalise_name)
    s1_chunk["norm_addr"] = s1_chunk["business_address"].fillna("").apply(normalise_name)
    s1_chunk["bkeys"]     = s1_chunk.apply(
        lambda r: get_keys(r["norm_name"], r["norm_addr"]), axis=1)

    s1_exp = s1_chunk[["entity_id", "bkeys"]].explode("bkeys").dropna(subset=["bkeys"])
    s1_exp = s1_exp[s1_exp["bkeys"].str.len() > 4].rename(columns={"bkeys": "bkey"})

    if s1_exp.empty:
        return EMPTY

    # Only compute s23 keys that are relevant for THIS chunk (saves memory)
    valid_s1_keys = set(s1_exp["bkey"].unique())

    s23_exp_rows = []
    for row in s23.itertuples(index=False):
        nn = normalise_name(row.business_name)
        na = normalise_name(getattr(row, "business_address", "") or "")
        for k in get_keys(nn, na):
            if k in valid_s1_keys:
                s23_exp_rows.append((row.entity_id, k))

    if not s23_exp_rows:
        return EMPTY

    s23_exp = pd.DataFrame(s23_exp_rows, columns=["entity_id", "bkey"])
    s23_exp = s23_exp[s23_exp["bkey"].str.len() > 4]

    # Hot-key capping
    c1  = s1_exp["bkey"].value_counts()
    c23 = s23_exp["bkey"].value_counts()
    safe_keys = set(k for k, n1 in c1.items() if n1 * c23.get(k, 0) <= MAX_PAIRS_PER_KEY)

    pairs = (s1_exp[s1_exp["bkey"].isin(safe_keys)]
             .merge(s23_exp[s23_exp["bkey"].isin(safe_keys)]
                    .rename(columns={"entity_id": "cid"}), on="bkey", how="inner")
             [["entity_id", "cid"]].drop_duplicates())

    if pairs.empty:
        return EMPTY

    # --- Feature engineering (single-threaded — Colab safe) ---
    s1d = s1_chunk.set_index("entity_id")[["business_name", "business_address"]].to_dict("index")

    s1_names = [str(s1d.get(i, {}).get("business_name",   "")).lower().strip() for i in pairs["entity_id"]]
    s1_addrs = [str(s1d.get(i, {}).get("business_address","")).lower().strip() for i in pairs["entity_id"]]
    c_names  = [str(s23_dict.get(i, {}).get("business_name",   "")).lower().strip() for i in pairs["cid"]]
    c_addrs  = [str(s23_dict.get(i, {}).get("business_address","")).lower().strip() for i in pairs["cid"]]

    pairs = pairs.copy()
    pairs["name_jw"]  = [safe_jw(a, b)  for a, b in zip(s1_names, c_names)]
    pairs["name_lev"] = [safe_lev(a, b) for a, b in zip(s1_names, c_names)]
    pairs["addr_jw"]  = [safe_jw(a, b)  for a, b in zip(s1_addrs, c_addrs)]
    pairs["addr_lev"] = [safe_lev(a, b) for a, b in zip(s1_addrs, c_addrs)]
    pairs["num_overlap"] = [num_overlap(a, b) for a, b in zip(s1_addrs, c_addrs)]

    # Use the SAVED vectorizers — consistent vocabulary with training
    s1n_v = vec_name.transform(s1_names); cn_v  = vec_name.transform(c_names)
    s1a_v = vec_addr.transform(s1_addrs); ca_v  = vec_addr.transform(c_addrs)

    pairs["name_tfidf_cosine"] = np.array(s1n_v.multiply(cn_v).sum(axis=1)).flatten()
    pairs["addr_tfidf_cosine"] = np.array(s1a_v.multiply(ca_v).sum(axis=1)).flatten()
    pairs["name_x_addr"]       = pairs["name_tfidf_cosine"] * pairs["addr_tfidf_cosine"]
    pairs["lookalike_flag"]    = ((pairs["name_jw"] > 0.90) & (pairs["addr_jw"] < 0.50)).astype(int)

    # --- Predict & filter ---
    pairs["prob"] = model.predict_proba(pairs[FEATURES])[:, 1]
    return pairs[pairs["prob"] >= threshold][["entity_id", "cid"]]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 60)
    print("v3 — Full Test Set Inference (Colab-safe, micro-chunked)")
    print("=" * 60)

    # Load model artifacts
    for fname in ["v3_classifier.pkl", "v3_vec_name.pkl", "v3_vec_addr.pkl", "v3_threshold.txt"]:
        if not os.path.exists(os.path.join(MODELS_DIR, fname)):
            print(f"ERROR: {fname} not found in models/. Run v3_classifier.py first.")
            return

    model     = pickle.load(open(os.path.join(MODELS_DIR, "v3_classifier.pkl"),  "rb"))
    vec_name  = pickle.load(open(os.path.join(MODELS_DIR, "v3_vec_name.pkl"),    "rb"))
    vec_addr  = pickle.load(open(os.path.join(MODELS_DIR, "v3_vec_addr.pkl"),    "rb"))
    threshold = float(open(os.path.join(MODELS_DIR, "v3_threshold.txt")).read().strip())
    print(f"  Loaded model | threshold = {threshold:.2f}")

    print("\nLoading Test Dataset (this may take ~1 min)...")
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
        if c_s1.empty:
            continue

        print(f"\n[{country}] S1={len(c_s1):,}, S23={len(c_s23):,}")

        # FIX: build s23_dict ONCE per country, not per chunk (big memory win)
        s23_dict = c_s23.set_index("entity_id")[
            ["business_name", "business_address"]].to_dict("index")

        n_chunks = (len(c_s1) + S1_CHUNK_SIZE - 1) // S1_CHUNK_SIZE
        for i, start in enumerate(range(0, len(c_s1), S1_CHUNK_SIZE)):
            end   = min(start + S1_CHUNK_SIZE, len(c_s1))
            chunk = c_s1.iloc[start:end]
            print(f"  chunk {i+1}/{n_chunks}  rows {start:,}–{end:,}", end="  ", flush=True)
            t1 = time.time()
            df_m = process_chunk(chunk, c_s23, s23_dict, model, vec_name, vec_addr, threshold)
            all_matches.append(df_m)
            print(f"-> {len(df_m):,} matches  ({time.time()-t1:.1f}s)")

    print("\nFormatting for leaderboard...")
    # FIX: safe concat even when all_matches contains only empty DataFrames
    final_matches = pd.concat(
        [df for df in all_matches if not df.empty],
        ignore_index=True
    ) if any(not df.empty for df in all_matches) else pd.DataFrame(columns=["entity_id", "cid"])

    res = (final_matches
           .groupby("entity_id")["cid"]
           .apply(lambda x: ",".join(x.unique()))
           .reset_index())
    res.columns = ["source1_entity_id", "matched_entity_ids"]

    # All S1 entities must appear (singletons get empty matched_entity_ids)
    out_df = (pd.DataFrame({"source1_entity_id": s1["entity_id"]})
              .merge(res, on="source1_entity_id", how="left")
              .fillna(""))

    out_path = os.path.join(OUTPUT_DIR, "matching_results.tsv")
    out_df.to_csv(out_path, sep="\t", index=False)

    matched = (out_df["matched_entity_ids"] != "").sum()
    print(f"\nSUCCESS! {out_path}")
    print(f"  Total S1 rows:    {len(out_df):,}")
    print(f"  Entities matched: {matched:,}  ({matched/len(out_df):.1%})")
    print(f"  Total time:       {time.time()-t0:.0f}s")
    print("\nNext step — validate before submitting:")
    print("  python utils/validate_submission.py \\")
    print("      --matching output/matching_results.tsv \\")
    print("      --candidate output/candidate_pairs.tsv \\")
    print("      --test-dir dataset/test")


if __name__ == "__main__":
    main()
