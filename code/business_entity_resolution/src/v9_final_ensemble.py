"""
v9 — final · ensemble + polish
================================
The final full-test-set inference pipeline. Combines everything built in
v1-v8:

  - Blocking: TF-IDF word keys + Soundex (v2) + Double Metaphone (v6),
    country-filtered, hot-key capped.
  - Scoring: v6's XGBoost classifier on 11 features (string distance +
    TF-IDF cosine + multilingual embedding cosine).
  - Threshold: re-swept on the FULL mini_train set (not just val) now
    that architecture is locked, per VERSIONS.md v9 spec.
  - LLM overrides: if output/v7_llm_decisions.tsv exists, any (s1_id, cid)
    pair judged "no match" by Qwen is removed from the final output even
    if the classifier said yes (precision-first: LLM veto only, never an
    LLM-only add -- the classifier already gated what reached the jury).
  - Defensive cleanup: dedupe matched_entity_ids per row, ensure every
    test S1 entity has exactly one row (including singletons).

Run order to reach this point:
    v1_baseline.py -> v2_blocking.py -> v3_classifier.py ->
    v4_calibration.py -> v5_embeddings.py ->
    v6_blocking_metaphone.py -> v6_xgboost.py ->
    v4_calibration.py again (re-calibrate v6's scores) ->
    v7_qwen_jury.py (optional but recommended) ->
    v9_final_ensemble.py  <- YOU ARE HERE (generates the submission)
    v8_graph_consistency.py  <- run AFTER v9, it edits matching_results.tsv in place
    utils/validate_submission.py  <- ALWAYS run before uploading
"""

import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import os, re, time, pickle, argparse
import numpy as np
import pandas as pd
import jellyfish
from sentence_transformers import SentenceTransformer

try:
    from metaphone import doublemetaphone
    HAVE_DOUBLE_METAPHONE = True
except ImportError:
    HAVE_DOUBLE_METAPHONE = False

REPO_ROOT   = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
                os.path.abspath(__file__)))))
TRAIN_DIR   = os.path.join(REPO_ROOT, "dataset", "mini_train")
TEST_DIR    = os.path.join(REPO_ROOT, "dataset", "test")
OUTPUT_DIR  = os.path.join(REPO_ROOT, "output")
MODELS_DIR  = os.path.join(REPO_ROOT, "models")

EMBED_MODEL_NAME  = "paraphrase-multilingual-MiniLM-L12-v2"
EMBED_BATCH_SIZE  = 256
MAX_PAIRS_PER_KEY = 150_000
PER_ENTITY_CAP    = 100
S1_CHUNK_SIZE     = 5_000
K                 = 50
FEATURE_CHUNK_SIZE = 200_000   # used by final_threshold_sweep() -- same fix as
                                 # v6_xgboost.py's MemoryError (13.7M-row candidate
                                 # set can't be vec.transform()'d / np.stack()'d whole)

FEATURES = [
    "name_jw", "name_lev", "name_tfidf_cosine",
    "addr_jw", "addr_lev", "addr_tfidf_cosine",
    "num_overlap", "name_x_addr", "lookalike_flag",
    "name_embed_cosine", "addr_embed_cosine",
]

_PUNCT_RE = re.compile(r'[^\w\s]')
_SPACE_RE = re.compile(r'\s+')

def norm(text):
    return _SPACE_RE.sub(' ', _PUNCT_RE.sub(' ', str(text).lower().strip())).strip()

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

def metaphone_codes(word):
    if len(word) < 4:
        return []
    if HAVE_DOUBLE_METAPHONE:
        p, a = doublemetaphone(word)
        return [c for c in (p, a) if c]
    code = jellyfish.metaphone(word)
    return [code] if code else []

def get_keys(nn, na):
    nw, aw = nn.split(), na.split()
    keys = set()
    if len(nw) >= 2:
        acr = "".join(w[0] for w in nw if w)
        if len(acr) >= 2: keys.add(f"acr:{acr}")
    for w in nw + aw:
        if len(w) >= 3: keys.add(f"w:{w}")
        if len(w) >= 5: keys.add(f"s:{jellyfish.soundex(w)}")
        for mp in metaphone_codes(w): keys.add(f"mp:{mp}")
    return list(keys)

def cos_rows(a, b):
    num = (a * b).sum(axis=1)
    den = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
    den[den == 0] = 1e-9
    return num / den


def process_chunk(chunk, s23_key_df, s23_names, s23_addrs, s23_name_vecs, s23_addr_vecs,
                   s23_id2idx, name2vec, addr2vec, model, vec_name, vec_addr):
    # NOTE: no `threshold` param -- this now returns the full top-K scored
    # pool. The caller (main()) splits it into candidates (everything) vs
    # matches (prob >= threshold).
    EMPTY = pd.DataFrame(columns=["entity_id", "cid", "prob"])
    if chunk.empty: return EMPTY

    chunk = chunk.copy()
    chunk["bkeys"] = chunk.apply(lambda r: get_keys(r["nn"], r["na"]), axis=1)
    s1_exp = (chunk[["entity_id", "bkeys"]].explode("bkeys")
              .dropna(subset=["bkeys"]).rename(columns={"bkeys": "bkey"}))
    s1_exp = s1_exp[s1_exp["bkey"].str.len() > 3]
    if s1_exp.empty: return EMPTY

    valid_keys = set(s1_exp["bkey"].unique())
    s23_rel = s23_key_df[s23_key_df["bkey"].isin(valid_keys)]
    if s23_rel.empty: return EMPTY

    c1, c23 = s1_exp["bkey"].value_counts(), s23_rel["bkey"].value_counts()
    safe = set(k for k, n1 in c1.items() if n1 * c23.get(k, 0) <= MAX_PAIRS_PER_KEY)

    joined = (s1_exp[s1_exp["bkey"].isin(safe)]
             .merge(s23_rel[s23_rel["bkey"].isin(safe)].rename(columns={"entity_id": "cid"}), on="bkey")
             [["entity_id", "cid"]])
             
    if joined.empty: return EMPTY

    scored = joined.groupby(["entity_id", "cid"]).size().reset_index(name="shared_key_count")
    scored = scored.sort_values(["entity_id", "shared_key_count"], ascending=[True, False])
    pairs = scored.groupby("entity_id").head(PER_ENTITY_CAP)[["entity_id", "cid"]]

    s1n_full = chunk["business_name"].fillna("").str.lower()
    s1a_full = chunk["business_address"].fillna("").str.lower()
    s1n_v = vec_name.transform(s1n_full)
    s1a_v = vec_addr.transform(s1a_full)
    s1_id2idx = {eid: i for i, eid in enumerate(chunk["entity_id"])}

    pairs["s1_idx"]  = pairs["entity_id"].map(s1_id2idx)
    pairs["s23_idx"] = pairs["cid"].map(s23_id2idx)

    M_CHUNK = 500_000
    n_scores, a_scores = [], []
    for st in range(0, len(pairs), M_CHUNK):
        ch = pairs.iloc[st:st+M_CHUNK]
        v1n, v23n = s1n_v[ch["s1_idx"].values], s23_name_vecs[ch["s23_idx"].values]
        n_scores.extend(np.array(v1n.multiply(v23n).sum(axis=1)).flatten())
        v1a, v23a = s1a_v[ch["s1_idx"].values], s23_addr_vecs[ch["s23_idx"].values]
        a_scores.extend(np.array(v1a.multiply(v23a).sum(axis=1)).flatten())
    pairs["name_tfidf_cosine"], pairs["addr_tfidf_cosine"] = n_scores, a_scores
    pairs["coarse_score"] = pairs["name_tfidf_cosine"] + pairs["addr_tfidf_cosine"]

    pairs = pairs.sort_values(["entity_id", "coarse_score"], ascending=[True, False]).groupby("entity_id").head(K)

    s1_rows_k  = [s1_id2idx[eid] for eid in pairs["entity_id"]]
    s23_rows_k = [s23_id2idx[cid] for cid in pairs["cid"]]
    s1n = s1n_full.values[s1_rows_k]; s1a = s1a_full.values[s1_rows_k]
    cn  = s23_names[s23_rows_k];      ca  = s23_addrs[s23_rows_k]

    pairs["name_jw"]  = [safe_jw(a, b)  for a, b in zip(s1n, cn)]
    pairs["name_lev"] = [safe_lev(a, b) for a, b in zip(s1n, cn)]
    pairs["addr_jw"]  = [safe_jw(a, b)  for a, b in zip(s1a, ca)]
    pairs["addr_lev"] = [safe_lev(a, b) for a, b in zip(s1a, ca)]
    pairs["num_overlap"] = [num_overlap(a, b) for a, b in zip(s1a, ca)]
    pairs["name_x_addr"]    = pairs["name_tfidf_cosine"] * pairs["addr_tfidf_cosine"]
    pairs["lookalike_flag"] = ((pairs["name_jw"] > 0.90) & (pairs["addr_jw"] < 0.50)).astype(int)

    v1n_e = np.stack([name2vec[t] for t in s1n]); v2n_e = np.stack([name2vec[t] for t in cn])
    v1a_e = np.stack([addr2vec[t] for t in s1a]); v2a_e = np.stack([addr2vec[t] for t in ca])
    pairs["name_embed_cosine"] = cos_rows(v1n_e, v2n_e)
    pairs["addr_embed_cosine"] = cos_rows(v1a_e, v2a_e)

    pairs["prob"] = model.predict_proba(pairs[FEATURES])[:, 1]
    # Return the FULL top-K scored pool (not just pairs that passed threshold).
    # This is what candidate_pairs.tsv is supposed to contain per the spec --
    # "the final candidate list just before the ML model scores them" -- and
    # what candidate_entity_ids the score gets computed against, not just the
    # winners. The caller splits this into candidates vs matches.
    return pairs[["entity_id", "cid", "prob"]]


def final_threshold_sweep(model, vec_name, vec_addr, name2vec, addr2vec):
    """Re-sweeps the threshold on the FULL mini_train set (train+val together,
    architecture now locked -- per VERSIONS.md v9 spec) using v6's candidate
    pairs. Falls back to v6_threshold.txt if anything is missing.

    CHUNKED (fixed after the same MemoryError pattern hit v6_xgboost.py at
    13.7M candidate rows): this used to call vec_name.transform()/np.stack()
    on the entire candidate_pairs.tsv in one shot -- exactly the bug that
    crashed v6_xgboost.py's feature-building step. Now processes
    FEATURE_CHUNK_SIZE rows at a time and only keeps the small resulting
    feature columns between chunks, same pattern as v6_xgboost.py and
    generate_debug_scores.py.
    """
    fallback_path = os.path.join(MODELS_DIR, "v6_threshold.txt")
    fallback = float(open(fallback_path).read().strip()) if os.path.exists(fallback_path) else 0.5

    cands_path = os.path.join(OUTPUT_DIR, "candidate_pairs.tsv")
    gt_path    = os.path.join(TRAIN_DIR, "train_ground_truth.tsv")
    if not (os.path.exists(cands_path) and os.path.exists(gt_path)):
        print(f"  (skipping full-set sweep, missing files -- using v6 threshold {fallback:.2f})")
        return fallback

    s1  = pd.read_csv(os.path.join(TRAIN_DIR, "train_source1.tsv"), sep="\t", dtype=str).fillna("")
    s23 = pd.concat([
        pd.read_csv(os.path.join(TRAIN_DIR, "train_source2.tsv"), sep="\t", dtype=str).fillna(""),
        pd.read_csv(os.path.join(TRAIN_DIR, "train_source3.tsv"), sep="\t", dtype=str).fillna(""),
    ], ignore_index=True)
    gt = pd.read_csv(gt_path, sep="\t", dtype=str).fillna("")
    cands_df = pd.read_csv(cands_path, sep="\t", dtype=str).fillna("")

    gt_map = {}
    for row in gt.itertuples(index=False):
        gt_map[row.source1_entity_id] = set(row.matched_entity_ids.split(",")) - {""}

    pairs = []
    for row in cands_df.itertuples(index=False):
        s1_id, c_ids = row.source1_entity_id, set(row.candidate_entity_ids.split(",")) - {""}
        true_set = gt_map.get(s1_id, set())
        for cid in c_ids:
            pairs.append({"s1_id": s1_id, "cid": cid, "label": int(cid in true_set)})
    df = pd.DataFrame(pairs)
    print(f"  {len(df):,} full-set candidate pairs to score (chunked at {FEATURE_CHUNK_SIZE:,} rows)")

    s1_dict  = s1.set_index("entity_id")[["business_name", "business_address"]].to_dict("index")
    s23_dict = s23.set_index("entity_id")[["business_name", "business_address"]].to_dict("index")
    s1n_all = [str(s1_dict.get(i, {}).get("business_name",  "")).lower().strip() for i in df["s1_id"]]
    s1a_all = [str(s1_dict.get(i, {}).get("business_address","")).lower().strip() for i in df["s1_id"]]
    cn_all  = [str(s23_dict.get(i, {}).get("business_name",  "")).lower().strip() for i in df["cid"]]
    ca_all  = [str(s23_dict.get(i, {}).get("business_address","")).lower().strip() for i in df["cid"]]

    n = len(df)
    s1_ids_arr = df["s1_id"].to_numpy(); cids_arr = df["cid"].to_numpy(); labels_arr = df["label"].to_numpy()
    n_chunks = (n + FEATURE_CHUNK_SIZE - 1) // FEATURE_CHUNK_SIZE
    parts = []
    t_chunk = time.time()

    for i, start in enumerate(range(0, n, FEATURE_CHUNK_SIZE)):
        end = min(start + FEATURE_CHUNK_SIZE, n)
        c_s1n, c_s1a = s1n_all[start:end], s1a_all[start:end]
        c_cn,  c_ca  = cn_all[start:end],  ca_all[start:end]

        chunk = pd.DataFrame({"s1_id": s1_ids_arr[start:end], "cid": cids_arr[start:end],
                               "label": labels_arr[start:end]})
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

        missing_n = [t for t in set(c_s1n + c_cn) if t not in name2vec]
        missing_a = [t for t in set(c_s1a + c_ca) if t not in addr2vec]
        if missing_n or missing_a:
            embedder = SentenceTransformer(EMBED_MODEL_NAME)
            if missing_n:
                for t, v in zip(missing_n, embedder.encode(missing_n, convert_to_numpy=True)):
                    name2vec[t] = v
            if missing_a:
                for t, v in zip(missing_a, embedder.encode(missing_a, convert_to_numpy=True)):
                    addr2vec[t] = v
        v1n = np.stack([name2vec[t] for t in c_s1n]); v2n = np.stack([name2vec[t] for t in c_cn])
        v1a = np.stack([addr2vec[t] for t in c_s1a]); v2a = np.stack([addr2vec[t] for t in c_ca])
        chunk["name_embed_cosine"] = cos_rows(v1n, v2n)
        chunk["addr_embed_cosine"] = cos_rows(v1a, v2a)

        chunk["prob"] = model.predict_proba(chunk[FEATURES])[:, 1]
        parts.append(chunk[["s1_id", "cid", "prob", "label"]])
        del s1n_v, cn_v, s1a_v, ca_v, v1n, v2n, v1a, v2a
        if (i + 1) % 10 == 0 or (i + 1) == n_chunks:
            print(f"    chunk {i+1}/{n_chunks} ({end:,}/{n:,} rows, {time.time()-t_chunk:.0f}s elapsed)")

    df = pd.concat(parts, ignore_index=True)
    unique_s1 = df["s1_id"].unique()

    def macro_f05_at(t):
        pm = {}
        for row in df.itertuples(index=False):
            pm.setdefault(row.s1_id, set())
            if row.prob >= t:
                pm[row.s1_id].add(row.cid)
        scores = []
        for s1_id in unique_s1:
            gt_set, pred_set = gt_map.get(s1_id, set()), pm.get(s1_id, set())
            if not gt_set and not pred_set:
                scores.append(1.0); continue
            tp = len(gt_set & pred_set)
            p = tp/len(pred_set) if pred_set else 0.0
            r = tp/len(gt_set) if gt_set else 0.0
            scores.append(1.25*p*r/(0.25*p+r) if (p+r) > 0 else 0.0)
        return sum(scores)/len(scores) if scores else 0.0

    best_t, best_f05 = fallback, -1.0
    for t in np.arange(0.05, 0.97, 0.02):
        f05 = macro_f05_at(t)
        if f05 > best_f05:
            best_f05, best_t = f05, t
    print(f"  Full-set threshold sweep: best_t={best_t:.2f}, F0.5={best_f05:.4f}")
    return best_t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-inference", action="store_true",
                     help="Skip the expensive blocking+scoring pass over the test set "
                          "and instead reload output/test_candidate_scores.tsv (written "
                          "by a prior full run). Use this for a cheap second pass after "
                          "running v7_qwen_jury.py --source test against the scores this "
                          "script exported -- avoids recomputing all test-set candidates "
                          "just to apply the LLM overrides + threshold.")
    args = ap.parse_args()

    print("=" * 60)
    print("v9 — final ensemble + full test inference")
    print("=" * 60)
    t0 = time.time()

    for fname in ["v6_classifier.pkl", "v6_vec_name.pkl", "v6_vec_addr.pkl"]:
        if not os.path.exists(os.path.join(MODELS_DIR, fname)):
            print(f"ERROR: {fname} missing. Run v6_xgboost.py first.")
            return

    model    = pickle.load(open(os.path.join(MODELS_DIR, "v6_classifier.pkl"), "rb"))
    vec_name = pickle.load(open(os.path.join(MODELS_DIR, "v6_vec_name.pkl"),   "rb"))
    vec_addr = pickle.load(open(os.path.join(MODELS_DIR, "v6_vec_addr.pkl"),   "rb"))

    print("\n[1/5] Re-sweeping threshold on the full training set...")
    embedder_cache_name, embedder_cache_addr = {}, {}
    threshold = final_threshold_sweep(model, vec_name, vec_addr, embedder_cache_name, embedder_cache_addr)

    s1 = pd.read_csv(os.path.join(TEST_DIR, "test_source1.tsv"), sep="\t", dtype=str).fillna("")
    scores_path = os.path.join(OUTPUT_DIR, "test_candidate_scores.tsv")

    if args.skip_inference:
        print(f"\n[2/5] --skip-inference: reloading {os.path.basename(scores_path)} "
              "instead of recomputing...")
        if not os.path.exists(scores_path):
            print(f"ERROR: {scores_path} missing -- can't skip inference without a prior "
                  "full run to reload from. Run without --skip-inference first.")
            return
        final_candidates = pd.read_csv(scores_path, sep="\t", dtype={"s1_id": str, "cid": str})
        final_candidates = final_candidates.rename(columns={"s1_id": "entity_id"})
        print(f"  Reloaded {len(final_candidates):,} candidate scores")
    else:
        print(f"\n[2/5] Loading test set + embedder ({EMBED_MODEL_NAME})...")
        embedder = SentenceTransformer(EMBED_MODEL_NAME)
        s23 = pd.concat([
            pd.read_csv(os.path.join(TEST_DIR, "test_source2.tsv"), sep="\t", dtype=str).fillna(""),
            pd.read_csv(os.path.join(TEST_DIR, "test_source3.tsv"), sep="\t", dtype=str).fillna(""),
        ], ignore_index=True)
        s1["nn"]  = s1["business_name"].apply(norm)
        s1["na"]  = s1["business_address"].fillna("").apply(norm)
        s23["nn"] = s23["business_name"].apply(norm)
        s23["na"] = s23["business_address"].fillna("").apply(norm)
        print(f"  S1: {len(s1):,}  |  S23: {len(s23):,}")

        all_matches, all_candidates = [], []

        # NOTE: country is treated as an open string label -- iterating over
        # whatever values appear (US, India, France, or anything else) rather
        # than a hardcoded list. This is the France-generalization requirement.
        for country in s1["country"].unique():
            c_s1  = s1[s1["country"] == country].reset_index(drop=True)
            c_s23 = s23[s23["country"] == country].reset_index(drop=True)
            if c_s1.empty: continue
            print(f"\n[{country}] S1={len(c_s1):,}, S23={len(c_s23):,}")

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

            # Cache embeddings for this country's S23 texts once
            for t in set(s23_names):
                if t not in embedder_cache_name:
                    pass  # filled in batch below
            uniq_names = sorted(set(s23_names) - embedder_cache_name.keys())
            uniq_addrs = sorted(set(s23_addrs) - embedder_cache_addr.keys())
            if uniq_names:
                for t, v in zip(uniq_names, embedder.encode(uniq_names, batch_size=EMBED_BATCH_SIZE,
                                                              show_progress_bar=True, convert_to_numpy=True)):
                    embedder_cache_name[t] = v
            if uniq_addrs:
                for t, v in zip(uniq_addrs, embedder.encode(uniq_addrs, batch_size=EMBED_BATCH_SIZE,
                                                              show_progress_bar=True, convert_to_numpy=True)):
                    embedder_cache_addr[t] = v

            n_chunks = (len(c_s1) + S1_CHUNK_SIZE - 1) // S1_CHUNK_SIZE
            for i, start in enumerate(range(0, len(c_s1), S1_CHUNK_SIZE)):
                end = min(start + S1_CHUNK_SIZE, len(c_s1))
                chunk = c_s1.iloc[start:end]

                # Cache this chunk's own text embeddings too
                cn_texts = chunk["business_name"].fillna("").str.lower().tolist()
                ca_texts = chunk["business_address"].fillna("").str.lower().tolist()
                new_n = [t for t in set(cn_texts) if t not in embedder_cache_name]
                new_a = [t for t in set(ca_texts) if t not in embedder_cache_addr]
                if new_n:
                    for t, v in zip(new_n, embedder.encode(new_n, convert_to_numpy=True)):
                        embedder_cache_name[t] = v
                if new_a:
                    for t, v in zip(new_a, embedder.encode(new_a, convert_to_numpy=True)):
                        embedder_cache_addr[t] = v

                df_pool = process_chunk(chunk, s23_key_df, s23_names, s23_addrs, s23_name_vecs, s23_addr_vecs,
                                         s23_id2idx, embedder_cache_name, embedder_cache_addr,
                                         model, vec_name, vec_addr)
                all_candidates.append(df_pool)
                print(f"  chunk {i+1}/{n_chunks} rows {start:,}-{end:,} -> "
                      f"{len(df_pool):,} candidates ({(df_pool['prob'] >= threshold).sum():,} above threshold)")

        final_candidates = pd.concat([d for d in all_candidates if not d.empty], ignore_index=True) \
                            if any(not d.empty for d in all_candidates) else pd.DataFrame(columns=["entity_id", "cid", "prob"])

        # Export the FULL scored pool to disk BEFORE thresholding/LLM overrides.
        # This is what makes `v7_qwen_jury.py --source test` possible -- it needs
        # real test-set (s1_id, cid, prob) rows to find the actual ambiguous band,
        # not the mini_train validation rows in debug_scores.tsv. Without this
        # export, any v7_llm_decisions.tsv you generate would reference training
        # entity IDs that don't exist in the test set and would silently apply
        # zero vetoes to the real submission.
        tmp_path = scores_path + ".tmp"
        final_candidates.rename(columns={"entity_id": "s1_id"}).to_csv(tmp_path, sep="\t", index=False)
        os.replace(tmp_path, scores_path)
        print(f"\n  Exported {len(final_candidates):,} candidate scores to "
              f"{os.path.basename(scores_path)} (for `v7_qwen_jury.py --source test`)")

    print("\n[3/5] Applying LLM overrides (if v7_llm_decisions.tsv exists)...")
    final_matches = final_candidates[final_candidates["prob"] >= threshold].copy()

    llm_path = os.path.join(OUTPUT_DIR, "v7_llm_decisions.tsv")
    if os.path.exists(llm_path):
        llm_df = pd.read_csv(llm_path, sep="\t", dtype=str)
        llm_df["llm_match"] = llm_df["llm_match"].astype(str).str.lower() == "true"
        veto = set(zip(llm_df.loc[~llm_df["llm_match"], "s1_id"], llm_df.loc[~llm_df["llm_match"], "cid"]))
        before = len(final_matches)
        # LLM veto only removes pairs from the FINAL matches -- a vetoed pair
        # was still genuinely considered (it was in the ambiguous band the
        # classifier + blocking produced), so it correctly stays in
        # candidate_pairs.tsv per the spec ("every ID in matching_results.tsv
        # should appear here" -- the reverse isn't required, and a vetoed
        # candidate is real evidence of what the pipeline considered).
        final_matches = final_matches[~final_matches.apply(lambda r: (r["entity_id"], r["cid"]) in veto, axis=1)]
        print(f"  Applied {before - len(final_matches):,} LLM vetoes")
    else:
        print("  No v7_llm_decisions.tsv found -- skipping LLM override step "
              "(run v7_qwen_jury.py first if you want it).")

    print("\n[4/5] Writing candidate_pairs.tsv and matching_results.tsv...")
    cand_res = (final_candidates.groupby("entity_id")["cid"]
                .apply(lambda x: ",".join(sorted(set(x)))).reset_index())
    cand_res.columns = ["source1_entity_id", "candidate_entity_ids"]
    cand_out = (pd.DataFrame({"source1_entity_id": s1["entity_id"]})
                .merge(cand_res, on="source1_entity_id", how="left").fillna(""))
    cand_out.to_csv(os.path.join(OUTPUT_DIR, "candidate_pairs.tsv"), sep="\t", index=False)

    match_res = (final_matches.groupby("entity_id")["cid"]
                 .apply(lambda x: ",".join(sorted(set(x)))).reset_index())
    match_res.columns = ["source1_entity_id", "matched_entity_ids"]
    match_out = (pd.DataFrame({"source1_entity_id": s1["entity_id"]})
                 .merge(match_res, on="source1_entity_id", how="left").fillna(""))
    match_out.to_csv(os.path.join(OUTPUT_DIR, "matching_results.tsv"), sep="\t", index=False)

    matched = (match_out["matched_entity_ids"] != "").sum()
    avg_candidates = len(final_candidates) / len(s1) if len(s1) else 0.0
    print(f"\n[5/5] SUCCESS!")
    print(f"  Total S1 rows:          {len(match_out):,}")
    print(f"  Entities matched:       {matched:,}  ({matched/len(match_out):.1%})")
    print(f"  Avg candidates/entity:  {avg_candidates:.1f}  "
          f"(this is now a scored criterion per Amazon's update -- lower is better,")
    print(f"                          provided recall/F0.5 hold)")
    print(f"  Threshold used:         {threshold:.2f}")
    print(f"  Total time:             {time.time()-t0:.0f}s")
    print("\n  NEXT: run v8_graph_consistency.py to prune inconsistent triangles,")
    print("        then utils/validate_submission.py before uploading.")

if __name__ == "__main__":
    main()