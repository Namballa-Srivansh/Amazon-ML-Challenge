"""
v10_final_ensemble.py
=======================
Architectural rewrite of the final test-set inference pipeline, designed
to fit a hard 16GB RAM ceiling at full scale (US alone: S1=663k rows,
S23=3.8M rows). Supersedes v9_final_ensemble.py and generate_v3_submission.py.

WHAT WENT WRONG IN v9 AND WHAT IS STRUCTURALLY DIFFERENT HERE

  1. Embeddings held in a Python dict (~22GB at float32 for the S23 corpus).
     -> embedding_store.py: SQLite-backed. Encode once, persist, and pull
        only the vectors the CURRENT chunk needs into RAM. The full corpus
        is never resident. Also resumable across runs and crash-safe
        (commits every batch), replacing the fragile "pickle after the loop".

  2. Every chunk's result appended to a list, concatenated at the end.
     -> Every chunk is written to disk as soon as it is scored (scores,
        candidate_pairs.tsv and matching_results.tsv are all streamed).
        Peak RAM is bounded by one chunk, not by the test set.

  3. Blocking via a pandas MERGE of S1 keys against S23 keys. A merge has
     no natural bound: one key shared by thousands of records on each side
     yields a cross-product for that key alone (the 86M-row chunks).
     -> Capped inverted index {key: [cid, ...]} built once per country
        (each key's list capped at HOT_KEY_CAP), then per-entity dict
        lookups with Counter-based ranking. Work is
        O(keys_per_entity x list_len) per entity; a bad constant costs
        recall on a hot key, never an OOM or a 10-minute stall. Because it
        is bounded by construction, Double Metaphone keys are safe to use.

  4. `.explode()` of S23 keys into a 15M+ row DataFrame before the loop.
     -> The index is built incrementally from S23 in chunks straight into
        a dict; the exploded intermediate never exists.

No gc.collect() / torch.cuda.empty_cache() calls: if those seem necessary,
something is being retained longer than it should be, and that should be
fixed in the data flow, not with GC hints.

NOT MEASURED: the memory/time figures quoted in comments are estimates.
Run it on a small slice first (see --limit-s1 / --limit-s23) and watch RSS
before launching a full run.

Usage:
    python v10_final_ensemble.py --threshold 0.95
    python v10_final_ensemble.py --limit-s1 5000 --limit-s23 200000   # dry run
"""

import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import os, re, time, pickle, argparse, shutil
from collections import Counter, defaultdict
import numpy as np
import pandas as pd
import jellyfish

from embedding_store import EmbeddingStore

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

EMBED_MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"

# These bound the WORST case. A too-small value costs recall on individual
# hot keys/entities; it cannot cause an OOM or a combinatorial stall.
HOT_KEY_CAP     = 2_000    # max candidates kept per blocking key
PER_ENTITY_CAP  = 150      # max candidates per S1 entity after key-overlap ranking
K               = 50       # final top-K per entity by coarse TF-IDF score
S1_CHUNK_SIZE   = 2_000    # S1 rows scored (and flushed to disk) per iteration
S23_INDEX_CHUNK = 50_000   # S23 rows consumed per iteration when building the index
SWEEP_CHUNK     = 200_000  # pairs per iteration in the training threshold sweep

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
    if len(word) < 4: return []
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
    return keys

def cos_rows(a, b):
    num = (a * b).sum(axis=1)
    den = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
    den[den == 0] = 1e-9
    return num / den

def tfidf_pair_cosine(vec, texts_a, texts_b):
    """Row-wise TF-IDF cosine between aligned text lists, vectorizing each
    UNIQUE string exactly once. Pair lists repeat the same S1 text once per
    candidate (up to PER_ENTITY_CAP times) and popular S23 texts across many
    entities; calling vec.transform() on the raw pair lists re-does that work
    every time and dominated runtime in profiling (~80%). Same numbers,
    far fewer transforms."""
    ua = {t: i for i, t in enumerate(dict.fromkeys(texts_a))}
    ub = {t: i for i, t in enumerate(dict.fromkeys(texts_b))}
    ma = vec.transform(list(ua))
    mb = vec.transform(list(ub))
    ia = np.fromiter((ua[t] for t in texts_a), dtype=np.int64, count=len(texts_a))
    ib = np.fromiter((ub[t] for t in texts_b), dtype=np.int64, count=len(texts_b))
    return np.asarray(ma[ia].multiply(mb[ib]).sum(axis=1)).ravel()


# ---------------------------------------------------------------------------
# Blocking: capped inverted index (replaces the pandas merge)
# ---------------------------------------------------------------------------
def build_inverted_index(s23_df: pd.DataFrame) -> dict:
    """{blocking_key: [cid, ...]}, built chunk by chunk. A key that reaches
    HOT_KEY_CAP is closed (later occurrences skipped in O(1), so a generic
    token cannot cost unbounded memory or time) and is removed at the end."""
    index = defaultdict(list)
    closed = set()
    n = len(s23_df)
    t0 = time.time()
    for start in range(0, n, S23_INDEX_CHUNK):
        end = min(start + S23_INDEX_CHUNK, n)
        part = s23_df.iloc[start:end]
        for eid, nn, na in zip(part["entity_id"], part["nn"], part["na"]):
            for key in get_keys(nn, na):
                if key in closed:
                    continue
                lst = index[key]
                lst.append(eid)
                if len(lst) >= HOT_KEY_CAP:
                    closed.add(key)
        if ((end // S23_INDEX_CHUNK) % 10 == 0) or end == n:
            print(f"    index: {end:,}/{n:,} S23 rows | {len(index):,} keys | "
                  f"{len(closed):,} capped | {time.time()-t0:.0f}s")
    # A capped key holds only an arbitrary first-come sample of >= HOT_KEY_CAP
    # records -- noise as a blocking signal, and it costs time on every lookup.
    # Treat it as a stop-word and remove it (also frees memory).
    for key in closed:
        del index[key]
    print(f"    dropped {len(closed):,} stop-word keys; {len(index):,} keys retained")
    return dict(index)


def score_s1_chunk(chunk, inv_index, s23_text, vec_name, vec_addr,
                    embed_store, embedder, model) -> pd.DataFrame:
    """Candidates for one S1 chunk -> scored DataFrame [entity_id, cid, prob].
    Everything here is sized by the chunk (<= S1_CHUNK_SIZE x K rows)."""
    empty = pd.DataFrame(columns=["entity_id", "cid", "prob"])

    rows = []
    for eid, nn, na in zip(chunk["entity_id"], chunk["nn"], chunk["na"]):
        counter = Counter()
        # sorted(): set iteration order depends on the per-process hash seed, and
        # it decides how ties in shared-key count are broken. Without this the
        # candidate set differs between runs on identical input.
        for key in sorted(get_keys(nn, na)):
            for cid in inv_index.get(key, ()):
                counter[cid] += 1
        for cid, _ in counter.most_common(PER_ENTITY_CAP):
            rows.append((eid, cid))
    if not rows:
        return empty

    pairs = pd.DataFrame(rows, columns=["entity_id", "cid"])
    s1_text = {e: (nm.lower().strip(), ad.lower().strip()) for e, nm, ad in
               zip(chunk["entity_id"], chunk["business_name"], chunk["business_address"])}

    def texts(df):
        return (df["entity_id"].map(lambda e: s1_text[e][0]).values,
                df["entity_id"].map(lambda e: s1_text[e][1]).values,
                df["cid"].map(lambda c: s23_text[c][0]).values,
                df["cid"].map(lambda c: s23_text[c][1]).values)

    s1n, s1a, cn, ca = texts(pairs)
    n_v = tfidf_pair_cosine(vec_name, s1n, cn)
    a_v = tfidf_pair_cosine(vec_addr, s1a, ca)
    pairs["name_tfidf_cosine"], pairs["addr_tfidf_cosine"] = n_v, a_v
    pairs["coarse_score"] = n_v + a_v
    pairs = (pairs.sort_values(["entity_id", "coarse_score", "cid"], ascending=[True, False, True])
             .groupby("entity_id").head(K).reset_index(drop=True))

    s1n, s1a, cn, ca = texts(pairs)
    pairs["name_jw"]  = [safe_jw(a, b)  for a, b in zip(s1n, cn)]
    pairs["name_lev"] = [safe_lev(a, b) for a, b in zip(s1n, cn)]
    pairs["addr_jw"]  = [safe_jw(a, b)  for a, b in zip(s1a, ca)]
    pairs["addr_lev"] = [safe_lev(a, b) for a, b in zip(s1a, ca)]
    pairs["num_overlap"] = [num_overlap(a, b) for a, b in zip(s1a, ca)]
    pairs["name_x_addr"]    = pairs["name_tfidf_cosine"] * pairs["addr_tfidf_cosine"]
    pairs["lookalike_flag"] = ((pairs["name_jw"] > 0.90) & (pairs["addr_jw"] < 0.50)).astype(int)

    # Only this chunk's post-cap texts are fetched from disk.
    needed = list(set(s1n) | set(cn) | set(s1a) | set(ca))
    embed_store.encode_and_store_missing(needed, embedder, show_progress=False)
    vecs = embed_store.get_many(needed)
    f32 = lambda arr: np.stack([vecs[t].astype(np.float32) for t in arr])
    pairs["name_embed_cosine"] = cos_rows(f32(s1n), f32(cn))
    pairs["addr_embed_cosine"] = cos_rows(f32(s1a), f32(ca))

    pairs["prob"] = model.predict_proba(pairs[FEATURES])[:, 1]
    return pairs[["entity_id", "cid", "prob"]]


class StreamingTSVWriter:
    """One open handle, header written once, rows appended as produced."""
    def __init__(self, path, columns):
        self.columns = columns
        self.f = open(path, "w", newline="", encoding="utf-8")
        self.f.write("\t".join(columns) + "\n")

    def write_rows(self, rows):
        for r in rows:
            self.f.write("\t".join(r) + "\n")
        self.f.flush()

    def close(self):
        self.f.close()


# ---------------------------------------------------------------------------
# Training-side threshold sweep
# ---------------------------------------------------------------------------
def ensure_train_candidates_backup():
    """The sweep needs the TRAINING candidate file, but this script later
    overwrites output/candidate_pairs.tsv with TEST candidates. To keep a
    re-run from silently sweeping against test IDs, the training version is
    copied to train_candidate_pairs.tsv the first time, and a marker records
    that candidate_pairs.tsv has since been overwritten."""
    live   = os.path.join(OUTPUT_DIR, "candidate_pairs.tsv")
    backup = os.path.join(OUTPUT_DIR, "train_candidate_pairs.tsv")
    marker = os.path.join(OUTPUT_DIR, ".v10_test_outputs_written")
    if os.path.exists(backup):
        return backup
    if os.path.exists(marker) or not os.path.exists(live):
        return None   # live file is (or may be) test-derived; do not guess
    shutil.copyfile(live, backup)
    return backup


def final_threshold_sweep(model, vec_name, vec_addr, embed_store, embedder):
    fallback_path = os.path.join(MODELS_DIR, "v6_threshold.txt")
    fallback = float(open(fallback_path).read().strip()) if os.path.exists(fallback_path) else None

    cands_path = ensure_train_candidates_backup()
    gt_path = os.path.join(TRAIN_DIR, "train_ground_truth.tsv")
    if cands_path is None or not os.path.exists(gt_path):
        if fallback is None:
            # No sweep possible and no saved threshold: a made-up default (e.g.
            # 0.5) would be badly wrong for an F0.5 metric, so refuse.
            raise SystemExit("ERROR: cannot sweep (no trustworthy training candidates) and "
                             "models/v6_threshold.txt is missing. Re-run with --threshold <value>.")
        print(f"  No trustworthy training candidate file -- using saved v6 threshold {fallback:.2f}. "
              f"(Pass --threshold to set it explicitly.)")
        return fallback

    s1  = pd.read_csv(os.path.join(TRAIN_DIR, "train_source1.tsv"), sep="\t", dtype=str).fillna("")
    s23 = pd.concat([
        pd.read_csv(os.path.join(TRAIN_DIR, "train_source2.tsv"), sep="\t", dtype=str).fillna(""),
        pd.read_csv(os.path.join(TRAIN_DIR, "train_source3.tsv"), sep="\t", dtype=str).fillna(""),
    ], ignore_index=True)
    gt = pd.read_csv(gt_path, sep="\t", dtype=str).fillna("")
    cands_df = pd.read_csv(cands_path, sep="\t", dtype=str).fillna("")

    gt_map = {r.source1_entity_id: set(r.matched_entity_ids.split(",")) - {""}
              for r in gt.itertuples(index=False)}
    s1_d  = {e: (str(n).lower().strip(), str(a).lower().strip()) for e, n, a in
             zip(s1["entity_id"], s1["business_name"], s1["business_address"])}
    s23_d = {e: (str(n).lower().strip(), str(a).lower().strip()) for e, n, a in
             zip(s23["entity_id"], s23["business_name"], s23["business_address"])}

    pairs = [(r.source1_entity_id, cid, int(cid in gt_map.get(r.source1_entity_id, set())))
             for r in cands_df.itertuples(index=False)
             for cid in set(r.candidate_entity_ids.split(",")) - {""}]
    df = pd.DataFrame(pairs, columns=["s1_id", "cid", "label"])
    print(f"  {len(df):,} training candidate pairs to score")

    parts, t0 = [], time.time()
    for start in range(0, len(df), SWEEP_CHUNK):
        ch = df.iloc[start:start + SWEEP_CHUNK].copy()
        s1n = ch["s1_id"].map(lambda i: s1_d.get(i, ("", ""))[0]).values
        s1a = ch["s1_id"].map(lambda i: s1_d.get(i, ("", ""))[1]).values
        cn  = ch["cid"].map(lambda i: s23_d.get(i, ("", ""))[0]).values
        ca  = ch["cid"].map(lambda i: s23_d.get(i, ("", ""))[1]).values
        ch["name_jw"]  = [safe_jw(a, b)  for a, b in zip(s1n, cn)]
        ch["name_lev"] = [safe_lev(a, b) for a, b in zip(s1n, cn)]
        ch["addr_jw"]  = [safe_jw(a, b)  for a, b in zip(s1a, ca)]
        ch["addr_lev"] = [safe_lev(a, b) for a, b in zip(s1a, ca)]
        ch["num_overlap"] = [num_overlap(a, b) for a, b in zip(s1a, ca)]
        ch["name_tfidf_cosine"] = tfidf_pair_cosine(vec_name, s1n, cn)
        ch["addr_tfidf_cosine"] = tfidf_pair_cosine(vec_addr, s1a, ca)
        ch["name_x_addr"]    = ch["name_tfidf_cosine"] * ch["addr_tfidf_cosine"]
        ch["lookalike_flag"] = ((ch["name_jw"] > 0.90) & (ch["addr_jw"] < 0.50)).astype(int)
        needed = list(set(s1n) | set(cn) | set(s1a) | set(ca))
        embed_store.encode_and_store_missing(needed, embedder, show_progress=False)
        vecs = embed_store.get_many(needed)
        f32 = lambda arr: np.stack([vecs[t].astype(np.float32) for t in arr])
        ch["name_embed_cosine"] = cos_rows(f32(s1n), f32(cn))
        ch["addr_embed_cosine"] = cos_rows(f32(s1a), f32(ca))
        ch["prob"] = model.predict_proba(ch[FEATURES])[:, 1]
        parts.append(ch[["s1_id", "cid", "prob", "label"]])
        print(f"    sweep {min(start + SWEEP_CHUNK, len(df)):,}/{len(df):,} ({time.time()-t0:.0f}s)")

    scored = pd.concat(parts, ignore_index=True)
    unique_s1 = scored["s1_id"].unique()

    def macro_f05_at(t):
        pm = defaultdict(set)
        for s, c, p in zip(scored["s1_id"], scored["cid"], scored["prob"]):
            if p >= t:
                pm[s].add(c)
        vals = []
        for s in unique_s1:
            g, pr = gt_map.get(s, set()), pm.get(s, set())
            if not g and not pr:
                vals.append(1.0); continue
            tp = len(g & pr)
            p = tp / len(pr) if pr else 0.0
            r = tp / len(g) if g else 0.0
            vals.append(1.25 * p * r / (0.25 * p + r) if (p + r) > 0 else 0.0)
        return sum(vals) / len(vals)

    best_t, best = (fallback if fallback is not None else 0.5), -1.0
    for t in np.arange(0.05, 0.97, 0.02):
        f = macro_f05_at(t)
        if f > best:
            best, best_t = f, t
    print(f"  Sweep result: best_t={best_t:.2f}, F0.5={best:.4f}")
    return best_t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threshold", type=float, default=None,
                     help="Skip the training sweep and use this value.")
    ap.add_argument("--limit-s1", type=int, default=None, help="Dry run: only first N S1 rows per country.")
    ap.add_argument("--limit-s23", type=int, default=None, help="Dry run: only first N S23 rows per country.")
    args = ap.parse_args()

    print("=" * 60)
    print("v10 — final ensemble (memory-architected for 16GB RAM)")
    print("=" * 60)
    t0 = time.time()

    for fname in ["v6_classifier.pkl", "v6_vec_name.pkl", "v6_vec_addr.pkl"]:
        if not os.path.exists(os.path.join(MODELS_DIR, fname)):
            print(f"ERROR: {fname} missing. Run v6_xgboost.py first.")
            return
    model    = pickle.load(open(os.path.join(MODELS_DIR, "v6_classifier.pkl"), "rb"))
    vec_name = pickle.load(open(os.path.join(MODELS_DIR, "v6_vec_name.pkl"),   "rb"))
    vec_addr = pickle.load(open(os.path.join(MODELS_DIR, "v6_vec_addr.pkl"),   "rb"))

    from sentence_transformers import SentenceTransformer   # lazy: only needed at runtime
    embedder = SentenceTransformer(EMBED_MODEL_NAME)
    embed_store = EmbeddingStore(os.path.join(MODELS_DIR, "embeddings.db"))
    print(f"\nEmbedding store: {embed_store.count():,} vectors on disk (persists across runs)")

    print("\n[1/3] Threshold...")
    if args.threshold is not None:
        threshold = args.threshold
        print(f"  Using --threshold {threshold:.2f}")
    else:
        threshold = final_threshold_sweep(model, vec_name, vec_addr, embed_store, embedder)

    print("\n[2/3] Loading test set...")
    s1 = pd.read_csv(os.path.join(TEST_DIR, "test_source1.tsv"), sep="\t", dtype=str).fillna("")
    s23 = pd.concat([
        pd.read_csv(os.path.join(TEST_DIR, "test_source2.tsv"), sep="\t", dtype=str).fillna(""),
        pd.read_csv(os.path.join(TEST_DIR, "test_source3.tsv"), sep="\t", dtype=str).fillna(""),
    ], ignore_index=True)
    s1["nn"]  = s1["business_name"].apply(norm)
    s1["na"]  = s1["business_address"].apply(norm)
    s23["nn"] = s23["business_name"].apply(norm)
    s23["na"] = s23["business_address"].apply(norm)
    print(f"  S1: {len(s1):,}  |  S23: {len(s23):,}")

    veto = set()
    llm_path = os.path.join(OUTPUT_DIR, "v7_llm_decisions.tsv")
    if os.path.exists(llm_path):
        d = pd.read_csv(llm_path, sep="\t", dtype=str)
        no = d[d["llm_match"].astype(str).str.lower() != "true"]
        veto = set(zip(no["s1_id"], no["cid"]))
        print(f"  Loaded {len(veto):,} LLM vetoes (applied to matches only)")

    scores_w = StreamingTSVWriter(os.path.join(OUTPUT_DIR, "test_candidate_scores.tsv"),
                                   ["s1_id", "cid", "prob"])
    cand_w  = StreamingTSVWriter(os.path.join(OUTPUT_DIR, "candidate_pairs.tsv.partial"),
                                  ["source1_entity_id", "candidate_entity_ids"])
    match_w = StreamingTSVWriter(os.path.join(OUTPUT_DIR, "matching_results.tsv.partial"),
                                  ["source1_entity_id", "matched_entity_ids"])
    n_entities = n_cands = n_matched_entities = n_matches = 0

    print("\n[3/3] Blocking + scoring, streamed to disk per chunk...")
    for country in s1["country"].unique():          # open-set: never hard-code countries
        c_s1  = s1[s1["country"] == country].reset_index(drop=True)
        c_s23 = s23[s23["country"] == country].reset_index(drop=True)
        if args.limit_s1:  c_s1  = c_s1.head(args.limit_s1)
        if args.limit_s23: c_s23 = c_s23.head(args.limit_s23)
        if c_s1.empty: continue
        print(f"\n[{country}] S1={len(c_s1):,}, S23={len(c_s23):,}")

        print("  Building inverted index...")
        inv_index = build_inverted_index(c_s23)
        s23_text = {e: (n.lower().strip(), a.lower().strip()) for e, n, a in
                    zip(c_s23["entity_id"], c_s23["business_name"], c_s23["business_address"])}

        n_chunks = (len(c_s1) + S1_CHUNK_SIZE - 1) // S1_CHUNK_SIZE
        tc = time.time()
        for i, start in enumerate(range(0, len(c_s1), S1_CHUNK_SIZE)):
            chunk = c_s1.iloc[start:start + S1_CHUNK_SIZE]
            res = score_s1_chunk(chunk, inv_index, s23_text, vec_name, vec_addr,
                                  embed_store, embedder, model)

            scores_w.write_rows(zip(res["entity_id"], res["cid"], res["prob"].map("{:.6f}".format)))

            by_e_c, by_e_m = defaultdict(list), defaultdict(list)
            for e, c, p in zip(res["entity_id"], res["cid"], res["prob"]):
                by_e_c[e].append(c)
                if p >= threshold and (e, c) not in veto:
                    by_e_m[e].append(c)
            ids = list(chunk["entity_id"])           # one row per S1 entity, incl. singletons
            cand_w.write_rows((e, ",".join(sorted(by_e_c.get(e, [])))) for e in ids)
            match_w.write_rows((e, ",".join(sorted(by_e_m.get(e, [])))) for e in ids)

            n_entities += len(ids); n_cands += len(res)
            n_matched_entities += sum(1 for e in ids if by_e_m.get(e))
            n_matches += sum(len(v) for v in by_e_m.values())
            if (i + 1) % 10 == 0 or (i + 1) == n_chunks:
                print(f"    chunk {i+1}/{n_chunks} ({time.time()-tc:.0f}s) | "
                      f"{len(res):,} candidates in this chunk")

    scores_w.close(); cand_w.close(); match_w.close()
    # Atomic finalize: only replace the real files once everything finished.
    for name in ["candidate_pairs.tsv", "matching_results.tsv"]:
        os.replace(os.path.join(OUTPUT_DIR, name + ".partial"), os.path.join(OUTPUT_DIR, name))
    open(os.path.join(OUTPUT_DIR, ".v10_test_outputs_written"), "w").write(time.ctime())

    print("\nSUCCESS")
    print(f"  S1 entities written:    {n_entities:,}")
    print(f"  Entities with a match:  {n_matched_entities:,} ({n_matched_entities/max(n_entities,1):.1%})")
    print(f"  Avg candidates/entity:  {n_cands/max(n_entities,1):.1f}")
    print(f"  Total matches:          {n_matches:,}")
    print(f"  Threshold:              {threshold:.2f}")
    print(f"  Elapsed:                {time.time()-t0:.0f}s")
    embed_store.close()
    print("\n  NEXT: python v8_graph_consistency.py, then utils/validate_submission.py")

if __name__ == "__main__":
    main()
