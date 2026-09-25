"""
v2 — blocking (Multi-Key + Hot-Key Capping)
============================================
Processes each country separately to stay within Colab's 12GB RAM.
Fixes applied (audit + Colab OOM 2026-09-25):
  - Country-by-country processing (no single giant merge)
  - Soundex only for words >= 5 chars (prevents huge collision buckets)
  - Word keys only for words >= 3 chars
  - makedirs for OUTPUT_DIR and MODELS_DIR
  - score filter removed (let v3 classifier decide)
  - address punctuation cleaned
  - acronym from name only
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
# Paths & Hyperparams
# ---------------------------------------------------------------------------
REPO_ROOT   = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
                os.path.abspath(__file__)))))
STUDENT_RES = os.path.join(REPO_ROOT, "6ab10eb3b23ba_student_resource", "student_resource")
TRAIN_DIR   = os.path.join(STUDENT_RES, "dataset", "mini_train")
TEST_DIR    = os.path.join(STUDENT_RES, "dataset", "test")
OUTPUT_DIR  = os.path.join(STUDENT_RES, "output")
MODELS_DIR  = os.path.join(REPO_ROOT, "models")

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

K                 = 50
MAX_FEATURES      = 100_000
MAX_PAIRS_PER_KEY = 100_000   # per-country cap — safe for Colab RAM

# ---------------------------------------------------------------------------
# Text normalisation & Keys
# ---------------------------------------------------------------------------
_PUNCT_RE = re.compile(r'[^\w\s]')
_SPACE_RE = re.compile(r'\s+')

def normalise_name(text: str) -> str:
    return _SPACE_RE.sub(' ', _PUNCT_RE.sub(' ', str(text).lower().strip())).strip()

def get_keys(norm_name: str, norm_addr: str) -> list:
    name_words = norm_name.split()
    addr_words = norm_addr.split()
    keys = set()

    # Acronym from name only (avoids address word contamination)
    if len(name_words) >= 2:
        acr = "".join(w[0] for w in name_words if w)
        if len(acr) >= 2:
            keys.add(f"acr:{acr}")

    # Word keys >= 3 chars (cuts noisy 1-2 letter words)
    # Soundex only >= 5 chars (prevents massive soundex collision buckets)
    for w in name_words + addr_words:
        if len(w) >= 3:
            keys.add(f"w:{w}")
        if len(w) >= 5:
            keys.add(f"s:{jellyfish.soundex(w)}")

    return list(keys)

def prep_df(df: pd.DataFrame) -> tuple:
    df = df.copy()
    df["norm_name"] = df["business_name"].apply(normalise_name)
    df["norm_addr"] = df["business_address"].fillna("").apply(normalise_name)
    df["tfidf_text"] = df["norm_name"] + " " + df["norm_addr"]
    df["bkeys"] = df.apply(lambda r: get_keys(r["norm_name"], r["norm_addr"]), axis=1)

    df_exp = df.explode("bkeys").dropna(subset=["bkeys"])
    df_exp = df_exp[df_exp["bkeys"].str.len() > 3]
    df_exp = df_exp.rename(columns={"bkeys": "bkey"})

    return df, df_exp[["entity_id", "bkey"]]

# ---------------------------------------------------------------------------
# Block one country slice
# ---------------------------------------------------------------------------
def block_country(s1_c: pd.DataFrame, s23_c: pd.DataFrame,
                  s1_keys: pd.DataFrame, s23_keys: pd.DataFrame,
                  vec: TfidfVectorizer, country: str) -> dict:

    c1  = s1_keys["bkey"].value_counts()
    c23 = s23_keys["bkey"].value_counts()

    safe = set(k for k, n1 in c1.items()
                if n1 * c23.get(k, 0) <= MAX_PAIRS_PER_KEY)

    sk1  = s1_keys[s1_keys["bkey"].isin(safe)]
    sk23 = s23_keys[s23_keys["bkey"].isin(safe)]

    pairs = (sk1.merge(sk23.rename(columns={"entity_id": "cid"}), on="bkey")
               [["entity_id", "cid"]].drop_duplicates())

    n_pairs = len(pairs)
    dropped = len(c1) - len(safe)
    print(f"    [{country}] {len(safe):,} safe keys (dropped {dropped:,}), "
          f"{n_pairs:,} candidate pairs")

    if pairs.empty:
        return {}

    # Rank pairs by TF-IDF cosine — DO NOT filter out (let v3 decide)
    s1_vecs  = vec.transform(s1_c["tfidf_text"])
    s23_vecs = vec.transform(s23_c["tfidf_text"])

    s1_idx  = {eid: i for i, eid in enumerate(s1_c["entity_id"])}
    s23_idx = {eid: i for i, eid in enumerate(s23_c["entity_id"])}

    pairs["s1i"]  = pairs["entity_id"].map(s1_idx)
    pairs["s23i"] = pairs["cid"].map(s23_idx)

    CHUNK = 1_000_000
    scores = []
    for start in range(0, len(pairs), CHUNK):
        ch = pairs.iloc[start:start+CHUNK]
        v1  = s1_vecs[ch["s1i"].values]
        v23 = s23_vecs[ch["s23i"].values]
        scores.extend(np.array(v1.multiply(v23).sum(axis=1)).flatten())

    pairs["score"] = scores
    pairs = (pairs.sort_values(["entity_id", "score"], ascending=[True, False])
                  .groupby("entity_id").head(K))

    cands = {}
    for row in pairs.itertuples(index=False):
        cands.setdefault(row.entity_id, {})[row.cid] = row.score

    return cands

# ---------------------------------------------------------------------------
# Recall scorer
# ---------------------------------------------------------------------------
def score_recall(cands: dict, gt_path: str):
    gt = pd.read_csv(gt_path, sep="\t", dtype=str).fillna("")
    found = total = 0
    for row in gt.itertuples(index=False):
        gt_ids = set(row.matched_entity_ids.split(",")) - {""}
        if not gt_ids: continue
        total  += len(gt_ids)
        found  += len(gt_ids & set(cands.get(row.source1_entity_id, {}).keys()))
    recall = found / total if total else 0.0
    print(f"\n  Blocking recall: {found:,} / {total:,} = {recall:.4f} ({recall:.1%})")
    print("  GATE PASSED" if recall >= 0.95 else "  GATE FAILED (target >= 95%)")
    return recall

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 60)
    print("v2 — blocking (Country-by-Country, Colab-safe)")
    print("=" * 60)

    t0 = time.time()

    print("\n[1/3] Loading MINI TRAIN data...")
    s1_tr = pd.read_csv(os.path.join(TRAIN_DIR, "train_source1.tsv"), sep="\t", dtype=str).fillna("")
    s23_tr = pd.concat([
        pd.read_csv(os.path.join(TRAIN_DIR, "train_source2.tsv"), sep="\t", dtype=str).fillna(""),
        pd.read_csv(os.path.join(TRAIN_DIR, "train_source3.tsv"), sep="\t", dtype=str).fillna(""),
    ], ignore_index=True)
    print(f"  S1: {len(s1_tr):,}  |  S23: {len(s23_tr):,}")

    print("\n[2/3] Fitting TF-IDF vectorizer on full mini_train vocab...")
    s1_tr, _  = prep_df(s1_tr)
    s23_tr, _ = prep_df(s23_tr)

    vec = TfidfVectorizer(analyzer="word", ngram_range=(1,2),
                          max_features=MAX_FEATURES, sublinear_tf=True)
    vec.fit(pd.concat([s1_tr["tfidf_text"], s23_tr["tfidf_text"]]))
    pickle.dump(vec, open(os.path.join(MODELS_DIR, "tfidf_vectorizer.pkl"), "wb"))
    print("  Vectorizer saved.")

    print("\n[3/3] Blocking per country...")
    all_cands = {}

    for country in s1_tr["country"].unique():
        c_s1  = s1_tr[s1_tr["country"]  == country].copy()
        c_s23 = s23_tr[s23_tr["country"] == country].copy()
        if c_s1.empty or c_s23.empty:
            continue

        print(f"\n  Country: {country}  (S1={len(c_s1):,}, S23={len(c_s23):,})")
        c_s1,  s1k  = prep_df(c_s1)
        c_s23, s23k = prep_df(c_s23)

        t1 = time.time()
        cands = block_country(c_s1, c_s23, s1k, s23k, vec, country)
        all_cands.update(cands)
        print(f"    Done in {time.time()-t1:.1f}s  |  {len(cands):,} S1 entities with candidates")

    score_recall(all_cands, os.path.join(TRAIN_DIR, "train_ground_truth.tsv"))

    print("\nSaving candidate_pairs.tsv for v3...")
    rows = [{"source1_entity_id": sid,
             "candidate_entity_ids": ",".join(all_cands.get(sid, {}).keys())}
            for sid in s1_tr["entity_id"]]
    pd.DataFrame(rows).to_csv(
        os.path.join(OUTPUT_DIR, "candidate_pairs.tsv"), sep="\t", index=False)
    print(f"  Saved {len(rows):,} rows.")
    print(f"\n[Done] Total time: {time.time()-t0:.1f}s")

if __name__ == "__main__":
    main()
