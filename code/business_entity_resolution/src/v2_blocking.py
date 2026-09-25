"""
v2 — blocking (Multi-Key with Hot-Key Capping)
==============================================
Memory-safe blocking using highly-specific keys.
Fixes applied (audit 2026-09-25):
  - Added os.makedirs for OUTPUT_DIR and MODELS_DIR
  - Removed score > 0.05 filter that silently dropped soundex/acronym pairs
  - Address punctuation now cleaned via normalise_name()
  - Acronyms generated from name only (not address)
  - Min-length lowered to 2 so IBM/SAP/BP get keys
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
TRAIN_DIR   = os.path.join(STUDENT_RES, "dataset", "mini_train")   # mini for fast dev
TEST_DIR    = os.path.join(STUDENT_RES, "dataset", "test")
OUTPUT_DIR  = os.path.join(STUDENT_RES, "output")
MODELS_DIR  = os.path.join(REPO_ROOT, "models")

# FIX: create directories so Colab / fresh environments don't crash
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

K                 = 50
MAX_FEATURES      = 100_000
MAX_PAIRS_PER_KEY = 250_000

# ---------------------------------------------------------------------------
# Text normalisation & Keys
# ---------------------------------------------------------------------------
_PUNCT_RE = re.compile(r'[^\w\s]')
_SPACE_RE = re.compile(r'\s+')

def normalise_name(text: str) -> str:
    text = str(text).lower().strip()
    return _SPACE_RE.sub(' ', _PUNCT_RE.sub(' ', text)).strip()

def get_keys(norm_name: str, norm_addr: str) -> list:
    """
    Generate blocking keys from NORMALISED name and address separately.
    - Acronym comes from name words only (not address words).
    - All words >= 2 chars get a word key and soundex key.
    - Min-length 2 captures IBM, BP, etc.
    """
    name_words = norm_name.split()
    addr_words = norm_addr.split()
    keys = set()

    # 1. Acronym from name words only (avoid address contamination)
    if len(name_words) >= 2:
        acr = "".join(w[0] for w in name_words if w)
        if len(acr) >= 2:            # allow 2-char: GM, HP, BP
            keys.add(f"acr:{acr}")

    # 2. Every word >= 2 chars from name + address
    for w in name_words + addr_words:
        if len(w) >= 2:
            keys.add(f"w:{w}")
            keys.add(f"s:{jellyfish.soundex(w)}")

    return list(keys)

def prep_df(df: pd.DataFrame) -> tuple:
    df = df.copy()
    # FIX: normalise both name AND address (removes punctuation from address too)
    df["norm_name"] = df["business_name"].apply(normalise_name)
    df["norm_addr"] = df["business_address"].fillna("").apply(normalise_name)
    df["tfidf_text"] = df["norm_name"] + " " + df["norm_addr"]

    print("    Extracting blocking keys from name + address...")
    df["bkeys"] = df.apply(lambda r: get_keys(r["norm_name"], r["norm_addr"]), axis=1)
    df_exploded = df.explode("bkeys").dropna(subset=["bkeys"])
    # Filter out empty-string keys that arise from empty names
    df_exploded = df_exploded[df_exploded["bkeys"].str.len() > 4]
    df_exploded["bkey"] = df_exploded["bkeys"] + "|" + df_exploded["country"].str.strip()

    return df, df_exploded[["entity_id", "bkey"]]

# ---------------------------------------------------------------------------
# Process pipeline
# ---------------------------------------------------------------------------
def process_split(s1, s23, s1_keys, s23_keys, vec) -> dict:
    t0 = time.time()

    print("    Counting key frequencies...")
    c1  = s1_keys["bkey"].value_counts()
    c23 = s23_keys["bkey"].value_counts()

    valid_keys = set(
        k for k, count1 in c1.items()
        if count1 * c23.get(k, 0) <= MAX_PAIRS_PER_KEY
    )
    print(f"    Kept {len(valid_keys):,} safe keys (dropped {len(c1)-len(valid_keys):,} hot keys).")

    s1_keys  = s1_keys[s1_keys["bkey"].isin(valid_keys)]
    s23_keys = s23_keys[s23_keys["bkey"].isin(valid_keys)]

    print("    Merging on safe blocking keys...")
    pairs = s1_keys.merge(s23_keys.rename(columns={"entity_id": "cid"}), on="bkey", how="inner")
    pairs = pairs[["entity_id", "cid"]].drop_duplicates()
    print(f"    Generated {len(pairs):,} unique candidate pairs in {time.time()-t0:.1f}s.")

    if len(pairs) == 0:
        return {}

    # TF-IDF similarity — used to RANK candidates, NOT to filter them out.
    # FIX: removed `pairs["score"] > 0.05` filter that was silently dropping
    # soundex/acronym pairs whose word-level cosine similarity happened to be 0.
    print("    Calculating TF-IDF similarities for ranking...")
    t1 = time.time()

    s1_vecs  = vec.transform(s1["tfidf_text"])
    s23_vecs = vec.transform(s23["tfidf_text"])

    s1_idx_map  = {eid: idx for idx, eid in enumerate(s1["entity_id"])}
    s23_idx_map = {eid: idx for idx, eid in enumerate(s23["entity_id"])}

    pairs["s1_idx"]  = pairs["entity_id"].map(s1_idx_map)
    pairs["s23_idx"] = pairs["cid"].map(s23_idx_map)

    CHUNK  = 2_000_000
    scores = []
    for start in range(0, len(pairs), CHUNK):
        chunk    = pairs.iloc[start:start+CHUNK]
        v1_chunk = s1_vecs[chunk["s1_idx"].values]
        v23_chunk = s23_vecs[chunk["s23_idx"].values]
        sims     = np.array(v1_chunk.multiply(v23_chunk).sum(axis=1)).flatten()
        scores.extend(sims)

    pairs["score"] = scores
    print(f"    Calculated similarities in {time.time()-t1:.1f}s.")

    # Keep top-K per S1 entity by TF-IDF score (no hard cutoff that removes true matches)
    print(f"    Keeping top {K} candidates per S1 entity...")
    pairs = pairs.sort_values(["entity_id", "score"], ascending=[True, False])
    pairs = pairs.groupby("entity_id").head(K)

    cands = {}
    for row in pairs.itertuples(index=False):
        sid = row.entity_id
        if sid not in cands:
            cands[sid] = {}
        cands[sid][row.cid] = row.score

    return cands

def score_recall(cands: dict, gt_path: str):
    gt = pd.read_csv(gt_path, sep="\t", dtype=str).fillna("")
    found = total = 0
    for row in gt.itertuples(index=False):
        s1_id  = row.source1_entity_id
        gt_ids = set(row.matched_entity_ids.split(",")) - {""}
        if not gt_ids:
            continue
        total  += len(gt_ids)
        pred_ids = set(cands.get(s1_id, {}).keys())
        found  += len(gt_ids & pred_ids)

    recall = found / total if total else 0.0
    print(f"\n  Blocking recall: {found:,} / {total:,} = {recall:.4f}  ({recall:.1%})")
    print("  GATE PASSED" if recall >= 0.95 else "  GATE FAILED (Target >= 95%)")

def main():
    print("=" * 60)
    print("v2 — blocking (Multi-Key + Hot-Key Capping) [MINI TRAIN]")
    print("=" * 60)

    print("\n[1/3] Loading MINI TRAIN data...")
    s1_tr = pd.read_csv(os.path.join(TRAIN_DIR, "train_source1.tsv"), sep="\t", dtype=str).fillna("")
    s23_tr = pd.concat([
        pd.read_csv(os.path.join(TRAIN_DIR, "train_source2.tsv"), sep="\t", dtype=str).fillna(""),
        pd.read_csv(os.path.join(TRAIN_DIR, "train_source3.tsv"), sep="\t", dtype=str).fillna(""),
    ], ignore_index=True)

    s1_tr, s1_tr_keys   = prep_df(s1_tr)
    s23_tr, s23_tr_keys = prep_df(s23_tr)

    print("\n[2/3] Fitting TF-IDF vectorizer (word n-grams, for ranking)...")
    vec = TfidfVectorizer(analyzer="word", ngram_range=(1, 2),
                          max_features=MAX_FEATURES, sublinear_tf=True)
    vec.fit(pd.concat([s1_tr["tfidf_text"], s23_tr["tfidf_text"]]))
    with open(os.path.join(MODELS_DIR, "tfidf_vectorizer.pkl"), "wb") as f:
        pickle.dump(vec, f)
    print(f"    Vectorizer saved to models/tfidf_vectorizer.pkl")

    print("\n[3/3] Processing TRAIN split...")
    cands_tr = process_split(s1_tr, s23_tr, s1_tr_keys, s23_tr_keys, vec)
    score_recall(cands_tr, os.path.join(TRAIN_DIR, "train_ground_truth.tsv"))

    print("\nSaving candidate pairs for v3...")
    rows_c = [
        {"source1_entity_id": sid, "candidate_entity_ids": ",".join(cands_tr.get(sid, {}).keys())}
        for sid in s1_tr["entity_id"]
    ]
    pd.DataFrame(rows_c).to_csv(os.path.join(OUTPUT_DIR, "candidate_pairs.tsv"), sep="\t", index=False)
    print(f"    Saved to output/candidate_pairs.tsv")

    print("\n[Done] Skipping TEST split during local development.")
    print("       Switch TRAIN_DIR -> train and run again to generate test candidates.")

if __name__ == "__main__":
    main()
