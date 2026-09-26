"""
v6 — phonetic-blocking (part 1 of 2)
=====================================
Adds Double Metaphone as a third blocking key, unioned with the
TF-IDF-word + Soundex keys already used in v2_blocking.py, then
re-measures recall against train_ground_truth.tsv.

Double Metaphone vs jellyfish's Soundex (already in v2):
  - Soundex is coarse and English-centric; it under-blocks non-English
    transliterations (relevant for India + France records).
  - Double Metaphone produces two codes per word (primary/alternate) and
    handles more phonetic edge cases -- better recall on name variants
    like "Xiomara"/"Ksiomara" or transliterated business names.

Dependency note: jellyfish only ships single Metaphone, not Double
Metaphone. This script prefers the `metaphone` package (pure-Python,
MIT licensed, `pip install metaphone`) and falls back to jellyfish's
single Metaphone with a printed warning if it isn't installed --
the pipeline still runs, just with slightly weaker phonetic recall.

Output: overwrites output/candidate_pairs.tsv with the union of the
existing (v2) candidates and the new metaphone-only candidates, capped
the same way as v2 (hot-key capping) to keep memory bounded.
"""

import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import os, re, time
import pandas as pd

try:
    from metaphone import doublemetaphone
    HAVE_DOUBLE_METAPHONE = True
except ImportError:
    import jellyfish
    HAVE_DOUBLE_METAPHONE = False
    print("WARNING: `metaphone` package not installed (pip install metaphone).")
    print("         Falling back to jellyfish single Metaphone -- weaker recall on")
    print("         name variants. Install `metaphone` for the full v6 upgrade.")

REPO_ROOT   = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
                os.path.abspath(__file__)))))
TRAIN_DIR   = os.path.join(REPO_ROOT, "dataset", "mini_train")
OUTPUT_DIR  = os.path.join(REPO_ROOT, "output")

MAX_PAIRS_PER_KEY = 250_000   # same cap as v2's hot-key capping

_PUNCT_RE = re.compile(r'[^\w\s]')
_SPACE_RE = re.compile(r'\s+')

def norm(text):
    return _SPACE_RE.sub(' ', _PUNCT_RE.sub(' ', str(text).lower().strip())).strip()

def metaphone_codes(word: str):
    if len(word) < 3:
        return []
    if HAVE_DOUBLE_METAPHONE:
        primary, alternate = doublemetaphone(word)
        return [c for c in (primary, alternate) if c]
    else:
        code = jellyfish.metaphone(word)
        return [code] if code else []

def get_metaphone_keys(name: str, addr: str):
    keys = set()
    for w in (name.split() + addr.split()):
        if len(w) >= 4:
            for code in metaphone_codes(w):
                keys.add(f"mp:{code}")
    return list(keys)

def build_key_table(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["nn"] = df["business_name"].apply(norm)
    df["na"] = df["business_address"].fillna("").apply(norm)
    df["bkeys"] = df.apply(lambda r: get_metaphone_keys(r["nn"], r["na"]), axis=1)
    exp = (df[["entity_id", "country", "bkeys"]].explode("bkeys")
           .dropna(subset=["bkeys"]).rename(columns={"bkeys": "bkey"}))
    return exp[exp["bkey"].str.len() > 4]

def recall_against_gt(cands_df: pd.DataFrame, gt: pd.DataFrame) -> float:
    cand_map = {}
    for row in cands_df.itertuples(index=False):
        cand_map[row.source1_entity_id] = set(row.candidate_entity_ids.split(",")) - {""}
    hit, total = 0, 0
    for row in gt.itertuples(index=False):
        true_ids = set(row.matched_entity_ids.split(",")) - {""}
        if not true_ids:
            continue
        cand_ids = cand_map.get(row.source1_entity_id, set())
        hit   += len(true_ids & cand_ids)
        total += len(true_ids)
    return hit / total if total else 0.0

def main():
    print("=" * 60)
    print("v6 — phonetic blocking upgrade (Double Metaphone)")
    print("=" * 60)
    t0 = time.time()

    print("\n[1/4] Loading source data and existing v2 candidate pairs...")
    s1  = pd.read_csv(os.path.join(TRAIN_DIR, "train_source1.tsv"), sep="\t", dtype=str).fillna("")
    s2  = pd.read_csv(os.path.join(TRAIN_DIR, "train_source2.tsv"), sep="\t", dtype=str).fillna("")
    s3  = pd.read_csv(os.path.join(TRAIN_DIR, "train_source3.tsv"), sep="\t", dtype=str).fillna("")
    s23 = pd.concat([s2, s3], ignore_index=True)
    gt  = pd.read_csv(os.path.join(TRAIN_DIR, "train_ground_truth.tsv"), sep="\t", dtype=str).fillna("")

    cands_path = os.path.join(OUTPUT_DIR, "candidate_pairs.tsv")
    old_cands = pd.read_csv(cands_path, sep="\t", dtype=str).fillna("") if os.path.exists(cands_path) \
                else pd.DataFrame(columns=["source1_entity_id", "candidate_entity_ids"])
    old_recall = recall_against_gt(old_cands, gt)
    print(f"  Existing (v2) blocking recall: {old_recall:.2%}")

    print("\n[2/4] Building Double Metaphone keys for S1 and S2+S3...")
    s1_keys  = build_key_table(s1)
    s23_keys = build_key_table(s23)

    print("\n[3/4] Joining on shared metaphone keys (hot-key capped)...")
    c1  = s1_keys["bkey"].value_counts()
    c23 = s23_keys["bkey"].value_counts()
    safe_keys = set(k for k, n1 in c1.items() if n1 * c23.get(k, 0) <= MAX_PAIRS_PER_KEY)
    print(f"  {len(c1):,} S1 metaphone keys, {len(safe_keys):,} kept after hot-key capping")

    joined = (s1_keys[s1_keys["bkey"].isin(safe_keys)]
              .merge(s23_keys[s23_keys["bkey"].isin(safe_keys)],
                     on=["bkey", "country"], suffixes=("_s1", "_s23"))
              [["entity_id_s1", "entity_id_s23"]]
              .drop_duplicates()
              .rename(columns={"entity_id_s1": "source1_entity_id",
                                "entity_id_s23": "cid"}))
    print(f"  New metaphone-only candidate pairs: {len(joined):,}")

    print("\n[4/4] Unioning with existing candidates and re-measuring recall...")
    new_grouped = joined.groupby("source1_entity_id")["cid"].apply(set).to_dict()
    old_map = {row.source1_entity_id: set(row.candidate_entity_ids.split(",")) - {""}
               for row in old_cands.itertuples(index=False)}

    all_s1_ids = set(s1["entity_id"]) | set(old_map.keys()) | set(new_grouped.keys())
    merged_rows = []
    for s1_id in all_s1_ids:
        merged = (old_map.get(s1_id, set()) | new_grouped.get(s1_id, set()))
        merged_rows.append({"source1_entity_id": s1_id,
                             "candidate_entity_ids": ",".join(sorted(merged))})
    merged_df = pd.DataFrame(merged_rows)

    new_recall = recall_against_gt(merged_df, gt)
    print(f"  Recall before (v2 only):        {old_recall:.2%}")
    print(f"  Recall after (v2 + metaphone):  {new_recall:.2%}")
    print(f"  Target: >= 97% (VERSIONS.md v6 done-criteria)")

    merged_df.to_csv(cands_path, sep="\t", index=False)
    print(f"\n[Done] Overwrote {cands_path} in {time.time()-t0:.1f}s")

    if new_recall < 0.97:
        print("\n  NOTE: recall is still below the 97% target. Consider widening the")
        print("        NearestNeighbors K in v2_blocking.py before moving to v6's")
        print("        classifier upgrade -- blocking misses can never be recovered.")

if __name__ == "__main__":
    main()
