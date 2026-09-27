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
existing (v2) candidates and the new metaphone-only candidates.

PER-ENTITY CAP (added after the first real run on mini_train blew up
candidate_pairs.tsv from ~5M to ~69.5M rows at 0.49% positive -- hot-key
capping alone doesn't stop the explosion because many *individually
common but not "hot"* metaphone codes union together per entity):
  - Metaphone-only candidates are scored by how many distinct metaphone
    keys a given (s1_id, cid) pair shares -- more shared phonetic keys is
    a real (if crude, label-free) confidence signal, and costs nothing
    extra since we already have the exploded key table.
  - Existing v2 candidates are always kept (they already passed a more
    selective process and are what v2's measured recall depends on).
  - New metaphone-only candidates then fill remaining slots up to
    PER_ENTITY_CAP per S1 entity, highest shared-key-count first.
  - This mirrors the top-K coarse-filter already used at inference time
    in generate_v3_submission.py / v9_final_ensemble.py, so training-time
    candidate_pairs.tsv stays consistent with what test-time blocking
    would actually produce (no label leakage -- the cap never looks at
    ground truth, only at key-overlap counts).

Safety net (added after a real run corrupted candidate_pairs.tsv): a prior
buggy version of this script overwrote candidate_pairs.tsv with an
already-bloated 69.5M-row file, and a LATER re-run of the fixed script then
read that bloated file back in, mistook it for "the sacred v2 baseline",
and dutifully kept all 69.5M rows untouched. To make that failure mode
impossible going forward:
  - Before touching anything, this script backs up whatever is currently
    at candidate_pairs.tsv to candidate_pairs_pre_v6_metaphone.tsv --
    ONLY on the first run (it will not overwrite an existing backup, so
    the backup always reflects the true pre-v6 state, not a corrupted one).
  - It also writes a small marker file (.v6_metaphone_applied) after a
    successful run. If that marker is already present, the script refuses
    to run again (and would otherwise re-treat its own already-merged
    output as "the v2 baseline" to preserve) unless you pass --force,
    in which case it warns loudly and proceeds anyway.
"""

import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import os, re, time, argparse
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
PER_ENTITY_CAP    = 100       # NEW: hard cap on total candidates per S1 entity
                               # (existing v2 candidates always kept; this caps
                               # how many *additional* metaphone-only candidates
                               # get unioned in, ranked by shared-key count)

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
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true",
                     help="Re-run even if .v6_metaphone_applied marker exists "
                          "(the file it reads as 'existing v2 candidates' will "
                          "already include the previous metaphone additions -- "
                          "only use this if you know that's what you want).")
    args = ap.parse_args()

    print("=" * 60)
    print("v6 — phonetic blocking upgrade (Double Metaphone)")
    print("=" * 60)
    t0 = time.time()

    marker_path = os.path.join(OUTPUT_DIR, ".v6_metaphone_applied")
    if os.path.exists(marker_path) and not args.force:
        print(f"\nERROR: {marker_path} already exists -- this script appears to have")
        print("       run successfully before. Re-running would read its own already-")
        print("       merged candidate_pairs.tsv and treat it as 'the v2 baseline',")
        print("       silently keeping everything in it (this is exactly the bug that")
        print("       caused the 69.5M-row corruption).")
        print(f"\n       If you need a clean v2 baseline, re-run v2_blocking.py first --")
        print(f"       it will overwrite candidate_pairs.tsv with a fresh, uninflated set.")
        print(f"       If you intentionally want to layer metaphone matching again on")
        print(f"       top of the current file, re-run with --force.")
        return

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

    # Back up whatever is currently in candidate_pairs.tsv BEFORE we touch it,
    # but only if a backup doesn't already exist -- if this is a second run,
    # an existing backup already holds the true pre-v6 state and must not be
    # clobbered by whatever (possibly already-bloated) file is here now.
    backup_path = os.path.join(OUTPUT_DIR, "candidate_pairs_pre_v6_metaphone.tsv")
    if os.path.exists(cands_path) and not os.path.exists(backup_path):
        old_cands.to_csv(backup_path, sep="\t", index=False)
        print(f"  Backed up pre-v6 candidates to {os.path.basename(backup_path)} "
              f"({len(old_cands):,} rows)")
    elif os.path.exists(backup_path):
        print(f"  (backup already exists at {os.path.basename(backup_path)} -- not overwriting)")

    print("\n[2/4] Building Double Metaphone keys for S1 and S2+S3...")
    s1_keys  = build_key_table(s1)
    s23_keys = build_key_table(s23)

    print("\n[3/4] Joining on shared metaphone keys (hot-key capped)...")
    c1  = s1_keys["bkey"].value_counts()
    c23 = s23_keys["bkey"].value_counts()
    safe_keys = set(k for k, n1 in c1.items() if n1 * c23.get(k, 0) <= MAX_PAIRS_PER_KEY)
    print(f"  {len(c1):,} S1 metaphone keys, {len(safe_keys):,} kept after hot-key capping")

    # NOTE: no drop_duplicates() here yet -- we keep every (s1_id, cid, bkey)
    # row so the next step can count how many distinct keys each pair shares.
    # That count is the ranking signal for the per-entity cap below.
    joined = (s1_keys[s1_keys["bkey"].isin(safe_keys)]
              .merge(s23_keys[s23_keys["bkey"].isin(safe_keys)],
                     on=["bkey", "country"], suffixes=("_s1", "_s23"))
              [["entity_id_s1", "entity_id_s23"]]
              .rename(columns={"entity_id_s1": "source1_entity_id",
                                "entity_id_s23": "cid"}))
    raw_pair_rows = len(joined)

    scored = (joined.groupby(["source1_entity_id", "cid"])
              .size().reset_index(name="shared_key_count"))
    print(f"  Raw joined rows (pre-dedup): {raw_pair_rows:,}")
    print(f"  Unique metaphone-only candidate pairs: {len(scored):,}")

    print(f"\n[4/4] Applying per-entity cap (<= {PER_ENTITY_CAP} candidates/entity), "
          "unioning with existing candidates, and re-measuring recall...")

    old_map = {row.source1_entity_id: set(row.candidate_entity_ids.split(",")) - {""}
               for row in old_cands.itertuples(index=False)}

    # Drop any new candidate that's already an existing (v2) candidate --
    # it doesn't need a "slot" from the cap since it's kept unconditionally.
    # Vectorized via a concatenated string key (fast even at tens of millions
    # of rows) -- avoid row-wise .apply()/tuple lookups here, that's the same
    # class of slowdown that made the safe_jw/safe_lev step painful.
    old_pair_keys = {f"{s1_id}||{cid}" for s1_id, cids in old_map.items() for cid in cids}
    scored_keys = scored["source1_entity_id"].astype(str) + "||" + scored["cid"].astype(str)
    scored_new_only = scored[~scored_keys.isin(old_pair_keys)]
    print(f"  Of those, genuinely new (not already in v2 candidates): {len(scored_new_only):,}")

    # Rank by shared_key_count (desc) within each entity, keep top PER_ENTITY_CAP
    scored_new_only = scored_new_only.sort_values(
        ["source1_entity_id", "shared_key_count"], ascending=[True, False])
    capped_new = scored_new_only.groupby("source1_entity_id").head(PER_ENTITY_CAP)
    print(f"  After per-entity cap: {len(capped_new):,} new candidate pairs kept "
          f"(dropped {len(scored_new_only) - len(capped_new):,} low-signal extras)")

    new_grouped = capped_new.groupby("source1_entity_id")["cid"].apply(set).to_dict()

    all_s1_ids = set(s1["entity_id"]) | set(old_map.keys()) | set(new_grouped.keys())
    merged_rows = []
    for s1_id in all_s1_ids:
        merged = (old_map.get(s1_id, set()) | new_grouped.get(s1_id, set()))
        merged_rows.append({"source1_entity_id": s1_id,
                             "candidate_entity_ids": ",".join(sorted(merged))})
    merged_df = pd.DataFrame(merged_rows)

    total_pairs = sum(len(r["candidate_entity_ids"].split(",")) if r["candidate_entity_ids"] else 0
                       for r in merged_rows)
    new_recall = recall_against_gt(merged_df, gt)
    print(f"\n  Total candidate pairs in final file: {total_pairs:,} "
          f"(compare: v5 was ~5,000,000, uncapped v6 was 69,571,877)")
    print(f"  Recall before (v2 only):        {old_recall:.2%}")
    print(f"  Recall after (v2 + metaphone):  {new_recall:.2%}")
    print(f"  Target: >= 97% (VERSIONS.md v6 done-criteria)")

    if new_recall < old_recall:
        print("\n  WARNING: recall DROPPED vs v2-only. The per-entity cap likely evicted a")
        print("           true-positive candidate ranked below PER_ENTITY_CAP on shared-key")
        print("           count. Consider raising PER_ENTITY_CAP (e.g. to 150-200) and re-running --")
        print("           it's a pure runtime/memory trade-off, not a correctness one.")

    merged_df.to_csv(cands_path, sep="\t", index=False)
    with open(marker_path, "w") as f:
        f.write(f"v6_blocking_metaphone.py completed successfully at {time.ctime()}\n"
                f"final_recall={new_recall:.4f}\ntotal_pairs={total_pairs}\n")
    print(f"\n[Done] Overwrote {cands_path} in {time.time()-t0:.1f}s")
    print(f"       Wrote {os.path.basename(marker_path)} -- delete it (or use --force) "
          "to intentionally re-run this script.")

    if new_recall < 0.97:
        print("\n  NOTE: recall is still below the 97% target. Consider widening the")
        print("        NearestNeighbors K in v2_blocking.py before moving to v6's")
        print("        classifier upgrade -- blocking misses can never be recovered.")

if __name__ == "__main__":
    main()