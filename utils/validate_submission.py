"""
validate_submission.py — local format checker
==============================================
Checks output/matching_results.tsv and output/candidate_pairs.tsv against
every rule in the problem statement BEFORE you spend a submission slot on it.

Usage:
    python utils/validate_submission.py \
        --matching output/matching_results.tsv \
        --candidate output/candidate_pairs.tsv \
        --test-dir dataset/test

Prints PASS (exit 0) or a numbered list of issues (exit 1).
stdlib only — no external dependencies.
"""

import argparse
import csv
import os
import sys


def load_tsv(path: str) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        return list(reader)


def load_entity_ids(test_dir: str) -> tuple[set, set, set]:
    """Return (s1_ids, s2_ids, s3_ids) from the test source files."""
    ids = {}
    for src in [1, 2, 3]:
        path = os.path.join(test_dir, f"test_source{src}.tsv")
        with open(path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f, delimiter="\t")
            ids[src] = {row["entity_id"].strip() for row in reader}
    return ids[1], ids[2], ids[3]


def validate(matching_path: str, candidate_path: str, test_dir: str) -> list[str]:
    issues: list[str] = []

    # Load test entity IDs
    try:
        s1_ids, s2_ids, s3_ids = load_entity_ids(test_dir)
    except FileNotFoundError as e:
        return [f"Cannot load test source files: {e}"]

    valid_match_ids = s2_ids | s3_ids   # matched_entity_ids must come from here

    # -----------------------------------------------------------------------
    # Validate matching_results.tsv
    # -----------------------------------------------------------------------
    try:
        matching_rows = load_tsv(matching_path)
    except FileNotFoundError:
        return [f"matching_results.tsv not found at: {matching_path}"]

    # Check columns
    if matching_rows:
        cols = set(matching_rows[0].keys())
        for required in ["source1_entity_id", "matched_entity_ids"]:
            if required not in cols:
                issues.append(f"matching_results.tsv missing column: '{required}'")

    seen_s1 = set()
    for i, row in enumerate(matching_rows, start=2):   # row 1 = header
        s1_id = row.get("source1_entity_id", "").strip()

        # Every S1 ID must exist in test set
        if s1_id not in s1_ids:
            issues.append(f"matching_results.tsv row {i}: "
                          f"'{s1_id}' not in test_source1.tsv")

        # No duplicate S1 rows
        if s1_id in seen_s1:
            issues.append(f"matching_results.tsv: duplicate source1_entity_id '{s1_id}'")
        seen_s1.add(s1_id)

        # Validate matched_entity_ids
        matched_str = row.get("matched_entity_ids", "").strip()
        if matched_str:
            matched_ids = [x.strip() for x in matched_str.split(",")]
            seen_local: set[str] = set()
            for mid in matched_ids:
                if mid not in valid_match_ids:
                    issues.append(f"matching_results.tsv row {i}: "
                                  f"'{mid}' is not a valid S2/S3 test entity ID")
                if mid in seen_local:
                    issues.append(f"matching_results.tsv row {i}: "
                                  f"duplicate id '{mid}' in matched_entity_ids")
                seen_local.add(mid)

    # Every S1 entity must have exactly one row
    missing_s1 = s1_ids - seen_s1
    if missing_s1:
        issues.append(f"matching_results.tsv: {len(missing_s1)} S1 entities missing — "
                      f"e.g. {sorted(missing_s1)[:5]}")

    # -----------------------------------------------------------------------
    # Validate candidate_pairs.tsv
    # -----------------------------------------------------------------------
    try:
        candidate_rows = load_tsv(candidate_path)
    except FileNotFoundError:
        issues.append(f"candidate_pairs.tsv not found at: {candidate_path}")
        candidate_rows = []

    if candidate_rows:
        cols = set(candidate_rows[0].keys())
        for required in ["source1_entity_id", "candidate_entity_ids"]:
            if required not in cols:
                issues.append(f"candidate_pairs.tsv missing column: '{required}'")

    # Build candidate set for cross-check
    candidate_map: dict[str, set] = {}
    seen_s1_cand: set[str] = set()
    for i, row in enumerate(candidate_rows, start=2):
        s1_id = row.get("source1_entity_id", "").strip()
        if s1_id in seen_s1_cand:
            issues.append(f"candidate_pairs.tsv: duplicate source1_entity_id '{s1_id}'")
        seen_s1_cand.add(s1_id)

        cand_str = row.get("candidate_entity_ids", "").strip()
        if cand_str:
            cands = {x.strip() for x in cand_str.split(",")}
        else:
            cands = set()
        candidate_map[s1_id] = cands

    # Every matched ID must appear in candidates
    for i, row in enumerate(matching_rows, start=2):
        s1_id = row.get("source1_entity_id", "").strip()
        matched_str = row.get("matched_entity_ids", "").strip()
        if not matched_str:
            continue
        matched_ids = {x.strip() for x in matched_str.split(",")}
        cands = candidate_map.get(s1_id, set())
        leaked = matched_ids - cands
        if leaked:
            issues.append(f"matching_results.tsv row {i}: "
                          f"matched IDs not in candidate_pairs — {leaked} "
                          f"(pipeline bug: model predicted a pair it never evaluated)")

    return issues


def main():
    parser = argparse.ArgumentParser(description="Validate submission files.")
    parser.add_argument("--matching",  required=True, help="Path to matching_results.tsv")
    parser.add_argument("--candidate", required=True, help="Path to candidate_pairs.tsv")
    parser.add_argument("--test-dir",  required=True, help="Path to dataset/test/ directory")
    args = parser.parse_args()

    issues = validate(args.matching, args.candidate, args.test_dir)

    if not issues:
        print("PASS — submission files look valid. Safe to upload.")
        sys.exit(0)
    else:
        print(f"FAIL — {len(issues)} issue(s) found:\n")
        for i, issue in enumerate(issues, start=1):
            print(f"  {i}. {issue}")
        sys.exit(1)


if __name__ == "__main__":
    main()
