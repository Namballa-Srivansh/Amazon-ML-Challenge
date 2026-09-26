"""
v8 — match-graph · graph consistency post-processing
======================================================
Catches false-positive merges that pairwise scoring alone can't see:
inconsistent triangles where S1-A matches both S2-B and S3-C, but B and C
look nothing like each other. Since a single real business should look
like itself everywhere, a broken B<->C edge is a strong signal that the
weaker of the two S1 edges is wrong.

Input:  output/matching_results.tsv produced by v6/v9 inference (pairwise
        classifier decisions, before this post-processing step) plus the
        raw source files for computing B<->C consistency scores.
Output: output/matching_results.tsv is REWRITTEN with weak edges pruned.
        The pre-pruning version is backed up to
        output/matching_results_pre_graph.tsv so this step is reversible.

Algorithm:
  1. Build an undirected weighted graph: nodes = every entity_id that
     appears in matching_results.tsv, edges = each (s1_id, cid) match
     with weight = the classifier's confidence for that pair (from
     debug_scores.tsv / v7 decisions where available; defaults to 1.0
     for pairs with no stored score).
  2. For each S1 node with >=2 matched neighbors from DIFFERENT sources
     (one S2 neighbor, one S3 neighbor), compute a name similarity
     between the S2 and S3 neighbor directly (Jaro-Winkler on normalized
     names -- cheap and license-free, no need to reload TF-IDF/embedding
     models here).
  3. If that cross-neighbor similarity is below CONSISTENCY_THRESHOLD,
     the triangle is inconsistent -> drop the LOWER-CONFIDENCE of the
     two S1 edges (keep the one the classifier was more sure about).
  4. Singletons (no edges) are untouched and stay empty, as required.
"""

import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import os, time
import pandas as pd
import jellyfish
import networkx as nx

REPO_ROOT   = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
                os.path.abspath(__file__)))))
TEST_DIR    = os.path.join(REPO_ROOT, "dataset", "test")
OUTPUT_DIR  = os.path.join(REPO_ROOT, "output")

CONSISTENCY_THRESHOLD = 0.55   # below this, an S2<->S3 neighbor pair is "inconsistent"
DEFAULT_EDGE_WEIGHT   = 1.0    # used when no per-pair confidence is available

def load_names(test_dir: str) -> dict:
    names = {}
    for fname in ["test_source1.tsv", "test_source2.tsv", "test_source3.tsv"]:
        path = os.path.join(test_dir, fname)
        if not os.path.exists(path):
            continue
        df = pd.read_csv(path, sep="\t", dtype=str).fillna("")
        for row in df.itertuples(index=False):
            names[row.entity_id] = str(row.business_name).lower().strip()
    return names

def load_edge_weights() -> dict:
    """Best-effort load of per-pair confidence from debug_scores.tsv, keyed
    by (s1_id, cid). Missing pairs fall back to DEFAULT_EDGE_WEIGHT."""
    path = os.path.join(OUTPUT_DIR, "debug_scores.tsv")
    weights = {}
    if os.path.exists(path):
        df = pd.read_csv(path, sep="\t", dtype=str)
        if "prob" in df.columns:
            for row in df.itertuples(index=False):
                weights[(row.s1_id, row.cid)] = float(row.prob)
    return weights

def main():
    print("=" * 60)
    print("v8 — match-graph consistency pruning")
    print("=" * 60)
    t0 = time.time()

    match_path = os.path.join(OUTPUT_DIR, "matching_results.tsv")
    if not os.path.exists(match_path):
        print(f"ERROR: {match_path} missing. Run the inference script "
              "(generate_v3_submission.py or v9_final_ensemble.py) first.")
        return

    print("\n[1/4] Loading matches, names, and edge confidences...")
    matches = pd.read_csv(match_path, sep="\t", dtype=str).fillna("")
    names = load_names(TEST_DIR)
    weights = load_edge_weights()

    # Backup pre-pruning version
    backup_path = os.path.join(OUTPUT_DIR, "matching_results_pre_graph.tsv")
    matches.to_csv(backup_path, sep="\t", index=False)
    print(f"  Backed up pre-pruning results to {os.path.basename(backup_path)}")

    print("\n[2/4] Building match graph...")
    G = nx.Graph()
    edge_rows = []  # (s1_id, cid, weight) for triangle scan
    for row in matches.itertuples(index=False):
        s1_id = row.source1_entity_id
        cids  = [c for c in str(row.matched_entity_ids).split(",") if c]
        for cid in cids:
            w = weights.get((s1_id, cid), DEFAULT_EDGE_WEIGHT)
            G.add_edge(s1_id, cid, weight=w, source1=s1_id)
            edge_rows.append((s1_id, cid, w))
    print(f"  Nodes: {G.number_of_nodes():,}  |  Edges: {G.number_of_edges():,}")
    print(f"  Connected components: {nx.number_connected_components(G):,}")

    print("\n[3/4] Scanning for inconsistent triangles (S1 -> S2 & S3, S2<->S3 mismatch)...")
    from collections import defaultdict
    by_s1 = defaultdict(list)
    for s1_id, cid, w in edge_rows:
        by_s1[s1_id].append((cid, w))

    edges_to_drop = set()
    n_checked, n_flagged = 0, 0
    for s1_id, neighbors in by_s1.items():
        s2_neighbors = [(c, w) for c, w in neighbors if c.startswith("S2-")]
        s3_neighbors = [(c, w) for c, w in neighbors if c.startswith("S3-")]
        if not s2_neighbors or not s3_neighbors:
            continue
        for c2, w2 in s2_neighbors:
            for c3, w3 in s3_neighbors:
                n_checked += 1
                n2, n3 = names.get(c2, ""), names.get(c3, "")
                if not n2 or not n3:
                    continue
                sim = jellyfish.jaro_winkler_similarity(n2, n3)
                if sim < CONSISTENCY_THRESHOLD:
                    n_flagged += 1
                    weaker = (s1_id, c2) if w2 < w3 else (s1_id, c3)
                    edges_to_drop.add(weaker)

    print(f"  Triangles checked: {n_checked:,}  |  Inconsistent: {n_flagged:,}")
    print(f"  Edges pruned: {len(edges_to_drop):,}")

    print("\n[4/4] Rebuilding matching_results.tsv from pruned graph...")
    pruned_rows = []
    for s1_id, neighbors in by_s1.items():
        kept = [c for c, w in neighbors if (s1_id, c) not in edges_to_drop]
        pruned_rows.append({"source1_entity_id": s1_id,
                             "matched_entity_ids": ",".join(sorted(set(kept)))})

    # Preserve every S1 row from the original file (including singletons
    # that had zero matches to begin with -- they never entered by_s1)
    pruned_df = pd.DataFrame(pruned_rows)
    out_df = matches[["source1_entity_id"]].merge(pruned_df, on="source1_entity_id", how="left")
    out_df["matched_entity_ids"] = out_df["matched_entity_ids"].fillna("")

    out_df.to_csv(match_path, sep="\t", index=False)

    n_before = (matches["matched_entity_ids"] != "").sum()
    n_after  = (out_df["matched_entity_ids"] != "").sum()
    print(f"  Entities with >=1 match before: {n_before:,}")
    print(f"  Entities with >=1 match after:  {n_after:,}")
    print(f"\n[Done] v8 finished in {time.time()-t0:.1f}s. Overwrote {match_path}.")
    print(f"  (Pre-pruning backup at {os.path.basename(backup_path)} if you need to revert.)")

if __name__ == "__main__":
    main()
