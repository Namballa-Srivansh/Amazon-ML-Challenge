import sys
import csv
import os

def inject(match_file, cand_file):
    print("Loading candidate_pairs.tsv...")
    seen = {}
    with open(cand_file, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            s1 = row["source1_entity_id"]
            cands = row.get("candidate_entity_ids", "")
            if cands:
                seen[s1] = set(cands.split(","))
            else:
                seen[s1] = set()

    missing_injections = 0
    with open(match_file, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            s1 = row["source1_entity_id"]
            matched = row.get("matched_entity_ids", "")
            if not matched: continue
            
            m_set = set(matched.split(","))
            if s1 not in seen:
                seen[s1] = set()
            
            leaked = m_set - seen[s1]
            if leaked:
                seen[s1].update(leaked)
                missing_injections += 1

    if missing_injections == 0:
        print("No missing candidates found. candidate_pairs.tsv is already perfect!")
        return

    print(f"Injecting missing candidates for {missing_injections} S1 entities...")
    tmp = cand_file + ".tmp"
    with open(tmp, "w", encoding="utf-8", newline="") as out:
        writer = csv.writer(out, delimiter="\t")
        writer.writerow(["source1_entity_id", "candidate_entity_ids"])
        for s1, cands in seen.items():
            writer.writerow([s1, ",".join(sorted(list(cands)))])
            
    os.replace(tmp, cand_file)
    print("Injection complete!")

if __name__ == "__main__":
    inject(sys.argv[1], sys.argv[2])
