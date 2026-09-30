"""
smoke_test_v10.py
==================
Runs the REAL v10_final_ensemble.py end-to-end on a small synthetic dataset
in a throwaway temp directory, in seconds, before you ever launch a full run.

Why this exists: every failure in the original v9 attempt (86M-row merges,
22GB dict cache, list accumulation) was discovered only after starting a
multi-hour run on the full data. All of them show up at small scale if you
look for them.

What it checks
  1. Pipeline runs end to end (incl. an UNSEEN country, i.e. open-set country
     handling, and a deliberately hot blocking token that must be capped).
  2. Output format rules from the problem statement (one row per S1 entity,
     no duplicate IDs, IDs exist in the test set, matches subset of
     candidates), plus the organizers' utils/validate_submission.py if present.
  3. LLM-veto handling: vetoed pair stays a candidate, leaves matches.
  4. Determinism: byte-identical outputs across different PYTHONHASHSEEDs
     (results must be reproducible for the final review).
  5. Blocking recall on known true pairs stays above a floor.
  6. Peak memory of the pipeline process stays small on the small slice.

The embedder and classifier are stand-ins (deterministic hash-based vectors,
a dummy logistic regression), so this validates PLUMBING and scaling
behaviour, not model quality.

Usage:   python tests/smoke_test_v10.py
Needs:   pandas, numpy, scikit-learn, jellyfish   (metaphone optional)
"""
import os, sys, shutil, random, pickle, subprocess, tempfile, hashlib, resource
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

HERE = os.path.dirname(os.path.abspath(__file__))
SRC  = os.path.join(os.path.dirname(HERE), "src")
VALIDATOR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(HERE))), "utils", "validate_submission.py")

RECALL_FLOOR = 0.95
COLS = ["entity_id", "business_name", "business_address", "country"]

STUB = '''import hashlib, numpy as np
class SentenceTransformer:
    def __init__(self, name): pass
    def encode(self, texts, batch_size=256, convert_to_numpy=True, show_progress_bar=False):
        out = []
        for t in texts:
            v = np.zeros(32, dtype=np.float32); s = f"  {t} "
            for i in range(len(s) - 2):
                v[int(hashlib.md5(s[i:i+3].encode()).hexdigest(), 16) % 32] += 1.0
            out.append(v)
        return np.stack(out)
'''

def make_country(country, n1, n_filler, rng):
    words = ["acme","zenith","royal","blue","star","apex","nova","delta","orion","summit",
             "pioneer","atlas","lotus","vertex","harbor","cedar","maple","falcon","ember","quartz"]
    streets = ["main","oak","park","lake","hill","river","sunset","broad"]
    def typo(s):
        if len(s) > 4 and rng.random() < 0.4:
            i = rng.randrange(1, len(s) - 1); s = s[:i] + s[i+1:]
        return s
    s1, s2, s3, gt = [], [], [], []
    for i in range(n1):
        name = f"{rng.choice(words)} {rng.choice(words)} {rng.choice(['llc','inc','ltd','sarl'])}"
        addr = f"{rng.randint(1,999)} {rng.choice(streets)} street"
        eid = f"S1-{country}-{i:05d}"; s1.append((eid, name, addr, country)); m = []
        for src, rows in (("S2", s2), ("S3", s3)):
            if rng.random() < 0.6:
                cid = f"{src}-{country}-{i:05d}"
                rows.append((cid, typo(name), addr if rng.random() < 0.7 else typo(addr), country)); m.append(cid)
        gt.append((eid, ",".join(m)))
    for j in range(n_filler):        # ~60% carry the hot token "company"
        src = "S2" if j % 2 == 0 else "S3"
        nm = f"{rng.choice(words)} company" if rng.random() < 0.6 else f"{rng.choice(words)} {rng.choice(words)}"
        (s2 if src == "S2" else s3).append((f"{src}-{country}-F{j:06d}", nm,
                                            f"{rng.randint(1,999)} {rng.choice(streets)} road", country))
    return s1, s2, s3, gt

def build_tree(root):
    rng = random.Random(1); np.random.seed(1)
    src = os.path.join(root, "code", "business_entity_resolution", "src")
    for d in (src, os.path.join(root, "dataset", "mini_train"), os.path.join(root, "dataset", "test"),
              os.path.join(root, "output"), os.path.join(root, "models"), os.path.join(root, "stubs", "sentence_transformers")):
        os.makedirs(d, exist_ok=True)
    for f in ("v10_final_ensemble.py", "embedding_store.py"):
        shutil.copy(os.path.join(SRC, f), src)
    open(os.path.join(root, "stubs", "sentence_transformers", "__init__.py"), "w").write(STUB)

    t1, t2, t3 = [], [], []
    for c, n1, nf in (("US", 900, 9000), ("FR", 300, 1500)):        # FR = country unseen in training
        a, b, d, _ = make_country(c, n1, nf, rng); t1 += a; t2 += b; t3 += d
    td = os.path.join(root, "dataset", "test")
    for n, rows in ((1, t1), (2, t2), (3, t3)):
        pd.DataFrame(rows, columns=COLS).to_csv(f"{td}/test_source{n}.tsv", sep="\t", index=False)

    a, b, d, gt = make_country("US", 300, 600, rng)
    tr = os.path.join(root, "dataset", "mini_train")
    for n, rows in ((1, a), (2, b), (3, d)):
        pd.DataFrame(rows, columns=COLS).to_csv(f"{tr}/train_source{n}.tsv", sep="\t", index=False)
    pd.DataFrame(gt, columns=["source1_entity_id", "matched_entity_ids"]).to_csv(f"{tr}/train_ground_truth.tsv", sep="\t", index=False)
    pool = [r[0] for r in b + d]; rows = []
    for eid, m in gt:
        rows.append((eid, ",".join(sorted((set(m.split(",")) - {""}) | set(rng.sample(pool, 8))))))
    pd.DataFrame(rows, columns=["source1_entity_id", "candidate_entity_ids"]).to_csv(
        os.path.join(root, "output", "candidate_pairs.tsv"), sep="\t", index=False)

    vn = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4)).fit([r[1] for r in a + b + d])
    va = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4)).fit([r[2] for r in a + b + d])
    X = np.random.rand(2000, 11); y = (X[:, 0] + X[:, 2] + 0.3 * np.random.randn(2000) > 1.0).astype(int)
    for name, obj in (("v6_classifier.pkl", LogisticRegression().fit(X, y)), ("v6_vec_name.pkl", vn), ("v6_vec_addr.pkl", va)):
        pickle.dump(obj, open(os.path.join(root, "models", name), "wb"))
    pd.DataFrame([("S1-US-00001", "S2-US-00001", False)], columns=["s1_id", "cid", "llm_match"]).to_csv(
        os.path.join(root, "output", "v7_llm_decisions.tsv"), sep="\t", index=False)
    return src

def run_v10(root, src, hashseed):
    env = dict(os.environ, PYTHONHASHSEED=str(hashseed), PYTHONPATH=os.path.join(root, "stubs"), PYTHONWARNINGS="ignore")
    wrapper = ("import resource,runpy,sys; sys.argv=['v10_final_ensemble.py','--threshold','0.9'];"
               "runpy.run_path('v10_final_ensemble.py',run_name='__main__');"
               "print('PEAK_RSS_MB',resource.getrusage(resource.RUSAGE_SELF).ru_maxrss//1024)")
    r = subprocess.run([sys.executable, "-c", wrapper], cwd=src, env=env, capture_output=True, text=True)
    assert r.returncode == 0, f"v10 crashed:\n{r.stdout[-1500:]}\n{r.stderr[-1500:]}"
    return int([l for l in r.stdout.splitlines() if l.startswith("PEAK_RSS_MB")][0].split()[1])

def md5(path): return hashlib.md5(open(path, "rb").read()).hexdigest()

def main():
    root = tempfile.mkdtemp(prefix="v10_smoke_")
    ok = True
    try:
        src = build_tree(root); out = os.path.join(root, "output"); td = os.path.join(root, "dataset", "test")
        peaks, sigs = [], []
        for seed in (1, 2):
            peaks.append(run_v10(root, src, seed))
            sigs.append({f: md5(os.path.join(out, f)) for f in ("candidate_pairs.tsv", "matching_results.tsv", "test_candidate_scores.tsv")})

        def check(name, cond, detail=""):
            nonlocal ok; ok &= bool(cond); print(f"  [{'PASS' if cond else 'FAIL'}] {name} {detail}")

        print("v10 smoke test")
        check("deterministic across PYTHONHASHSEED", sigs[0] == sigs[1])

        s1 = pd.read_csv(f"{td}/test_source1.tsv", sep="\t", dtype=str)
        ids23 = set(pd.concat([pd.read_csv(f"{td}/test_source{i}.tsv", sep="\t", dtype=str) for i in (2, 3)]).entity_id)
        m = pd.read_csv(f"{out}/matching_results.tsv", sep="\t", dtype=str).fillna("")
        c = pd.read_csv(f"{out}/candidate_pairs.tsv", sep="\t", dtype=str).fillna("")
        cm = dict(zip(c.source1_entity_id, c.candidate_entity_ids.map(lambda x: set(filter(None, x.split(","))))))
        mm = dict(zip(m.source1_entity_id, m.matched_entity_ids.map(lambda x: [t for t in x.split(",") if t])))

        check("one row per S1 entity (incl. unseen country FR)",
              len(m) == len(s1) and m.source1_entity_id.is_unique and set(m.source1_entity_id) == set(s1.entity_id))
        check("no duplicate ids / all ids exist / matches subset of candidates",
              all(len(v) == len(set(v)) and set(v) <= ids23 and set(v) <= cm[e] for e, v in mm.items()))
        check("LLM veto: stays a candidate, removed from matches",
              "S2-US-00001" in cm["S1-US-00001"] and "S2-US-00001" not in mm["S1-US-00001"])

        for ctry in ("US", "FR"):
            t = h = 0
            for e in s1.entity_id[s1.entity_id.str.contains(f"-{ctry}-")]:
                for src_ in ("S2", "S3"):
                    cid = f"{src_}-{ctry}-{e.split('-')[2]}"
                    if cid in ids23: t += 1; h += cid in cm[e]
            check(f"blocking recall {ctry} >= {RECALL_FLOOR:.0%}", h / t >= RECALL_FLOOR, f"({h}/{t})")

        check("peak memory small on the small slice", max(peaks) < 1500, f"({max(peaks)} MB)")

        if os.path.exists(VALIDATOR):
            r = subprocess.run([sys.executable, VALIDATOR, "--matching", f"{out}/matching_results.tsv",
                                "--candidate", f"{out}/candidate_pairs.tsv", "--test-dir", td], capture_output=True, text=True)
            check("official validate_submission.py", r.returncode == 0, r.stdout.strip().splitlines()[-1] if r.stdout.strip() else "")
        else:
            print("  [SKIP] official validator not found at", VALIDATOR)
    finally:
        shutil.rmtree(root, ignore_errors=True)
    print("\nRESULT:", "ALL CHECKS PASSED" if ok else "FAILURES ABOVE")
    sys.exit(0 if ok else 1)

if __name__ == "__main__":
    main()
