"""
v7 — llm-ambiguous-band · Qwen2.5-7B jury
==========================================
Routes ONLY the pairs whose v6 classifier score falls inside the
ambiguous band [low_thresh, high_thresh] (defined by v4_calibration.py --
re-run that against v6's scores first if you haven't) to a local
Qwen2.5-7B model via Ollama for a structured match/no-match decision.
Clear-cut pairs (score below low_thresh or above high_thresh) are left
exactly as v6 decided them.

License/size constraint check: Qwen2.5-7B-Instruct is Apache 2.0 and
7.6B params -- inside the MIT/Apache + <=8B rule in the problem statement.

MANDATORY FIRST STEP (per VERSIONS.md): measure throughput on a small
sample BEFORE running the full ambiguous band. This script's `--throughput-test`
mode does exactly that and estimates total wall-clock time for the real
ambiguous-band size, so you can tighten the band in v4 if it won't fit
your time budget.

Usage:
    # 1. One-time: `ollama pull qwen2.5:7b-instruct`
    # 2. Throughput sanity check (always do this first):
    python v7_qwen_jury.py --throughput-test --sample-size 100
    # 3. Full ambiguous-band run:
    python v7_qwen_jury.py --time-budget-min 30
"""

import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import os, re, time, json, pickle, argparse
import numpy as np
import pandas as pd

REPO_ROOT   = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
                os.path.abspath(__file__)))))
TRAIN_DIR   = os.path.join(REPO_ROOT, "dataset", "mini_train")
OUTPUT_DIR  = os.path.join(REPO_ROOT, "output")
MODELS_DIR  = os.path.join(REPO_ROOT, "models")

OLLAMA_MODEL = "qwen2.5:7b-instruct"

PROMPT_TEMPLATE = """You are auditing whether two business records refer to the SAME real-world business. \
Records come from noisy, independently-collected sources (typos, abbreviations, partial \
addresses, transliteration variants, DBA/trade names are all normal noise -- do not treat \
noise alone as disqualifying).

Record A:
  Name: {name_a}
  Address: {addr_a}
  Country: {country_a}

Record B:
  Name: {name_b}
  Address: {addr_b}
  Country: {country_b}

Respond with ONLY a JSON object, no other text:
{{"match": true or false, "reasoning": "one short sentence"}}"""


def call_ollama(prompt: str) -> dict:
    """Call the local Ollama server. Returns {"match": bool, "reasoning": str}
    or a safe default (no-match) if anything about the call/parse fails --
    given F0.5's 2x precision weight, a failed judgment should never merge."""
    try:
        import ollama
        resp = ollama.chat(
            model=OLLAMA_MODEL,
            messages=[{"role": "user", "content": prompt}],
            format="json",
            options={"temperature": 0.0},
        )
        content = resp["message"]["content"]
        parsed = json.loads(content)
        return {"match": bool(parsed.get("match", False)),
                "reasoning": str(parsed.get("reasoning", ""))[:200]}
    except Exception as e:
        return {"match": False, "reasoning": f"ERROR: {e}"}


def run_throughput_test(sample_size: int):
    print(f"Running throughput test on {sample_size} sample pairs...")
    samples = [
        ("Acme Corp", "123 Main St, Springfield", "US",
         "Acme Corporation", "123 Main Street, Springfield", "US"),
        ("Sharma Traders Pvt Ltd", "Near SBI ATM, MG Road, Pune", "India",
         "Sharma Trading Co", "MG Rd, Pune", "India"),
    ] * (sample_size // 2 + 1)
    samples = samples[:sample_size]

    t0 = time.time()
    for name_a, addr_a, country_a, name_b, addr_b, country_b in samples:
        prompt = PROMPT_TEMPLATE.format(name_a=name_a, addr_a=addr_a, country_a=country_a,
                                         name_b=name_b, addr_b=addr_b, country_b=country_b)
        call_ollama(prompt)
    elapsed = time.time() - t0
    pairs_per_sec = sample_size / elapsed if elapsed > 0 else 0

    print(f"\n  {sample_size} pairs in {elapsed:.1f}s  ->  {pairs_per_sec:.2f} pairs/sec")
    print(f"  Estimated time for 10,000 ambiguous pairs:  {10_000/pairs_per_sec/60:.1f} min"
          if pairs_per_sec > 0 else "  Estimated time: N/A (0 throughput)")
    print(f"  Estimated time for 100,000 ambiguous pairs: {100_000/pairs_per_sec/60:.1f} min"
          if pairs_per_sec > 0 else "")
    print("\n  If your actual ambiguous-band size / this rate exceeds your time budget,")
    print("  go back to v4_calibration.py and shrink [low_thresh, high_thresh].")
    return pairs_per_sec


def load_ambiguous_pairs(debug_scores_path: str, low_thresh: float, high_thresh: float,
                          s1_dict: dict, s23_dict: dict) -> pd.DataFrame:
    df = pd.read_csv(debug_scores_path, sep="\t", dtype=str)
    df["prob"] = df["prob"].astype(float)
    band = df[(df["prob"] >= low_thresh) & (df["prob"] <= high_thresh)].copy()

    def lookup(d, eid, field):
        return str(d.get(eid, {}).get(field, ""))

    band["name_a"] = band["s1_id"].apply(lambda e: lookup(s1_dict, e, "business_name"))
    band["addr_a"] = band["s1_id"].apply(lambda e: lookup(s1_dict, e, "business_address"))
    band["country_a"] = band["s1_id"].apply(lambda e: lookup(s1_dict, e, "country"))
    band["name_b"] = band["cid"].apply(lambda e: lookup(s23_dict, e, "business_name"))
    band["addr_b"] = band["cid"].apply(lambda e: lookup(s23_dict, e, "business_address"))
    band["country_b"] = band["cid"].apply(lambda e: lookup(s23_dict, e, "country"))
    return band


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--throughput-test", action="store_true")
    ap.add_argument("--sample-size", type=int, default=100)
    ap.add_argument("--time-budget-min", type=float, default=30.0)
    args = ap.parse_args()

    print("=" * 60)
    print("v7 — Qwen2.5-7B ambiguous-band jury")
    print("=" * 60)

    if args.throughput_test:
        run_throughput_test(args.sample_size)
        return

    for fname in ["v4_band_low.txt", "v4_band_high.txt"]:
        if not os.path.exists(os.path.join(MODELS_DIR, fname)):
            print(f"ERROR: {fname} missing. Run v4_calibration.py first "
                  "(re-run it against v6 scores if you upgraded the classifier).")
            return

    low_thresh  = float(open(os.path.join(MODELS_DIR, "v4_band_low.txt")).read().strip())
    high_thresh = float(open(os.path.join(MODELS_DIR, "v4_band_high.txt")).read().strip())
    print(f"Ambiguous band: [{low_thresh:.2f}, {high_thresh:.2f}]")

    debug_path = os.path.join(OUTPUT_DIR, "debug_scores.tsv")
    if not os.path.exists(debug_path):
        print(f"ERROR: {debug_path} missing. Run the classifier script "
              "(v3/v5/v6) first -- it writes debug_scores.tsv.")
        return

    print("\nLoading source text for ambiguous pairs...")
    s1  = pd.read_csv(os.path.join(TRAIN_DIR, "train_source1.tsv"), sep="\t", dtype=str).fillna("")
    s23 = pd.concat([
        pd.read_csv(os.path.join(TRAIN_DIR, "train_source2.tsv"), sep="\t", dtype=str).fillna(""),
        pd.read_csv(os.path.join(TRAIN_DIR, "train_source3.tsv"), sep="\t", dtype=str).fillna(""),
    ], ignore_index=True)
    s1_dict  = s1.set_index("entity_id")[["business_name", "business_address", "country"]].to_dict("index")
    s23_dict = s23.set_index("entity_id")[["business_name", "business_address", "country"]].to_dict("index")

    band = load_ambiguous_pairs(debug_path, low_thresh, high_thresh, s1_dict, s23_dict)
    print(f"  {len(band):,} ambiguous pairs to judge")

    # Pre-flight throughput check against the ACTUAL band size
    rate = run_throughput_test(min(20, max(1, len(band))))
    est_min = (len(band) / rate / 60) if rate > 0 else float("inf")
    print(f"\nEstimated time for full band: {est_min:.1f} min (budget: {args.time_budget_min} min)")
    if est_min > args.time_budget_min:
        print("  Band is too large for the time budget. Aborting -- shrink the band in "
              "v4_calibration.py (lower MAX_BAND_FRACTION) and re-run before retrying v7.")
        return

    print("\nRunning Qwen jury on ambiguous pairs...")
    decisions = []
    t0 = time.time()
    for i, row in enumerate(band.itertuples(index=False)):
        prompt = PROMPT_TEMPLATE.format(
            name_a=row.name_a, addr_a=row.addr_a, country_a=row.country_a,
            name_b=row.name_b, addr_b=row.addr_b, country_b=row.country_b,
        )
        result = call_ollama(prompt)
        decisions.append({"s1_id": row.s1_id, "cid": row.cid,
                           "llm_match": result["match"], "reasoning": result["reasoning"]})
        if (i + 1) % 50 == 0:
            print(f"  {i+1:,}/{len(band):,} judged  ({time.time()-t0:.0f}s elapsed)")

    dec_df = pd.DataFrame(decisions)
    dec_path = os.path.join(OUTPUT_DIR, "v7_llm_decisions.tsv")
    dec_df.to_csv(dec_path, sep="\t", index=False)

    overturned = (~dec_df["llm_match"]).sum() if len(dec_df) else 0
    print(f"\n[Done] Judged {len(dec_df):,} ambiguous pairs in {time.time()-t0:.0f}s")
    print(f"  LLM said NO MATCH (overturns a positive-leaning band score) for "
          f"{overturned:,} pairs")
    print(f"  Saved: {dec_path}")
    print("  Feed this file into v9_final_ensemble.py to apply these overrides "
          "to the final matching_results.tsv.")

if __name__ == "__main__":
    main()
