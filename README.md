# First Light — Business Entity Resolution

**Amazon ML Challenge 2026** · Team **First Light**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
![Python](https://img.shields.io/badge/python-3.11-blue.svg)
![Status](https://img.shields.io/badge/status-post--deadline%20rewrite-orange.svg)

**Team:** Tanish Kumar · Namballa Srivansh · Aryan Yadav

---

## What this is

A solution to Amazon's Business Entity Resolution challenge: given business
records from three independent, noisy data sources, determine which records
across sources refer to the same real-world business — with no external data
lookups allowed, and a precision-weighted metric (F₀.₅) that punishes false
merges twice as hard as missed matches.

**Status: no leaderboard submission.** The team missed the 72-hour deadline
while debugging a memory-scaling failure in the final inference stage. This
repo documents the full pipeline as built (v1 through v9, run on a validation
sample and scoring 0.95 F₀.₅), the specific way it broke at full test-set
scale, and a post-deadline rewrite (v10) that fixes the underlying
architecture issue and has been validated on a real dry run — but has not yet
been run end-to-end against the actual test set. This README exists in place
of the competition's methodology document, which was never filled in.

Everything here was built and run on a single Windows 11 laptop (AMD Ryzen 7
7435HS, RTX 4060 8GB VRAM, 16GB RAM) — no cloud instances, no institutional
compute, no server access.

---

## Results at a glance

| Metric | Value | Where |
|---|---|---|
| Best validation F₀.₅ | **0.9503** | v6 (XGBoost, 11 features) + v4 Platt calibration, on mini_train |
| Blocking recall | **97.92%** | v6, TF-IDF + Soundex + Double Metaphone keys |
| Avg candidates per entity | **6.2** | v6 blocking (13.7M candidate pairs / 2.2M S1 entities) |
| Leaderboard score | **None** | Deadline passed before a submission was made |
| v10 dry run (4,000 S1 rows/country) | **9.1GB peak RAM, 0 crashes** | Own machine, 16GB physical RAM, zero pagefile swapping |

The validation-set number (0.9503) is real and reproducible on mini_train.
It is **not** a leaderboard score, and full-test-set behavior at v6's
classifier quality has never been confirmed end-to-end (see [Post-mortem](#post-mortem--what-actually-happened)).

---

## Architecture

Two-stage pipeline, run per country (country is treated as an open string
set — the test data contains France, which never appears in training):

```mermaid
flowchart LR
    A[Source 1<br/>reference] -->|blocking keys| C{Inverted<br/>Index}
    B[Source 2 + 3<br/>noisy records] -->|blocking keys| C
    C -->|candidate pairs,<br/>capped per entity| D[Feature Engineering]
    D -->|9-11 features| E[XGBoost Classifier]
    E -->|calibrated probability| F{Ambiguous?}
    F -->|no| G[Final Decision]
    F -->|yes, ~0.4% of pairs| H[Qwen2.5-7B Jury]
    H -->|veto only| G
    G --> I[Graph Consistency Check]
    I --> J[matching_results.tsv]
```

**Blocking** determines the recall ceiling — nothing downstream can recover
a true match that blocking never proposed as a candidate — so it got the
most iteration of any stage. **Matching** turns each candidate pair into a
feature vector and a classifier decision. **Calibration** turns raw
classifier scores into a threshold and an "ambiguous band" where the
classifier is least confident. The **LLM jury** only ever reviews that
narrow band, and only ever downgrades a match to a non-match — never the
reverse, since a false merge costs twice as much as a miss under F₀.₅.
**Graph consistency** is a final cross-check: if Source 1 record A matches
both a Source 2 and a Source 3 record, but those two records look nothing
like each other, the weaker of the two edges is dropped.

---

## Version history

| Version | What it added | Result |
|---|---|---|
| v1 | Exact-match blocking, output format validation | 0.1937 F₀.₅ — establishes the baseline and the format contract |
| v2 | Real blocking: TF-IDF word keys + Soundex, hot-key capped | 95.7% blocking recall |
| v3 | Logistic regression, 9 hand-engineered features (Jaro-Winkler, Levenshtein, TF-IDF cosine, digit overlap, a name/address "lookalike" flag), entity-level train/val split, hard negatives sourced from real blocking candidates | 0.9332 val F₀.₅ |
| v4 | Platt scaling on the classifier's raw score; data-driven "ambiguous band" (probability bins where the positive rate isn't near 0 or 1), capped at <20% of candidates | Recalibrated threshold + band definition for the LLM jury |
| v5 | Added multilingual sentence-transformer embedding cosine features (name + address) | 0.9302 F₀.₅ — **worse** than v4. Embeddings didn't help a linear model; see [what didn't work](#what-didnt-work) |
| v6 | Double Metaphone blocking keys (phonetic matching for transliteration variants); classifier upgraded to XGBoost | 97.92% blocking recall, 0.9411 raw / 0.9503 calibrated F₀.₅ |
| v7 | Qwen2.5-7B (Apache 2.0, 7.6B params) as a structured-output jury for the ambiguous band only, via local Ollama | Never run against real test data — see post-mortem |
| v8 | Post-hoc graph consistency check: flags inconsistent Source 2 ↔ Source 3 triangles and prunes the weaker edge | Implemented, never run at full scale |
| v9 | Full test-set inference tying v6–v8 together | **Never completed** — repeated OOM/stalls at real scale; this is where the deadline was lost |
| v10 | Post-deadline rewrite: blocking as a capped inverted index instead of a pandas merge, embeddings in a disk-backed SQLite store instead of an in-memory dict, streaming output instead of list accumulation | Validated on a dry run (9.1GB peak RAM, zero crashes); full-scale run not yet attempted |

---

## Methodology

### Blocking

Business records from independent sources rarely share exact strings, so
candidate generation combines several signal types per record:

- **Word keys** — individual tokens ≥3 characters from the normalized name and address
- **Soundex** — phonetic code for tokens ≥5 characters, catches simple misspellings
- **Double Metaphone** — a stronger phonetic code that catches transliteration
  variants (relevant for India and France records specifically); adding this
  in v6 lifted recall from 96.23% to 97.92%
- **Acronym keys** — first letters of multi-word names, for abbreviated vs.
  full company names

Records sharing any key become candidate pairs. This is a standard
entity-resolution blocking approach, but it has a sharp failure mode at
scale: joining on shared keys via a database-style merge has no natural
bound — one common key shared by thousands of records on each side produces
a full cross-product for that key alone. **This is exactly what broke v9**
(see post-mortem). v10 replaces the merge with a capped inverted index
instead.

### Feature engineering (9→11 features)

Per candidate pair: Jaro-Winkler similarity and normalized Levenshtein
distance on both name and address; TF-IDF (character n-gram) cosine
similarity on both fields; a numeric-token overlap score (catches shared
postal codes / building numbers even when the rest of the address differs);
an interaction term (name similarity × address similarity); a "lookalike"
flag (very similar name, very different address — a specific false-positive
pattern); and, from v5 onward, multilingual sentence-embedding cosine
similarity on name and address.

### Classifier and calibration

v3 used Logistic Regression; v6 upgraded to XGBoost, which better captures
nonlinear interactions between features (e.g., the interaction term and
lookalike flag dominate XGBoost's feature importance — over 75% combined —
while the two embedding features contribute under 2%). Raw classifier
probabilities are then Platt-scaled: a sigmoid is fit on a held-out
"calibration" split (disjoint from the split used to pick the final
threshold, to avoid double-dipping the same data for two different jobs),
and the merge threshold is re-swept on a third, disjoint "eval" split.

### Ambiguous-band LLM jury

Only pairs whose calibrated probability falls in a narrow band around the
decision threshold — 0.60 to 0.90, roughly 0.4% of candidates — are sent to
a locally-run Qwen2.5-7B-Instruct model (Apache 2.0, 7.6B parameters, well
under the competition's 8B cap) for a structured match/no-match judgment
with reasoning. The jury is **veto-only**: it can downgrade a classifier
"yes" to "no," never the reverse, which matches F₀.₅'s bias toward avoiding
false merges. In an offline test against validation data, roughly 53% of
pairs the jury reviewed in a narrow high-score slice were overturned to
"no match" — a strong signal the classifier is genuinely weak in that range,
though this was never applied to a real submission (see post-mortem).

### Graph consistency

A final pass builds a graph from all predicted matches and checks: if a
Source 1 record matches both a Source 2 and a Source 3 record, do those two
records resemble each other directly? If not, the weaker of the two edges
(by classifier confidence) is dropped. Implemented in `v8_graph_consistency.py`,
never run at full scale.

---

## What didn't work

Worth stating plainly, since a lot of engineering time went into confirming
these negative results:

- **Multilingual sentence embeddings hurt more than they helped.** Adding
  them to Logistic Regression (v5) *reduced* F₀.₅ from 0.9356 to 0.9302.
  In XGBoost (v6), they scored under 2% combined feature importance. TF-IDF
  character n-grams and string-edit-distance features already captured
  nearly everything useful for short, structured business names and
  addresses — semantic embeddings added noise, not signal, at this task
  and this data scale.
- **A single pandas `merge()` is the wrong tool for blocking at this scale.**
  It has no natural per-key or per-entity bound. A bounded approach — an
  inverted index with capped candidate lists per key, as in v10 — is safer
  by construction and should have been the design from the start rather
  than something reached after three separate OOM incidents.

---

## Post-mortem — what actually happened

The team built a working, well-validated pipeline through v6 (0.9503
calibrated F₀.₅ on held-out data) inside the 72-hour window. The deadline
was lost entirely in `v9_final_ensemble.py` — the script meant to tie
everything together and run inference on the real test set
(S1 ≈ 663k rows, S2+S3 ≈ 3.8M rows for the US subset alone).

Three failures surfaced in sequence, all with the same underlying cause —
an in-memory data structure sized to the full corpus instead of to the
current unit of work — rather than three unrelated bugs:

1. **Blocking explosion.** The S1↔S23 key merge produced an 86-million-row
   intermediate DataFrame from a single 5,000-row chunk once the data hit
   real scale, because Double Metaphone keys (added in v6 for good recall
   reasons) multiplied the number of shared-key matches far beyond what
   the training-scale validation runs had shown.
2. **Embedding cache OOM.** A Python dict intended to hold every S23
   record's embedding vector approached 22GB at float32 — over the
   machine's full 16GB RAM budget before any other stage even ran.
3. **List accumulation.** Per-chunk results were appended to a Python list
   and concatenated only at the end, so memory grew for the entire
   duration of the run instead of staying bounded.

Each was patched reactively under deadline pressure (smaller chunks, lower
per-key caps, `float16` casts, manual `gc.collect()` calls, disk-append
workarounds) rather than redesigned, and the deadline passed during the
last of these attempts. A safe fallback existed the whole time —
`generate_v3_submission.py`, an earlier v3-only inference path whose
simpler blocking (no Double Metaphone) never had this failure mode — but
it was identified too late in the process to run before time ran out.

Post-deadline, with no time pressure, the two root-cause components were
rewritten rather than patched further:

- `embedding_store.py` — a SQLite-backed key-value store. Vectors are
  encoded once, persisted to disk, and read back only for the specific
  texts the current chunk needs. The full corpus is never resident in
  Python memory at once, and encoding is resumable across process
  restarts for free (every batch commits immediately).
- `v10_final_ensemble.py` — blocking rewritten as a capped inverted index
  (`{key: [cid, ...]}`, each key's list capped, built incrementally so the
  intermediate exploded table never exists) instead of a merge, with
  results streamed to disk per chunk instead of accumulated in memory.

A dry run (`--limit-s1 4000`, both countries, cold embedding cache) on the
same 16GB machine completed cleanly: 9.1GB peak RAM, zero pagefile
swapping, zero crashes, the inverted index built over the full 3.8M-row S23
corpus in about 3.5 minutes, and no tuning of the blocking caps was needed
against real data. **This has not yet been run at full scale** (the dry run
covered 12,000 of what would be hundreds of thousands of S1 queries), and
no leaderboard submission was ever made — v10 exists to make the pipeline
correct and reproducible for this repository, not to claim a competition
result that doesn't exist.

---

## Repository layout

```
code/business_entity_resolution/
├── src/
│   ├── v1_baseline.py              # exact-match blocking, format validation
│   ├── v2_blocking.py              # TF-IDF word keys + Soundex blocking
│   ├── v3_classifier.py            # Logistic regression, 9 features
│   ├── v4_calibration.py           # Platt scaling + ambiguous-band definition
│   ├── v5_embeddings.py            # + sentence-embedding features (regressed — see above)
│   ├── v6_blocking_metaphone.py    # + Double Metaphone blocking keys
│   ├── v6_xgboost.py               # XGBoost classifier on 11 features
│   ├── v7_qwen_jury.py             # Qwen2.5-7B ambiguous-band jury (Ollama)
│   ├── v8_graph_consistency.py     # triangle-consistency graph pruning
│   ├── v10_final_ensemble.py       # CURRENT final inference pipeline
│   ├── embedding_store.py          # SQLite-backed embedding cache (used by v10)
│   ├── generate_debug_scores.py    # lightweight val-set scoring without retraining
│   ├── retrain_10k.py, v4_fast_inference.py, v4_gpu_inference.py
│                                    # early GPU/VRAM-fit inference experiments
├── tests/
│   └── smoke_test_v10.py           # end-to-end synthetic test for v10, runs in seconds
├── archive/                        # v9_final_ensemble.py, generate_v3_submission.py
│                                    # (superseded by v10 — see archive/README.md)
├── requirements.txt
└── README.md                       # this file
```

---

## Running it

```bash
pip install -r requirements.txt
```

**Always smoke-test before a full run.** Every failure described above
would have shown up in seconds at small scale instead of after a multi-hour
run at full scale:

```bash
python code/business_entity_resolution/tests/smoke_test_v10.py
```

**Dry run** on your own data before committing to a full pass — watch RAM
and per-chunk timing:

```bash
cd code/business_entity_resolution/src
python v10_final_ensemble.py --threshold 0.95 --limit-s1 4000
```

**Full pipeline**, in order, from `src/`:

```bash
python v1_baseline.py
python v2_blocking.py
python v3_classifier.py
python v4_calibration.py                 # --model-prefix v3 (default is v6, run this after v6 instead)
python v5_embeddings.py
python v6_blocking_metaphone.py
python v6_xgboost.py
python v4_calibration.py                 # re-run, defaults to --model-prefix v6
python generate_debug_scores.py          # scores val split from the saved v6 model, no retraining
python v7_qwen_jury.py --throughput-test --sample-size 100   # always test throughput first
python v7_qwen_jury.py --source test     # judges the real ambiguous band (needs a v10 run first for scores)
python v10_final_ensemble.py             # full test-set inference — THIS IS THE CURRENT FINAL STAGE
python v8_graph_consistency.py           # triangle-consistency pruning, edits matching_results.tsv in place
```

Then validate before trusting the output:

```bash
python utils/validate_submission.py \
    --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv \
    --test-dir dataset/test
```

`archive/v9_final_ensemble.py` and `archive/generate_v3_submission.py` are
kept for history only — do not run them; see `archive/README.md`.

---

## License

MIT — see [`LICENSE`](LICENSE).

## Team

**First Light**

- **Tanish Kumar** — pipeline architecture, blocking, classifier, calibration, LLM jury design, memory-architecture rewrite (v10)
- **Namballa Srivansh**
- **Aryan Yadav**
