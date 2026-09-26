# Business Entity Resolution — Run Instructions

## Environment Setup

```bash
pip install -r requirements.txt
```

## How to Reproduce Results (End-to-End)

### Step 0: Explore the data (first time only)
```bash
python utils/explore_data.py
```

### Step 1: Run the full pipeline, in order

Each script depends on outputs from the ones before it. Run them in this
exact order (all paths relative to `code/business_entity_resolution/src/`):

```bash
python v1_baseline.py                # sanity-check format, writes first output/*.tsv
python v2_blocking.py                # real blocking -> output/candidate_pairs.tsv (recall >= 95% gate)
python v3_classifier.py              # LR classifier -> models/v3_*.pkl (val F0.5 baseline)
python v4_calibration.py             # Platt scaling + ambiguous band -> models/v4_*
python v5_embeddings.py              # + multilingual embedding features -> models/v5_*
python v6_blocking_metaphone.py      # adds Double Metaphone keys, overwrites candidate_pairs.tsv (recall >= 97% gate)
python v6_xgboost.py                 # XGBoost on 11 features -> models/v6_*
python v4_calibration.py             # re-run to re-calibrate v6's scores (uses v3 model path by default --
                                      # if you want it against v6 specifically, point MODEL/VEC paths at v6_*)
python v7_qwen_jury.py --throughput-test --sample-size 100   # ALWAYS test throughput first
python v7_qwen_jury.py --time-budget-min 30                  # judges the ambiguous band -> output/v7_llm_decisions.tsv
python v9_final_ensemble.py          # full test-set inference + LLM overrides -> output/matching_results.tsv
python v8_graph_consistency.py       # triangle-consistency pruning, edits matching_results.tsv in place
```

`generate_v3_submission.py` is kept as a standalone v3-only inference path
(useful for a quick early leaderboard submission before v6 is ready) --
it is superseded by `v9_final_ensemble.py` for the final submission.

### Step 2: Validate before submitting
```bash
python utils/validate_submission.py \
    --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv \
    --test-dir dataset/test
```
Must print `PASS` before uploading.

### Step 3: Upload to leaderboard
Upload `output/matching_results.tsv` to the challenge portal.

---

## Directory Structure

```
amazon-ml-challenge/
├── dataset/
│   ├── train/          # train_source1/2/3.tsv + train_ground_truth.tsv
│   └── test/           # test_source1/2/3.tsv
├── output/             # matching_results.tsv + candidate_pairs.tsv (generated)
├── utils/
│   ├── validate_submission.py
│   └── explore_data.py
├── code/
│   └── business_entity_resolution/
│       ├── src/        # Pipeline scripts (v1_baseline.py, v2_blocking.py, ...)
│       ├── README.md   # This file
│       └── requirements.txt
├── HANDOFF.md          # Progress tracking for AI assistants
├── VERSIONS.md         # 9-version build plan
└── Documentation_template.md
```

---

## Version History

| Version | Script(s) | Description | Status |
|---------|-----------|-------------|--------|
| v1 | `src/v1_baseline.py` | Exact-match blocking, format validation | Done |
| v2 | `src/v2_blocking.py` | TF-IDF + KNN + phonetic blocking | Done (95.7% recall on mini_train) |
| v3 | `src/v3_classifier.py`, `src/generate_v3_submission.py` | Logistic regression + 9 engineered features | Done (val F0.5 = 0.9332 on mini_train) |
| v4 | `src/v4_calibration.py` | Platt scaling + ambiguous-band definition | Written, not yet run on real data |
| v5 | `src/v5_embeddings.py` | Multilingual sentence-transformer features | Written, not yet run on real data |
| v6 | `src/v6_blocking_metaphone.py`, `src/v6_xgboost.py` | Double Metaphone blocking + XGBoost classifier | Written, not yet run on real data |
| v7 | `src/v7_qwen_jury.py` | Qwen2.5-7B jury on the ambiguous band (via Ollama) | Written, not yet run on real data |
| v8 | `src/v8_graph_consistency.py` | Triangle-consistency graph pruning | Written, not yet run on real data |
| v9 | `src/v9_final_ensemble.py` | Full test-set inference, LLM overrides, final threshold sweep | Written, not yet run on real data |

`src/retrain_10k.py`, `src/v4_fast_inference.py`, `src/v4_gpu_inference.py` are
earlier exploratory scripts (10k-feature retrain and sparse/GPU inference
speedups) built ahead of the versioned track above — see HANDOFF.md Session 3.

**Important:** v4-v9 above were written against the problem spec and the
existing v1-v3 code patterns, but this repo copy has no dataset files, so
none of them have been executed end-to-end. Run them locally in the order
in Step 1 and report back errors, recall/F0.5 numbers, or throughput
results — do not assume they're bug-free on first run.
