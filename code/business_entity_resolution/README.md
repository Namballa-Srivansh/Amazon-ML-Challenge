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

### Step 1: Run the pipeline for your target version

**v1 — Baseline (exact-match)**
```bash
python code/business_entity_resolution/src/v1_baseline.py
```

*(Future versions will be listed here as they are built)*

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

| Version | Script | Description |
|---------|--------|-------------|
| v1 | `src/v1_baseline.py` | Exact-match blocking, format validation |
| v2 | `src/v2_blocking.py` | TF-IDF + KNN + phonetic blocking |
| v3 | `src/v3_classifier.py` | Logistic regression + 9 engineered features |
| v4 | `src/v4_threshold_tuned.py` | Platt calibration + F₀.₅ threshold |
| v5 | `src/v5_embedding_juror.py` | Multilingual sentence-transformer features |
| v6 | `src/v6_xgboost.py` | XGBoost + Double Metaphone blocking |
| v7 | `src/v7_llm_band.py` | Qwen2.5-7B on ambiguous band |
| v8 | `src/v8_match_graph.py` | Graph consistency post-processing |
| v9 | `src/v9_final.py` | Ensemble + final polish |
