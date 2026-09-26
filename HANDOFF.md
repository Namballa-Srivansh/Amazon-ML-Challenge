# 🤖 AI Handoff File — Amazon ML Challenge 2026
## Business Entity Resolution · Living Progress Document

> **For any AI picking this up:** Read this file first — top to bottom — before touching any code.
> Update the **Current Status**, **Session Log**, and **File Map** sections after every working session.

---

## ⚡ Quick Context (30-second brief)

**What we're building:** An ML pipeline that matches business records across 3 noisy data sources
(entity resolution). Source 1 is the clean reference; find all matching records in Source 2 and
Source 3 for every S1 entity.

**Metric:** Macro-averaged **F₀.₅** — precision weighted **2×** over recall. A false merge hurts
twice as much as a miss. Every threshold/design decision should bias toward precision.

**Constraint:** No external APIs. Model must be MIT/Apache 2.0 licensed, ≤ 8B params.
Qwen2.5-7B and sentence-transformers are pre-approved.

**Critical gotcha:** Test set has France records. Training only has US + India. Treat `country`
as an open string label everywhere — never hardcode `{US, India}`.

**Key files to read for full context:**
- `amazon_ml_challenge_2026_strategy.md` — architecture decisions and reasoning
- `amazon_ml_challenge_2026_improvements.md` — 5 specific technical improvements
- `VERSIONS.md` — full 9-version build plan with done-criteria (in project root)

---

## 🚦 Current Status

```
VERSION:    v4-v9 — Code written, NOT yet run against real data.
STAGE:      All 9 versions now have implementations in src/. v1-v3 were run
            and validated on mini_train. v4-v9 were written against the spec
            and v1-v3's established patterns but have never been executed --
            this repo copy has no dataset files in it.
NEXT TASK:  Run v4_calibration.py through v9_final_ensemble.py locally, in
            the order given in README.md Step 1, against the real
            mini_train/train/test data. Expect bugs on first run (untested
            code) -- report errors and numbers back before trusting any of
            the v4-v9 F0.5 figures.
BLOCKER:    None -- just needs a real run + debug pass.
```

---

## ✅ Completed

- [x] Read and understood problem statement (PDF)
- [x] Created strategy document (`amazon_ml_challenge_2026_strategy.md`)
- [x] Created improvements document (`amazon_ml_challenge_2026_improvements.md`)
- [x] Created 9-version build roadmap (`VERSIONS.md`)
- [x] Created this handoff file (`HANDOFF.md`)
- [x] Created full directory structure (`dataset/`, `output/`, `utils/`, `code/`)
- [x] Wrote `code/business_entity_resolution/src/v1_baseline.py`
- [x] Wrote `utils/validate_submission.py` (stdlib only, no deps)
- [x] Wrote `utils/explore_data.py` (run once after dataset lands)
- [x] Wrote `code/business_entity_resolution/README.md`
- [x] Wrote `code/business_entity_resolution/requirements.txt`
- [x] Wrote `utils/create_mini_dataset.py` (created 5% sample for fast local dev)
- [x] Wrote `code/business_entity_resolution/src/v2_blocking.py` (multi-key + hot-key capping)
- [x] Wrote `code/business_entity_resolution/src/v3_classifier.py` (LR, 9 features, entity-split, hard negatives) — val F0.5 = 0.9332 on mini_train
- [x] Wrote `code/business_entity_resolution/src/generate_v3_submission.py` (memory-safe full-test inference for v3)
- [x] Wrote `code/business_entity_resolution/src/retrain_10k.py`, `v4_fast_inference.py`, `v4_gpu_inference.py` (VRAM-fit + GPU inference experiments, done ahead of the versioned track — see Session 3)
- [x] Wrote `code/business_entity_resolution/src/v4_calibration.py` (Platt scaling + ambiguous band) — **not yet run**
- [x] Wrote `code/business_entity_resolution/src/v5_embeddings.py` (multilingual sentence-transformer features) — **not yet run**
- [x] Wrote `code/business_entity_resolution/src/v6_blocking_metaphone.py` (Double Metaphone blocking) — **not yet run**
- [x] Wrote `code/business_entity_resolution/src/v6_xgboost.py` (XGBoost classifier upgrade) — **not yet run**
- [x] Wrote `code/business_entity_resolution/src/v7_qwen_jury.py` (Qwen2.5-7B ambiguous-band jury via Ollama, includes throughput test) — **not yet run**
- [x] Wrote `code/business_entity_resolution/src/v8_graph_consistency.py` (triangle-consistency graph pruning) — **not yet run**
- [x] Wrote `code/business_entity_resolution/src/v9_final_ensemble.py` (final full-test inference + LLM overrides + full-set threshold sweep) — **not yet run**

---

## 🔄 In Progress

- [ ] Run and debug v4→v9 against real mini_train/train/test data (code exists, execution doesn't)
- [ ] Re-run v4_calibration.py specifically against v6's XGBoost scores (it currently calibrates whichever model/vectorizer paths you point it at — default in the file is v3's; see inline comment in README Step 1)

---

## ⏳ Up Next (in order)

- [x] **v1** — Set up directory structure, load TSVs, dumb exact-match baseline, validate format
- [x] **v2** — TF-IDF + KNN blocking, phonetic blocking, recall ≥ 95% gate
- [x] **v3** — Logistic regression classifier, 9 engineered features, hard negatives, entity-level split (code + one real run)
- [x] **v4** — Platt calibration, threshold tuning, ambiguous band definition (code written, needs a run)
- [x] **v5** — Multilingual sentence-transformer features (code written, needs a run)
- [x] **v6** — Double Metaphone blocking + XGBoost upgrade (code written, needs a run)
- [x] **v7** — Qwen2.5-7B on ambiguous band, throughput test included (code written, needs a run + Ollama set up)
- [x] **v8** — Match graph + triangle consistency pruning (code written, needs a run)
- [x] **v9** — Final ensemble, polish, submit (code written, needs a run — this is what actually generates the leaderboard file)

---

## 🗂️ File Map

### Project Root: `c:/Projects/amazon-ml-challenge/`

```
amazon-ml-challenge/
├── dataset/
│   ├── train/
│   ├── test/
│   └── mini_train/                     # 5% sample for rapid development
├── output/                             # ← generated by pipeline
│   ├── matching_results.tsv            
│   └── candidate_pairs.tsv             
├── utils/
│   ├── validate_submission.py          # run before EVERY submission
│   ├── explore_data.py                 # explore dataset statistics
│   └── create_mini_dataset.py          # samples 100k records for fast training
├── code/
│   └── business_entity_resolution/
│       ├── src/
│       │   ├── v1_baseline.py             # exact-match blocking
│       │   ├── v2_blocking.py             # multi-key + hot-key capping blocking
│       │   ├── v3_classifier.py           # LR + 9 features, trains models/v3_*
│       │   ├── generate_v3_submission.py  # v3-only full-test inference (memory-safe)
│       │   ├── retrain_10k.py             # 10k-vocab retrain for GPU VRAM fit
│       │   ├── v4_fast_inference.py       # sparse dot-product full-test inference
│       │   ├── v4_gpu_inference.py        # CuPy/cuSPARSE tiled GPU inference
│       │   ├── v4_calibration.py          # Platt scaling + ambiguous band -> models/v4_*
│       │   ├── v5_embeddings.py           # + multilingual embedding features -> models/v5_*
│       │   ├── v6_blocking_metaphone.py   # Double Metaphone blocking (overwrites candidate_pairs.tsv)
│       │   ├── v6_xgboost.py              # XGBoost on 11 features -> models/v6_*
│       │   ├── v7_qwen_jury.py            # Qwen2.5-7B ambiguous-band jury (Ollama)
│       │   ├── v8_graph_consistency.py    # triangle-consistency pruning (edits matching_results.tsv)
│       │   └── v9_final_ensemble.py       # FINAL full-test inference + LLM overrides + threshold sweep
│       ├── README.md                   
│       └── requirements.txt            
├── models/
│   ├── tfidf_vectorizer.pkl            # saved from v2 for downstream use
│   └── v3_*.pkl, v3_threshold.txt      # v3 classifier + vectorizers + threshold
│       (v4_*, v5_*, v6_* artifacts land here once those scripts are run)
├── HANDOFF.md                          # ✅ This file
├── VERSIONS.md                         # 9-version build plan
├── amazon_ml_challenge_2026_strategy.md
├── amazon_ml_challenge_2026_improvements.md
└── Documentation_template.md          
```

---

## 🧠 Architecture Decisions (Don't Re-debate These)

| Decision | Choice | Reason |
|---|---|---|
| Blocking method | Multi-Key Pandas Merge with Hot-Key Capping (max 250k) | Avoids dense matrix memory explosions and pure O(N*M) compute |
| Classifier | Logistic Regression → XGBoost (upgraded in v6) | LR first for interpretability; XGBoost for nonlinear interactions |
| Embedding model | `paraphrase-multilingual-MiniLM-L12-v2` | MIT license, handles French out-of-the-box, fast enough for feature generation |
| LLM juror | Qwen2.5-7B via Ollama | MIT/Apache, ≤8B, structured output, only on ambiguous band |
| Shared TF-IDF vectorizer | Fit once across all 3 sources | Consistent vector space for both blocking and scoring |
| Train/val split | By S1 entity ID (not by pair) | Prevents leakage — pairs of same entity stay on same side |
| Negatives | Hard negatives only (blocking candidates ∖ ground truth) | Easy negatives teach nothing; hard negatives directly target precision |
| Country handling | Open string label, never hardcoded | France in test set has no training signal; must generalize |
| Final metric | F₀.₅ (not accuracy, not F1) | The actual leaderboard metric; tune threshold specifically for this |

---

## ⚠️ Known Risks & Watchpoints

1. **Qwen2.5-7B throughput** — Unknown inference speed at challenge scale. Test pairs/sec the
   moment real data arrives before building the full v7 pipeline. If the ambiguous band is too
   large for the time budget, tighten `[low_thresh, high_thresh]` to shrink it.

2. **Blocking recall is unrecoverable** — A missed true match in blocking can never be recovered
   downstream. Do not move to v3 until recall ≥ 95% against `train_ground_truth.tsv`.

3. **France generalization** — No French training examples exist. Use multilingual embedding model
   (v5+). Test all normalization code against synthetic French business names before final submit.

4. **Singleton precision** — Wrongly predicting any match for a singleton scores 0.0 (not just a
   partial penalty). Threshold calibration is critical.

---

## 📊 Leaderboard Submission Log

| # | Date | Version | Held-out F₀.₅ | Public LB F₀.₅ | What changed | Outcome |
|---|------|---------|--------------|----------------|--------------|---------|
| 1 | — | v1 | — | — | Baseline, format validation | — |
| 2 | — | v2 | 95.7% Recall | — | Real blocking | PASS (mini dataset) |
| 3 | — | v3 | — | — | Classifier + features | — |

*(5 submissions/day limit — add rows as you go)*

---

## 🔑 Key Numbers to Track

| Metric | Target | Current |
|---|---|---|
| Blocking recall (train GT) | ≥ 95% before v3 | 95.7% (v2 on mini_train) |
| Blocking recall (train GT) | ≥ 97% after v6 | — |
| Macro Precision — v1 baseline | — | 0.2588 (dragged down by 10.8M FPs) |
| Macro Recall — v1 baseline | — | 0.1575 |
| Val F₀.₅ — v1 baseline | — | **0.1937** |
| Val F₀.₅ — v3 classifier | First real baseline to beat | **0.9332 (thresh=0.90)** |
| Val F₀.₅ — best so far | — | 0.9332 (v3) |
| Public LB F₀.₅ — best so far | — | — (not yet submitted) |
| Singletons correct | > 90% | 67.6% (39,895 wrongly matched) |
| Total FP (false merges) | Minimise | 10,865,787 (v1) |
| Ambiguous band size (% of candidates) | < 20% for Qwen budget | — (v4_calibration.py enforces this cap when run) |
| Qwen2.5-7B throughput | Measure before v7 | — pairs/sec (run `v7_qwen_jury.py --throughput-test` first) |
| Val F0.5 — v5 (embeddings) | Improve over v4 | — not yet run |
| Val F0.5 — v6 (XGBoost + metaphone) | Improve over v5 | — not yet run |
| Blocking recall — v6 (+ metaphone) | ≥ 97% | — not yet run |
| Val F0.5 — v9 (final ensemble) | Maximum | — not yet run |

---

## 📝 Session Log

### Session 4 — 2026-09-26

**Who:** User + Claude (via uploaded project zip)
**What happened:**
- Reviewed the full project deeply: confirmed v3's methodology is sound (no leakage,
  proper entity-level split, hard negatives from real blocking candidates, correct
  macro-F0.5 scorer including singletons).
- Flagged that this file (HANDOFF.md) had gone stale relative to the actual `src/`
  contents — `retrain_10k.py`, `v4_fast_inference.py`, and `v4_gpu_inference.py`
  existed but were never logged here. Fixed in this session's edits.
- Wrote all remaining versions per VERSIONS.md spec: `v4_calibration.py`,
  `v5_embeddings.py`, `v6_blocking_metaphone.py`, `v6_xgboost.py`,
  `v7_qwen_jury.py`, `v8_graph_consistency.py`, `v9_final_ensemble.py`.
- Updated `README.md` with the full v1→v9 run order and per-version status.

**Decisions made:**
- v4's ambiguous band is defined data-drivenly (probability bins where positive
  rate isn't close to 0 or 1) rather than a fixed margin around the threshold,
  with a hard cap enforcing VERSIONS.md's <20% budget target.
- v6's "Double Metaphone" uses the `metaphone` PyPI package (true double
  metaphone) with a graceful fallback to jellyfish's single Metaphone if that
  package isn't installed, since jellyfish doesn't ship double metaphone.
- v7's LLM jury is a **veto only** — it can downgrade a classifier "yes" to "no"
  for ambiguous-band pairs, but never upgrades a "no" to "yes". This matches
  the precision-first spirit of F0.5 and the fact that v9 only sends
  classifier-approved pairs into the ambiguous band to begin with.
- v8's graph consistency runs strictly AFTER v9 generates matching_results.tsv
  (it edits that file in place and keeps a `_pre_graph` backup).

**Known limitation of this session's work:** none of v4-v9 were executed —
this repo copy has no dataset files, so nothing could be run end-to-end.
Treat all new scripts as **unvalidated first drafts**: run them locally in
the order given in README.md, and expect to file bugs against this session's
work before trusting any F0.5 number they report.

**Files created/modified:**
- `code/business_entity_resolution/src/v4_calibration.py` (new)
- `code/business_entity_resolution/src/v5_embeddings.py` (new)
- `code/business_entity_resolution/src/v6_blocking_metaphone.py` (new)
- `code/business_entity_resolution/src/v6_xgboost.py` (new)
- `code/business_entity_resolution/src/v7_qwen_jury.py` (new)
- `code/business_entity_resolution/src/v8_graph_consistency.py` (new)
- `code/business_entity_resolution/src/v9_final_ensemble.py` (new)
- `code/business_entity_resolution/README.md` (updated run order + version table)
- `code/business_entity_resolution/requirements.txt` (added `metaphone`, `ollama`)
- `HANDOFF.md` (this file — synced to actual code state)

**Next session should start at:** Run `v4_calibration.py` first against the
real mini_train data and report the printed reliability table + band size
back — that determines whether v5/v6 are worth chasing before v7's LLM step.

---

### Session 3 — 2026-09-25

**Who:** User + Antigravity (Gemini 3.1 Pro)
**What happened:**
- Discovered that full TF-IDF cross-product on 24M records causes out-of-memory (OOM) errors (requested 40GB+ RAM).
- Refactored `v2_blocking.py` repeatedly to solve memory issues.
- Implemented **Multi-Key Blocking with Hot-Key Capping**: drops any blocking key that generates > 250,000 candidate pairs. This keeps the memory strictly bounded.
- Wrote `create_mini_dataset.py` to down-sample S1 to 100,000 records (5% sample) while strictly preserving all corresponding S2/S3 true matches and adding random noise.
- Extracted aggressive blocking keys (every word, acronyms, and phonetic soundex equivalents) from *both* name and address.
- Achieved **95.7% Blocking Recall** on the mini dataset in ~4.5 minutes.
- Committed and tagged as `v2-blocking`.

**Decisions made:**
- Always develop v3, v4, v5, etc. against `dataset/mini_train/` to ensure lightning-fast iteration (seconds instead of hours).
- Only run the full `train` and `test` splits when ready to generate the final leaderboard submission.

**Files created/modified:**
- `utils/create_mini_dataset.py`
- `code/business_entity_resolution/src/v2_blocking.py`

**Next session should start at:** Building `v3_classifier.py` using `mini_train` candidate pairs.

---

### Session 2 — 2026-09-25

**Who:** User + Antigravity (Claude Sonnet 4.6 Thinking)
**What happened:**
- Created full directory structure (`dataset/train/`, `dataset/test/`, `output/`, `utils/`, `code/business_entity_resolution/src/`)
- Wrote `v1_baseline.py` — exact-match blocking, correct TSV output format, self-scoring F₀.₅ on training data
- Wrote `utils/validate_submission.py` — full format checker (stdlib only, no deps), catches all rules from problem statement
- Wrote `utils/explore_data.py` — run once after dataset arrives to understand row counts, country distribution, null rates, match cardinality
- Wrote `code/business_entity_resolution/README.md` — full run instructions
- Wrote `code/business_entity_resolution/requirements.txt` — all deps for v1–v9 pre-listed

**Decisions made:** None new.
**Files created this session:**
- `code/business_entity_resolution/src/v1_baseline.py`
- `utils/validate_submission.py`
- `utils/explore_data.py`
- `code/business_entity_resolution/README.md`
- `code/business_entity_resolution/requirements.txt`

**Blocker:** Dataset not yet downloaded. `Unconfirmed 316023.crdownload` still in root.
**Next session should start at:** Dataset arrives → `python utils/explore_data.py` → `python code/business_entity_resolution/src/v1_baseline.py` → `python utils/validate_submission.py --matching output/matching_results.tsv --candidate output/candidate_pairs.tsv --test-dir dataset/test` → submit

---

### Session 1 — 2026-09-25

**Who:** User + Antigravity (Claude Sonnet 4.6 Thinking)
**What happened:**
- Parsed problem statement PDF in full
- Read strategy doc and improvements doc in full
- Created 9-version build roadmap (`VERSIONS.md`)
- Created this handoff file (`HANDOFF.md`)

**Decisions made:** Planning only — no code written yet.
**Files created this session:** `VERSIONS.md`, `HANDOFF.md`

---

*(Paste a new session block here at the start of each working session, above older ones)*

---

## 🛑 Rules for Any AI Reading This File

1. **Always update this file** at the end of a session — Current Status, Completed list, Key Numbers, Session Log.
2. **Never skip a version** without confirming its done-criteria are met.
3. **Run `validate_submission.py` before any leaderboard upload.** A wasted submission slot is a real cost.
4. **Do not hardcode country values.** `{US, India}` anywhere in code is a bug.
5. **Check the File Map before creating new files** — respect the established structure.
6. **Log every submission** in the table above, including held-out F₀.₅ alongside the LB score.
7. When in doubt about a design decision, refer to the **Architecture Decisions table** — those debates are settled. Don't re-open them without a concrete experimental result.
