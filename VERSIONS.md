# Amazon ML Challenge 2026 — Version Roadmap

> **Metric:** Macro-averaged F₀.₅ (precision weighted 2× over recall)
> **Rule:** One idea per version. Never submit without running `validate_submission.py` first.
> **Log:** After each submission, record held-out F₀.₅ + leaderboard F₀.₅ in HANDOFF.md.

---

## Submission Log

| Version | Held-out F₀.₅ | Public LB F₀.₅ | Notes |
|---------|--------------|----------------|-------|
| v1 | — | — | |
| v2 | — | — | |
| v3 | — | — | |
| v4 | — | — | |
| v5 | — | — | |
| v6 | — | — | |
| v7 | — | — | |
| v8 | — | — | |
| v9 | — | — | |

---

## v1 — `baseline` · Format Validation + Dumb Exact Match

**Goal:** Get a valid submission accepted by the leaderboard. Score doesn't matter yet.

**What it does:**
- Load all 6 source TSVs with `sep="\t"`
- Block by exact `country` match only
- Match on exact `business_name` string (lowercased, stripped)
- Write `matching_results.tsv` and `candidate_pairs.tsv` with correct columns
- Run `validate_submission.py` → must print PASS before uploading

**What it does NOT do:** Any real ML. That's fine.

**Done when:** Leaderboard returns a SCORED status (not a format error).

---

## v2 — `blocking` · TF-IDF + KNN Candidate Generation

**Goal:** Build the real blocking stage. Establish recall ceiling ≥ 95% against `train_ground_truth.tsv`.

**What it adds:**
- Normalize `business_name` and `business_address`:
  - Lowercase, strip punctuation
  - Expand legal suffixes: `Corp→Corporation`, `Pvt→Private`, `Ltd→Limited`, `Inc→Incorporated`
  - Expand address abbreviations: `St→Street`, `Rd→Road`, `Ave→Avenue`, `Blvd→Boulevard`
- Fit a single `TfidfVectorizer` across all S1+S2+S3 normalized names (reused in v3+)
- `NearestNeighbors` top-K retrieval (start K=10, tune based on recall check)
- **Country filter:** only block S1 against S2/S3 records with the same `country` value (treats country as open string label — no hardcoding)
- **Secondary blocking:** Soundex/Metaphone phonetic keys, unioned with TF-IDF candidates
- **Measure recall** against `train_ground_truth.tsv` — print the number. Must hit 95%+ before moving on.

**Done when:** Recall ≥ 95% on training ground truth AND `validate_submission.py` passes.

---

## v3 — `classifier` · Logistic Regression Juror + Engineered Features

**Goal:** First real ML model. A complete, submittable precision-tuned pipeline.

**What it adds:**
- **Hard negatives:** For each S1 entity, take its candidate list from v2 blocking → remove ground-truth match IDs → everything left = hard negative training examples
- **Entity-level train/val split:** Split on S1 entity IDs (80/20), assign ALL candidate pairs for an entity to the same side (prevents leakage)
- **Feature engineering per candidate pair:**
  - `name_jaro_winkler` — Jaro-Winkler similarity on normalized names
  - `name_levenshtein` — normalized edit distance on names
  - `name_tfidf_cosine` — TF-IDF cosine (reuse vectorizer from v2)
  - `addr_jaro_winkler` — address Jaro-Winkler
  - `addr_levenshtein` — address edit distance
  - `addr_tfidf_cosine` — address TF-IDF cosine
  - `numeric_token_overlap` — Jaccard of numeric tokens (building numbers, PINs, ZIP codes) extracted from addresses
  - `name_x_addr` — interaction term: `name_tfidf_cosine × addr_tfidf_cosine`
  - `lookalike_flag` — high name similarity + different address or different country (binary)
- Train `LogisticRegression` on above features
- **Threshold tuning:** sweep thresholds, pick the one maximizing held-out F₀.₅ (will be > 0.5)
- Add `debug_scores.tsv` output: per candidate pair → entity IDs, all feature values, final score, merge decision

**Done when:** Held-out F₀.₅ is better than v2 AND submission passes validation.

---

## v4 — `threshold-tuned` · Calibration + Optimized Threshold

**Goal:** Make the classifier confidence scores meaningful before routing pairs to the LLM jury.

**What it adds:**
- Apply **Platt scaling** (`CalibratedClassifierCV(method='sigmoid')`) on top of the v3 classifier
- Re-sweep thresholds on calibrated probabilities and re-pick best F₀.₅ threshold
- Define the **ambiguous band** boundaries: `[low_thresh, high_thresh]` — pairs in this band go to Qwen in v7
- Verify calibration: plot predicted probability vs actual positive rate (reliability diagram)

**Same model, same features — only calibration and threshold change.**

**Done when:** Calibrated held-out F₀.₅ ≥ v3 AND ambiguous band size is known + reasonable.

---

## v5 — `embedding-juror` · Sentence Embedding Similarity Features

**Goal:** Add semantic signal that character-level features miss (reordering, abbreviation, structural variation).

**What it adds:**
- Encode `business_name` and `business_address` with a sentence-transformer:
  - Model: `paraphrase-multilingual-MiniLM-L12-v2` (MIT licensed, handles French)
- Add to feature set:
  - `name_embed_cosine` — cosine similarity of name embeddings
  - `addr_embed_cosine` — cosine similarity of address embeddings
- Retrain the logistic regression juror with these additional features
- Re-tune threshold on held-out F₀.₅

**Note:** Multilingual model is intentional — handles France test data with no training signal.

**Done when:** Held-out F₀.₅ improves over v4 AND submission passes validation.

---

## v6 — `phonetic-blocking` · Improved Blocking + XGBoost Upgrade

**Goal:** Improve blocking recall and upgrade the classifier to handle nonlinear interactions.

**What it adds:**
- **Blocking:** Add Double Metaphone as a third blocking key (unioned with TF-IDF + Soundex from v2). Tune K in NearestNeighbors if recall is still below 97%.
- **Classifier upgrade:** Replace `LogisticRegression` with `XGBClassifier` (or `LGBMClassifier`) — handles `name_x_addr` interaction and `lookalike_flag` nonlinearly
- Re-tune threshold on held-out F₀.₅
- Update `debug_scores.tsv` with new model's scores

**Done when:** Blocking recall ≥ 97% AND held-out F₀.₅ improves over v5.

---

## v7 — `llm-ambiguous-band` · Qwen2.5-7B Jury for Ambiguous Pairs

**Goal:** Use LLM judgment only on the pairs neither the classifier nor embeddings can confidently resolve.

**What it adds:**
- Set up Qwen2.5-7B via **Ollama** with structured JSON output
- Route only pairs where calibrated score ∈ `[low_thresh, high_thresh]` (defined in v4) to Qwen
- Prompt template: provide both records' name + address + country, ask for binary match decision + 1-sentence reasoning
- LLM decision overrides classifier for ambiguous band pairs only; clear predictions unchanged
- **⚠️ Throughput test FIRST:** before building the full pipeline, measure pairs/second on a 100-pair sample. If ambiguous band is too large for the time budget, tighten the band boundaries.

**Done when:** LLM jury runs within time budget AND held-out F₀.₅ improves over v6.

---

## v8 — `match-graph` · Graph Consistency Post-Processing

**Goal:** Catch false positive merges that pairwise scoring cannot — inconsistent triangles.

**What it adds:**
- Build undirected graph: nodes = all records (S1+S2+S3), edges = pairs above merge threshold, edge weight = confidence score
- Run **connected components** — each component is a proposed merged entity cluster
- **Triangle inconsistency check:** if S1-A matched S2-B and S1-A matched S3-C, but S2-B ↔ S3-C similarity is low → flag the weaker edge and re-evaluate (drop or down-weight it)
- Recompute `matching_results.tsv` from cleaned graph
- **Singleton preservation:** S1 nodes with no edges → empty `matched_entity_ids`

**Done when:** Held-out F₀.₅ (especially precision) improves over v7.

---

## v9 — `final` · Ensemble + Polish

**Goal:** Maximum score. Clean up, harden, and squeeze remaining performance.

**What it adds (pick whichever moves the needle after v8):**
- Ensemble v6 classifier + v7 LLM scores with a learned meta-weight (stacking)
- Expand normalization: handle `&` ↔ `and`, DBA names, transliteration variants
- Address numeric token comparison: normalize PIN/ZIP formats before comparison
- Final threshold sweep on the full training set (architecture is now locked)
- Deduplicate `matched_entity_ids` per row (defensive check before final output)
- Run `validate_submission.py` → PASS → submit
- Fill in `Documentation_template.md` for the final zip

**Done when:** This is the final submission. All deliverables ready for the zip archive.

---

## Build Order

```
v1 → v2 → v3 → v4 → v5 → v6 → v7 → v8 → v9
      ↑         ↑              ↑
   recall    foundation     test Qwen
   gate      for all        throughput
   (95%)     later work     FIRST
```

> **Never skip ahead.** A working v3 beats a broken v7.
> Finish each version end-to-end (including a leaderboard submission) before starting the next.

> **France warning:** Test every normalization and country-filtering code path against
> synthetic France-like records. Any implicit assumption about {US, India} format will
> silently fail on the private leaderboard.
