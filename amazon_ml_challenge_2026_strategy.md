# Amazon ML Challenge 2026 — Business Entity Resolution
## Strategy Document

---

## 1. The Problem, Precisely

Three data sources describe overlapping sets of businesses, with no shared ID:

- **Source 1** — the deduplicated reference list. Every entity here is unique.
- **Source 2, Source 3** — noisier, fragmented vendor data. May contain duplicates, typos, abbreviations, transliteration variants, and deliberate look-alikes.

For every Source 1 entity, the task is to find **all** matching records in Source 2 and Source 3 — zero, one, or many. Fields available: `entity_id` (prefixed S1-/S2-/S3-), `business_name`, `business_address`, `country`.

**Critical nuance:** `country` is an *open set*. Training data only has US and India; the test set adds France. Any logic that hardcodes or filters on `{US, India}` will silently break on French records. Treat country as an arbitrary string label, not an enum.

---

## 2. Constraints That Shape Every Decision

| Constraint | Why it matters |
|---|---|
| No external data lookup (no APIs, geocoding, DBs) | Rules out the "easy" real-world solution (call a business registry). Everything must come from the provided files. |
| Model must be MIT/Apache 2.0 licensed, ≤8B params | Qwen2.5-7B, Mistral-7B-v0.3, DeBERTa all qualify. Llama-family and commercial APIs do not. |
| Scored on macro-averaged **F₀.₅** | Precision weighted 2× over recall. A false merge costs roughly twice what a miss costs. This should bias *every* threshold decision toward "don't merge when unsure." |
| Singletons scored explicitly | A Source 1 entity with no true match: predict empty → 1.0. Predict any match → 0.0. There is no partial credit for a wrong guess on a singleton — it's actively worse than guessing nothing. |
| 5 leaderboard submissions/day | Enough to iterate deliberately, not enough to spray-and-pray. Each submission should test a specific hypothesis. |
| `candidate_pairs.tsv` must be the *final* candidate set | Not an early loose blocking pass — it's whatever your matching model actually ran inference over, last-stage. Matches must be a subset of candidates or the validator flags a pipeline bug. |

---

## 3. Pipeline Architecture

Two stages, as the challenge itself frames it — but the real work is in how each stage is built.

### Stage 1 — Blocking (Candidate Generation)

**Goal:** maximize recall. This is the ceiling on your entire score — a true match that never appears as a candidate can never be recovered downstream, no matter how good Stage 2 is.

**Approach:**
1. Normalize `business_name` and `business_address`: lowercase, strip legal suffixes (Corp/Corporation, Pvt/Private, Ltd/Limited), expand common address abbreviations (St/Street, Rd/Road).
2. Generate candidates via TF-IDF + NearestNeighbors (or lightweight embeddings/FAISS) — retrieve top-K most similar Source 2/3 records per Source 1 entity.
3. **Filter by country** — a France record should never block against a US record. This shrinks the candidate space essentially for free.
4. **Immediately measure recall against `train_ground_truth.tsv`**: what fraction of true matches actually appear somewhere in your candidate lists. Target 95%+ before moving to Stage 2. This number is non-negotiable — it's the one thing that can't be fixed later.

**Secondary blocking signal worth adding:** phonetic hashing (Soundex/Metaphone) as a second key, unioned with the TF-IDF candidates. Catches sound-alike variants that token overlap misses, and costs very little to implement.

### Stage 2 — Matching (The Jury System)

This is the part we spent the most time designing, and it's worth restating in full because the reasoning matters as much as the mechanics.

**The core idea:** rather than one model deciding match/no-match, multiple signals ("jurors") each independently score a candidate pair, and a final classifier ("judge") combines them. This works because the jurors fail in *different, mostly uncorrelated* ways:

- **Logistic regression / XGBoost on engineered features** — fast, deterministic, runs on every candidate pair. Catches surface-level typos and formatting noise directly.
- **Embedding similarity (BERT-family / sentence-transformers)** — catches semantic and structural variation (reordering, abbreviation) that character-level metrics miss. Weakness: can be fooled by topically-similar-but-distinct businesses — exactly the kind of look-alike this challenge deliberately includes ("Acme Robotics" vs "Acme Robotix").
- **Qwen2.5-7B (via Ollama, structured output)** — the strongest judgment on genuinely ambiguous cases, but by far the slowest. **This is reserved for the ambiguous confidence band only** — pairs where the first two jurors disagree or land in a middle zone — not run over every candidate pair. At scale (potentially hundreds of thousands of candidate pairs), running an LLM over everything risks blowing the time budget before you even know your dataset size. Test throughput the moment real data arrives.

**Why this beats a single model:** the three jurors were chosen because they're mechanistically different (not just different hyperparameters of the same idea), so their errors don't cluster. A pair that fools edit-distance-based features (a real typo) is unlikely to also fool embeddings; a pair that fools embeddings (semantic closeness without true identity) is exactly what LLM judgment on the ambiguous band is for. This is a legitimate application of stacked ensembling, not over-engineering — as long as it's built in the right order (see §5).

**Feature engineering for the logistic regression juror, specifically:**
- Compute **name similarity and address similarity as separate features**, never blended into one score — a pair can have a near-identical name but a different address (different branch) or vice versa, and the classifier needs both signals separately to learn the right weighting.
- Extract and compare **numeric tokens from addresses** (building numbers, PIN codes) as their own feature — strong signal even when surrounding text is noisy.
- Add an explicit **interaction term**: `name_similarity × address_similarity`. Logistic regression is linear and can't discover multiplicative interactions on its own; a true match usually needs both components reasonably high, and this feature captures that jointly — directly helps precision, which is what F₀.₅ rewards.
- Add a **look-alike flag**: high name similarity combined with a different address or different country. The challenge deliberately includes near-miss traps ("Acme Bakery LLC" at the same address as a true "Acme Robotics" match, in the video example) — an explicit feature for this pattern gives the classifier a direct signal rather than hoping it's inferred from raw scores.

**Threshold tuning:** never default to 0.5. Hold out a validation split from the training data (test labels aren't provided — the challenge explicitly expects you to self-score), compute F₀.₅ directly at several thresholds, and pick the one that maximizes it. Given the 2:1 precision weighting, this will almost always land above 0.5.

### Stage 3 — Post-Processing (Match Graph, Optional but High-Value)

Build a graph after Stage 2 scoring: nodes = records across all three sources, edges = candidate pairs above threshold, weighted by confidence. Run connected components.

**Why this helps precision specifically:** if S1-A matched S2-B and separately S1-A matched S3-C, but S2-B and S3-C look nothing alike, that's a contradiction your pairwise classifier would never catch on its own — it never directly compares S2-B to S3-C. Flagging or down-weighting these inconsistent triangles catches a class of error invisible to per-pair scoring, directly serving the metric that matters most here.

This is a genuinely good use of the RAG/KG-adjacent experience — not as literal retrieval-augmented generation, but as a graph-consistency layer, which is where that skill set actually transfers to this problem.

---

## 4. Build Order (What To Actually Do, In Sequence)

1. **Get a complete, dumb pipeline submitted within the first few hours.** Exact-match or near-exact blocking, no classifier — just to validate file format and get a real baseline score. Submission #1 matters more early than submission-quality.
2. **Blocking, properly** — TF-IDF/embeddings + country filter, recall-verified against ground truth.
3. **Logistic regression juror** — engineered features (§3), trained and threshold-tuned on a held-out F₀.₅ split. This alone is a complete, submittable pipeline.
4. **Embedding juror** — added as more features into the same classifier, not a separate system. Natural continuation of step 3.
5. **Qwen juror** — reserved for later, and ideally for when both teammates are available, since prompt design + structured output + throughput tuning benefits from a second opinion before committing engineering time.
6. **Match-graph post-processing** — once the pairwise pipeline is stable and scored.
7. **Iterate against real leaderboard feedback** — public leaderboard score tells you where to invest remaining time (low recall → invest in blocking; bleeding precision → tighten threshold or improve features). Verify every change with a submission rather than assuming it helped.

---

## 5. Versioning Plan

Git-based, one commit per change that could plausibly move the score — not per file save.

**Suggested version tags:**
- `v1-baseline` — dumb exact-match blocking, no classifier, format validation only
- `v2-blocking` — real TF-IDF/embedding blocking, recall-verified
- `v3-classifier` — logistic regression juror, engineered features
- `v4-threshold-tuned` — same model, threshold tuned specifically on held-out F₀.₅
- `v5-embedding-juror`, `v6-phonetic-blocking`, `v7-llm-ambiguous-band`, `v8-match-graph` — one idea per version, added incrementally

**Maintain a `SUBMISSIONS.md` log:** version → what changed → your own held-out F₀.₅ → public leaderboard F₀.₅. This is what lets you attribute score movement to specific changes instead of guessing, and it becomes direct evidence for the `Documentation_template.md` methodology write-up at the end — the top teams' packages get reviewed in detail, and a clean version history is exactly what that review rewards.

**Operational rules:**
- Never upload to the leaderboard without running `utils/validate_submission.py` locally first — a submission burned on a formatting error is genuinely wasted.
- Reserve at least one of the five daily submissions as a safety net — resubmit your best known-good version if you're about to test something risky, so a bad experiment doesn't cost you your standing score for the day.

---

## 6. Honest Assessment — Where This Is Strong, Where It's Risky

**Strong:**
- The jury/stacking design is sound engineering, not overkill, because the chosen signals genuinely fail differently. This is the single best idea in the plan.
- Explicitly designing for the F₀.₅ precision bias (threshold tuning, look-alike features, the match-graph consistency check) shows the metric was understood, not just noted. Most teams will optimize toward accuracy or a default threshold and lose points here.
- Reusing prior project experience (graph-based post-processing from Sylvan Eye/Product Intelligence, embedding pipelines from the YouTube summarizer, RAG-adjacent thinking) onto a genuinely different problem shape, rather than forcing a template, is the right instinct — the graph-consistency layer in particular is a non-obvious, high-value transfer.

**Risky, worth respecting:**
- **Qwen throughput is the single biggest unknown.** Nothing in the plan is wrong if the LLM judges only the ambiguous band, but if that band turns out larger than expected, or if the dataset is bigger than assumed, this becomes the bottleneck. Test this the moment real data lands, before assuming the architecture works at scale.
- **Blocking recall is the one mistake that can't be fixed later.** If a true match never becomes a candidate, no amount of classifier or LLM sophistication recovers it. This deserves disproportionate early attention relative to how "interesting" it is compared to the jury system.
- **France-in-test-only is a trap for implicit assumptions.** Any code path that works correctly for US/India by accident (because their formats happen to fit some assumption) needs to be explicitly tested against synthetic France-like data, since there's no training signal for it at all.
- **Solo build risk:** with the team partially unavailable at times, the temptation is to build multiple jurors in parallel and integrate later. Better to finish one juror completely (including its threshold-tuned score) before starting the next — a single working classifier beats three half-built ones when submission time arrives.

**Net view:** this is a well-reasoned, appropriately ambitious plan for the time available — more thoughtful than "throw an LLM at it," and more realistic than "pure classical ML only." The main risk isn't the design, it's sequencing under time pressure: the instinct to build the most interesting piece (the LLM jury) first needs to be resisted in favor of the boring-but-load-bearing pieces (blocking recall, a working baseline classifier) that everything else depends on.
