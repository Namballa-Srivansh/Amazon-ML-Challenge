# Amazon ML Challenge 2026 — Pipeline Improvements
## Refinements Beyond the Base Strategy

These are not part of the original architecture — they're specific technical improvements that address failure modes the base strategy doesn't cover on its own. Each one is small to implement but meaningfully affects score quality or debugging speed.

---

## 1. Negative Sampling Strategy for the Classifier

`train_ground_truth.tsv` gives positive pairs directly, but the logistic regression juror also needs negative examples (candidate pairs that are *not* true matches) to train on.

**Don't use random negatives.** A randomly sampled unrelated business is too easy to reject and teaches the classifier very little.

**Use hard negatives instead:** candidate pairs that came out of your blocking stage (so they were similar enough to be retrieved) but are *not* true matches according to ground truth. These are exactly the confusable cases — like "Acme Robotics" vs. "Acme Robotix" — that the classifier most needs to learn to reject. Training on hard negatives directly targets precision, which is what F₀.₅ rewards most.

**How to generate them:** for each Source 1 entity, take its candidate list from blocking, remove the IDs that appear in its ground-truth match list — everything left over is a hard negative for that entity.

---

## 2. Entity-Level Validation Split (Avoid Leakage)

When holding out data to self-score F₀.₅ (required, since no test labels are provided), split by **Source 1 entity**, not by individual candidate pair.

**Why it matters:** if pairs belonging to the same Source 1 entity end up split across both your training set and validation set, the model has effectively already seen related examples before being "tested" on them. Your validation F₀.₅ will look better than your real leaderboard performance — a false confidence signal at exactly the point where you need accurate signal most (deciding whether a change actually helped).

**Fix:** split the list of Source 1 entity IDs first (e.g. 80/20), then assign *all* candidate pairs for a given entity to whichever side its ID landed on.

---

## 3. Calibrate Classifier Confidence, Don't Just Threshold It

If ambiguous pairs are being routed to the Qwen juror based on classifier confidence (per the jury design), raw logistic regression probabilities aren't necessarily well-calibrated — a predicted score of 0.7 doesn't reliably mean "70% of pairs scored this way are true matches."

**Why it matters here specifically:** the boundary of your "ambiguous band" controls two things at once — how many pairs get expensive LLM judgment (throughput budget) and where your F₀.₅ threshold effectively sits (precision/recall tradeoff). An uncalibrated score makes both of these harder to reason about precisely.

**Fix:** a quick calibration pass — Platt scaling, or sklearn's `CalibratedClassifierCV` — on top of the trained classifier before using its output to define the ambiguous band or the final merge threshold.

---

## 4. Handle Near-Duplicate Candidates Within One Entity's List

Source 2 or Source 3 may themselves contain near-duplicate records (a data quality issue in the source, not something you caused) that both plausibly match the same Source 1 entity.

**Risk:** output logic that isn't careful here can either accidentally drop a valid match or violate the "no duplicate entity IDs within a single ID list" validation rule.

**Fix:** explicitly deduplicate `matched_entity_ids` per row before writing output, and add a check for this case in whatever local validation you run before submission — don't rely on `validate_submission.py` alone to catch it, since it checks format, not necessarily this specific data pathology.

---

## 5. Log Merge-Decision Reasoning, Not Just the Decision

Keep a `debug_scores.tsv` (or similar) alongside the real output — per candidate pair: entity IDs, individual juror scores, final combined score, and the merge/no-merge decision.

**Why it's worth the small extra effort:**
- **Faster debugging** — when your F₀.₅ score is lower than expected, you can inspect actual false positives/negatives directly instead of guessing which stage is at fault.
- **Direct input to the methodology write-up** — the `Documentation_template.md` deliverable asks for feature engineering and approach details; this log is raw evidence for that section rather than something you'd have to reconstruct from memory afterward.

---

## 6. TF-IDF + KNN — Confirming Its Two Roles

Worth stating explicitly, since it's easy to conflate: TF-IDF + KNN already appears twice in the pipeline, doing two different jobs.

**Role A — Blocking (candidate generation).** Vectorize `business_name` (optionally concatenated with `business_address`) using TF-IDF, then use `NearestNeighbors` to retrieve the top-K most similar Source 2/3 records for each Source 1 entity. This *is* the mechanism that produces `candidate_pairs.tsv`. Tune `K` based on the recall check against `train_ground_truth.tsv` (§ base strategy, Stage 1) — too low and true matches get missed; too high and the classifier has to work harder to filter noise.

**Role B — A scoring feature (matching stage).** For each candidate pair that blocking already produced, compute TF-IDF cosine similarity again — this time as one input feature into the logistic regression juror, alongside Levenshtein, Jaro-Winkler, and the other engineered features. This is a *reuse* of the same vectorization, not a new blocking pass.

**Practical note:** these can share the same fitted `TfidfVectorizer` (fit once across all three sources' normalized text) rather than refitting separately for each role — saves time and keeps the vector space consistent between blocking and scoring.

---

## Priority If Time Is Short

If not all five fit before deadline, in order of impact:

1. **Hard negative sampling** (#1) — directly affects classifier quality and precision, the metric that matters most.
2. **Entity-level validation split** (#2) — without this, every other tuning decision is based on inflated, unreliable signal.
3. **Debug logging** (#5) — cheap to add, pays for itself the first time something looks wrong.
4. **Duplicate handling** (#4) — a correctness/validation-safety fix, low effort.
5. **Calibration** (#3) — valuable but the least likely to break the submission if skipped; the Qwen band boundary can be tuned empirically instead if time runs out.
