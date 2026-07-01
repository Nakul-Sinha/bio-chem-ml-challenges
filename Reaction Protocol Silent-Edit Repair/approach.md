# Approach — Reaction Protocol Silent-Edit Repair

**Time spent:** ~4 hours (EDA, augmentation design, CV, finalization)

## Summary
A **T5-small encoder** with **six per-slot classifier heads** (one 6-way head per slot), fine-tuned
to map the full prompt (reaction-family header + noisy protocol_note + silent-edit correction_notice)
to the six-slot canonical sequence `prep;activation;order;control;quench;workup`, then **ensembled
over several random seeds** by averaging the per-slot probabilities. Compliant with the rule that the
submitted method must *only fine-tune a model* on the public examples — there is no handwritten
template parser and no id→answer map in the prediction path.

Earlier iterations used a T5-small **encoder–decoder** that generated the slot string left-to-right.
On an H100 I confirmed two things: (1) **model size is not the bottleneck** — t5-base scores the same
as t5-small (the value signal per tag is inherently noisy), and (2) left-to-right generation
**collapses the last slot (workup 0.40)** and slightly under-serves the two highest-weight slots.
Switching to symmetric classifier heads fixes workup (→0.70) and lifts control/quench, and unlike
seq2seq the seeds **ensemble effectively** (probability averaging), moving CV 0.726 → **0.736**.

## What made this work: test-matched augmentation
The decisive observation is a **train↔test distribution shift**:
- **Train** notes are clean: all six operations present, each labelled (`setup/activation/order/…`)
  and in canonical order, with a bench tag whose **word-prefix** (`word1-word2`) encodes the
  (slot, value) and whose numeric+letter suffix is noise.
- **Test** notes are degraded: **unlabelled** generic phrasings, only **3 of 6** operations shown,
  an explicit "a background operation is missing" sentence, and **unseen tag suffixes** (only the
  33 word-prefixes are shared train↔test; 71/72 full tags are new).

Training naively on the clean train format learns the wrong input distribution. Instead, each train
row is **degraded into K=6 test-style examples**: unlabelled note phrasing sampled from the test
templates, 3 randomly-shown operations, randomized tag suffixes, a missing-operation sentence, and
the original correction. Targets remain the full corrected six-slot sequence. This forces the model
to (a) decode prefix→value position-free, (b) apply the correction, and (c) **infer the 2–3 hidden
slots** from the family + visible slots (prep↔control and quench↔workup are correlated).

## Model / preprocessing / post-processing
- Input = raw prompt, mean-pooled T5-small encoder embedding → six independent 6-way softmax heads
  (one per slot). Per-slot argmax over the seed-averaged probabilities; every value is by
  construction a valid learned class, so the 524-row submission is always structurally valid.
- Runtime: `solution.py` is **self-timing** — it trains ensemble seeds one at a time and stops once
  another seed would exceed a 23-min budget (≈3 seeds on A10G, 5 on an H100), guaranteeing the
  ~30-min A10G limit on any GPU.

## Local validation
- CV mirrors the hidden split by degrading a 15% family-stratified holdout to the exact test format
  and scoring with the official weighted metric.
- **CV weighted score ≈ 0.736** for the 5-seed classifier ensemble (3 degradation reps), up from
  **0.726** for the earlier single seq2seq. References: random/family-mode ≈ 0.271; realistic
  deterministic prefix decoder ≈ 0.654. Recipes are ~85% unique (1197/1395 singletons) and every
  test prefix is seen in train, so the CV is not inflated by memorisation.
- Per-slot accuracy (ensemble): order 0.86, activation 0.83, **control 0.79 (w 3.0)**, prep 0.73,
  **quench 0.67 (w 4.0)**, **workup 0.70 (w 0.25)**. Quench is bounded by an irreducible same-slot
  prefix ambiguity; the classifier heads recover workup (the seq2seq collapsed it to 0.40 by
  generating it last).

## What was tried / considered (H100 experiments)
- **t5-base seq2seq** — CV 0.726, identical to t5-small: model capacity is not the bottleneck.
- **Target reordering** (generate quench/control first) — lifts the seq2seq to 0.733 by giving the
  high-weight slots the reliable early-generation positions; superseded by the classifier.
- **Classifier heads** (submitted) — 0.736 ensembled; symmetric slots, effective seed-ensembling,
  fixes workup, gives calibrated probabilities.
- **seq2seq+classifier blend** — 0.739 at 30% seq2seq weight, but a seq2seq is too slow to retrain
  on A10G within 30 min, so it was dropped: the +0.003 is within seed noise and not worth the
  runtime risk.
- **Slot-weighted loss** — 0.718, worse (over-focuses on high-weight slots); discarded.

## Reproducibility / compliance
- Fixed seeds; self-contained `solution.py`; reads `./dataset[/public]/`, writes
  `./working/submission.csv` and `./submission.csv`; uses only standard Kaggle libraries
  (torch + transformers). No external data, no leaderboard probing, no hardcoded predictions.
