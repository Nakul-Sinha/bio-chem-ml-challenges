# Approach — Reaction Protocol Silent-Edit Repair

**Time spent:** ~4 hours (EDA, augmentation design, CV, finalization)

## Summary
A single **T5-small** encoder–decoder, fine-tuned to map the full prompt (reaction-family header
+ noisy protocol_note + silent-edit correction_notice) to the six-slot canonical sequence
`prep;activation;order;control;quench;workup`. Compliant with the rule that the submitted method
must *only fine-tune a model* on the public examples — there is no handwritten template parser
and no id→answer map in the prediction path.

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

## Preprocessing / post-processing
- Input = raw prompt; target = `slot=value;…` string. Greedy decoding (max 48 new tokens).
- Post-processing (allowed diagnostics): each emitted slot value is snapped to the valid per-slot
  vocabulary (6 values/slot) learned from train; invalid/missing slots fall back to the
  family-mode value. Guarantees a structurally valid 524-row submission.

## Local validation
- CV mirrors the hidden split by degrading a 15% family-stratified holdout to the exact test format
  and scoring with the official weighted metric.
- **CV weighted score ≈ 0.726** (3 degradation reps). References: random/family-mode ≈ 0.283;
  realistic deterministic decoder ≈ 0.693; oracle (perfect visible decode + conditional hidden) ≈
  0.734. The fine-tuned model reaches ~99% of the oracle ceiling.
- Per-slot accuracy: order 0.88, activation 0.84, control 0.76 (w 3.0), prep 0.73, quench 0.67
  (w 4.0), workup 0.39 (w 0.25). Quench is bounded by an irreducible same-slot prefix ambiguity
  (`tavo-nori`→acidic_quench/brine_split); workup is intentionally traded off (lowest weight) for
  the heavier slots it shares ambiguous prefixes with.

## What was tried / considered
- Deterministic prefix decoder (reference only, not submitted) — 0.693; the model beats it because
  it disambiguates quench from context and infers hidden slots better.
- t5-base — not pursued: the model is already within 0.008 of the oracle ceiling, so headroom is
  negligible and runtime/robustness favor t5-small (fits the A10G ~30-min budget comfortably).

## Reproducibility / compliance
- Fixed seeds; self-contained `solution.py`; reads `./dataset[/public]/`, writes
  `./working/submission.csv` and `./submission.csv`; uses only standard Kaggle libraries
  (torch + transformers). No external data, no leaderboard probing, no hardcoded predictions.
