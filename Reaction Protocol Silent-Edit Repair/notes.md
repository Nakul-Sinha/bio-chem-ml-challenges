# Reaction Protocol Silent-Edit Repair — notes

## Task
- seq2seq: prompt (family header + noisy protocol_note + correction_notice) -> 6-slot canonical
  sequence `prep;activation;order;control;quench;workup`. Each slot is one of **6** values.
- Metric: Operation-Weighted Repair Sequence Score; per-slot exact match, weights
  prep 2.20, activation 0.85, order 0.60, control 3.00, quench 4.00, workup 0.25 (÷10.90),
  averaged within hidden reaction/edit groups. Higher = better. Range 0..1.
- Rules: MUST fine-tune a model (no handwritten-parser main solution, no id→answer map,
  must use the correction_notice). 1676 train / 524 test.

## Key EDA findings
- Each slot has exactly 6 valid values (≈balanced). 8 reaction families, all present in both
  train and test (no held-out family). Output is structurally clean in train (all 6 slots).
- **Bench tags = `word1-word2-NN+Letter`.** The **word-prefix** (word1-word2) is the real signal;
  the numeric+letter suffix is essentially noise.
  - Full-tag → (slot,value) is 100% deterministic in train (108 tags, ~3 aliases per slot-value).
  - BUT test uses **different full tags** (71/72 unseen) — only the **33 word-prefixes are shared**.
    => a full-tag lookup fails on test; the model must key on the prefix. (This is why the rules
    require a learned model, not a memorized table.)
- **Correction_notice** parses deterministically: 4 templates × 6 slot-descriptions →
  (corrected_slot, new_tag). Descriptions: opening handling line=prep, line before reactive
  contact=activation, line describing which material waits=order, condition maintained during the
  hold=control, operation that ends reactivity=quench, cleanup operation=workup.
- **Train vs test distribution shift (critical):** train notes show all 6 slots, labeled and in
  order. **Test notes are UNLABELED, show only 3 of 6 tags**, always include a "missing
  operation" sentence, and use unseen suffixes. So at test ~2-3 slots are hidden and must be
  inferred from family + visible slots + the correction.
- 3 prefixes are ambiguous: `tavo-prax`→activation/ workup, `lumo-sava`→control/workup (cross-slot,
  resolvable by context), `tavo-nori`→ quench acidic_quench/brine_split (**same-slot, irreducible**).
- Slots are NOT independent: prep↔control and quench↔workup are correlated
  (e.g. control mode-acc 0.264 alone → 0.577 given prep). Exploitable for hidden slots.

## Score references (5-fold CV, val degraded to test format, weighted metric)
- Pure family-mode (ignore note):                 **0.283**
- Oracle-decode visible + family-mode hidden:      **0.692**
- Oracle-decode visible + conditional hidden:      **0.734**  (approx ceiling)
- Realistic deterministic decoder (NOT submitted): **0.693**  (quench 0.582 = bottleneck, w=4.0)

## Approach (rules-compliant)
- Fine-tune **T5** (seq2seq) prompt→sequence.
- **Test-matched augmentation**: degrade each train row into K test-style examples — unlabeled
  phrasing, 3 randomly-shown slots, randomized tag suffixes, a missing-operation sentence, keep
  the correction. Teaches decode (prefix→value, position-free), correction application, and
  hidden-slot inference, all in the test input distribution.
- Post-processing (allowed): snap each emitted slot value to the valid per-slot vocab; fall back
  to family-mode for invalid/missing → guarantees a valid submission.

## Experiments
- t5-small, K=6 aug, 8 epochs, lr 3e-4, bs 16: **CV weighted 0.7259** (val_reps=3, 15% holdout).
  Per-slot acc: prep .733 act .844 order .878 control .755 quench .673 workup .387.
- Final: trained on all 1676 rows (11,732 augmented pairs, 8 epochs) -> submission.csv (524 rows).
  Independent strict validation PASSED; slot distributions match train marginals except workup
  (collapses to organic_extract 68% — weight-0.25 trade-off, ~+0.006 max to fix, not pursued).
- Decision: ship t5-small (within 0.008 of oracle ceiling 0.734; fits A10G 30-min budget). t5-base
  not pursued (negligible headroom vs the ceiling; effort better spent on C2/C3).

## Deliverables
- `solution.py` (self-contained, reads ./dataset[/public]/, writes ./working/submission.csv + ./submission.csv)
- `submission.csv` (524 rows, validated)
- `approach.md`, `notes.md`
