# Interrogation findings: Spectral Route Image Classification

## Data
- 926 train / 199 test / 199 sample_submission. All images 224×224 RGB, clean. Layout flat: `dataset/images/{train,test}/<id>.jpg`, csvs in `dataset/`.
- 6 classes, imbalanced: driftwood 224, fjord 156, cygnus 150, equinox 146, **aphelion 130 (anchor, 14%)**, borealis 120.
- `target_id`: 0 aphelion,1 borealis,2 cygnus,3 driftwood,4 equinox,5 fjord.

## Split / leakage
- `artifact_signature` = **degradation-profile id, not source id**: 11 unique, all impure (span classes), 8 shared train/test. Useless as a source-group key.
- Frozen convnext_tiny.fb_in22k embeddings: **random 5-fold kNN macroF1 0.584 vs grouped-by-near-dup 0.587** → within-train same-source leakage does NOT inflate random CV. **Stratified CV is trustworthy.** (near-dup clusters confounded by degradation-collapse: cos≥0.99 pairs are 92% *different* class = content-wiped images.)
- Test is genuinely held-out: test→train max-cos p50 0.79 / p99 0.95 (below train internal near-dup mode).

## Signal
- **Frozen ImageNet-22k kNN already = 0.58 macroF1** (no finetune/aug/meta) → maps to ≈0.55 Final, already > current best 0.517. Task is solvable; best is beatable. Fine-tune should clear it.
- Metadata-only GBM = **0.387 macroF1** (>> 0.167 chance): acquisition scores (RGB means, entropy, texture, edges) partly encode content = real but secondary signal. Use as optional fusion; keep only if it survives the shift.

## Stress subset (15% of Final)
- **stress_flag == (sensor_noise_score ≥ 0.5757)**, 100% accurate on test (AUC 1.000). `artifact_burden_score ≥ 0.463` = 99.5%.
- **Acquisition shift**: by that rule train is **12% stressed**, test is **52% stressed**. Test is much more degraded → robustness to heavy degradation is the key skill, and stressed training data is scarce (~113 imgs).

## Implications for modeling
1. Primary signal = image; fine-tune a pretrained backbone (allowed by rules §15).
2. **Heavy degradation-simulating augmentation** (the description's op list) is the central lever to bridge the 12%→52% stress shift.
3. CV = stratified 5-fold + exact 4-component Final; stress proxy = sensor_noise≥0.576; ALSO evaluate a re-degraded "test-sim" val to read stress robustness honestly (natural 12% slice is optimistic).
4. Decision-rule calibration matters (MacroF1+BalAcc = 75% of score → tune priors/thresholds on OOF).
5. Offline constraint at ship time → bundle backbone weights (no download at grading).
