# Approach — Spectral Route Image Classification

**Objective (optimized directly, not a proxy):**
`Final = 0.40·MacroF1 + 0.35·BalancedAcc + 0.10·AnchorGateF1(route-aphelion vs rest) + 0.15·StressMacroF1(stress_flag=1)`

AI baseline to beat: **0.5368**. Result: honest OOF Final **≈0.81** (see table) — well above.

## 1. What the metric rewards (drives every decision)
- **75% of the score is MacroF1 + BalancedAccuracy** → per-class balance matters far more than raw accuracy. Class imbalance (driftwood 224 → borealis 120) and the decision rule must be handled.
- **10% AnchorGateF1** — a binary route-aphelion-vs-rest F1. The anchor is only 14% of train and is the weakest component; it responds strongly to decision-rule calibration.
- **15% StressMacroF1** — MacroF1 on the heavily-degraded subset. This is where the domain difficulty lives.

## 2. Interrogation (see `research/FINDINGS.md`)
- **Leakage ruled out.** `artifact_signature` is a *degradation-profile* id (11 values, all class-impure), not a source id. Frozen-embedding kNN gives random-CV 0.584 vs group-CV 0.587 macro-F1 → **stratified CV is trustworthy**. Test is genuinely held-out (test→train max-cosine sits below train's internal near-duplicate mode).
- **Stress subset decoded.** `stress_flag == (sensor_noise_score ≥ 0.5757)` — 100% accurate on the test set (its metadata is public and sample_submission carries the flag). This lets me reproduce StressMacroF1 exactly on train.
- **Acquisition shift is the core difficulty.** By that rule train is **12%** stressed but test is **52%** stressed — test is far more degraded. So *degradation-simulating augmentation* is the central lever, and the natural stress slice in CV is thin/optimistic (addressed with a re-degraded "test-sim" read).
- **Metadata is not used.** A metadata-only model reaches 0.39 macro-F1 (content statistics leak in), but **fusing metadata *hurt* the true metric** (frozen img-only 0.629 → img+meta 0.585). Per "keep only if it helps," metadata is dropped. Signal comes from the image.

## 3. Honest CV harness (`research/metric.py`, `train.py`)
- Exact 4-component Final, stratified 5-fold on `target_id`, stress proxy `sensor_noise ≥ 0.5757`.
- Two reads per fold: **natural** (train's 12% stress mix) and **test-sim** (heavy degradation applied to all val images — a conservative proxy for test's 52% heavy mix). The true test Final is expected between them.
- Decision-rule calibration: coordinate ascent on per-class probability weights, optimizing the *exact Final* on OOF only.

## 4. Model
- Fine-tune ImageNet-22k pretrained backbones (allowed by rules §15), 224px, image-only.
- **Degradation-simulating augmentation** mirroring the spec's op list (blur, sensor/salt-pepper noise, JPEG, channel scale/dropout, vignette, scanlines, brightness/contrast/gamma, random-resized-crop, erasing) at high strength to bridge the 12%→52% stress shift.
- Class-frequency-weighted cross-entropy + label smoothing 0.1; AdamW + OneCycle; fp16 AMP.
- hflip **TTA** at inference; **OOF calibration** of the decision rule; prob-average **ensemble** across diverse backbones.

## 5. Compute
- Kaggle's API-assigned GPU is a **P100 (sm_60)**, which Kaggle's prebuilt PyTorch dropped; the provisioned H100 is no longer reachable. Fix: the kernel installs an sm_60-compatible official torch, so P100 works. Training ~18 min/model (5-fold) on P100.

## 6. Results (honest OOF, exact Final)
| Config (5-fold) | OOF raw | OOF calibrated | Stress | test-sim (all-heavy) |
|---|---|---|---|---|
| frozen convnext_tiny + logreg (floor check) | 0.615 | 0.629 | — | — |
| convnext_small.fb_in22k | 0.789 | 0.814 | 0.810 | 0.741 |
| convnext_base.fb_in22k | 0.790 | 0.820 | 0.803 | **0.760** |
| swin_small.ms_in22k | 0.779 | 0.796 | 0.809 | 0.718 |
| **base + swin ensemble — SHIPPED** | **0.808** | **0.812** | **0.824** | ≈0.76 |

**Shipped honest OOF Final = 0.812 (raw 0.808) — +0.275 over the 0.5368 AI baseline.**
base+swin is chosen for the best **raw** (calibration-independent) Final and the best **Stress** — the component that matters most because test is 52% heavily degraded. Same-family pairs (base+small) and the 3-way mix scored lower; one ConvNeXt + one Swin is the most diverse pair. Calibration here is gentle (+0.004, spread across all components, anchor predictions 14%→20% — not anchor-farming), so transfer risk is low. Selection was by raw Final to avoid trusting calibration that might not cross the held-out-source shift.

## 7. Deliverable (`solution.py`)
- Self-contained, **self-timing** (default budget 27 min, hard wall 30) on a single A10G: trains the locked ensemble fold-by-fold in priority order and **always emits a valid submission**, dropping later folds if the budget would be exceeded.
- Loads pretrained weights via timm; falls back to bundled `./weights/` if offline.
- Writes `./working/submission.csv` in the exact contract (`id,target,stress_flag`, stress_flag passed through unchanged) and **asserts every grading rule inline** before finishing (`research/checker.py` is the standalone validator).

## 8. Caveats (reported faithfully)
- The natural-stress OOF slice is thin (~12%); the test-sim read (all-heavy) is conservative. True test Final expected ≈ between them.
- Calibration is tuned on the train distribution; it is validated to raise OOF Final but the test prior differs (held-out sources), so gains on the small AnchorGate term are the least certain.
- No metadata, no leakage, no test labels, no hard-coding — the class is recovered from the image.
