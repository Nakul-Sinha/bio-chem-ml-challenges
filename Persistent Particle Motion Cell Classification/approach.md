# Approach — Persistent Particle Motion Cell Classification (Bedload Flow)

## Problem framing (geometry first)
Each sample is a 200×96 crop pair: **left panel** = frame *t* with the target bead
marked (red cross+circle) at the exact center (48,48); an 8-px gray separator;
**right panel** = frame *t+horizon*, cropped around the *same image location*
(so the right-panel center is the zero-displacement reference). The label is a
**deterministic quantization of a continuous 2-D displacement** (dx,dy):

```
x_band: 0:dx<-30  1:-30..-22  2:-22..-14  3:-14..-6  4:dx>=-6 (catch-all: 0/rightward)
y_band: 0:dy<-2   1:-2..0     2:0..2      3:dy>=2
motion_class = 5*y_band + x_band            # 0..19, scored by exact accuracy
```
The flow is predominantly **leftward** (x-bands live on the negative side; band 4
catches everything ≥ −6). The y-bands are **fine and symmetric around 0** (±2 px),
so the y-axis needs ~1-px precision and is the hard sub-problem. `horizon` (2/3/4)
scales the displacement magnitude → larger horizon = bigger, harder motion.

Because the label is an *exact function of (dx,dy)*, we can measure a continuous
displacement per sample and quantize it with the published edges — this is genuine
per-sample tracking, not a flow prior.

## Priors we must beat (in-sample)
- global majority class: **0.112**
- horizon-conditioned majority: **0.143**
So the majority/flow prior is ~0.14; anything real must clear that decisively.

## Leakage / honest CV
These crops come from continuous high-speed sequences, so sequence-adjacency is the
default leakage suspicion. Checks (downsampled-panel cosine NN):
- **test→train** max cosine ≈ 0.955, **zero** test images with NN ≥ 0.98 → the test
  set is cleanly held out; there is no near-duplicate to exploit.
- within-train only ~10 near-duplicate pairs out of 900; crucially, background
  similarity does **not** predict the label (samples at cosine ≥ 0.90 share a class
  only ~28% of the time). So a model cannot cheat via background — it must track.
Conclusion: **stratified 5-fold on motion_class is honest** here. The classical
matcher additionally has **no trained parameters**, so its in-sample accuracy is a
leakage-free estimate of test accuracy.

## What was tried (all on honest CV / leakage-free)
| Method | exact | x-band | y-band | notes |
|---|---|---|---|---|
| horizon-cond majority prior | 0.143 | — | — | baseline to beat |
| Classical NCC, naive | 0.458 | 0.662 | 0.578 | masked template @center |
| + plausibility window + subpixel | 0.549 | 0.72 | 0.64 | reject wrong-bead/far peaks |
| + multi-scale × multi-channel grid fusion | **0.606** | 0.773 | 0.684 | gray+clahe+green, halves 10–16 |
| (dx,dy) affine calibration | 0.606 | — | — | matcher already unbiased → no gain |
| local phase-corr subpixel refine | 0.587 | — | — | **worse** — dropped |
| CNN (aligned 6-ch, resnet18, xb+yb heads) | 0.627 | **0.862** | 0.692 | fixes gross x-errors via learned association |
| CNN resnet18 + vflip aug[yb→3−yb] + vflip TTA | 0.639 | 0.868 | 0.712 | vflip lifts the fine y-band |
| CNN resnet34 + vflip aug + TTA | 0.650 | 0.879 | 0.713 | architectural diversity |
| CNN ensemble (resnet18+resnet34, vflip) | 0.660 | 0.883 | 0.720 | |
| **Blend: CNN ensemble(log) ⊕ classical grid-marginals** | **0.676** | ~0.89 | ~0.74 | nested-5-fold-CV honest (WX=0.85,WY=0.65) |

**Headline honest number: 0.676 exact (nested 5-fold CV)** vs 0.517 AI baseline and 0.143
flow prior. Every step was gated on this honest metric; reverting was used freely
(phase-corr refine and affine calibration were tried and dropped for lack of gain).

### Why each ingredient
- **Classical grid fusion**: single-bead template matching is brittle under clutter;
  fusing masked NCC over template sizes and channels (gray/CLAHE/green) and taking a
  subpixel peak inside a plausible displacement window gives a robust (dx,dy) *and* a
  full correlation surface (secondary peaks = alternative bead candidates).
- **CNN on spatially-aligned 6-channel input** (left RGB ⊕ right RGB, marker kept):
  a conv sees both frames at the same pixel → proper inductive bias for displacement.
  It learns appearance-based association and lifts x-band 0.773→0.862 (kills the
  wrong-bead gross errors the rigid matcher makes at large horizon).
- **Vertical-flip augmentation is label-exact** (y-band edges symmetric about 0 ⇒
  yb→3−yb); it doubles data for the hard y-axis and enables vflip TTA. (Horizontal
  flip is *not* usable: x-band edges are asymmetric and we only have the quantized
  label, so dx→−dx has no clean band image.)
- **Blend**: the CNN owns x-band; the classical grid-marginal owns fine y precision.
  A log-domain blend with per-band weights (tuned by nested 5-fold CV) combines them.

## Error geometry (drove the effort)
On the classical matcher, the dominant error was **y-band off-by-one with x correct
(≈16% of samples)**, concentrated at the dy=0 boundary — pure estimation variance
(not a correctable bias). Second was **gross x-errors at horizon=4** (wrong bead).
The CNN attacks the gross x-errors; the vflip-augmented y-head + blend attack the y
boundary.

## Final solution (shipped)
`solution.py` (self-timing, A10G ≤30 min, torch + pretrained weights, no pip):
1. classical fused-grid matcher → test (dx,dy) + grid x/y-band marginals;
2. self-timed CNN ensemble (resnet18/34 seeds) trained on all 900 public images,
   vflip aug + vflip TTA, horizon scalar input;
3. log-blend CNN ⊕ classical per band (WX, WY), quantize with published edges;
4. write `working/submission.csv` (strict schema; order matches sample_submission).

Compliance: only image content + public `horizon`; no ids/hashes/order/timestamps;
no private files; no per-file hardcoding; reproducible in the expected runtime.
