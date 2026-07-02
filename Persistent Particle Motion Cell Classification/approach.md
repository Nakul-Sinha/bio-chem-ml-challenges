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
| small custom CNN, from scratch (6-ch) | 0.552 | 0.853 | 0.636 | too weak on 900 imgs → dropped |
| hybrid CNN (classical grid fed as input) | 0.587–0.619 | — | — | late blend beats hybrid → dropped |
| CNN resnet18 **from scratch** (6-ch, vflip aug[yb→3−yb]+TTA) | 0.618 | **0.870** | 0.700 | BN+residual+heavy aug train well from scratch |
| CNN resnet34 from scratch | 0.614 | 0.860 | 0.698 | architectural diversity |
| + translation augmentation | 0.55–0.57 | — | — | **worse** — shifts target off center → dropped |
| CNN ensemble, 3 models (from scratch) | 0.629 | 0.876 | 0.706 | image-only |
| CNN ensemble, **7 models** (resnet18×4 + resnet34×3) | 0.636 | 0.871 | 0.716 | more seeds ⇒ +0.7% CNN, +1.1% blend |
| **Blend: CNN ensemble(log) ⊕ classical grid-marginals** | **~0.66** | 0.88 | 0.736 | per-axis-tuned WX=0.70, WY=0.50 |

**Headline honest number: ~0.66 exact** (7-model ensemble + classical blend) vs 0.517 AI
baseline and 0.143 flow prior — nested-5-fold-CV 0.650, per-axis-tuned fixed weights 0.660.
Every step was gated on this honest metric; reverting was used freely (translation aug, the
classical-grid hybrid, phase-corr refine and affine calibration were all tried and dropped).

**Blend weights — 1-D per-axis tuning.** x-band and y-band are independent argmaxes, so
each weight was tuned on its own 1-D curve (robust, low-DOF). x-band is flat over WX∈[0.5,0.7]
(classical & CNN make *complementary* x-errors — the blend beats CNN-alone 0.871→0.884);
y-band peaks sharply at WY=0.50 (0.736). The from-scratch CNN is weak enough that classical
earns near-equal weight — unlike the pretrained variant (WX=0.85).

**Compliance note.** The CNNs are trained **only on the provided public train images**
(`weights=None` — no ImageNet/external pretrained weights), to strictly honor the rule
*"ensembles trained only from the provided public training data and images."* An earlier
variant initialized from ImageNet-pretrained backbones scored 0.676 CV; it was dropped
to remove any ambiguity about external data (the ~0.036 gap is the compliance cost). The
training-free classical matcher (0.606) is compliant under the strictest reading.

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
`solution.py` (self-timing, A10G ≤30 min, torch only — no pip, no pretrained weights):
1. classical fused-grid matcher → test (dx,dy) + grid x/y-band marginals;
2. self-timed CNN ensemble (resnet18/34 seeds, `weights=None`) trained from scratch on
   all 900 public images, vflip aug + vflip TTA, horizon scalar input;
3. log-blend CNN ⊕ classical per band (WX=0.70, WY=0.50), quantize with published edges;
4. write `working/submission.csv` (strict schema; order matches sample_submission).

Compliance: only image content + public `horizon`; models trained **only on provided
public images** (no external/pretrained weights); no ids/hashes/order/timestamps; no
private files; no per-file hardcoding; reproducible in the expected runtime.

## Pretrained standby (`solution_pretrained.py`, pending maintainer OK)
Kept ready in case ImageNet-pretrained backbones are permitted (the rules list
"computer vision models / supervised image classification" as allowed; the one
ambiguity is whether "trained only from the provided public training data" bars the
ImageNet *initialization*). Identical pipeline but the CNN ensemble is
ImageNet-initialized ResNet18/34/**50** (downloaded at runtime, then fine-tuned only on
the 900 provided images — no committed/uploaded weights, no hardcoded predictions),
WX=0.85 WY=0.62. Honest nested-5-fold CV **0.679** (resnet50 adds the best single model,
0.667, and lifts the y-band). The earlier resnet18/34-only pretrained build scored 0.676
CV → **0.66 real**, so this ~0.68 build is the higher-scoring option if pretrained is
allowed. Output: `submission_pretrained.csv`. Does **not** replace the shipped
from-scratch `solution.py`/`submission.csv` until approved.
