# Spectral Route Image Classification

## Problem Description

### 1. Overview
Image-recognition systems are often expected to operate on samples captured under imperfect conditions. In practical acquisition pipelines, an image may contain blur, field loss, compression damage, sensor noise, lighting variation, color-channel imbalance, scanline artifacts, vignetting, or local occlusion. A reliable model should recover the underlying visual class even when these acquisition effects alter the image appearance.

Spectral Route Attribution is a six-class robust visual classification benchmark. The dataset is created from six curated source image groups. During preparation, each source sample is converted into a standardized degraded RGB JPEG image using deterministic acquisition-style transformations. These transformations simulate realistic capture stress while removing direct public references to the original source-group names.

The original class names are replaced with anonymized benchmark labels called spectral routes. Participants must learn to identify the correct route from the degraded image and optional public auxiliary acquisition features.

The objective is:

Given a degraded image and its public auxiliary metadata, predict the correct spectral route label.

Each test image belongs to exactly one of six classes:

- route-aphelion
- route-borealis
- route-cygnus
- route-driftwood
- route-equinox
- route-fjord

The challenge rewards models that perform well across all six classes, correctly separate the anchor route from the remaining routes, and remain stable on heavily degraded acquisition-stress samples.

The spectral route names are anonymized competition labels only. They are not medical diagnoses, biological labels, or semantic category descriptions.

### 2. Task
Task type: multi-class image classification.

For every row in `public/test.csv`, participants must predict:

- target

The prediction must be one of the six valid route labels:

- route-aphelion
- route-borealis
- route-cygnus
- route-driftwood
- route-equinox
- route-fjord

The grader also accepts integer class ids from 0 to 5, but route-label strings are recommended.

### 3. Dataset Preparation Context
The benchmark is generated from six source image groups. The preparation process removes source folder names, original source labels, and original file paths from the public dataset files. Public CSV files expose only anonymized sample ids, prepared image paths, training labels, and non-leaky acquisition metadata.

Each prepared image is saved as a 224 × 224 RGB JPEG image.

The preparation pipeline applies acquisition profiles with different degradation strengths. These profiles may introduce:

- random resized cropping
- horizontal flipping
- brightness variation
- contrast variation
- color variation
- channel scaling
- channel dropout
- additive sensor noise
- salt-and-pepper noise
- vignetting
- scanline artifacts
- random erasing
- Gaussian blur
- JPEG compression

Some profiles produce mild or moderate distortion, while stronger profiles produce heavier acquisition stress.

Training samples are generated with degraded views from the training portion of the source images. Test samples are generated from held-out source images with one prepared degraded image per test row.

The benchmark is designed to test whether a model can recover the underlying spectral route rather than memorizing simple acquisition artifacts.

### 4. Dataset Structure
The dataset has the following structure:

```
dataset/
├── public/
│   ├── train.csv
│   ├── test.csv
│   ├── sample_submission.csv
│   └── images/
│       ├── train/
│       └── test/
└── private/
    └── answers.csv
```

`public/train.csv` contains training ids, image paths, target labels, target ids, and public auxiliary acquisition features.

`public/test.csv` contains test ids, image paths, and public auxiliary acquisition features.

`public/sample_submission.csv` shows the exact submission schema required by the grader.

`private/answers.csv` contains the hidden ground-truth target labels and authoritative stress flags used for scoring.

Images should be loaded using paths relative to the public dataset folder:

```
./dataset/public/
```

Example image paths:

```
./dataset/public/images/train/<id>.jpg
./dataset/public/images/test/<id>.jpg
```

### 5. Target Labels
The target mapping is:

- 0 → route-aphelion
- 1 → route-borealis
- 2 → route-cygnus
- 3 → route-driftwood
- 4 → route-equinox
- 5 → route-fjord

The target column may contain either the canonical route-label string or the corresponding integer class id.

Recommended target values:

- route-aphelion
- route-borealis
- route-cygnus
- route-driftwood
- route-equinox
- route-fjord

### 6. train.csv
`train.csv` contains one row per prepared training image.

Columns in `train.csv`, in exact order:

- id
- image_path
- target
- target_id
- artifact_signature
- sensor_noise_score
- field_retention_score
- focus_integrity_score
- illumination_stability_score
- chromatic_balance_score
- compression_resilience_score
- artifact_burden_score
- image_brightness_score
- image_contrast_score
- color_variation_score
- channel_shift_score
- edge_density_score
- texture_energy_score
- image_entropy_score
- sharpness_proxy_score
- red_mean_score
- green_mean_score
- blue_mean_score
- local_uniformity_score
- acquisition_complexity_score

Column descriptions:

`id` is the anonymized sample identifier.

`image_path` is the relative path to the prepared JPEG image inside `dataset/public/`.

`target` is the ground-truth spectral route label for the training sample.

`target_id` is the integer class id from 0 to 5.

`artifact_signature` is a categorical acquisition signature derived from the degradation profile and compression-quality bucket. It is a string identifier such as `aq-xxxxxx`.

`sensor_noise_score` is a numeric score from 0 to 1 estimating the strength of synthetic sensor noise. Higher values indicate stronger noise.

`field_retention_score` is a numeric score from 0 to 1 estimating how much image field remains after cropping. Higher values indicate less crop loss.

`focus_integrity_score` is a numeric score from 0 to 1 estimating retained focus quality. Higher values indicate less blur.

`illumination_stability_score` is a numeric score from 0 to 1 estimating lighting consistency. Higher values indicate more stable illumination.

`chromatic_balance_score` is a numeric score from 0 to 1 estimating color-channel balance. Higher values indicate less chromatic disturbance.

`compression_resilience_score` is a numeric score from 0 to 1 derived from JPEG quality. Higher values indicate milder compression.

`artifact_burden_score` is a numeric score from 0 to 1 summarizing the combined acquisition degradation burden. Higher values indicate stronger overall corruption.

`image_brightness_score` is a numeric score from 0 to 1 measuring average grayscale brightness.

`image_contrast_score` is a numeric score from 0 to 1 measuring grayscale contrast.

`color_variation_score` is a numeric score from 0 to 1 measuring saturation-like color spread across channels.

`channel_shift_score` is a numeric score from 0 to 1 measuring imbalance between color-channel means and channel standard deviations.

`edge_density_score` is a numeric score from 0 to 1 measuring local edge energy from grayscale differences.

`texture_energy_score` is a numeric score from 0 to 1 measuring local texture residual energy.

`image_entropy_score` is a numeric score from 0 to 1 measuring normalized grayscale entropy.

`sharpness_proxy_score` is a numeric score from 0 to 1 measuring gradient-based sharpness.

`red_mean_score` is a numeric score from 0 to 1 measuring mean red-channel intensity.

`green_mean_score` is a numeric score from 0 to 1 measuring mean green-channel intensity.

`blue_mean_score` is a numeric score from 0 to 1 measuring mean blue-channel intensity.

`local_uniformity_score` is a numeric score from 0 to 1 estimating local smoothness or uniformity. Higher values indicate more uniform local structure.

`acquisition_complexity_score` is a numeric score from 0 to 1 combining artifact burden, entropy, texture energy, and edge density into one acquisition-complexity proxy.

All numeric auxiliary features are public metadata generated from the degraded images. They are optional model inputs and should not be treated as direct labels.

### 7. test.csv
`test.csv` contains one row per prepared test image.

Columns in `test.csv`, in exact order:

- id
- image_path
- artifact_signature
- sensor_noise_score
- field_retention_score
- focus_integrity_score
- illumination_stability_score
- chromatic_balance_score
- compression_resilience_score
- artifact_burden_score
- image_brightness_score
- image_contrast_score
- color_variation_score
- channel_shift_score
- edge_density_score
- texture_energy_score
- image_entropy_score
- sharpness_proxy_score
- red_mean_score
- green_mean_score
- blue_mean_score
- local_uniformity_score
- acquisition_complexity_score

The test labels are hidden. Participants must predict the target label for every test id.

The test file does not contain `target` or `target_id`.

### 8. Public Auxiliary Features
The auxiliary feature columns are public acquisition descriptors generated from the degraded images.

Participants may use:

- image-only models
- metadata-only models
- image-plus-metadata fusion models

The auxiliary features describe degradation strength, retained field, focus quality, illumination stability, chromatic balance, compression quality, brightness, contrast, color variation, channel shift, edge density, texture energy, entropy, sharpness, local uniformity, and acquisition complexity.

These features are non-leaky proxy features. They describe image condition and acquisition quality, but they do not directly reveal the target class.

### 9. Submission Format
Participants must submit a file named:

```
submission.csv
```

The file must be saved to:

```
./working/submission.csv
```

The submission must contain exactly these columns, in this exact order:

```
id,target,stress_flag
```

- No extra columns are allowed.
- No missing columns are allowed.
- The id column must exactly match the test ids in the same row order as `public/sample_submission.csv`.
- The target column must contain one valid route label or one valid integer class id for each test sample.
- The stress_flag column is required by the submission contract and is explicitly allowed. It is a pass-through column, not a target to predict. Participants should copy the stress_flag column from `public/sample_submission.csv` unchanged.

Valid submission example:

```
id,target,stress_flag
17b9d96ec4f42a90,route-aphelion,0
8f14e45fceea167a,route-equinox,1
```

The grader scores only the submitted target predictions. For the stress-subset metric, the grader uses the authoritative stress_flag values from `private/answers.csv`.

### 10. stress_flag Meaning
The stress_flag marks whether a test image belongs to the stronger acquisition-stress subset.

- stress_flag = 0 → normal or moderately degraded acquisition condition
- stress_flag = 1 → heavily degraded acquisition condition

Images with stress_flag = 1 may contain stronger crop loss, blur, noise, lighting shift, compression artifacts, channel disturbance, vignetting, scanline effects, or occlusion-like corruption.

The stress_flag column must be present in the submission, but participants are not required to predict it.

### 11. Evaluation
Submissions are evaluated using the Spectral Route Robustness Score.

The score combines four components:

- Six-class Macro F1-score
- Six-class Balanced Accuracy
- Anchor-vs-Rest Gate F1-score
- Stress-Subset Macro F1-score

Only the target column is scored.

The stress_flag column is used only to identify the stressed subset for the fourth metric component.

### 12. Metric Definitions
Let:

- y_i = true class for sample i
- p_i = predicted class for sample i
- C = set of six spectral route classes

For each class c:

- TP_c = number of samples where y_i = c and p_i = c
- FP_c = number of samples where y_i ≠ c and p_i = c
- FN_c = number of samples where y_i = c and p_i ≠ c

#### 12.1 Macro F1-score
For each class c:

```
F1_c = 2·TP_c / (2·TP_c + FP_c + FN_c)
```

The six-class Macro F1-score is:

```
Macro_F1 = (1 / |C|) × Σ F1_c for all c in C
```

Macro F1 rewards balanced performance across all six spectral route classes.

#### 12.2 Balanced Accuracy
For each class c:

```
Recall_c = TP_c / (TP_c + FN_c)
```

Balanced Accuracy is:

```
Balanced_Accuracy = (1 / |C|) × Σ Recall_c for all c in C
```

Balanced Accuracy gives each class equal importance, reducing the effect of class imbalance.

#### 12.3 Anchor-vs-Rest Gate F1-score
The anchor class is:

```
route-aphelion
```

For this component, the six-class prediction is converted into a binary decision:

- positive class = route-aphelion
- negative class = all other route labels

The Anchor-vs-Rest Gate F1-score is:

```
Gate_F1 = 2·TP_anchor / (2·TP_anchor + FP_anchor + FN_anchor)
```

This component rewards models that correctly identify the anchor route without confusing it with the other five classes.

#### 12.4 Stress-Subset Macro F1-score
The stress subset contains samples where the authoritative private answer file has:

```
stress_flag = 1
```

The Stress-Subset Macro F1-score is calculated only on these stressed samples:

```
Stress_Macro_F1 = Macro F1 calculated on samples with stress_flag = 1
```

If the stress subset contains fewer than two samples or fewer than two unique true classes, the grader falls back to the overall Macro F1-score:

```
Stress_Macro_F1 = Macro_F1
```

This component rewards models that remain reliable under heavier acquisition degradation.

### 13. Final Score Formula
The final score is a weighted sum:

```
Final_Score =
    0.40 × Macro_F1
  + 0.35 × Balanced_Accuracy
  + 0.10 × Gate_F1
  + 0.15 × Stress_Macro_F1
```

The weights sum to 1.00.

The score ranges from 0 to 1.

Higher is better.

The weighting emphasizes balanced six-class classification while also rewarding anchor-route detection and robustness under stressed acquisition conditions.

### 14. Strict Grading Rules
A submission is rejected if:

- required columns are missing
- extra columns are present
- columns are in the wrong order
- the row count is incorrect
- ids are missing or empty
- ids are duplicated
- ids do not match the answer ids
- ids are correct but in the wrong row order
- target values are invalid
- the metric produces a non-finite score

The only accepted submission columns are:

```
id,target,stress_flag
```

The stress_flag column is explicitly required and allowed. It is not treated as a forbidden helper column.

Forbidden extra columns include:

- target_id
- visibility
- answer
- answers
- label
- class
- class_id
- y
- fold
- split
- is_private
- private
- public
- prediction

Forbidden extra column prefixes include:

- `unnamed:`
- `hint_`
- `evidence_`
- `target_`
- `true_`
- `gt_`
- `score_route_`

Because the grader enforces the exact three-column schema, any extra probability, helper, score, fold, split, prediction, label, or class column will be rejected.

### 15. Allowed Methods
Participants may use:

- pretrained image models
- custom convolutional models
- custom transformer models
- public auxiliary metadata features
- image-plus-metadata fusion models
- stratified validation
- class balancing
- robust image augmentation
- test-time augmentation
- ensembles trained only on public training data

Participants may use the public training data, public test images, and public test metadata for inference.

### 16. Prohibited Methods
Participants must not:

- use `private/answers.csv` for training
- use private labels
- treat the test set as labeled training data
- hardcode answer patterns
- infer labels from hidden or private metadata
- reorder submission ids
- add probability, score, helper, fold, split, prediction, label, or class columns
- remove the stress_flag column
- output invalid target labels
- submit ids in a different order from `sample_submission.csv`

### 17. Data Secrecy and Anonymization
The public dataset is designed to avoid exposing original source-category names.

Public files do not include:

- original source folder names
- original source labels
- original file paths
- private answer columns
- hidden split indicators
- helper score columns
- route hint columns

The public target names are anonymized route labels. Prepared image ids are anonymized hashes. The public auxiliary features describe acquisition condition and image quality only.

Participants should solve the task using the released training images, released training labels, released test images, and released public metadata.

### 18. Baseline Expectations
A majority-class submission should produce a low but non-zero score.

A basic pretrained CNN with correct preprocessing should beat the majority-class baseline.

A stronger solution should combine robust image augmentation, class-balanced training, stratified validation, and optional metadata fusion.

Top-performing solutions should perform well across all six route classes, avoid overfitting to acquisition artifacts, identify the anchor route reliably, and remain stable on heavily degraded samples where:

```
stress_flag = 1
```

## Expected Output
Output to `./working/submission.csv`
