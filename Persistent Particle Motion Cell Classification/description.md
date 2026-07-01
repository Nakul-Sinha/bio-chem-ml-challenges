# Persistent Particle Motion Cell Classification

## Problem Description

### Persistent Particle Motion Cell Classification in Bedload Flow

You are given short two-frame crop pairs from laboratory particle-flow imagery. Each image sample contains two side-by-side panels extracted from a high-speed image sequence of visually similar beads moving through a dense flow scene.

In the left panel, the particle to track is marked with a small red cross and circle at the crop center. The right panel shows the same local image region after a short time gap. Your task is to classify which motion cell contains the same particle in the right panel.

This is a fine-grained visual tracking and image classification problem. The particles are visually similar, the scene contains many nearby particles, and the target may move several pixels between frames. A good solution must use visual evidence and temporal continuity rather than only a global average flow prior.

### Public files

- **train.csv** — Labeled training rows with columns:
  - `sample_id`: unique sample identifier
  - `image_path`: path to the side-by-side crop-pair image
  - `horizon`: number of source frames between the left and right panel
  - `motion_class`: integer class label from 0 to 19
- **test.csv** — Test rows with columns:
  - `sample_id`
  - `image_path`
  - `horizon`
- **sample_submission.csv** — Required submission format with columns:
  - `sample_id`
  - `motion_class`
- **motion_class_map.csv** and **motion_class_map.json** — Class definitions for all 20 motion classes, including each class's `x_band`, `y_band`, `x_range`, and `y_range`.
- **images/train/*.jpg** — Labeled training crop-pair images referenced by `train.csv`.
- **images/test/*.jpg** — Unlabeled test crop-pair images referenced by `test.csv`.

### Image interpretation

Each crop-pair image has two panels. The left panel is frame t. The right panel is frame t + horizon. The marked target starts at the exact center of the left panel. The right panel is cropped around the same original image location, not around the target's future location.

### Axis convention

- `dx` is measured in image x-coordinates. Negative dx means the particle moved left in the right panel; positive dx means the particle moved right.
- `dy` is measured in image y-coordinates. Negative dy means upward motion; positive dy means downward motion.

### Prediction target

For every test sample, predict `motion_class`, an integer from 0 to 19. The class is based on the target particle's displacement from the left-panel center to the same particle's position in the right panel.

The classes are defined by a 5-by-4 grid of motion cells: five horizontal x-bands and four vertical y-bands. The x-axis follows normal image-coordinate convention: positive x is to the right in the image, and negative x is to the left. The final x-band explicitly includes every displacement with dx >= -6, including small leftward, zero, and positive rightward displacements.

- `x_band` is determined from the horizontal displacement dx:
  - `x_band = 0` when dx < -30
  - `x_band = 1` when -30 <= dx < -22
  - `x_band = 2` when -22 <= dx < -14
  - `x_band = 3` when -14 <= dx < -6
  - `x_band = 4` when dx >= -6, including zero and positive/rightward displacements
- `y_band` is determined from the vertical displacement dy:
  - `y_band = 0` when dy < -2
  - `y_band = 1` when -2 <= dy < 0
  - `y_band = 2` when 0 <= dy < 2
  - `y_band = 3` when dy >= 2

The final class formula is:

```
motion_class = 5 * y_band + x_band
```

Therefore valid motion_class values are the integers 0 through 19 inclusive.

### Submission format

Submit a CSV with exactly these columns, in this order:

```
sample_id,motion_class
```

The submission must contain exactly one row for every sample_id in test.csv. Extra columns, missing columns, reordered columns, duplicate sample IDs, missing sample IDs, missing values, NaN, null, infinity, non-numeric predictions, non-integer predictions, or class values outside 0 through 19 are invalid.

### Scoring

Submissions are scored by exact classification accuracy.

For each test sample i:

- `predicted_i` is the submitted motion_class
- `true_i` is the hidden correct motion_class
- `sample_score_i = 1` if predicted_i equals true_i, otherwise 0

The final score is the mean of sample_score_i across all N test samples:

```
score = (1 / N) * sum(sample_score_i for i = 1..N)
```

The score is bounded between 0 and 1. A score of 1 means every test sample was assigned the correct motion_class. A score of 0 means no test sample was assigned the correct motion_class, or the submission was invalid.

### What not to use

- Do not use private answer files, hidden trajectory files, or any data outside the prepared public files for this challenge.
- Do not reconstruct labels from sample_id hashes, file ordering, filesystem timestamps, image hashes, or any other identifier/ordering shortcut.
- Do not hardcode predictions for specific public test files or sample IDs.
- Do not use private leaderboard feedback to tune per-sample outputs or manually override individual test predictions.
- Do not submit NaN, null, infinite, blank, non-numeric, non-integer, duplicate, missing, reordered-column, or extra-column outputs. The grader treats invalid submissions as score 0.
- Allowed approaches include computer vision models, optical flow, template matching, local feature tracking, supervised image classification, and ensembles trained only from the provided public training data and images.

## Expected Output

Output to `./working/submission.csv`
