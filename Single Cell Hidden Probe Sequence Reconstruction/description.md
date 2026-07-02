# Single Cell Hidden Probe Sequence Reconstruction

## Problem Description

### Single Cell Hidden Probe Sequence Reconstruction

### Overview

This challenge is a sequence-to-sequence learning problem on single-cell expression data.

Each biological observation is represented as an anonymized source sequence. The source sequence encodes the observed part of a small expression panel, biological and acquisition context, assay and sample metadata, panel damage indicators, and quantized observed expression tokens.

For each training row, the target is a hidden probe-signature sequence. The target is an ordered token sequence describing which hidden probes are active and how strongly they are expressed.

For each test row, your task is to generate the hidden probe-signature sequence.

The private evaluation set contains several robustness stress cases:

- Rows from a shifted acquisition or biological context domain.
- Held-out sample groups that reduce near-duplicate leakage.
- Rows where part of the observed panel has been deterministically zeroed.
- Rare hidden signature patterns that are uncommon in training.

A strong solution should learn to translate the observed expression-context sequence into the hidden probe-signature sequence. Random row validation will likely overestimate private performance.

### Files

The prepared public data contains:

- `train.csv`: training examples with source and target sequences.
- `test.csv`: test examples with source sequences only.
- `sample_submission.csv`: required submission format.
- `sequence_vocabulary.csv`: list of valid source and target tokens.
- `observed_panel.csv`: anonymized metadata for observed expression tokens.
- `target_panel.csv`: anonymized metadata for hidden target tokens.
- `README.md`: short summary of the prepared files.

The private data contains:

- `answers.csv`: private ground truth target sequences and hidden evaluation subset flags.

### Task

For every row in `test.csv`, generate one `predicted_sequence`.

The model input is the `source_sequence` column.

The desired model output is the hidden `target_sequence`.

This is a sequence reconstruction problem. The output length varies by row. Some rows have no active hidden probes, while other rows have many active hidden probes.

### Source Sequence Format

Each `source_sequence` is a whitespace-separated sequence of tokens.

Example:

```
DOM_004 ASSAY_001 CTX_018 GENCTX_003 COND_002 SEX_001 STAGE_000 SAMPLE_001 PANEL_NORMAL TOTAL_Q18 NZ_Q42 O000_Q00 O001_Q03 O002_Q00 O003_Q11 O004_Q01
```

Token groups include:

- `DOM_*`: anonymized coarse acquisition or context domain.
- `ASSAY_*`: anonymized assay or measurement technology.
- `CTX_*`: anonymized fine-grained biological context.
- `GENCTX_*`: anonymized higher-level biological context.
- `COND_*`: anonymized condition or status category.
- `SEX_*`: anonymized sex metadata where available.
- `STAGE_*`: anonymized development or stage metadata where available.
- `SAMPLE_*`: anonymized sample or suspension type category.
- `PANEL_NORMAL` or `PANEL_DAMAGE_*`: observed panel condition.
- `TOTAL_Q*`: quantized total observed panel signal.
- `NZ_Q*`: quantized number of nonzero observed probes.
- `O000_Q*` through `O079_Q*`: quantized observed expression probe tokens.

The order of tokens in the source sequence is fixed and meaningful.

### Target Sequence Format

Each target sequence is a whitespace-separated sequence of hidden probe tokens.

Valid hidden probe tokens have this form:

```
T##_B#
```

where:

- `T00` through `T15` identify the hidden target probe.
- `B1`, `B2`, and `B3` indicate low, medium, and high positive hidden signal.
- Hidden probes with no or near-zero signal are omitted from the sequence.

The canonical target sequence is ordered by descending hidden signal bin, then by target index.

Example:

```
T06_B3 T13_B3 T03_B2 T09_B2 T01_B1 T07_B1 T10_B1 T14_B1
```

If no hidden probe is active, the target sequence is:

```
NONE
```

Important details:

- `B0` is never written in the target sequence.
- Zero or near-zero hidden probes are represented by omission.
- The token order matters.
- Duplicate target tokens are not valid and will hurt the score.
- Unknown or malformed tokens are treated as sequence errors.

### Columns In `train.csv`

`train.csv` contains:

- `id`: stable row identifier.
- `source_sequence`: input sequence for the model.
- `target_sequence`: ground-truth output sequence for the model.

### Columns In `test.csv`

`test.csv` contains:

- `id`: stable row identifier.
- `source_sequence`: input sequence for the model.

It does not contain `target_sequence`.

### Sequence Vocabulary File

`sequence_vocabulary.csv` contains valid tokens used by the prepared dataset.

Columns:

- `token`: token string.
- `token_type`: one of `metadata`, `observed_expression`, `summary`, `target_probe`, or `special`.
- `description`: short anonymized description.
- `train_count`: number of occurrences in training source or target sequences where applicable.

### Observed Panel File

`observed_panel.csv` contains metadata for observed expression tokens.

Columns:

- `observed_index`: observed probe position.
- `source_token_prefix`: matching source token prefix such as `O000`.
- `feature_code`: anonymized observed feature identifier.
- `damage_group`: deterministic missing-probe group affecting this observed feature, or `none`.

### Target Panel File

`target_panel.csv` contains metadata for hidden target tokens.

Columns:

- `target_index`: hidden target probe position.
- `target_token_prefix`: target token prefix such as `T00`.
- `target_code`: anonymized hidden target identifier.
- `train_active_count`: number of training rows where this target appears in the target sequence.
- `train_b1_count`: number of training rows where this target appears as `B1`.
- `train_b2_count`: number of training rows where this target appears as `B2`.
- `train_b3_count`: number of training rows where this target appears as `B3`.

### Submission Format

Submit a CSV file with exactly one row for every `id` in `test.csv`.

The required columns are:

- `id`
- `predicted_sequence`

Example:

```
id,predicted_sequence
cell_0a12f4b8e91d,T06_B3 T13_B3 T03_B2 T09_B2 T01_B1 T07_B1
cell_3be7d0c91a44,NONE
cell_f829a6610c02,T05_B3 T14_B3 T02_B2 T03_B1 T11_B1
```

The `predicted_sequence` column must contain a whitespace-separated sequence of valid target tokens, or `NONE`.

Missing IDs, duplicate IDs, extra IDs, missing predictions, or extra submission columns are invalid submissions.

### Evaluation

Submissions are scored using a weighted sequence reconstruction score. Higher is better. The minimum possible score is 0 and the maximum possible score is 1.

The final leaderboard score is:

```
score =
    0.45 * sequence_score_all_private_rows
  + 0.25 * sequence_score_shifted_domain_rows
  + 0.20 * sequence_score_damaged_panel_rows
  + 0.10 * sequence_score_rare_signature_rows
```

For each evaluated row, three sequence-level similarities are computed:

```
row_score =
    0.50 * normalized_edit_similarity
  + 0.30 * exact_token_f1
  + 0.20 * order_lcs_similarity
```

The components are:

- `normalized_edit_similarity` rewards generating the correct sequence with few insertions, deletions, or substitutions.
- `exact_token_f1` rewards recovering the correct hidden probe and bin tokens.
- `order_lcs_similarity` rewards placing the recovered target tokens in the correct order.

The subset scores are the mean row score over the corresponding private rows.

Malformed target tokens are treated as wrong sequence tokens. Duplicate target IDs, unknown target IDs, invalid bin values, and non-canonical tokens reduce the score.

### Modeling Notes

Good approaches may include:

- Treating `source_sequence` and `target_sequence` as text-like token sequences.
- Encoder-decoder transformers or other sequence models.
- Retrieval-augmented nearest-neighbor sequence generation.
- Sequence-aware validation split by held-out domain, sample group, or panel condition.
- Data augmentation by masking observed probe tokens.
- Hybrid approaches that infer latent hidden probe activity and decode into a canonical sequence.

Do not rely only on a random row split. Related rows from the same context, sample group, or acquisition domain can be very similar, and random validation can overestimate private performance.

### Restrictions

Use only the prepared public files provided in this challenge.

Do not use external datasets, internet lookup, source reconstruction, row-ID memorization, private labels, hidden answer files, or leaderboard probing.

### Expected Output

Output to `./working/submission.csv`
