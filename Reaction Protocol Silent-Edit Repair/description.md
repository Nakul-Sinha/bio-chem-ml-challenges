Problem Description
Reaction Protocol Silent-Edit Repair
Overview
Chemistry lab notebooks often contain terse protocol notes that are later corrected by an audit trail. A useful model must not only read the recorded procedure, but also apply the silent-edit notice and regenerate the corrected protocol in the lab's canonical sequence format.

Your task is to generate a repaired six-slot reaction protocol sequence from a lab note and a correction notice. Each input gives a reaction-family header, a noisy protocol note, and one silent-edit correction written as local bench shorthand. Some copied notes are mixed-order, use local bench tags, or omit a background operation, so a useful model must infer the complete repaired procedure from the family, visible note evidence, and correction notice. The output is a semicolon-separated canonical sequence covering setup preparation, activation, addition order, control condition, quench, and workup.

This is a Chemistry x sequence-to-sequence fine-tuning task. The practical goal is protocol-log normalization: when a corrected audit note changes part of an experiment record, downstream robotic execution, safety review, and reproducibility checks need the complete repaired protocol sequence rather than a single local tag.

Dataset
File descriptions
train.jsonl -- 1,676 instruction/input/output examples for sequence-model training or fine-tuning.
test.jsonl -- 524 held-out instruction/input examples without repaired outputs.
train.csv -- CSV mirror of train.jsonl with prompt text and the repaired_sequence output.
test.csv -- CSV mirror of test.jsonl with prompt text but without repaired_sequence.
sample_submission.csv -- A template showing the required submission format with random valid-looking repaired sequences.
Column descriptions
id (string) -- Unique 12-character hex identifier for each protocol repair case.
prompt (string) -- Full input text containing the reaction-family header, protocol note, correction notice, and generation request.
protocol_note (string) -- The recorded lab-note text before applying the silent-edit correction.
correction_notice (string) -- The audit correction that overrides one recorded protocol slot through local bench shorthand.
instruction (string) -- JSONL instruction telling a model to generate only the repaired canonical protocol sequence.
input (string) -- JSONL input text equivalent to prompt.
output (string) -- Target field in train.jsonl only. Same content as repaired_sequence.
repaired_sequence (string) -- Target field in train.csv and submission column. The value must contain six ordered slot assignments separated by semicolons: prep, activation, order, control, quench, and workup.
Evaluation
Submissions are scored using Operation-Weighted Repair Sequence Score. Higher is better.

Each submitted repaired_sequence is parsed into six ordered slot assignments. The grader gives weighted credit for each slot that exactly matches the private repaired sequence, then averages scores within hidden reaction/edit groups before taking the final mean.

Setup preparation, control condition, and quench receive the largest weights because those silent edits most often change safety-critical handling. Activation, addition order, and workup still contribute to the score, but they carry less weight than the corrected operating-condition slots.

Numeric slot weights are prep=2.20, activation=0.85, order=0.60, control=3.00, quench=4.00, and workup=0.25. For one row, row_score = (2.20 prep_correct + 0.85 activation_correct + 0.60 order_correct + 3.00 control_correct + 4.00 quench_correct + 0.25 workup_correct) / 10.90, where each *_correct term is 1 if the predicted slot value exactly matches the private answer and 0 otherwise. The final score is the mean of row_score within each hidden reaction/edit group, averaged equally across groups.

The score ranges from 0 to 1. Random valid sequences should score near the low baseline. Stronger solutions should improve by training on the public examples, learning the local shorthand, and generating the full repaired sequence rather than only the edited slot.

Submission
Submit a CSV file with one generated repaired sequence for every row in test.jsonl or test.csv.

id (string) -- The 12-character hex identifier from the test file.
repaired_sequence (string) -- The generated canonical protocol sequence, exactly six semicolon-separated slot assignments in this order: prep, activation, order, control, quench, workup.
Example:

A valid submission file looks like this:

id,repaired_sequence

008d7291f5ca,prep=degas;activation=preactivate_oxidant;order=oxidant_portionwise;control=warm_hold;quench=no_quench;workup=organic_extract

00ac6657ce3d,prep=degas;activation=preactivate_oxidant;order=substrate_slow;control=vented_hold;quench=water_slow;workup=direct_concentrate

01870a3efc75,prep=dry_glass;activation=preactivate_oxidant;order=oxidant_portionwise;control=vented_hold;quench=no_quench;workup=solvent_swap

Requirements
The file must contain exactly 524 rows plus the header.
Every id from the test file must be present exactly once.
repaired_sequence must contain exactly six assignments separated by semicolons.
Slot names must appear exactly in this order: prep, activation, order, control, quench, workup.
Each assignment must use slot=value format with one of the valid values learned from public training examples.
Missing predictions, duplicate ids, malformed slot order, invalid values, extra columns, and wrong row counts are invalid.
File format: .csv only, with exact column names id,repaired_sequence.
What Not To Use
Do not build the main solution as a handwritten parser that only recognizes the visible note templates or bench-cue phrases. Diagnostic and post-processing rules are fine, but the submitted method should ONLY fine-tune a model on the public examples.
Do not submit a hardcoded map from test ids, full prompt strings, or memorized repaired_sequence values. The task is to learn the repair mapping from public examples.
Do not use externally shared answer tables, cached outputs, or leaked protocol-repair case maps to fill the held-out repaired sequences.
Do not ignore the correction_notice and predict only a common reaction-family sequence. The evaluated rows require applying the silent edit and regenerating all six slots.
 