Problem Description
Overview
Reaction conditions are a major part of chemical synthesis planning. Two reactions with similar reactants and products can require different solvents, temperatures, reaction times, or catalysts. This challenge asks you to infer structured reaction-condition targets from a single reaction SMILES string.

The task is not product prediction or retrosynthesis. The products are already present in the input. The goal is to complete missing experimental condition fields in a controlled output format. The hidden test set emphasizes distribution shift, rare condition regimes, catalyst-positive reactions, long reactions, and uncommon solvent settings, so a majority-condition baseline should not rank well.

Task
For each test reaction, predict:

pred_solvents - a pipe-separated set of solvent labels from solvent_vocabulary.csv.
pred_temp_bin - one temperature bin.
pred_time_bin - one time bin.
pred_catalyst_present - 1 if a catalyst is predicted to be present, otherwise 0.
The public input reaction string has this format:

reactants>>products

No reaction condition text is included in public test inputs.

Output Labels
Temperature bins:

cryogenic
cold
room
warm
hot
Time bins:

very_short
short
medium
long
very_long
overnight
Solvent labels:

Use labels from dataset/public/solvent_vocabulary.csv.
Use NONE when no solvent is predicted.
Use OTHER when the solvent appears outside the public controlled vocabulary.
Multiple solvent labels must be separated with |, for example CN(C)C=O|C1CCOC1.
Evaluation
Submissions are scored with a composite metric in [0, 1], where higher is better:

Score =
    0.18 * SolventSetF1
  + 0.10 * TemperatureBalancedAccuracy
  + 0.10 * TimeBalancedAccuracy
  + 0.08 * CatalystMacroF1
  + 0.20 * RareConditionTrackScore
  + 0.18 * ShiftedReactionTrackScore
  + 0.08 * CatalystPositiveTrackScore
  + 0.08 * ExactTupleAccuracy

SolventSetF1
For each reaction, the grader compares the predicted solvent set to the true solvent set using set F1:

precision = |predicted_solvents intersect true_solvents| / |predicted_solvents|
recall = |predicted_solvents intersect true_solvents| / |true_solvents|
SolventSetF1_row = 2 * precision * recall / (precision + recall)

Rows with an exact solvent-set match receive 1.0. Rows with no overlap receive 0.0. The final component is the mean over all private test reactions.

TemperatureBalancedAccuracy
Balanced accuracy over the hidden temperature bins. The grader computes recall separately for each true temperature bin and averages those recalls. This prevents the common room class from dominating the metric.

TimeBalancedAccuracy
Balanced accuracy over the hidden time bins. The grader computes recall separately for each true time bin and averages those recalls.

CatalystMacroF1
Macro F1 for catalyst presence. Catalyst-positive rows are a minority, so plain accuracy would reward predicting 0 for everything.

Row-Level Condition Score
Several hidden-track metrics use a row-level condition score:

RowScore =
    0.40 * SolventSetF1_row
  + 0.20 * I(pred_temp_bin == true_temp_bin)
  + 0.20 * I(pred_time_bin == true_time_bin)
  + 0.20 * I(pred_catalyst_present == true_catalyst_present)

I(...) is 1 when the condition is true and 0 otherwise.

RareConditionTrackScore
Mean RowScore on hidden rare-condition examples. This track emphasizes rare solvent settings, extreme temperature/time bins, complex reactions, and long reaction strings.

ShiftedReactionTrackScore
Mean RowScore on a hidden shifted-reaction subset. This rewards generalization to reactions drawn from a different source slice rather than memorization of common training-condition frequencies.

CatalystPositiveTrackScore
Mean RowScore on hidden catalyst-positive reactions. This track makes it hard to score well by predicting no catalyst or by treating catalyst use as an afterthought.

ExactTupleAccuracy
Exact match of all four outputs: solvent set, temperature bin, time bin, and catalyst-present label.

Dataset
Public files:

dataset/public/train.csv - training reactions with condition labels.
dataset/public/test.csv - test reactions without condition labels.
dataset/public/solvent_vocabulary.csv - controlled solvent vocabulary.
dataset/public/sample_submission.csv - valid submission template.
train.csv Columns
Column	Data type	Description
reaction_id	string	Anonymous row identifier for the training reaction.
reaction_smiles	string	Reaction SMILES in reactants>>products format.
reactant_count	integer	Number of dot-separated reactant components before >>.
product_count	integer	Number of dot-separated product components after >>.
smiles_length	integer	Character length of reaction_smiles.
solvent_labels	string	Pipe-separated true solvent label set using the public controlled vocabulary.
temp_bin	string	True temperature bin.
time_bin	string	True time bin.
catalyst_present	integer	1 if a catalyst is present, otherwise 0.
test.csv Columns
Column	Data type	Description
reaction_id	string	Anonymous row identifier for the test reaction.
reaction_smiles	string	Reaction SMILES in reactants>>products format.
reactant_count	integer	Number of dot-separated reactant components before >>.
product_count	integer	Number of dot-separated product components after >>.
smiles_length	integer	Character length of reaction_smiles.
solvent_vocabulary.csv Columns
Column	Data type	Description
solvent_label	string	Allowed solvent label for pred_solvents. Includes reserved labels NONE and OTHER.
source	string	Indicates whether the label is reserved or selected from common training solvents.
Private files used by the grader:

dataset/private/answers.csv - hidden condition labels and hidden track labels.
Submission Format
Submit a CSV at ./working/submission.csv with exactly these columns:

reaction_id,pred_solvents,pred_temp_bin,pred_time_bin,pred_catalyst_present
RXNTEST_abcd1234ef56,CN(C)C=O,room,short,0
RXNTEST_1234abcdef56,NONE,hot,long,1

 