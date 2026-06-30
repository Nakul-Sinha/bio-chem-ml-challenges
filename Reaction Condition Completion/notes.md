# Reaction Condition Completion — notes

## Task
Predict 4 condition targets from reaction SMILES (`reactants>>products`):
`pred_solvents` (multi-label set from 82-label vocab incl NONE/OTHER), `pred_temp_bin` (5),
`pred_time_bin` (6), `pred_catalyst_present` (binary).

## Metric (composite, higher better)
0.18 SolventSetF1 + 0.10 TempBalAcc + 0.10 TimeBalAcc + 0.08 CatMacroF1
+ 0.20 RareTrack + 0.18 ShiftTrack + 0.08 CatPosTrack + 0.08 ExactTuple.
RowScore (used by the 3 tracks) = 0.40 SolvF1 + 0.20 temp + 0.20 time + 0.20 cat.
=> effective weights ~ solvent 0.36, catalyst 0.25, temp 0.19, time 0.19. Tracks emphasize
rare/shifted/catalyst-positive rows; balanced-acc + macro-F1 penalize majority-only.

## EDA
- 18k train / 4k test. temp: room 58%/hot 17/warm 15/cold 8/cryo 2. time: short 32/long 23/
  med 20/overnight 14/very_long 6/very_short 5. catalyst 16% positive.
- Solvent sets: NONE 11.4%, singleton 76.7%, multi 11.8%. 80 train_top solvents + OTHER used.
- **Catalysts & solvents are NOT in the SMILES** (metals in only ~38/18000 rows; solvent appears
  as a reactant component 0.8%) => all 4 targets are genuine structure->conditions inference.
- Distribution shift: test reactions are longer/more complex (smiles_len 143 vs 129).

## Approach
- Features: Morgan FP (r=2, 2048) of reactants + products + difference + 49 descriptors
  (atom/charge counts, reagent-pattern presence). Cached.
- Multi-task MLP (1024->512 backbone, dropout) with heads: softmax PRIMARY solvent (NONE+81),
  sigmoid multilabel solvent (secondaries), temp(5), time(6), catalyst(1). 5-fold OOF.
- Decode tuned on OOF to the composite: prior-adjust alpha (temp/time balanced-acc), catalyst
  threshold (macro-F1), secondary-solvent threshold, NONE bias.

## Experiments (5-fold OOF proxy composite)
- v1 sigmoid-only multilabel solvent:           composite 0.380 (SolvF1 0.204, top-1 0.206)
- v2 + softmax primary-solvent head:            composite 0.430 (SolvF1 0.328, top-1 0.325)
  - key: softmax ranks top-1 far better than independent sigmoids for the 77% singleton case.
- decode + NONE-bias tuning: composite 0.4302 (NONE recall 0.34->0.45; negligible composite gain).
- FINAL: 3-seed ensemble of v2 on full data + tuned decode (a_t .7, a_tm 0, cat_thr .45,
  sec_thr .3, none_boost 1.5). submission.csv (4000 rows) validated; all format checks PASS.
  Validated CV estimate ~0.43 (single-model OOF; ensemble expected >=).

## Deliverables
- solution.py (self-contained, rdkit+torch), submission.csv (4000 rows, validated),
  approach.md, approach_short.md, notes.md, research/.

## Component breakdown (v2)
SolvF1 0.328, TempBA 0.466, TimeBA 0.248 (hard), CatMF1 0.814 (strong),
Rare 0.485, Shift 0.545, CatPos 0.426, Exact 0.063.
