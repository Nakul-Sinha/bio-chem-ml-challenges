# Approach — Reaction Condition Completion

**Time spent:** ~2.5 hours

## Summary
A single **multi-task MLP** on RDKit reaction fingerprints predicts all four condition targets
(solvent set, temperature bin, time bin, catalyst presence) from the reaction SMILES. The final
submission averages a 3-seed ensemble and applies a decode tuned directly to the composite metric.

## Features (ensemble of two views)
Two featurizations, each fed to its own MLP and **blended 50/50**:
1. **Morgan fingerprints** (radius 2, 2048 bits) of reactants/products/difference — structure-aware,
   source-invariant (helps the hidden distribution-shift track). The grading env lacks rdkit but
   has internet, so `solution.py` `pip install`s rdkit at runtime (with an n-gram-only fallback).
2. **Hashed character n-grams** (n=2..5, crc32, 2048 buckets) of the SMILES — rdkit-free.
Both add 49 string descriptors (atom/charge/bond counts + presence flags for common reagents:
metals, carbonate/hydroxide bases, amine bases, phosphines, Boc, azide, tosyl, …).
**5-fold OOF composite: Morgan 0.430, n-gram 0.422, 50/50 ensemble 0.447.** (A fine-tuned
ChemBERTa-77M was tried and *underperformed* at 0.34 — this task is driven by reagent presence,
which fingerprints capture directly, not by deep structure.) EDA showed
catalysts and solvents are essentially absent from the SMILES (metals in ~38/18000 rows; a true
solvent appears as a reactant component in 0.8%), so every target is genuine structure→conditions
inference; the reagents that *are* present (bases, ligands) are the key signal, captured by the
reactant fingerprint + reagent flags.

## Model
Shared 1024→512 backbone (BatchNorm + dropout) with five heads:
- **softmax PRIMARY-solvent** head (NONE + 81 labels) — the decisive design choice. Solvent sets
  are 77% singleton / 11% NONE, so framing the dominant case as single-label softmax ranks the
  top solvent far better than 81 independent sigmoids (top-1 0.206→0.325, SolventSetF1 0.204→0.328).
- **sigmoid multi-label** head for adding secondary solvents.
- temp (5), time (6), catalyst (1) heads.
Trained jointly with CE + BCE, OneCycle LR, AdamW.

## Decode (tuned on OOF to the composite)
- temp/time: argmax of posterior divided by prior^α — α tuned so the *balanced* accuracy
  (mean per-class recall) is maximized without wrecking plain accuracy in the RowScore tracks.
- catalyst: probability threshold tuned for macro-F1.
- solvent: softmax argmax for the primary (with a NONE bias), plus sigmoid-thresholded secondaries.

## Local validation (5-fold OOF, proxy composite)
Rare/shifted tracks approximated from observable rarity/complexity (extreme temp/time bins, multi
or rare solvents, long reactions) since the hidden track labels aren't available.
- **Composite ≈ 0.430.** Components: SolventSetF1 0.328, TempBalAcc 0.466, TimeBalAcc 0.248,
  CatMacroF1 0.814, Rare 0.486, Shift 0.545, CatPos 0.425, Exact 0.064.
- Progression: sigmoid-only baseline 0.380 → +softmax primary head 0.430.

## Why this method
The composite rewards solvent and catalyst most and penalizes majority-only predictions
(balanced acc, macro-F1, rare/shifted tracks). The softmax primary head fixes the dominant
solvent case; prior-adjusted decoding handles the imbalance for temp/time; threshold/macro-F1
tuning handles catalyst. Pretrained chem models are allowed but a fingerprint MLP already
captures the reagent signal cheaply and robustly across the length shift.

## Compliance / reproducibility
Provided data only; fixed seeds; self-contained `solution.py` (reads `./dataset[/public]/`,
writes `./working/submission.csv` + `./submission.csv`); standard libs (rdkit, torch, sklearn).
