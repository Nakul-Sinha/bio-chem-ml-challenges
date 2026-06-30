# Microbial DNA Compositional Motif Restoration — notes

## Task
Contig = top-16 most-frequent hexamer tokens (opaque ints 1..1024), rank-ordered. One mid-rank
token masked; predict ranked top-10. Metric: MRR@10. genome_context = top-16 pooled across the
genome's sibling contigs. Train/test genomes DISJOINT.

## EDA
- 7585 train / 1877 test contigs. Vocab 1024. All seqs length 16. Mask positions 4..15.
- TRAIN: 113 genomes, mean 67 contigs, **median 4** (many tiny genomes).
- TEST: 31 genomes, mean 60, **median 12**, with several huge (321, 298, 286, 202, ...).
  **77% of test contigs are in genomes of size >=151.** => test has much richer transductive
  (sibling) signal than the train-CV average; raw train-CV underestimates the test score.
- Gold token in genome_context only ~31%, but **in the sibling-token-union 94%** => recall is not
  the problem; RANKING is.
- k-mers within a contig are distinct => masked token is never among the 15 visible (excluded).

## Validation
- **Group-CV by genome** (disjoint, mirrors test). **Test-faithful one-mask-per-contig**, all
  transductive stats from VISIBLE tokens only (leak-free). Caught and removed two leaks:
  (a) genome co-occurrence inflated by the contig's own gold (+15), (b) global co-occurrence
  self-inclusion — fixed with leave-one-GENOME-out globals.
- Report both raw contig-weighted CV and a **test-calibrated** estimate (reweight per-genome-size
  MRR by the test genome-size distribution), since test skews to large genomes.

## Journey
- v1 (co-occurrence + genome-cooc + PMI blend): honest CV ~0.167 — BELOW the AI baseline. Rejected.
- Diagnosis: 94% sibling-union recall, but frequency/co-occurrence ranking caps ~0.17.
- **Breakthrough: transductive contig-similarity collaborative filtering (CF).** The gold appears
  in sibling contigs most SIMILAR to the target. Score a candidate by IDF-weighted
  similarity^power voting over siblings; discount genome-ubiquitous candidates (they're genuinely
  absent here, not the gold). This jumped CV to ~0.21 and test-calibrated ~0.22.
- Explored and REJECTED (none beat CF honestly): from-scratch masked-LM transformer (0.12 —
  overfits the ~90 train genomes, can't access the test genome's co-occurrence); LightGBM
  learning-to-rank reranker over CF+cooc+rank features (0.205, overfits cross-genome); linear /
  z-scored rank-fusion blends with gcond/cooc (all < CF). CF (non-parametric, transductive)
  generalizes to unseen genomes better than any learned combiner.

## Final model (CPU-only, transductive)
Per genome (test contigs pooled): IDF-weighted contig-similarity matrix S; candidate score
`score(t) = log1p( CF(t) / sibfrac(t)^0.5 ) + 0.1 * mean_global_cooc(t | visible)`,
CF(t) = sum_s S[i,s]^3 * [t visible in s]; exclude the 15 visible; top-10.
- **Faithful group-CV MRR@10: raw 0.215, test-calibrated 0.221** (power=3, gamma=0.5, cooc_w=0.1).
- Reference: AI baseline 0.191; co-occurrence baseline ~0.14. => beats baseline; ~rank 3 (leaderboard
  rank-2 = 0.227, rank-3 = 0.209 at time of writing).

## Deliverables
solution.py (self-contained, numpy/pandas), submission.csv (1877 rows, validated), approach.md,
approach_short.md, notes.md, research/.
