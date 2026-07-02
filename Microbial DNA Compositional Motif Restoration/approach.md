# Approach: Microbial DNA Compositional Motif Restoration

**Time spent:** ~5 hours

## Summary
A transductive **contig-similarity collaborative-filtering (CF)** ranker, CPU-only. It restores the
masked hexamer by voting over the genome's *most similar* sibling contigs. Faithful genome-grouped
CV: raw MRR@10 0.215, **test-calibrated 0.221** (the test set skews to large genomes with richer
sibling signal). Beats the AI baseline (0.191); ~leaderboard rank 3.

## Key insights (in order of impact)
1. **Group-CV by genome** (genome = sorted set of genome_context). Train/test genomes are disjoint,
   so random CV is meaningless. This was step 1 and reshaped everything.
2. **Recall is free; ranking is everything.** The masked token is in the sibling-token-union **94%**
   of the time. A frequency/co-occurrence baseline only reaches ~0.17 because it ranks poorly.
3. **Transductive contig-similarity CF.** The gold appears in the sibling contigs most *similar* to
   the target contig (same genomic region). Score candidate t by
   `sum over siblings s of similarity(i,s)^3 · [t visible in s]`, with IDF-weighted similarity so
   discriminative k-mers drive the match (not genome-ubiquitous ones). This is legitimate
   transductive use of test *inputs* (like the provided genome_context).
4. **Discount genome-ubiquitous candidates.** On large genomes, tokens present in *most* siblings get
   high CF scores but are genuinely absent from this region (not the gold). Dividing by
   `sibfrac^0.5` fixes this and lifts large-genome MRR (large genomes are 77% of the test).
5. **Leak-free evaluation.** All genome statistics use visible tokens only; global co-occurrence uses
   leave-one-genome-out. (An early version leaked the contig's own gold pairs and reported a
   misleading 0.28, caught and removed.)

## Final scoring
`score(t) = log1p( CF(t) / (sibfrac(t)+0.05)^0.5 ) + 0.1 · mean_{v∈visible} logP_global(t|v)`
CF from IDF-weighted similarity^3 sibling voting; global co-occurrence is a backoff for tiny
genomes; the 15 visible tokens are excluded; output top-10. (power=3, gamma=0.5, cooc_w=0.1, tuned
on faithful CV.)

## What didn't work
- From-scratch masked-LM transformer (contig + genome_context): 0.12. It overfits the ~90 training
  genomes and cannot access an unseen test genome's co-occurrence from just the 16-token context.
- LightGBM learning-to-rank reranker over CF + co-occurrence + sibling-rank features: 0.205, a
  learned cross-genome combiner generalizes *worse* to disjoint genomes than non-parametric CF.
- Linear and z-scored rank-fusion blends (CF + genome-cooc + global-cooc): all below CF alone.

## Compliance / reproducibility
Learns k-mer co-occurrence and genome context from the provided data only; no de-anonymization, no
external lookup, no hardcoded answers; deterministic. Self-contained `solution.py` (numpy/pandas),
reads `./dataset[/public]/`, writes `./working/submission.csv` + `./submission.csv`.
