# Microbial DNA Compositional Motif Restoration — notes

## Task
Each contig = top-16 most-frequent hexamer tokens (opaque ints 1..1024), rank-ordered. One
mid-rank token is masked; predict a ranked top-10 of candidate token ids. Metric: MRR@10 (KRR).
genome_context = top-16 k-mers pooled across the genome's sibling contigs. Train/test genomes
are DISJOINT. No external data / pretrained (tokens are anonymized ints anyway).

## EDA
- 7585 train / 1877 test contigs. Vocab exactly 1024 (ids 1..1024). All seqs length 16.
- **113 genomes** in train (group key = sorted set of genome_context). ~67 contigs/genome.
- Position encodes frequency rank (pos0 dominated by token 692/952). Mask positions = 4..15.
- Gold token is in the genome_context only ~31-37% of the time (matches the brief).
- kmer_seq tokens are distinct => the masked token is never among the 15 visible tokens (excluded
  from candidates).

## Validation (the critical part)
- **Group-CV by genome** (5-fold) so val genomes are held out — mirrors the disjoint-genome test.
- **Test-faithful masking**: one mask per contig; all transductive genome statistics
  (co-occurrence, frequency) are built from VISIBLE tokens only, so the masked token contributes
  no pairs from its own contig (no leak). Averaged over 3 mask assignments.
- Caught a leak: building genome co-occurrence from full sequences inflated gcond to 0.226; the
  honest value is ~0.166.

## Signals explored (honest faithful CV, MRR@10)
- position prior: weak.   global co-occurrence (cond): 0.125.   genome pool freq (gpool): 0.137.
- PMI (genome-specific vs global freq): weak alone (0.04) but complementary.
- positional consensus (gpos, sibling token at same rank): 0.102.
- **genome-specific co-occurrence (gcond, transductive)**: 0.166 — the strongest single signal.
- NN masked-LM transformer (contig+genome_context): 0.121 — overfits the ~90 train genomes,
  generalizes poorly to held-out genomes; adds only ~+0.003 in a blend, not worth the GPU cost.

## Final model
Transductive count-based blend, CPU-only:
`score(t) = 3.0*log P_genome(t|visible) + 0.5*log P_global(t|visible) + 0.5*PMI_genome(t)`
where P_genome co-occurrence/frequency are estimated from the genome's own (test) contigs'
visible tokens. Exclude the 15 visible tokens; take top-10.
- **Faithful group-CV MRR@10 = 0.167 ± 0.002** (vs co-occurrence baseline ~0.14).

## Deliverables
solution.py (self-contained, numpy/pandas only), submission.csv (1877 rows, validated),
approach.md, approach_short.md, notes.md, research/.
