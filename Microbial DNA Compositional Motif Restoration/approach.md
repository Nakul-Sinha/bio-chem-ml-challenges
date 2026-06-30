# Approach — Microbial DNA Compositional Motif Restoration

**Time spent:** ~3 hours

## Summary
A transductive, count-based ranking model that restores the masked hexamer token by combining
**genome-specific k-mer co-occurrence**, global co-occurrence, and a genome-specificity (PMI)
term. CPU-only, no neural net needed. Faithful genome-grouped CV MRR@10 ≈ **0.167**.

## Key ideas
1. **Group-CV by genome** (genome = sorted set of genome_context) because train/test genomes are
   disjoint — random CV would be wildly optimistic. The held-out folds are entire genomes.
2. **Transductive genome statistics.** For a genome, pool its contigs' *visible* tokens to estimate
   that genome's k-mer frequency and **k-mer–k-mer co-occurrence**. The masked token of one contig
   is usually visible in sibling contigs, so its association with the contig's visible tokens is
   recoverable. This is legitimate (uses test *inputs* only, exactly like the provided
   genome_context) and is the single biggest signal.
3. **Leak-free evaluation.** Genome co-occurrence is built from visible tokens only (one mask per
   contig), so the masked token contributes no pairs from its own contig. (Building it from full
   sequences inflated CV from 0.166 to a misleading 0.226 — caught and fixed.)
4. **Candidate restriction.** k-mers in a contig are distinct, so the 15 visible tokens are excluded
   from the candidate set.

## Final scoring
`score(t) = 3.0·logP_genome(t|visible) + 0.5·logP_global(t|visible) + 0.5·PMI_genome(t)`
- `P_genome(t|visible)`: sum of genome co-occurrence of t with the visible tokens (transductive).
- `P_global(t|visible)`: same from the global training co-occurrence (backoff for sparse genomes).
- `PMI_genome(t) = logP_genome(t) − logP_global(t)`: upweights genome-characteristic k-mers.
Weights tuned on faithful group-CV. Output = top-10 after excluding visible tokens.

## What didn't work / tradeoffs
- A from-scratch masked-LM transformer (contig + genome_context) reached only 0.12 — it overfits
  the ~90 training genomes and generalizes poorly to held-out genomes; it adds only ~0.003 in a
  blend, so it's omitted to keep the solution CPU-only and robust.
- Positional consensus (sibling token at the same rank) and raw genome frequency were weak alone.

## Compliance / reproducibility
Learns co-occurrence + genome context from the provided data only; no de-anonymization, no external
lookup, no hardcoded answers. Deterministic; self-contained `solution.py` (numpy/pandas), reads
`./dataset[/public]/`, writes `./working/submission.csv` + `./submission.csv`.
