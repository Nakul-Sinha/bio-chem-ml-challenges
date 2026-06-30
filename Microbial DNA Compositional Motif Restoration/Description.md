Problem Description
Microbial DNA Compositional Motif Restoration
Overview
Microbial genomes carry distinctive DNA compositional signatures: the frequencies of short nucleotide motifs (k-mers) vary systematically across species and even across genomic islands within a genome. This challenge poses the restoration of a missing compositional motif from a contig's signature.

Each genome contig is written as a short sequence of its most frequent hexamer (6-mer) motifs, in rank order, as opaque integer k-mer tokens — its "compositional signature." One mid-rank token has been removed (masked); the task is to restore the missing k-mer token, drawing on the contig's other top k-mers and on the k-mer profile across sibling contigs from the same genome (genome context).

The masked token is one hexamer from a fixed vocabulary of 1 024 tokens (the 1 024 most frequent hexamers in the corpus — the only tokens that ever appear in the data). Rather than a single guess, you submit a ranked list of up to 10 candidate tokens per contig and are scored by where the gold token lands in your ranking (partial credit), so a near-miss still counts. K-mer identities are anonymised (opaque integer ids), and genomes are disjointly split between train and test, so the restoration must be learned from the training contigs: no external genomic knowledge, atlas, or pretrained model applies.

Dataset
public/train.csv — id, genome_context, kmer_seq
kmer_seq (string): the contig's top-16 most frequent k-mers as space-separated opaque integer tokens, in rank order (the full, unmasked signature)
genome_context (string): the top-16 k-mers pooled across the contig's sibling contigs from the same genome — the genomic context
public/test.csv — id, genome_context, masked_kmer_seq, mask_index
masked_kmer_seq (string): the contig's signature with one mid-rank token replaced by ?
mask_index (int): 0-based position of the ?
public/sample_submission.csv — a valid baseline submission
Train and test contigs come from disjoint genomes. The vocabulary is fixed at the 1 024 tokens appearing in train.csv, and every masked k-mer is one of them — so the candidate set is fully known from the training data.

Evaluation
Submissions are scored with K-mer Restoration Rank (KRR) — the mean reciprocal rank of the gold token within each contig's submitted ranked candidate list, with a cutoff of 10. For one contig, if the gold token is the r-th entry of your ranked list (r ≤ 10) it scores 1/r; if it is absent (or below rank 10) it scores 0. KRR is the mean over all test contigs:

KRR = mean over contigs of ( 1 / rank_of_gold_in_your_list )   # 0 if gold not in top 10  

Score is in [0, 1]; higher is better. Unlike a top-1 exact match, this gives partial credit: ranking the gold token 3rd scores 1/3 instead of 0, so a stronger model that pushes the gold token up the list separates clearly on the leaderboard.

Reference baselines (public training data only)
The task is solvable well above the most-frequent floor by learning k-mer co-occurrence and genome context. It remains hard: the masked token is in the genome context only about a third of the time, so a co-occurrence ranker reaches only ~0.14, with the gold token landing in its top 10 about 31% of the time (Recall@10 ≈ 0.31). A stronger context-aware sequence model can realistically push KRR into roughly the 0.20–0.30 range — a clear, separable gain over the cheap baseline — by improving both recall and the rank of the gold token; a solver with no model of the anonymised tokens collapses toward the floor.

Submission Format
Submit a CSV with exactly two columns:

id: contig identifier from test.csv
predicted_kmer_ids: your ranked candidate tokens for the masked position — 1 to 10 integer k-mer ids, best first, space-separated, in one cell
id,predicted_kmer_ids  
3a1f9c2b7e4d0581,512 77 1024 9 311  
c8b2d90f3e61a744,77 512 6  
...  

Strict format requirements (submission rejected on violation):

Exactly the columns id and predicted_kmer_ids — no extra columns.
One row for every test id, no duplicate ids. Every test id must be present; extra rows whose id is not in the test set are ignored.
Each predicted_kmer_ids cell is a non-empty, duplicate-free list of at most 10 integer tokens, ranked best-first; no NaN / missing values.
Approaches
Masked sequence modeling: train a bidirectional model (transformer/BiLSTM) over the compositional signatures with random masking, conditioning on genome_context, then rank the vocabulary by the model's probability for the held-out position and submit the top 10.
Co-occurrence / association models: estimate k-mer–k-mer co-occurrence and k-mer–genome-context association from the training signatures, with backoff, and rank candidate tokens — a strong, cheap baseline.
Genome-context conditioning: use genome_context to bias the ranking toward k-mers consistent with the genome's overall compositional profile.
Submit a ranked top-10, not a single guess — partial credit rewards getting the gold token into your list even when it is not your top pick.
Validate KRR on a held-out split of the training contigs before predicting the test set.
Prohibited Methods
De-anonymisation / external lookup: recovering real k-mer sequences behind the opaque tokens, or matching signatures against any external genome database / reference / model.
Source identification: identifying or retrieving the originating organism or genome. Ids are opaque; tokens are anonymised.
Hard-coded answers: producing answers by any means other than a model built from the provided training data.
Valid solutions learn k-mer co-occurrence and genome context from train.csv and apply them to restore the masked tokens.

Configuration
Direction: maximize
Min Score: 0.0
Max Score: 1.0
Difficulty: Hard
GPU Tier: A10G