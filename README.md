# Bio and Chem ML Solutions

Six machine learning problems from biology and chemistry, each in its own folder
with its own description, notes and solution code.

| Folder | What the problem is |
|---|---|
| `Microbial DNA Compositional Motif Restoration/` | Restore a missing k-mer compositional motif from a contig's DNA signature |
| `Persistent Particle Motion Cell Classification/` | Classify particle motion from two-frame crop pairs of visually similar beads in dense bedload flow |
| `Reaction Condition Completion/` | Infer structured reaction conditions, solvent, temperature, time and catalyst, from a single reaction SMILES string |
| `Reaction Protocol Silent-Edit Repair/` | Read a terse lab protocol plus a silent-edit notice and regenerate the corrected protocol in canonical form |
| `Single Cell Hidden Probe Sequence Reconstruction/` | Sequence to sequence reconstruction of a hidden probe from anonymized single-cell expression data |
| `Spectral Route Image Classification/` | Classify images degraded by blur, field loss, compression damage, sensor noise and scanline artifacts |

The common thread is that none of them is a clean benchmark. Every one arrives
with some deliberate corruption, anonymization or missing modality, so most of the
work is figuring out what signal actually survived.

Datasets are not committed.
