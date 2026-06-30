"""Shared: data loading, genome grouping, MRR@10 metric, group-CV by genome."""
import numpy as np, pandas as pd
from pathlib import Path
DS=Path(__file__).resolve().parent.parent/"dataset"
VOCAB=1024  # tokens 1..1024

def toks(s): return [int(x) for x in str(s).split()]

def load():
    tr=pd.read_csv(DS/"train.csv"); te=pd.read_csv(DS/"test.csv")
    tr["kseq"]=tr["kmer_seq"].apply(toks); tr["gc"]=tr["genome_context"].apply(toks)
    te["gc"]=te["genome_context"].apply(toks)
    te["mseq"]=te["masked_kmer_seq"].apply(lambda s:[None if x=="?" else int(x) for x in str(s).split()])
    te["mask_index"]=te["mask_index"].astype(int)
    return tr,te

def genome_key(gc): return tuple(sorted(set(gc)))

def mrr10(gold, ranked):
    """gold: list of ints; ranked: list of lists (each up to 10 token ids)."""
    s=0.0
    for g,r in zip(gold,ranked):
        rr=0.0
        for i,t in enumerate(r[:10]):
            if t==g: rr=1.0/(i+1); break
        s+=rr
    return s/len(gold)

def group_folds(keys, n=5, seed=0):
    uk=list(dict.fromkeys(keys)); rng=np.random.default_rng(seed); rng.shuffle(uk)
    fold_of={k:i%n for i,k in enumerate(uk)}
    return np.array([fold_of[k] for k in keys])
