import sys, collections
from pathlib import Path
import numpy as np, pandas as pd

VOCAB=1024
MASK_TOKEN="?"
POWER=3.0
GAMMA=0.5
COOC_W=0.10
A_CO=0.1

def find_data_dir():
    here=Path(__file__).resolve().parent
    for c in [here/"dataset"/"public",here/"dataset",Path("dataset/public"),Path("dataset"),here,Path(".")]:
        if (c/"train.csv").exists() and (c/"test.csv").exists(): return c
    raise FileNotFoundError("train.csv/test.csv not found")

def toks(s): return [int(x) for x in str(s).split()]
def genome_key(gc): return tuple(sorted(set(gc)))

def main():
    DATA=find_data_dir(); print("data dir:",DATA)
    train=pd.read_csv(DATA/"train.csv"); test=pd.read_csv(DATA/"test.csv")
    train_ks=[toks(s) for s in train["kmer_seq"]]

    df=np.zeros(VOCAB+1); cooc=np.zeros((VOCAB+1,VOCAB+1)); tokc=np.zeros(VOCAB+1)
    for k in train_ks:
        for t in set(k): df[t]+=1
        for a in k:
            tokc[a]+=1
            for b in k:
                if a!=b: cooc[a,b]+=1
    IDF=np.log((len(train_ks)+1)/(df+1))+1.0
    LOGCOND=np.log((cooc+A_CO)/(tokc[:,None]+A_CO*VOCAB))

    test_gc=[toks(s) for s in test["genome_context"]]
    test_vis=[[int(x) for x in str(s).split() if x!=MASK_TOKEN] for s in test["masked_kmer_seq"]]
    test_keys=[genome_key(g) for g in test_gc]

    by=collections.defaultdict(list)
    for i,gk in enumerate(test_keys): by[gk].append(i)

    preds=[None]*len(test); colidx=np.arange(VOCAB)
    for gk,idxs in by.items():
        m=len(idxs)
        Vb=np.zeros((m,VOCAB+1),dtype=np.float32)
        for a,i in enumerate(idxs):
            for t in test_vis[i]: Vb[a,t]=1.0
        Vidf=Vb*IDF; sim=Vidf@Vb.T; np.fill_diagonal(sim,0.0); simp=sim**POWER
        cf=simp@Vb
        sibfrac=Vb.sum(0)/m
        disc=cf/((sibfrac+0.05)**GAMMA)
        for a,i in enumerate(idxs):
            vis=test_vis[i]
            co=LOGCOND[vis].mean(0) if vis else np.zeros(VOCAB+1)
            S=np.log1p(disc[a])+COOC_W*co
            S[0]=-1e9
            for v in vis:
                if 1<=v<=VOCAB: S[v]=-1e9
            s=S[1:]
            top=np.argpartition(-s,10)[:10]; top=top[np.argsort(-s[top])]
            preds[i]=" ".join(str(int(t)+1) for t in top)
    sub=pd.DataFrame({"id":test["id"],"predicted_kmer_ids":preds})
    assert list(sub.columns)==["id","predicted_kmer_ids"]
    assert len(sub)==len(test) and sub["id"].is_unique and not sub.isna().any().any()
    for s in sub["predicted_kmer_ids"]:
        ids=s.split(); assert 1<=len(ids)<=10 and len(ids)==len(set(ids))
        assert all(1<=int(x)<=VOCAB for x in ids)
    Path("working").mkdir(exist_ok=True)
    sub.to_csv("working/submission.csv",index=False); sub.to_csv("submission.csv",index=False)
    print("wrote submission.csv & working/submission.csv:",sub.shape); print(sub.head())

if __name__=="__main__":
    main()
