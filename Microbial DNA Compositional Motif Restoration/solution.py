import sys, collections
from pathlib import Path
import numpy as np, pandas as pd

VOCAB=1024
MASK_TOKEN="?"
WG=3.0
WC=0.5
WPMI=0.5
A_GC=0.05
A_CO=0.1
A_GP=0.3

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

    cooc=np.zeros((VOCAB+1,VOCAB+1)); tokcnt=np.zeros(VOCAB+1); glob=np.zeros(VOCAB+1)
    for k in train_ks:
        for a in k:
            tokcnt[a]+=1; glob[a]+=1
            for b in k:
                if a!=b: cooc[a,b]+=1
    logcond=np.log((cooc+A_CO)/(tokcnt[:,None]+A_CO*VOCAB))
    logglob=np.log((glob+0.5)/(glob.sum()+0.5*VOCAB))

    test_gc=[toks(s) for s in test["genome_context"]]
    test_vis=[]
    for s in test["masked_kmer_seq"]:
        test_vis.append([int(x) for x in str(s).split() if x!=MASK_TOKEN])
    test_keys=[genome_key(g) for g in test_gc]
    mask_index=test["mask_index"].astype(int).tolist()

    by=collections.defaultdict(list)
    for i,gk in enumerate(test_keys): by[gk].append(i)
    gcooc={}; gtot={}
    for gk,idxs in by.items():
        C=np.zeros((VOCAB+1,VOCAB+1)); T=np.zeros(VOCAB+1)
        for i in idxs:
            vis=test_vis[i]
            for t in vis: T[t]+=1
            for a in vis:
                for b in vis:
                    if a!=b: C[a,b]+=1
        gcooc[gk]=C; gtot[gk]=T

    colidx=np.arange(VOCAB); preds=[]
    for i in range(len(test)):
        gk=test_keys[i]; vis=test_vis[i]
        gc=gcooc[gk][vis].sum(0) if vis else np.zeros(VOCAB+1)
        gcond=np.log((gc[1:]+A_GC)/(gc.sum()+A_GC*VOCAB))
        cond=logcond[vis].mean(0)[1:] if vis else np.zeros(VOCAB)
        T=gtot[gk]; gpool=np.log((T[1:]+A_GP)/(T.sum()+A_GP*VOCAB))
        pmi=gpool-logglob[1:]
        S=WG*gcond+WC*cond+WPMI*pmi
        for v in vis:
            if 1<=v<=VOCAB: S[v-1]=-1e9
        top=np.argpartition(-S,10)[:10]; top=top[np.argsort(-S[top])]
        preds.append(" ".join(str(int(t)+1) for t in top))
    sub=pd.DataFrame({"id":test["id"],"predicted_kmer_ids":preds})
    assert list(sub.columns)==["id","predicted_kmer_ids"]
    assert len(sub)==len(test) and sub["id"].is_unique
    for s in sub["predicted_kmer_ids"]:
        ids=s.split(); assert 1<=len(ids)<=10 and len(ids)==len(set(ids))
        assert all(1<=int(x)<=VOCAB for x in ids)
    Path("working").mkdir(exist_ok=True)
    sub.to_csv("working/submission.csv",index=False); sub.to_csv("submission.csv",index=False)
    print("wrote submission.csv & working/submission.csv:",sub.shape); print(sub.head())

if __name__=="__main__":
    main()
