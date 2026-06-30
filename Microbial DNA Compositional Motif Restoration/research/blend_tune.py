"""Blend genome-pool + PMI + co-occurrence + position + gc. Tune on FULL 5-fold group-CV."""
import sys, numpy as np, pandas as pd, collections, itertools
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from common import load, genome_key, mrr10, group_folds, VOCAB
tr,te=load()
keys=[genome_key(g) for g in tr["gc"]]
folds=group_folds(keys,n=5,seed=0)
MASKPOS=list(range(4,16))
KS=[r for r in tr["kseq"]]; GC=[r for r in tr["gc"]]

def build(idxs):
    pos_cnt=np.zeros((16,VOCAB+1)); cooc=np.zeros((VOCAB+1,VOCAB+1)); tokcnt=np.zeros(VOCAB+1); glob=np.zeros(VOCAB+1)
    for i in idxs:
        k=KS[i]
        for pos,t in enumerate(k): pos_cnt[pos,t]+=1; glob[t]+=1
        for a in k:
            tokcnt[a]+=1
            for b in k:
                if a!=b: cooc[a,b]+=1
    logpos=np.log((pos_cnt+0.5)/(pos_cnt.sum(1,keepdims=True)+0.5*VOCAB))
    logcond=np.log((cooc+0.1)/(tokcnt[:,None]+0.1*VOCAB))
    logglob=np.log((glob+0.5)/(glob.sum()+0.5*VOCAB))
    return logpos,logcond,logglob
def gtotals(idxs):
    tot=collections.defaultdict(lambda: np.zeros(VOCAB+1))
    for i in idxs:
        for t in KS[i]: tot[keys[i]][t]+=1
    return tot

art={}
for f in range(5):
    tri=np.where(folds!=f)[0]; vai=np.where(folds==f)[0]
    logpos,logcond,logglob=build(tri); tot=gtotals(vai)
    art[f]=(vai,logpos,logcond,logglob,tot)

def fold_components(f):
    """Precompute per-(contig,pos) component scores once, return arrays for fast weight search."""
    vai,logpos,logcond,logglob,tot=art[f]
    rows=[]
    for idx in vai:
        k=KS[idx]; gk=keys[idx]; gcset=set(GC[idx]); base=tot[gk]
        gcv=np.zeros(VOCAB+1)
        for g in gcset: gcv[g]=1.0
        for p in MASKPOS:
            g=k[p]; vis=[k[i] for i in range(16) if i!=p]
            prof=base.copy(); prof[g]-=1
            logprof=np.log((prof+0.3)/(prof.sum()+0.3*VOCAB))
            pmi=logprof-logglob
            cond=logcond[vis].mean(0)
            mask=np.zeros(VOCAB+1); mask[0]=-1e9
            for v in vis: mask[v]=-1e9
            rows.append((g,logprof,pmi,cond,logpos[p],gcv,mask))
    return rows

comp={f:fold_components(f) for f in range(5)}
def score_fold(f,w):
    wprof,wpmi,wcond,wpos,wgc=w
    gold=[]; ranked=[]
    for g,logprof,pmi,cond,lpos,gcv,mask in comp[f]:
        s=wprof*logprof+wpmi*pmi+wcond*cond+wpos*lpos+wgc*gcv+mask
        top=np.argpartition(-s,10)[:10]; top=top[np.argsort(-s[top])]
        gold.append(g); ranked.append(list(top))
    return mrr10(gold,ranked)
def cv(w): return np.mean([score_fold(f,w) for f in range(5)])

best=(-1,None)
grid=dict(wprof=[0,1,2],wpmi=[0,1,2,3],wcond=[0,1,2],wpos=[0,0.5],wgc=[0,1,3,5])
import itertools as it
for vals in it.product(*grid.values()):
    w=dict(zip(grid.keys(),vals)).values(); w=tuple(w)
    m=cv(w)
    if m>best[0]: best=(m,w)
print(f"BEST CV MRR@10={best[0]:.4f} (wprof,wpmi,wcond,wpos,wgc)={best[1]}")
for f in range(5): print(f"  fold{f}={score_fold(f,best[1]):.4f}")
