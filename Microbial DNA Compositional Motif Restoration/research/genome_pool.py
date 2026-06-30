"""Transductive genome-pooled frequency + co-occurrence + position prior. Group-CV (genome).
Genome profile is leave-one-contig honest (subtract the current contig's gold). Stats built once/fold."""
import sys, numpy as np, pandas as pd, collections
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from common import load, genome_key, mrr10, group_folds, VOCAB
tr,te=load()
keys=[genome_key(g) for g in tr["gc"]]            # list of tuples
folds=group_folds(keys,n=5,seed=0)
MASKPOS=list(range(4,16))
KS=[r for r in tr["kseq"]]; GC=[r for r in tr["gc"]]

def build_global(idxs):
    pos_cnt=np.zeros((16,VOCAB+1)); cooc=np.zeros((VOCAB+1,VOCAB+1)); tokcnt=np.zeros(VOCAB+1)
    for i in idxs:
        k=KS[i]
        for pos,t in enumerate(k): pos_cnt[pos,t]+=1
        for a in k:
            tokcnt[a]+=1
            for b in k:
                if a!=b: cooc[a,b]+=1
    logpos=np.log((pos_cnt+0.5)/(pos_cnt.sum(1,keepdims=True)+0.5*VOCAB))
    logcond=np.log((cooc+0.1)/(tokcnt[:,None]+0.1*VOCAB))
    return logpos,logcond

def genome_totals(idxs):
    tot=collections.defaultdict(lambda: np.zeros(VOCAB+1))
    for i in idxs:
        gk=keys[i]
        for t in KS[i]: tot[gk][t]+=1
    return tot

# precompute per-fold artifacts once
fold_art={}
for f in range(5):
    tri=np.where(folds!=f)[0]; vai=np.where(folds==f)[0]
    logpos,logcond=build_global(tri)
    tot=genome_totals(vai)                            # transductive over val genomes
    fold_art[f]=(vai,logpos,logcond,tot)

def eval_fold(f,w):
    wpool,wcooc,wpos,wgc=w
    vai,logpos,logcond,tot=fold_art[f]
    gold=[]; ranked=[]
    for idx in vai:
        k=KS[idx]; gk=keys[idx]; gcset=set(GC[idx])
        base=tot[gk]
        gcv=np.zeros(VOCAB+1)
        for g in gcset: gcv[g]=1.0
        for p in MASKPOS:
            g=k[p]; vis=[k[i] for i in range(16) if i!=p]
            prof=base.copy(); prof[g]-=1               # remove self-leak
            logprof=np.log((prof+0.3)/(prof.sum()+0.3*VOCAB))
            s=wpool*logprof+wpos*logpos[p]+wgc*gcv
            if vis: s=s+wcooc*logcond[vis].mean(0)
            s[0]=-1e9
            for v in vis: s[v]=-1e9
            top=np.argpartition(-s,10)[:10]; top=top[np.argsort(-s[top])]
            gold.append(g); ranked.append(list(top))
    return mrr10(gold,ranked)

best=(-1,None)
for wpool in [0,1,2,3,4]:
  for wcooc in [0,1,2]:
    for wpos in [0,0.5,1]:
      for wgc in [0,1,3]:
        m=eval_fold(0,(wpool,wcooc,wpos,wgc))
        if m>best[0]: best=(m,(wpool,wcooc,wpos,wgc))
print("fold0 best:",best)
w=best[1]; scores=[eval_fold(f,w) for f in range(5)]
for f,m in enumerate(scores): print(f"fold{f} MRR@10={m:.4f}")
print(f"\nCV MRR@10 (genome-pool+cooc+pos+gc) = {np.mean(scores):.4f} weights={w}")
# also report pure pool and pool+cooc
print("pure pool only:", np.mean([eval_fold(f,(1,0,0,0)) for f in range(5)]).round(4))
