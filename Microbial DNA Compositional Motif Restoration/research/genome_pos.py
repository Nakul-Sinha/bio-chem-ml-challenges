"""Test transductive positional consensus (sibling tokens at same rank) + genome co-occurrence,
blended with global co-occurrence and PMI. Full group-CV, MRR@10 with proper tie-break."""
import sys, numpy as np, collections
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from common import load, genome_key, mrr10, group_folds, VOCAB
tr,te=load()
keys=[genome_key(g) for g in tr["gc"]]
folds=group_folds(keys,n=5,seed=0)
MASKPOS=list(range(4,16)); KS=[r for r in tr["kseq"]]
colidx=np.arange(VOCAB)

def build_global(idxs):
    cooc=np.zeros((VOCAB+1,VOCAB+1)); tokcnt=np.zeros(VOCAB+1); glob=np.zeros(VOCAB+1)
    for i in idxs:
        for t in KS[i]: glob[t]+=1
        for a in KS[i]:
            tokcnt[a]+=1
            for b in KS[i]:
                if a!=b: cooc[a,b]+=1
    return np.log((cooc+0.1)/(tokcnt[:,None]+0.1*VOCAB)), np.log((glob+0.5)/(glob.sum()+0.5*VOCAB))

def gen_artifacts(vai):
    """per genome: total token counts, per-position counts, and genome-cooc."""
    tot=collections.defaultdict(lambda: np.zeros(VOCAB+1))
    poscnt=collections.defaultdict(lambda: np.zeros((16,VOCAB+1)))
    gcooc=collections.defaultdict(lambda: np.zeros((VOCAB+1,VOCAB+1)))
    by=collections.defaultdict(list)
    for i in vai: by[keys[i]].append(i)
    for gk,idxs in by.items():
        for i in idxs:
            k=KS[i]
            for p,t in enumerate(k): tot[gk][t]+=1; poscnt[gk][p,t]+=1
            for a in k:
                for b in k:
                    if a!=b: gcooc[gk][a,b]+=1
    return tot,poscnt,gcooc,by

def eval_all(weights):
    """weights: dict name->w. Components: gpool, gpos, gcond, cond, pmi."""
    rr=[]
    for f in range(5):
        tri=np.where(folds!=f)[0]; vai=np.where(folds==f)[0]
        logcond,logglob=build_global(tri); tot,poscnt,gcooc,by=gen_artifacts(vai)
        for i in vai:
            k=KS[i]; gk=keys[i]
            gtokcnt=tot[gk].sum()
            for p in MASKPOS:
                g=k[p]; vis=[k[j] for j in range(16) if j!=p]
                S=np.zeros(VOCAB)
                if weights.get("gpool"):
                    prof=tot[gk].copy(); prof[g]-=1
                    S+=weights["gpool"]*np.log((prof[1:]+0.3)/(prof.sum()+0.3*VOCAB))
                if weights.get("gpos"):
                    pc=poscnt[gk][p].copy(); pc[g]-=1
                    S+=weights["gpos"]*np.log((pc[1:]+0.1)/(pc.sum()+0.1*VOCAB))
                if weights.get("gcond"):
                    gc=gcooc[gk][vis].sum(0).copy()
                    S+=weights["gcond"]*np.log((gc[1:]+0.1)/(gc.sum()+0.1*VOCAB))
                if weights.get("cond"):
                    S+=weights["cond"]*logcond[vis].mean(0)[1:]
                if weights.get("pmi"):
                    prof=tot[gk].copy(); prof[g]-=1
                    S+=weights["pmi"]*(np.log((prof[1:]+0.3)/(prof.sum()+0.3*VOCAB))-logglob[1:])
                for v in vis:
                    if 1<=v<=VOCAB: S[v-1]=-1e9
                gs=S[g-1]; rank=int((S>gs).sum()+(((S==gs)&(colidx<g-1)).sum()))+1
                rr.append(1.0/rank if rank<=10 else 0.0)
    return np.mean(rr)

import itertools
print("gpos only:", round(eval_all({"gpos":1}),4))
print("gpool only:", round(eval_all({"gpool":1}),4))
print("gcond only:", round(eval_all({"gcond":1}),4))
print("cond only:", round(eval_all({"cond":1}),4))
print("gpos+cond:", round(eval_all({"gpos":1,"cond":1}),4))
print("gpos+gcond:", round(eval_all({"gpos":1,"gcond":1}),4))
print("gpos+gpool+cond:", round(eval_all({"gpos":1,"gpool":1,"cond":1}),4))
print("gpos+gpool+gcond+cond:", round(eval_all({"gpos":1,"gpool":1,"gcond":1,"cond":1}),4))
