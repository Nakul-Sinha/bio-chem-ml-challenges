"""Comprehensive vectorized faithful blend: CF + genome-cooc + global-cooc + genome_context boost."""
import sys, time, numpy as np, collections
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from common import load, genome_key, group_folds, VOCAB
tr,te=load()
keys=[genome_key(g) for g in tr["gc"]]; folds=group_folds(keys,n=5,seed=0)
KS=[np.array(r,dtype=np.int64) for r in tr["kseq"]]
GCS=[set(int(x) for x in r) for r in tr["gc"]]
by=collections.defaultdict(list)
for i,k in enumerate(keys): by[k].append(i)
MASKPOS=np.array(range(4,16))
df=np.zeros(VOCAB+1)
for k in KS:
    for t in set(int(x) for x in k): df[t]+=1
IDF=np.log((len(KS)+1)/(df+1))+1.0

def build_cond(idxs):
    cooc=np.zeros((VOCAB+1,VOCAB+1)); tokcnt=np.zeros(VOCAB+1)
    for i in idxs:
        for a in KS[i]:
            tokcnt[a]+=1
            for b in KS[i]:
                if a!=b: cooc[a,b]+=1
    return np.log((cooc+0.1)/(tokcnt[:,None]+0.1*VOCAB))

POWER=4.0
def fold_components(f, rng):
    vai=np.where(folds==f)[0]; tri=np.where(folds!=f)[0]
    lc=build_cond(tri)
    comps=[]  # each: dict of component vectors + gold + vismask
    fold_g=[gk for gk in by if by[gk][0] in set(vai.tolist())]
    for gk in fold_g:
        idxs=by[gk]; m=len(idxs); mp=rng.choice(MASKPOS,size=m)
        Vb=np.zeros((m,VOCAB+1),dtype=np.float32); gold=np.zeros(m,dtype=np.int64); gcvec=np.zeros((m,VOCAB+1),dtype=np.float32)
        for a,j in enumerate(idxs):
            k=KS[j]; p=mp[a]; gold[a]=k[p]
            for q in range(16):
                if q!=p: Vb[a,k[q]]=1.0
            for t in GCS[j]: gcvec[a,t]=1.0
        Vidf=Vb*IDF
        sim=Vidf@Vb.T; np.fill_diagonal(sim,0.0); simp=sim**POWER
        cf=simp@Vb                                   # CF vote
        C=Vb.T@Vb                                     # genome cooc (gold not in Vb[a] => no self-leak for gold)
        gcond=Vb@C                                    # genome-cooc vote
        cfl=np.log1p(cf); gcl=np.log1p(gcond)
        coocv=np.zeros((m,VOCAB+1),dtype=np.float32)
        for a in range(m):
            vis=np.where(Vb[a]>0)[0]; coocv[a]=lc[vis].mean(0)
        comps.append((cfl,gcl,coocv,gcvec,Vb,gold))
    return comps

def score_all(comps, w):
    wcf,wgc,wco,wgcb=w; rr=[]
    for cfl,gcl,coocv,gcvec,Vb,gold in comps:
        S=wcf*cfl+wgc*gcl+wco*coocv+wgcb*gcvec
        S[:,0]=-1e9; S[Vb>0]=-1e9
        for a in range(len(gold)):
            s=S[a]; g=gold[a]
            top=np.argpartition(-s,10)[:10]; top=top[np.argsort(-s[top])]
            rank=next((r+1 for r,t in enumerate(top) if t==g),None)
            rr.append(1.0/rank if rank else 0.0)
    return np.mean(rr)

t0=time.time()
rng=np.random.default_rng(100)
allc=[fold_components(f,rng) for f in range(5)]
comps=[c for fc in allc for c in fc]
print(f"components built [{time.time()-t0:.0f}s]")
print("refs: CF", round(score_all(comps,(1,0,0,0)),4), " gcond",round(score_all(comps,(0,1,0,0)),4),
      " cooc",round(score_all(comps,(0,0,1,0)),4), " gcboost",round(score_all(comps,(0,0,0,1)),4))
best=(-1,None)
for wcf in [0,1,2,3]:
  for wgc in [0,1,2,3]:
    for wco in [0,1,2,4]:
      for wgcb in [0,1,2,4]:
        if wcf==wgc==wco==wgcb==0: continue
        m=score_all(comps,(wcf,wgc,wco,wgcb))
        if m>best[0]: best=(m,(wcf,wgc,wco,wgcb))
print(f"BEST={best[0]:.4f} w(cf,gcond,cooc,gcboost)={best[1]} [{time.time()-t0:.0f}s]")
