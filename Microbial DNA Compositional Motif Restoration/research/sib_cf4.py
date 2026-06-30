"""Vectorized faithful CF: per-genome similarity^power voting, IDF-candidate weighting, cooc blend."""
import sys, time, numpy as np, collections
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from common import load, genome_key, group_folds, VOCAB
tr,te=load()
keys=[genome_key(g) for g in tr["gc"]]; folds=group_folds(keys,n=5,seed=0)
KS=[np.array(r,dtype=np.int64) for r in tr["kseq"]]
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

def eval4(power, beta=0.0, cooc_w=0.0, rep_seed=100):
    rng=np.random.default_rng(rep_seed); rr=[]
    for f in range(5):
        vai=np.where(folds==f)[0]; tri=np.where(folds!=f)[0]
        lc=build_cond(tri) if cooc_w else None
        fold_g=[gk for gk in by if by[gk][0] in set(vai.tolist())]
        for gk in fold_g:
            idxs=by[gk]; m=len(idxs)
            mp=rng.choice(MASKPOS,size=m)
            Vb=np.zeros((m,VOCAB+1),dtype=np.float32); gold=np.zeros(m,dtype=np.int64)
            for a,j in enumerate(idxs):
                k=KS[j]; p=mp[a]; gold[a]=k[p]
                for q in range(16):
                    if q!=p: Vb[a,k[q]]=1.0
            Vidf=Vb*IDF
            sim=(Vidf@Vb.T)                     # m x m, sim[a,b]=sum IDF over shared visible
            np.fill_diagonal(sim,0.0)
            simp=sim**power
            vote=Vb*(IDF**beta) if beta else Vb
            score=simp@vote                     # m x (V+1)
            if cooc_w:
                for a in range(m):
                    vis=np.where(Vb[a]>0)[0]
                    score[a]+=cooc_w*lc[vis].mean(0)
            score[:,0]=-1e9
            score[Vb>0]=-1e9                     # exclude visible
            for a in range(m):
                s=score[a]; g=gold[a]
                top=np.argpartition(-s,10)[:10]; top=top[np.argsort(-s[top])]
                rank=next((r+1 for r,t in enumerate(top) if t==g),None)
                rr.append(1.0/rank if rank else 0.0)
    return np.mean(rr)

t0=time.time()
print("=== power x beta (IDF candidate weight) ===")
for power in [4,6,8]:
  for beta in [0,0.5,1.0,1.5]:
    print(f"power={power} beta={beta}: {eval4(power,beta):.4f} [{time.time()-t0:.0f}s]")
