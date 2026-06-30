"""Leak-free comprehensive blend: gcond + global cond + pmi + gpool. Faithful one-mask-per-contig CV."""
import sys, time, numpy as np, collections, itertools
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from common import load, genome_key, group_folds, VOCAB
tr,te=load()
keys=[genome_key(g) for g in tr["gc"]]
folds=group_folds(keys,n=5,seed=0)
MASKPOS=np.array(range(4,16)); KS=[np.array(r) for r in tr["kseq"]]
colidx=np.arange(VOCAB)

def build_global(idxs):
    cooc=np.zeros((VOCAB+1,VOCAB+1)); tokcnt=np.zeros(VOCAB+1); glob=np.zeros(VOCAB+1)
    for i in idxs:
        for a in KS[i]:
            tokcnt[a]+=1; glob[a]+=1
            for b in KS[i]:
                if a!=b: cooc[a,b]+=1
    return np.log((cooc+0.1)/(tokcnt[:,None]+0.1*VOCAB)), np.log((glob+0.5)/(glob.sum()+0.5*VOCAB))

def build_components(seed):
    rng=np.random.default_rng(seed)
    GC=[];CO=[];PMI=[];GP=[];GOLD=[];VISR=[]
    for f in range(5):
        tri=np.where(folds!=f)[0]; vai=np.where(folds==f)[0]
        logcond,logglob=build_global(tri)
        mpos={int(i):int(rng.choice(MASKPOS)) for i in vai}
        tot=collections.defaultdict(lambda: np.zeros(VOCAB+1))
        gcooc=collections.defaultdict(lambda: np.zeros((VOCAB+1,VOCAB+1)))
        for i in vai:
            k=KS[i]; p=mpos[int(i)]; gk=keys[i]; vism=[k[j] for j in range(16) if j!=p]
            for t in vism: tot[gk][t]+=1
            for a in vism:
                for b in vism:
                    if a!=b: gcooc[gk][a,b]+=1
        for i in vai:
            k=KS[i]; p=mpos[int(i)]; gk=keys[i]; g=int(k[p]); vis=[int(k[j]) for j in range(16) if j!=p]
            gc=gcooc[gk][vis].sum(0)
            GC.append(np.log((gc[1:]+0.05)/(gc.sum()+0.05*VOCAB)).astype(np.float32))
            CO.append(logcond[vis].mean(0)[1:].astype(np.float32))
            T=tot[gk]; gp=np.log((T[1:]+0.3)/(T.sum()+0.3*VOCAB))
            GP.append(gp.astype(np.float32)); PMI.append((gp-logglob[1:]).astype(np.float32))
            GOLD.append(g); VISR.append(vis)
    GC=np.array(GC);CO=np.array(CO);PMI=np.array(PMI);GP=np.array(GP);GOLD=np.array(GOLD);n=len(GOLD)
    VIS=np.zeros((n,VOCAB),dtype=np.float32)
    for r,vis in enumerate(VISR):
        for v in vis:
            if 1<=v<=VOCAB: VIS[r,v-1]=-1e9
    return GC,CO,PMI,GP,GOLD,VIS

def mrr(comp,w):
    GC,CO,PMI,GP,GOLD,VIS=comp; n=len(GOLD); goldcol=GOLD-1
    S=w[0]*GC+w[1]*CO+w[2]*PMI+w[3]*GP+VIS
    gs=S[np.arange(n),goldcol]
    rank=(S>gs[:,None]).sum(1)+((S==gs[:,None])&(colidx[None,:]<goldcol[:,None])).sum(1)+1
    return np.where(rank<=10,1.0/rank,0.0).mean()

t0=time.time(); comp0=build_components(100); print(f"built [{time.time()-t0:.0f}s]")
best=(-1,None)
for wg in [0,2,3,4,5]:
  for wc in [0,0.5,1,1.5]:
    for wpmi in [0,0.5,1,1.5]:
      for wgp in [0,0.5,1]:
        if wg==wc==wpmi==wgp==0: continue
        m=mrr(comp0,(wg,wc,wpmi,wgp))
        if m>best[0]: best=(m,(wg,wc,wpmi,wgp))
print(f"rep0 BEST={best[0]:.4f} w(gcond,cond,pmi,gpool)={best[1]}")
print("  refs: pmi+cond",round(mrr(comp0,(0,1,1,0)),4)," gcond",round(mrr(comp0,(1,0,0,0)),4),
      " gcond+pmi+cond",round(mrr(comp0,(3,1,1,0)),4))
w=best[1]; ms=[best[0]]
for s in [200,300]: ms.append(mrr(build_components(s),w))
print(f"FAITHFUL CV = {np.mean(ms):.4f} +- {np.std(ms):.4f} weights={w}")
