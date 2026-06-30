"""Test-faithful gcond ranker: one mask per contig, genome co-occurrence from VISIBLE tokens only
(no leak). Group-CV. Precompute components for tuning + verify best weights over repeats."""
import sys, time, numpy as np, collections
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from common import load, genome_key, group_folds, VOCAB
tr,te=load()
keys=[genome_key(g) for g in tr["gc"]]
folds=group_folds(keys,n=5,seed=0)
MASKPOS=np.array(range(4,16)); KS=[np.array(r) for r in tr["kseq"]]
colidx=np.arange(VOCAB)

def build_global(idxs):
    cooc=np.zeros((VOCAB+1,VOCAB+1)); tokcnt=np.zeros(VOCAB+1)
    for i in idxs:
        for a in KS[i]:
            tokcnt[a]+=1
            for b in KS[i]:
                if a!=b: cooc[a,b]+=1
    return np.log((cooc+0.1)/(tokcnt[:,None]+0.1*VOCAB))

def build_components(rep_seed, store=False):
    """Assign one mask per contig (seeded), build genome gcooc/tot from visible-15, return components."""
    rng=np.random.default_rng(rep_seed)
    GC=[]; CO=[]; GP=[]; GOLD=[]; VISR=[]
    for f in range(5):
        tri=np.where(folds!=f)[0]; vai=np.where(folds==f)[0]
        logcond=build_global(tri)
        # assign masks; build visible-only genome artifacts
        mpos={int(i):int(rng.choice(MASKPOS)) for i in vai}
        tot=collections.defaultdict(lambda: np.zeros(VOCAB+1))
        gcooc=collections.defaultdict(lambda: np.zeros((VOCAB+1,VOCAB+1)))
        for i in vai:
            k=KS[i]; p=mpos[int(i)]; gk=keys[i]
            vismask=[k[j] for j in range(16) if j!=p]
            for t in vismask: tot[gk][t]+=1
            for a in vismask:
                for b in vismask:
                    if a!=b: gcooc[gk][a,b]+=1
        for i in vai:
            k=KS[i]; p=mpos[int(i)]; gk=keys[i]; g=int(k[p]); vis=[int(k[j]) for j in range(16) if j!=p]
            gc=gcooc[gk][vis].sum(0)
            gcond=np.log((gc[1:]+0.05)/(gc.sum()+0.05*VOCAB))
            cond=logcond[vis].mean(0)[1:]
            prof=tot[gk]; gpool=np.log((prof[1:]+0.3)/(prof.sum()+0.3*VOCAB))
            GC.append(gcond.astype(np.float32)); CO.append(cond.astype(np.float32)); GP.append(gpool.astype(np.float32))
            GOLD.append(g); VISR.append(vis)
    GC=np.array(GC); CO=np.array(CO); GP=np.array(GP); GOLD=np.array(GOLD); n=len(GOLD)
    VIS=np.zeros((n,VOCAB),dtype=np.float32)
    for r,vis in enumerate(VISR):
        for v in vis:
            if 1<=v<=VOCAB: VIS[r,v-1]=-1e9
    return GC,CO,GP,GOLD,VIS

def mrr(comp,wg,wc,wp):
    GC,CO,GP,GOLD,VIS=comp; n=len(GOLD); goldcol=GOLD-1
    S=wg*GC+wc*CO+wp*GP+VIS
    gs=S[np.arange(n),goldcol]
    rank=(S>gs[:,None]).sum(1)+((S==gs[:,None])&(colidx[None,:]<goldcol[:,None])).sum(1)+1
    return np.where(rank<=10,1.0/rank,0.0).mean()

t0=time.time()
comp0=build_components(100); print(f"components(rep0) built [{time.time()-t0:.0f}s] rows={len(comp0[3])}")
best=(-1,None)
for wg in [1,2,3,4,6]:
  for wc in [0,0.25,0.5,1]:
    for wp in [0,0.25,0.5,1]:
        m=mrr(comp0,wg,wc,wp)
        if m>best[0]: best=(m,(wg,wc,wp))
print(f"rep0 BEST MRR@10={best[0]:.4f} (wgcond,wcond,wgpool)={best[1]}")
print("gcond only rep0:",round(mrr(comp0,1,0,0),4))
# verify over 2 more repeats
w=best[1]; ms=[best[0]]
for s in [200,300]:
    c=build_components(s); ms.append(mrr(c,*w)); print(f"  rep{s} MRR={ms[-1]:.4f} [{time.time()-t0:.0f}s]")
print(f"FAITHFUL CV MRR@10 = {np.mean(ms):.4f} +- {np.std(ms):.4f} weights={w}")
