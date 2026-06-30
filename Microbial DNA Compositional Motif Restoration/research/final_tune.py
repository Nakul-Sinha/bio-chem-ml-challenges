"""Fine-tune disc-CF (power,gamma,cooc_w) -> raw + test-calibrated MRR. Mirrors solution.py exactly."""
import sys, numpy as np, collections
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from common import load, genome_key, VOCAB
tr,te=load()
keys=[genome_key(g) for g in tr["gc"]]; KS=[np.array(r,dtype=np.int64) for r in tr["kseq"]]
by=collections.defaultdict(list)
for i,k in enumerate(keys): by[k].append(i)
MASKPOS=np.array(range(4,16))
# LOGO-honest globals per genome for IDF & cooc
TOT_COOC=np.zeros((VOCAB+1,VOCAB+1)); TOT_TOK=np.zeros(VOCAB+1); TOT_DF=np.zeros(VOCAB+1)
for k in KS:
    for t in set(int(x) for x in k): TOT_DF[t]+=1
    for a in k:
        TOT_TOK[a]+=1
        for b in k:
            if a!=b: TOT_COOC[a,b]+=1
NDOC=len(KS)
buckets=[(1,3),(4,8),(9,20),(21,60),(61,150),(151,10**9)]
tek=[genome_key(g) for g in te["gc"]]; tec=collections.Counter(tek); test_w=collections.Counter()
for k in tek:
    sz=tec[k]
    for lo,hi in buckets:
        if lo<=sz<=hi: test_w[(lo,hi)]+=1; break
TOT=sum(test_w.values())

def precompute(seed):
    rng=np.random.default_rng(seed); per=[]
    for gk,idxs in by.items():
        m=len(idxs)
        cg=np.zeros((VOCAB+1,VOCAB+1)); tg=np.zeros(VOCAB+1); dg=np.zeros(VOCAB+1)
        for j in idxs:
            k=KS[j]
            for t in set(int(x) for x in k): dg[t]+=1
            for a in k:
                tg[a]+=1
                for b in k:
                    if a!=b: cg[a,b]+=1
        cooc=TOT_COOC-cg; tok=TOT_TOK-tg; dfo=TOT_DF-dg
        LOGCOND=np.log((cooc+0.1)/(tok[:,None]+0.1*VOCAB)); IDFo=np.log((NDOC-m+1)/(dfo+1))+1.0
        mp=rng.choice(MASKPOS,size=m); Vb=np.zeros((m,VOCAB+1),dtype=np.float32); gold=np.zeros(m,dtype=np.int64)
        for a,j in enumerate(idxs):
            k=KS[j]; p=mp[a]; gold[a]=k[p]
            for q in range(16):
                if q!=p: Vb[a,k[q]]=1.0
        Vidf=Vb*IDFo; sim=Vidf@Vb.T; np.fill_diagonal(sim,0.0)
        co=np.zeros((m,VOCAB+1),dtype=np.float32)
        for a in range(m): co[a]=LOGCOND[np.where(Vb[a]>0)[0]].mean(0)
        per.append((m,sim,Vb,gold,co))
    return per

def evalp(per, power, gamma, coocw):
    recs=[]
    for m,sim,Vb,gold,co in per:
        simp=sim**power; cf=simp@Vb; sibfrac=Vb.sum(0)/m; disc=cf/((sibfrac+0.05)**gamma)
        S=np.log1p(disc)+coocw*co; S[:,0]=-1e9; S[Vb>0]=-1e9
        for a in range(m):
            s=S[a,1:]; g=gold[a]-1; top=np.argpartition(-s,10)[:10]; top=top[np.argsort(-s[top])]
            rank=next((r+1 for r,t in enumerate(top) if t==g),None); recs.append((m,1.0/rank if rank else 0.0))
    recs=np.array(recs); raw=recs[:,1].mean(); bm={}
    for lo,hi in buckets:
        mask=(recs[:,0]>=lo)&(recs[:,0]<=hi)
        if mask.sum(): bm[(lo,hi)]=recs[mask,1].mean()
    cal=sum(bm.get(b,0)*test_w.get(b,0) for b in buckets)/TOT
    return raw,cal

per=precompute(100)
best=(-1,None)
for power in [2.5,3,3.5,4]:
  for gamma in [0.3,0.4,0.5,0.6]:
    for coocw in [0.0,0.1,0.2]:
        raw,cal=evalp(per,power,gamma,coocw)
        if cal>best[0]: best=(cal,(power,gamma,coocw),raw)
        print(f"p={power} g={gamma} cw={coocw}: cal={cal:.4f} raw={raw:.4f}")
print(f"\nBEST cal={best[0]:.4f} params(power,gamma,coocw)={best[1]} raw={best[2]:.4f}")
