"""Discriminative CF: penalize genome-ubiquitous candidates. score=cf / sibfrac^gamma.
Vectorized faithful; report raw CV + test-calibrated."""
import sys, numpy as np, collections
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from common import load, genome_key, VOCAB
tr,te=load()
keys=[genome_key(g) for g in tr["gc"]]; KS=[np.array(r,dtype=np.int64) for r in tr["kseq"]]
by=collections.defaultdict(list)
for i,k in enumerate(keys): by[k].append(i)
MASKPOS=np.array(range(4,16))
df=np.zeros(VOCAB+1)
for k in KS:
    for t in set(int(x) for x in k): df[t]+=1
IDF=np.log((len(KS)+1)/(df+1))+1.0
buckets=[(1,3),(4,8),(9,20),(21,60),(61,150),(151,10**9)]
tek=[genome_key(g) for g in te["gc"]]; tec=collections.Counter(tek); test_w=collections.Counter()
for k in tek:
    sz=tec[k]
    for lo,hi in buckets:
        if lo<=sz<=hi: test_w[(lo,hi)]+=1; break
TOT=sum(test_w.values())

def run(power, gamma, ublam=0.0):
    rng=np.random.default_rng(100); recs=[]
    for gk,idxs in by.items():
        m=len(idxs); mp=rng.choice(MASKPOS,size=m)
        Vb=np.zeros((m,VOCAB+1),dtype=np.float32); gold=np.zeros(m,dtype=np.int64)
        for a,j in enumerate(idxs):
            k=KS[j]; p=mp[a]; gold[a]=k[p]
            for q in range(16):
                if q!=p: Vb[a,k[q]]=1.0
        Vidf=Vb*IDF; sim=Vidf@Vb.T; np.fill_diagonal(sim,0.0); simp=sim**power
        cf=simp@Vb
        sibfrac=Vb.sum(0)/m                       # popularity in genome
        denom=(sibfrac+0.05)**gamma
        score=cf/denom - ublam*sibfrac
        score[:,0]=-1e9; score[Vb>0]=-1e9
        for a in range(m):
            s=score[a]; g=gold[a]; top=np.argpartition(-s,10)[:10]; top=top[np.argsort(-s[top])]
            rank=next((r+1 for r,t in enumerate(top) if t==g),None)
            recs.append((m,1.0/rank if rank else 0.0))
    recs=np.array(recs)
    raw=recs[:,1].mean()
    bm={}
    for lo,hi in buckets:
        mask=(recs[:,0]>=lo)&(recs[:,0]<=hi)
        if mask.sum(): bm[(lo,hi)]=recs[mask,1].mean()
    cal=sum(bm.get(b,0)*test_w.get(b,0) for b in buckets)/TOT
    return raw,cal

print("power=4 baseline (gamma=0):", [round(x,4) for x in run(4,0)])
print("\ngamma sweep (power=4): raw / test-calibrated")
for gamma in [0.3,0.5,0.8,1.0,1.5,2.0]:
    raw,cal=run(4,gamma); print(f"  gamma={gamma}: raw={raw:.4f} cal={cal:.4f}")
print("\npower x gamma (cal):")
for power in [3,4,6]:
  for gamma in [0.5,0.8,1.0,1.5]:
    raw,cal=run(power,gamma); print(f"  power={power} gamma={gamma}: cal={cal:.4f} (raw {raw:.4f})")
