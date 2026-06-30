"""CF MRR stratified by genome size; reweight to TEST contig distribution for a test-calibrated estimate."""
import sys, numpy as np, collections
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from common import load, genome_key, group_folds, VOCAB
tr,te=load()
keys=[genome_key(g) for g in tr["gc"]]
KS=[np.array(r,dtype=np.int64) for r in tr["kseq"]]
by=collections.defaultdict(list)
for i,k in enumerate(keys): by[k].append(i)
MASKPOS=np.array(range(4,16)); POWER=4.0
df=np.zeros(VOCAB+1)
for k in KS:
    for t in set(int(x) for x in k): df[t]+=1
IDF=np.log((len(KS)+1)/(df+1))+1.0
rng=np.random.default_rng(100)

# CF per contig, record (genome_size, rr)
recs=[]
for gk,idxs in by.items():
    m=len(idxs); mp=rng.choice(MASKPOS,size=m)
    Vb=np.zeros((m,VOCAB+1),dtype=np.float32); gold=np.zeros(m,dtype=np.int64)
    for a,j in enumerate(idxs):
        k=KS[j]; p=mp[a]; gold[a]=k[p]
        for q in range(16):
            if q!=p: Vb[a,k[q]]=1.0
    Vidf=Vb*IDF; sim=Vidf@Vb.T; np.fill_diagonal(sim,0.0); simp=sim**POWER; cf=simp@Vb
    cf[:,0]=-1e9; cf[Vb>0]=-1e9
    for a in range(m):
        s=cf[a]; g=gold[a]; top=np.argpartition(-s,10)[:10]; top=top[np.argsort(-s[top])]
        rank=next((r+1 for r,t in enumerate(top) if t==g),None)
        recs.append((m,1.0/rank if rank else 0.0))
recs=np.array(recs)
print(f"overall CF MRR (train-CV, contig-weighted): {recs[:,1].mean():.4f}")

# bucket by genome size
buckets=[(1,3),(4,8),(9,20),(21,60),(61,150),(151,10000)]
print("\nMRR by genome size bucket (train):")
bucket_mrr={}
for lo,hi in buckets:
    mask=(recs[:,0]>=lo)&(recs[:,0]<=hi)
    if mask.sum(): bucket_mrr[(lo,hi)]=recs[mask,1].mean(); print(f"  size {lo}-{hi}: n_contigs={int(mask.sum())} MRR={recs[mask,1].mean():.4f}")

# test contig distribution by genome size
tek=[genome_key(g) for g in te["gc"]]; tec=collections.Counter(tek)
test_sizes=[tec[k] for k in tek]  # per test contig, its genome size
test_w=collections.Counter()
for sz in test_sizes:
    for lo,hi in buckets:
        if lo<=sz<=hi: test_w[(lo,hi)]+=1; break
tot=sum(test_w.values())
print("\nTEST contig distribution by bucket:", {f"{k[0]}-{k[1]}":round(v/tot,3) for k,v in test_w.items()})
est=sum(bucket_mrr.get(b,0)*test_w.get(b,0) for b in buckets)/tot
print(f"\nTEST-CALIBRATED CF MRR estimate (reweighted): {est:.4f}")
