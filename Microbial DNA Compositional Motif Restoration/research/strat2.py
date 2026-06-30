"""CF power sweep on LARGE genomes (test-dominant) + check very-large genome MRR."""
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

def cf_mrr_for_genomes(gks, power, rng):
    rr=[]
    for gk in gks:
        idxs=by[gk]; m=len(idxs); mp=rng.choice(MASKPOS,size=m)
        Vb=np.zeros((m,VOCAB+1),dtype=np.float32); gold=np.zeros(m,dtype=np.int64)
        for a,j in enumerate(idxs):
            k=KS[j]; p=mp[a]; gold[a]=k[p]
            for q in range(16):
                if q!=p: Vb[a,k[q]]=1.0
        Vidf=Vb*IDF; sim=Vidf@Vb.T; np.fill_diagonal(sim,0.0); simp=sim**power; cf=simp@Vb
        cf[:,0]=-1e9; cf[Vb>0]=-1e9
        for a in range(m):
            s=cf[a]; g=gold[a]; top=np.argpartition(-s,10)[:10]; top=top[np.argsort(-s[top])]
            rank=next((r+1 for r,t in enumerate(top) if t==g),None); rr.append(1.0/rank if rank else 0.0)
    return np.mean(rr)

sizes={gk:len(idxs) for gk,idxs in by.items()}
large=[gk for gk,s in sizes.items() if s>=100]
vlarge=[gk for gk,s in sizes.items() if s>=200]
print(f"large(>=100) genomes: {len(large)} ({sum(sizes[g] for g in large)} contigs)")
print(f"vlarge(>=200) genomes: {len(vlarge)} ({sum(sizes[g] for g in vlarge)} contigs)")
print("test largest genome sizes: 321,298,286,202,174,165 ...")
rng=np.random.default_rng(100)
print("\nCF power sweep on LARGE(>=100) genomes:")
for power in [4,6,8,10,12,16,20]:
    rng=np.random.default_rng(100)
    print(f"  power={power}: MRR={cf_mrr_for_genomes(large,power,rng):.4f}")
print("\nCF power sweep on VLARGE(>=200):")
for power in [4,8,12,16]:
    rng=np.random.default_rng(100)
    print(f"  power={power}: MRR={cf_mrr_for_genomes(vlarge,power,rng):.4f}")
