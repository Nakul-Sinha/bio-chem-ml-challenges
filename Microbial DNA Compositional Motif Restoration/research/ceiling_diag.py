"""Diagnose: recall ceiling (gold in sibling union) + per-bucket achievable MRR."""
import sys, numpy as np, collections
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from common import load, genome_key, group_folds, VOCAB
tr,te=load()
keys=[genome_key(g) for g in tr["gc"]]
KS=[list(r) for r in tr["kseq"]]; GCs=[set(r) for r in tr["gc"]]
by=collections.defaultdict(list)
for i,k in enumerate(keys): by[k].append(i)
MASKPOS=list(range(4,16))
rng=np.random.default_rng(0)

in_gc=0; in_sib=0; in_sib_notgc=0; tot=0
sib_counts=[]
for gk,idxs in by.items():
    # union of tokens per contig
    for i in idxs:
        sibset=set()
        for j in idxs:
            if j!=i: sibset.update(KS[j])
        for p in MASKPOS:
            g=KS[i][p]; tot+=1
            in_gc+= g in GCs[i]
            in_sib+= g in sibset
            in_sib_notgc+= (g in sibset) and (g not in GCs[i])
print(f"contigs/genome: mean {np.mean([len(v) for v in by.values()]):.1f}")
print(f"gold in genome_context: {in_gc/tot:.3f}")
print(f"gold in sibling-union:  {in_sib/tot:.3f}   <-- RECALL CEILING")
print(f"gold in sibling-union but NOT gc: {in_sib_notgc/tot:.3f}")

# Among sibling-union candidates (excl visible), rank gold by sibling FREQUENCY. Achievable MRR.
def mrr_by_sibfreq():
    rr=[]
    for gk,idxs in by.items():
        for i in idxs:
            sibfreq=collections.Counter()
            for j in idxs:
                if j!=i:
                    for t in KS[j]: sibfreq[t]+=1
            for p in MASKPOS:
                g=KS[i][p]; vis=set(KS[i][q] for q in range(16) if q!=p)
                cand=[(c,t) for t,c in sibfreq.items() if t not in vis]
                cand.sort(reverse=True)
                rank=None
                for r,(c,t) in enumerate(cand[:10]):
                    if t==g: rank=r+1; break
                rr.append(1.0/rank if rank else 0.0)
    return np.mean(rr)
print(f"\nMRR@10 ranking sibling-union by sibling-frequency: {mrr_by_sibfreq():.4f}")
