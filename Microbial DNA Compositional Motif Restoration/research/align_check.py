import sys, numpy as np, pandas as pd, collections
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from common import load
tr,te=load()
# For each contig position p (4..15), if gold in gc, what's its gc rank?
rows=[]
gcrank_given_p=collections.defaultdict(list)
incnt=0; tot=0
for _,r in tr.iterrows():
    k=r["kseq"]; gc=r["gc"]; gcpos={t:i for i,t in enumerate(gc)}
    for p in range(4,16):
        t=k[p]; tot+=1
        if t in gcpos: incnt+=1; gcrank_given_p[p].append(gcpos[t])
print(f"gold-in-gc rate (pos4-15): {incnt/tot:.3f}")
print("\nmedian gc-rank of gold given contig position p (when in gc):")
for p in range(4,16):
    v=gcrank_given_p[p]
    if v: print(f"  p={p:2d}: n={len(v):4d} gc-rank median={np.median(v):.0f} mean={np.mean(v):.1f} within±2 of p:{np.mean([abs(x-p)<=2 for x in v]):.2f}")

# Strategy test: for in-gc cases, rank gc tokens (excl visible) by |gcrank-p|. MRR if always correct ordering?
# Simulate: predict candidates = gc tokens not in visible, ordered by |gcrank - p|, then fill.
from common import mrr10
gold=[]; ranked=[]
allk=collections.Counter(x for l in tr["kseq"] for x in l)
glob=[t for t,_ in allk.most_common(30)]
for _,r in tr.iterrows():
    k=r["kseq"]; gc=r["gc"]
    for p in range(4,16):
        g=k[p]; vis=set(k[i] for i in range(16) if i!=p)
        cand=[(abs(i-p),t) for i,t in enumerate(gc) if t not in vis]
        cand.sort()
        pred=[t for _,t in cand]
        for t in glob:
            if t not in vis and t not in pred: pred.append(t)
        gold.append(g); ranked.append(pred[:10])
print(f"\n[oracle-free] gc-by-|rank-p| + global fill MRR@10 (train, in-sample): {mrr10(gold,ranked):.4f}")
