"""Re-tune blend on saved components with deterministic tie-breaking (fixes all-ties artifact)."""
import sys, numpy as np
from pathlib import Path
d=np.load(Path(__file__).resolve().parent/"blend_components.npz")
PMI,COND,NN,GOLD,VIS=d["PMI"],d["COND"],d["NN"],d["GOLD"],d["VIS"]
n,V=PMI.shape; goldcol=GOLD-1
colidx=np.arange(V)
def mrr(w):
    wpmi,wcond,wnn=w
    S=wpmi*PMI+wcond*COND+wnn*NN+VIS
    gs=S[np.arange(n),goldcol]
    greater=(S>gs[:,None]).sum(1)
    ties=((S==gs[:,None])&(colidx[None,:]<goldcol[:,None])).sum(1)
    rank=greater+ties+1
    return np.where(rank<=10,1.0/rank,0.0).mean()
print("references: pmi+cond",round(mrr((1,1,0)),4)," nn",round(mrr((0,0,1)),4)," pmi",round(mrr((1,0,0)),4)," cond",round(mrr((0,1,0)),4))
best=(-1,None)
for wpmi in [0,0.5,1,1.5,2,3]:
  for wcond in [0,0.5,1,1.5,2,3]:
    for wnn in [0,0.5,1,1.5,2,3,4]:
        if wpmi==0 and wcond==0 and wnn==0: continue
        m=mrr((wpmi,wcond,wnn))
        if m>best[0]: best=(m,(wpmi,wcond,wnn))
print(f"BEST blend CV MRR@10={best[0]:.4f} (wpmi,wcond,wnn)={best[1]}")
# finer around best
bw=best[1]; best2=best
import itertools as it
rng=lambda c:[max(0,c-0.5),c-0.25,c,c+0.25,c+0.5,c+1]
for wpmi in rng(bw[0]):
  for wcond in rng(bw[1]):
    for wnn in rng(bw[2]):
        if wpmi<=0 and wcond<=0 and wnn<=0: continue
        m=mrr((round(wpmi,2),round(wcond,2),round(wnn,2)))
        if m>best2[0]: best2=(m,(round(wpmi,2),round(wcond,2),round(wnn,2)))
print(f"FINER BEST CV MRR@10={best2[0]:.4f} (wpmi,wcond,wnn)={best2[1]}")
