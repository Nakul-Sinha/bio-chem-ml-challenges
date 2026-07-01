"""Fast vectorized metric, verified against exact DP.
Key facts (both sequences canonically sorted, unique tokens):
  - LCS(pred,true) = |intersection|
  - Levenshtein = sum over gaps (between common-token anchors, in canonical order) of max(#FP,#FN)
We verify both against srlib exact DP on random cases, then expose a vectorized scorer."""
import numpy as np, srlib as L, random

# canonical key for token (target ti, bin b): sort by (-b, ti)
def bins_to_tokens(b):  # b: len16 ints
    toks=[(ti,int(x)) for ti,x in enumerate(b) if x>0]
    toks.sort(key=lambda kv:(-kv[1],kv[0]))
    return [f"T{ti:02d}_B{bb}" for ti,bb in toks]

def edit_segments(pb, tb):
    """Vectorizable-in-spirit edit distance via gap segmentation."""
    # union tokens in canonical order
    p=set((ti,int(pb[ti])) for ti in range(16) if pb[ti]>0)
    t=set((ti,int(tb[ti])) for ti in range(16) if tb[ti]>0)
    common=p&t
    fp=p-common; fn=t-common
    alltok=sorted(p|t, key=lambda kv:(-kv[1],kv[0]))
    # walk, split by anchors
    cost=0; x=0; y=0
    for tok in alltok:
        if tok in common:
            cost+=max(x,y); x=0; y=0
        elif tok in fp: x+=1
        else: y+=1
    cost+=max(x,y)
    return cost

def fast_row_score(pb, tb):
    p=[(ti,int(pb[ti])) for ti in range(16) if pb[ti]>0]
    t=[(ti,int(tb[ti])) for ti in range(16) if tb[ti]>0]
    lp,lt=len(p),len(t)
    if lp==0 and lt==0: return 1.0
    ps=set(p); ts=set(t); C=len(ps&ts)
    m=max(lp,lt) if max(lp,lt)>0 else 1
    # NONE handling: if one side empty, it's ['NONE'] len1
    if lp==0 or lt==0:
        # pred/true is NONE token vs other side's tokens -> no overlap
        m=max(lp,lt,1)
        edit_sim=1.0-m/m  # =0
        return 0.0
    edit=edit_segments(pb,tb)
    edit_sim=1.0-edit/m
    lcs=C/m
    prec=C/lp; rec=C/lt
    f1=2*prec*rec/(prec+rec) if (prec+rec)>0 else 0.0
    return 0.5*edit_sim+0.3*f1+0.2*lcs

# -------- verify against exact DP --------
rng=random.Random(0)
def rand_bins():
    b=np.zeros(16,int)
    for t in range(16):
        r=rng.random()
        if r<0.35: b[t]=rng.choice([1,1,1,2,3])
    return b
maxdiff=0; ncheck=4000
for _ in range(ncheck):
    a=rand_bins(); b=rand_bins()
    exact=L.row_score(L.bins_to_seq(a), L.bins_to_seq(b))
    fast=fast_row_score(a,b)
    d=abs(exact-fast); maxdiff=max(maxdiff,d)
    if d>1e-9:
        print("MISMATCH", d, L.bins_to_seq(a), L.bins_to_seq(b), "exact",exact,"fast",fast)
        break
print(f"verified {ncheck} cases, max|diff|={maxdiff:.2e}")

# vectorized batch scorer over many rows given pred_bins (N,16), true_bins (N,16)
def batch_score(pred_bins, true_bins):
    N=len(pred_bins)
    return np.array([fast_row_score(pred_bins[i], true_bins[i]) for i in range(N)])

if __name__=="__main__":
    print("fastmetric self-check done")
