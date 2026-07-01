"""Nested-CV decode comparison (fast): is the 16-threshold decode overfitting?
Tune decode on inner folds, evaluate on held-out outer fold. Compare 16-thresh vs global
vs param-free expected-count top-k. Report across norms (max/sum/ratio)."""
import numpy as np, pandas as pd, srlib as L
from fastexact import FastScorer
from sklearn.model_selection import GroupKFold
import warnings; warnings.filterwarnings('ignore')

D="../dataset/"; train=pd.read_csv(D+"train.csv"); N=len(train)
Y=np.load("Y.npy"); fs=FastScorer(Y)
gid,gkeys=L.make_groups(train)
gsize=pd.Series(gkeys).value_counts(); small=set(gsize[gsize<=gsize.quantile(0.35)].index)
rare_tokens,_=L.rare_token_set(train,160)
FL=L.subset_flags(train,np.arange(N),rare_tokens,small,gkeys)
STRONG=[5,6,12,13,14]

Os=[np.load(f) for f in ['oof_gbdt_grouped.npy','oof_lr_grouped.npy','oof_mlp_grouped.npy','oof_tlin.npy','oof_tmlp.npy']]
allowed=np.zeros((16,4),bool)
for t in range(16):
    allowed[t,0]=True
    for b in [1,2,3]:
        if (Y[:,t]==b).any(): allowed[t,b]=True
def norm_probs(o):
    o=o.copy()
    for t in range(16): o[:,t,~allowed[t]]=0
    return o/(o.sum(2,keepdims=True)+1e-9)
ens=norm_probs(np.mean(Os,0))
PACT=1-ens[:,:,0]; ARGB=ens[:,:,1:].argmax(2)+1

def bins_full(active_mask):   # active_mask (N,16) bool -> full-N bins
    b=np.where(active_mask,1,0)
    for t in STRONG: b[:,t]=np.where(active_mask[:,t],ARGB[:,t],0)
    return b
def dec_thresh16(tau): return bins_full(PACT>=tau[None,:])
def dec_global(tau):   return bins_full(PACT>=tau)
def dec_topk(alpha):
    k=np.clip(np.rint(alpha*PACT.sum(1)).astype(int),0,16); order=np.argsort(-PACT,axis=1)
    mask=np.zeros_like(PACT,bool)
    for r in range(N): mask[r,order[r,:k[r]]]=True
    return bins_full(mask)

def wscore(rs, idx):  # subset-weighted final over an index set, using precomputed row scores rs
    s=rs[idx]; fl={k:FL[k][idx] for k in ['shifted','damaged','rare']}
    def mn(m): m=np.asarray(m,bool); return float(s[m].mean()) if m.any() else float(s.mean())
    return 0.45*s.mean()+0.25*mn(fl['shifted'])+0.20*mn(fl['damaged'])+0.10*mn(fl['rare'])

def rows_of(bins,norm): return fs.rows(bins,norm)

folds=list(GroupKFold(5).split(np.arange(N),groups=gid))

def tune_global(tr,norm):
    best=(-1,0.1)
    for tau in np.arange(0.05,0.6,0.02):
        r=rows_of(dec_global(tau),norm); f=wscore(r,tr)
        if f>best[0]: best=(f,tau)
    return best[1]
def tune_alpha(tr,norm):
    best=(-1,1.0)
    for a in np.arange(0.55,1.5,0.05):
        r=rows_of(dec_topk(a),norm); f=wscore(r,tr)
        if f>best[0]: best=(f,a)
    return best[1]
def tune_tau16(tr,norm):
    tau=np.full(16,0.15)
    cur=wscore(rows_of(dec_thresh16(tau),norm),tr)
    for _ in range(2):
        for t in range(16):
            bt,bv=tau[t],cur
            for c in np.arange(0.03,0.55,0.03):
                tau[t]=c; f=wscore(rows_of(dec_thresh16(tau),norm),tr)
                if f>bv: bv,bt=f,c
            tau[t]=bt; cur=bv
    return tau

ALL=np.arange(N)
for norm in ['max','sum','ratio']:
    print(f"\n===== NORM={norm} : nested grouped 5-fold (honest) =====")
    hb={k:np.zeros((N,16),int) for k in ['thresh16','global','topk']}
    alphas=[]
    for tr,va in folds:
        hb['thresh16'][va]=dec_thresh16(tune_tau16(tr,norm))[va]
        hb['global'][va]=dec_global(tune_global(tr,norm))[va]
        a=tune_alpha(tr,norm); alphas.append(a); hb['topk'][va]=dec_topk(a)[va]
    for k in ['thresh16','global','topk']:
        r=rows_of(hb[k],norm); f=wscore(r,ALL)
        print(f"  {k:9s} FINAL={f:.4f}  all={r.mean():.4f}")
    # in-sample optimistic 16-thresh
    r=rows_of(dec_thresh16(tune_tau16(ALL,norm)),norm)
    print(f"  {'INSAMP16':9s} FINAL={wscore(r,ALL):.4f}  <-- optimistic (same rows tuned+scored)")
    print(f"  topk alphas per fold: {[round(a,2) for a in alphas]}")
print("\nDONE")
