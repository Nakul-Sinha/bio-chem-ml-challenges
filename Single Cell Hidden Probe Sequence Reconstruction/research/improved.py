"""Combined improvement measured on HONEST nested grouped CV:
 (1) recall-favoring decode tuned on mean(max,ratio) [avoids 'sum' under-prediction],
 (2) bin argmax only for predictable-bin targets {1,4,13,14}, B1 elsewhere,
 (3) per-target vs global threshold.
Report vs current shipped baseline under max & ratio norms."""
import numpy as np, pandas as pd, srlib as L
from fastexact import FastScorer
from sklearn.model_selection import GroupKFold
import warnings; warnings.filterwarnings('ignore')

train=pd.read_csv('../dataset/train.csv'); N=len(train)
Y=np.load('Y.npy'); fs=FastScorer(Y)
gid,gkeys=L.make_groups(train)
gsize=pd.Series(gkeys).value_counts(); small=set(gsize[gsize<=gsize.quantile(0.35)].index)
rare_tokens,_=L.rare_token_set(train,160); FL=L.subset_flags(train,np.arange(N),rare_tokens,small,gkeys)

Os=[np.load(f) for f in ['oof_gbdt_grouped.npy','oof_lr_grouped.npy','oof_mlp_grouped.npy','oof_tlin.npy','oof_tmlp.npy']]
allowed=np.zeros((16,4),bool)
for t in range(16):
    allowed[t,0]=True
    for b in [1,2,3]:
        if (Y[:,t]==b).any(): allowed[t,b]=True
def npb(o):
    o=o.copy()
    for t in range(16): o[:,t,~allowed[t]]=0
    return o/(o.sum(2,keepdims=True)+1e-9)
ens=npb(np.mean(Os,0)); PACT=1-ens[:,:,0]; ARGB=ens[:,:,1:].argmax(2)+1

BIN_TARGETS=[1,4,13,14]           # predictable bins (from bin_exp)
OLD_STRONG=[5,6,12,13,14]
def make_bins(active_mask, bin_set):
    b=np.where(active_mask,1,0)
    for t in bin_set: b[:,t]=np.where(active_mask[:,t],ARGB[:,t],0)
    return b

def wf(rs,idx):
    s=rs[idx]
    def mn(m): m=np.asarray(m,bool)[idx]; return float(rs[idx][m].mean()) if m.any() else float(s.mean())
    return 0.45*s.mean()+0.25*mn(FL['shifted'])+0.20*mn(FL['damaged'])+0.10*mn(FL['rare'])

folds=list(GroupKFold(5).split(np.arange(N),groups=gid))
OBJ=lambda pb,idx: 0.5*(wf(fs.rows(pb,'max'),idx)+wf(fs.rows(pb,'ratio'),idx))  # robust obj

def tune_global(tr,bin_set):
    best=(-1,0.2)
    for tau in np.arange(0.05,0.55,0.01):
        pb=make_bins(PACT>=tau,bin_set); f=OBJ(pb,tr)
        if f>best[0]: best=(f,tau)
    return best[1]
def tune_tau16(tr,bin_set):
    tau=np.full(16,0.15); cur=OBJ(make_bins(PACT>=tau[None,:],bin_set),tr)
    for _ in range(2):
        for t in range(16):
            bt,bv=tau[t],cur
            for c in np.arange(0.03,0.5,0.03):
                tau[t]=c; f=OBJ(make_bins(PACT>=tau[None,:],bin_set),tr)
                if f>bv: bv,bt=f,c
            tau[t]=bt; cur=bv
    return tau

ALL=np.arange(N)
configs=[('OLD strong{5,6,12,13,14} global', OLD_STRONG,'global'),
         ('NEW bins{1,4,13,14} global',       BIN_TARGETS,'global'),
         ('NEW bins{1,4,13,14} tau16',         BIN_TARGETS,'tau16'),
         ('no-bins allB1 global',              [],'global')]
print("honest nested grouped 5-fold (tuned on inner, scored on held-out):")
print(f"  {'config':34s} {'max':>7s} {'ratio':>7s} {'tok/row':>8s}")
for name,bset,mode in configs:
    hb=np.zeros((N,16),int)
    for tr,va in folds:
        if mode=='global': tau=tune_global(tr,bset); hb[va]=make_bins(PACT>=tau,bset)[va]
        else: tau=tune_tau16(tr,bset); hb[va]=make_bins(PACT>=tau[None,:],bset)[va]
    mx=wf(fs.rows(hb,'max'),ALL); ra=wf(fs.rows(hb,'ratio'),ALL); tok=(hb>0).sum(1).mean()
    print(f"  {name:34s} {mx:7.4f} {ra:7.4f} {tok:8.2f}")
print("\nshipped baseline (ref): max~0.252 ratio~0.322 tok~9.85")
