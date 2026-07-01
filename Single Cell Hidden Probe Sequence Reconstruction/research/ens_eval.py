"""Ensemble saved OOFs, tune per-target decode, report weighted score under all metric norms."""
import numpy as np, pandas as pd, srlib as L
D="../dataset/"; train=pd.read_csv(D+"train.csv"); N=len(train)
Y=np.load("Y.npy")
oofs={'gbdt':np.load("oof_gbdt_grouped.npy"),'lr':np.load("oof_lr_grouped.npy")}
gid,gkeys=L.make_groups(train)
gsize=pd.Series(gkeys).value_counts(); small=set(gsize[gsize<=gsize.quantile(0.35)].index)
rare_tokens,_=L.rare_token_set(train,160)
flags=L.subset_flags(train,np.arange(N),rare_tokens,small,gkeys)
true_seqs=[L.target_tokens(s) for s in train['target_sequence']]
STRONG=[5,6,12,13,14]

def wscore(pb,norm):
    ps=[L.bins_to_seq(pb[i]) for i in range(N)]
    return L.weighted_score(ps,true_seqs,flags,norm)

def decode(oof,tau,use_argmax_strong=True):
    pact=1-oof[:,:,0]
    bins=np.ones((N,16),int)
    if use_argmax_strong:
        ab=oof[:,:,1:].argmax(2)+1
        for t in STRONG: bins[:,t]=ab[:,t]
    return np.where(pact>=tau[None,:],bins,0)

def tune_pertarget(oof,norm):
    tau=np.full(16,0.08)
    cur=wscore(decode(oof,tau),norm)['final']
    for _ in range(3):
        for t in range(16):
            bt,bv=tau[t],cur
            for c in np.arange(0.02,0.55,0.02):
                tau[t]=c; v=wscore(decode(oof,tau),norm)['final']
                if v>bv: bv,bt=v,c
            tau[t]=bt; cur=bv
    return tau,cur

# build ensemble
ens=sum(oofs.values())/len(oofs)
# renormalize over allowed bins (some targets have no B2)
allowed=np.zeros((16,4),bool)
for t in range(16):
    allowed[t,0]=True
    for b in [1,2,3]:
        if (Y[:,t]==b).any(): allowed[t,b]=True
for t in range(16):
    ens[:,t,~allowed[t]]=0
ens/=ens.sum(2,keepdims=True)+1e-9

print("Per-model + ensemble, per-target-tuned decode, under 3 metric norms")
print(f"{'model':10s} {'norm':6s} {'FINAL':>7s} {'all':>7s} {'shift':>7s} {'damg':>7s} {'rare':>7s}")
for name,oof in list(oofs.items())+[('ENSEMBLE',ens)]:
    for norm in ['max','sum','ratio']:
        tau,f=tune_pertarget(oof,norm)
        r=wscore(decode(oof,tau),norm)
        print(f"{name:10s} {norm:6s} {r['final']:7.4f} {r['all']:7.4f} {r['shifted']:7.4f} {r['damaged']:7.4f} {r['rare']:7.4f}")
    print()
print("DONE")
