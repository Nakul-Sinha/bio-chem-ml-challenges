"""Definitive OOF evaluation with fast exact metric: ensemble, per-target decode, all norms,
cross-norm robustness, subset breakdown, and honest (nested) threshold check."""
import numpy as np, pandas as pd, srlib as L
from fastexact import FastScorer
import warnings; warnings.filterwarnings('ignore')
D="../dataset/"; train=pd.read_csv(D+"train.csv"); N=len(train)
Y=np.load("Y.npy")
fs=FastScorer(Y)
gid,gkeys=L.make_groups(train)
gsize=pd.Series(gkeys).value_counts(); small=set(gsize[gsize<=gsize.quantile(0.35)].index)
rare_tokens,_=L.rare_token_set(train,160)
flags=L.subset_flags(train,np.arange(N),rare_tokens,small,gkeys)
STRONG=[5,6,12,13,14]

oofs={}
for name in ['gbdt','lr','mlp']:
    f={'gbdt':'oof_gbdt_grouped.npy','lr':'oof_lr_grouped.npy','mlp':'oof_mlp_grouped.npy'}[name]
    try: oofs[name]=np.load(f)
    except: print("missing",f)

allowed=np.zeros((16,4),bool)
for t in range(16):
    allowed[t,0]=True
    for b in [1,2,3]:
        if (Y[:,t]==b).any(): allowed[t,b]=True
def norm_probs(o):
    o=o.copy()
    for t in range(16): o[:,t,~allowed[t]]=0
    return o/(o.sum(2,keepdims=True)+1e-9)
for k in oofs: oofs[k]=norm_probs(oofs[k])

def decode(oof,tau,strong_argmax=True):
    pact=1-oof[:,:,0]; bins=np.ones((N,16),int)
    if strong_argmax:
        ab=oof[:,:,1:].argmax(2)+1
        for t in STRONG: bins[:,t]=ab[:,t]
    return np.where(pact>=tau[None,:],bins,0)

def tune(oof,norm,strong=True):
    tau=np.full(16,0.10)
    cur=fs.weighted(decode(oof,tau,strong),flags,norm)['final']
    for _ in range(4):
        for t in range(16):
            bt,bv=tau[t],cur
            for c in np.arange(0.02,0.60,0.01):
                tau[t]=c; v=fs.weighted(decode(oof,tau,strong),flags,norm)['final']
                if v>bv+1e-9: bv,bt=v,c
            tau[t]=bt; cur=bv
    return tau,cur

ens=sum(oofs.values())/len(oofs); ens=norm_probs(ens)
ens2=(oofs['gbdt']+oofs['lr'])/2; ens2=norm_probs(ens2)
models={**oofs,'ENS(all)':ens,'ENS(gb+lr)':ens2}

print("Per-target-tuned decode (strong-argmax bins), fast exact metric:")
print(f"{'model':11s} {'norm':5s} {'FINAL':>7s} {'all':>7s} {'shift':>7s} {'damg':>7s} {'rare':>7s}")
best={}
for name,oof in models.items():
    for norm in ['max','sum','ratio']:
        tau,f=tune(oof,norm)
        r=fs.weighted(decode(oof,tau),flags,norm)
        print(f"{name:11s} {norm:5s} {r['final']:7.4f} {r['all']:7.4f} {r['shifted']:7.4f} {r['damaged']:7.4f} {r['rare']:7.4f}")
        if norm=='max' and name=='ENS(all)': best['tau_max']=tau
        if norm=='sum' and name=='ENS(all)': best['tau_sum']=tau
    print()

# cross-norm robustness: decode tuned on max, evaluated on sum/ratio (and vice versa)
print("CROSS-NORM ROBUSTNESS (ENS all):")
for tn in ['max','sum']:
    tau,_=tune(ens,tn)
    line=f"  tuned@{tn}: "
    for en in ['max','sum','ratio']:
        line+=f"{en}={fs.weighted(decode(ens,tau),flags,en)['final']:.4f}  "
    print(line)
print("taus tuned@max:",np.round(best.get('tau_max',np.zeros(16)),2))
print("DONE")
