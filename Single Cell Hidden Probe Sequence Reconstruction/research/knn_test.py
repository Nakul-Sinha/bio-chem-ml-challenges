"""Does retrieval (kNN in expression space) add ranking signal beyond the parametric ensemble?
Grouped OOF kNN: predict per-target activity = distance-weighted neighbor activity (neighbors
from OTHER groups only, to respect leakage). Measure AUC alone and blended with ensemble."""
import numpy as np, pandas as pd, srlib as L
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score
from fastexact import FastScorer
import warnings; warnings.filterwarnings('ignore')

train=pd.read_csv('../dataset/train.csv'); N=len(train)
Y=np.load('Y.npy'); ACT=(Y>0).astype(float); gid,gkeys=L.make_groups(train)
OV=np.zeros((N,80))
for i,s in enumerate(train.source_sequence):
    ov,_,_,_,_=L.parse_source(s); OV[i]=ov
# normalized log-expression for similarity
lg=np.log1p(OV); Z=(lg-lg.mean(0))/(lg.std(0)+1e-6)
Zn=Z/ (np.linalg.norm(Z,axis=1,keepdims=True)+1e-9)
S=Zn@Zn.T   # cosine similarity NxN
np.fill_diagonal(S,-1)

folds=list(GroupKFold(5).split(np.arange(N),groups=gid))
def knn_oof(K,temp):
    P=np.zeros((N,16))
    for tr,va in folds:
        trset=set(tr.tolist())
        for i in va:
            sims=S[i].copy()
            mask=np.zeros(N,bool); mask[tr]=True
            sims=np.where(mask,sims,-1)
            idx=np.argpartition(-sims,K)[:K]
            w=np.exp(sims[idx]/temp); w=w/(w.sum()+1e-9)
            P[i]=w@ACT[idx]
    return P

Os=[np.load(f) for f in ['oof_gbdt_grouped.npy','oof_lr_grouped.npy','oof_mlp_grouped.npy','oof_tlin.npy','oof_tmlp.npy']]
ens=np.mean(Os,0); ens_pact=1-ens[:,:,0]
gsize=pd.Series(gkeys).value_counts(); small=set(gsize[gsize<=gsize.quantile(0.35)].index)
rare_tokens,_=L.rare_token_set(train,160); FL=L.subset_flags(train,np.arange(N),rare_tokens,small,gkeys)
fs=FastScorer(Y); ALL=np.arange(N)
BIN=[1,4,13,14]; ARGB=ens[:,:,1:].argmax(2)+1
def wf(rs,idx):
    s=rs[idx]
    def mn(m): m=np.asarray(m,bool)[idx]; return float(rs[idx][m].mean()) if m.any() else float(s.mean())
    return 0.45*s.mean()+0.25*mn(FL['shifted'])+0.20*mn(FL['damaged'])+0.10*mn(FL['rare'])
def mk(pact,tau):
    b=np.where(pact>=tau,1,0)
    for t in BIN: b[:,t]=np.where(pact[:,t]>=tau,ARGB[:,t],0)
    return b
def best(pact):  # in-sample global tau on mean(max,ratio) — quick proxy for ranking quality
    bb=(-1,0,0,0)
    for tau in np.arange(0.05,0.6,0.01):
        b=mk(pact,tau); f=0.5*(wf(fs.rows(b,'max'),ALL)+wf(fs.rows(b,'ratio'),ALL))
        r=wf(fs.rows(b,'ratio'),ALL)
        if f>bb[0]: bb=(f,tau,r,(b>0).sum(1).mean())
    return bb
base_auc=np.mean([roc_auc_score(ACT[:,t],ens_pact[:,t]) for t in range(16)])
print('ensemble alone: mean-AUC=%.4f  ratio=%.4f'%(base_auc,best(ens_pact)[2]))
for K in [20,40,80]:
    for temp in [0.1,0.2]:
        Pk=knn_oof(K,temp)
        auc=np.mean([roc_auc_score(ACT[:,t],Pk[:,t]) for t in range(16)])
        # blend
        for w in [0.2,0.35,0.5]:
            bl=(1-w)*ens_pact+w*Pk
            bauc=np.mean([roc_auc_score(ACT[:,t],bl[:,t]) for t in range(16)])
            r=best(bl)[2]
            print('  K=%3d temp=%.1f knnAUC=%.4f | blend w=%.2f AUC=%.4f ratio=%.4f'%(K,temp,auc,w,bauc,r))
