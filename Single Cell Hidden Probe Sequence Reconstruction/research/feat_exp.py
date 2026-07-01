"""Does better feature engineering raise per-target AUC above 0.654?
Test feature sets with LightGBM per-target, grouped 5-fold OOF. Report mean active-AUC
and ratio-norm weighted CV with a recall-favoring decode."""
import numpy as np, pandas as pd, srlib as L
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score
import lightgbm as lgb, warnings; warnings.filterwarnings('ignore')

train=pd.read_csv('../dataset/train.csv'); N=len(train)
Y=np.load('Y.npy'); ACT=(Y>0).astype(int)
gid,gkeys=L.make_groups(train)

# ---- parse sources ----
OV=np.zeros((N,80)); TOT=np.zeros(N); NZ=np.zeros(N); PANEL=[]; META=[]
for i,s in enumerate(train.source_sequence):
    ov,tot,nz,panel,meta=L.parse_source(s); OV[i]=ov; TOT[i]=tot; NZ[i]=nz; PANEL.append(panel); META.append(meta)
def onehot(vals):
    u=sorted(set(vals)); idx={v:j for j,v in enumerate(u)}
    M=np.zeros((N,len(u)))
    for i,v in enumerate(vals): M[i,idx[v]]=1
    return M
COND=onehot([m.get('COND','?') for m in META]); SEX=onehot([m.get('SEX','?') for m in META])
STAGE=onehot([m.get('STAGE','?') for m in META]); DMG=onehot([p for p in PANEL])
metaF=np.hstack([COND,SEX,STAGE,DMG,TOT[:,None]/31,NZ[:,None]/80])

# feature sets
def libnorm(ov,tot):
    lib=ov.sum(1,keepdims=True)+1e-6
    rel=ov/lib                       # relative abundance
    logo=np.log1p(ov)
    z=(logo-logo.mean(0))/(logo.std(0)+1e-6)
    return np.hstack([ov, logo, rel*100, z])
FSETS={
 'raw80+meta': np.hstack([OV, metaF]),
 'libnorm+meta': np.hstack([libnorm(OV,TOT), metaF]),
}
# add PCA-augmented
from sklearn.decomposition import PCA
lg=np.log1p(OV); lgz=(lg-lg.mean(0))/(lg.std(0)+1e-6)
pca=PCA(20,random_state=0).fit_transform(lgz)
FSETS['libnorm+meta+pca']=np.hstack([libnorm(OV,TOT), metaF, pca])

folds=list(GroupKFold(5).split(np.arange(N),groups=gid))
def oof_lgb(X):
    P=np.zeros((N,16))
    for t in range(16):
        for tr,va in folds:
            m=lgb.LGBMClassifier(n_estimators=200,learning_rate=0.03,num_leaves=15,
                subsample=0.8,colsample_bytree=0.7,min_child_samples=30,reg_lambda=1.0,verbose=-1)
            m.fit(X[tr],ACT[tr,t]); P[va,t]=m.predict_proba(X[va])[:,1]
    return P

# recall decode + ratio weighted score
gsize=pd.Series(gkeys).value_counts(); small=set(gsize[gsize<=gsize.quantile(0.35)].index)
rare_tokens,_=L.rare_token_set(train,160); FL=L.subset_flags(train,np.arange(N),rare_tokens,small,gkeys)
from fastexact import FastScorer; fs=FastScorer(Y)
def wf(rs,idx=None):
    idx=np.arange(N) if idx is None else idx; s=rs[idx]
    def mn(m): m=np.asarray(m,bool)[idx]; return float(rs[idx][m].mean()) if m.any() else float(s.mean())
    return 0.45*s.mean()+0.25*mn(FL['shifted'])+0.20*mn(FL['damaged'])+0.10*mn(FL['rare'])
def decode_topk_by_thresh(P,tau):
    return np.where(P>=tau,1,0)   # all B1 (modal); bins refined elsewhere
def best_ratio(P):
    # single global threshold tuned for ratio (in-sample, quick proxy)
    best=(-1,0.5,0)
    for tau in np.arange(0.15,0.6,0.02):
        b=decode_topk_by_thresh(P,tau); f=wf(fs.rows(b,'ratio'))
        if f>best[0]: best=(f,tau,(b>0).sum(1).mean())
    return best

for name,X in FSETS.items():
    P=oof_lgb(X)
    aucs=[roc_auc_score(ACT[:,t],P[:,t]) for t in range(16)]
    f,tau,tok=best_ratio(P)
    print('%-20s mean-AUC=%.4f  ratioCV(global-tau)=%.4f  @tau=%.2f tok/row=%.1f'%(name,np.mean(aucs),f,tau,tok))
print('reference: prior ensemble mean-AUC=0.654')
