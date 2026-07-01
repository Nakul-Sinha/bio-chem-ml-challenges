"""Are bins predictable from source? For each target, among ACTIVE rows, predict the bin.
If B3-vs-B1 (and B2) carries signal, exact-token F1 / ratio-inter gets a real lift beyond
always-B1. Report per-target bin AUC and accuracy vs the modal-B1 baseline."""
import numpy as np, pandas as pd, srlib as L
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score
import lightgbm as lgb, warnings; warnings.filterwarnings('ignore')

train=pd.read_csv('../dataset/train.csv'); N=len(train)
Y=np.load('Y.npy'); gid,gkeys=L.make_groups(train)
OV=np.zeros((N,80)); TOT=np.zeros(N); NZ=np.zeros(N); META=[]
for i,s in enumerate(train.source_sequence):
    ov,tot,nz,panel,meta=L.parse_source(s); OV[i]=ov; TOT[i]=tot; NZ[i]=nz; meta['PANEL']=panel; META.append(meta)
def onehot(vals):
    u=sorted(set(vals)); idx={v:j for j,v in enumerate(u)}; M=np.zeros((N,len(u)))
    for i,v in enumerate(vals): M[i,idx[v]]=1
    return M
metaF=np.hstack([onehot([m.get('COND','?') for m in META]),onehot([m.get('SEX','?') for m in META]),
                 onehot([m.get('STAGE','?') for m in META]),onehot([m['PANEL'] for m in META]),TOT[:,None]/31,NZ[:,None]/80])
X=np.hstack([OV,np.log1p(OV),metaF])
folds=list(GroupKFold(5).split(np.arange(N),groups=gid))

print("per-target BIN predictability among active rows (grouped OOF):")
print("  T##  n_act  %B1  %B2  %B3 | modalACC  B3vsB1-AUC  hi-bin-AUC")
gain_targets=[]
for t in range(16):
    act=Y[:,t]>0; ya=Y[act,t]
    b1=(ya==1).mean(); b2=(ya==2).mean(); b3=(ya==3).mean()
    modal_acc=max(b1,b2,b3)
    # OOF prob of "high bin" (>=2) among active rows only
    hi=(Y[:,t]>=2).astype(int)   # among active, is it B2/B3
    P=np.full(N,np.nan)
    for tr,va in folds:
        tra=tr[Y[tr,t]>0]  # train only on active rows
        if hi[tra].sum()<5 or (1-hi[tra]).sum()<5:
            continue
        m=lgb.LGBMClassifier(n_estimators=150,learning_rate=0.03,num_leaves=15,min_child_samples=20,
            subsample=0.8,colsample_bytree=0.7,reg_lambda=1.0,verbose=-1)
        m.fit(X[tra],hi[tra])
        vaa=va[Y[va,t]>0];
        if len(vaa): P[vaa]=m.predict_proba(X[vaa])[:,1]
    mask=~np.isnan(P) & act
    if mask.sum()>20 and len(set(hi[mask]))>1:
        auc=roc_auc_score(hi[mask],P[mask])
    else:
        auc=float('nan')
    # B3 vs B1 only (drop B2)
    m31=act & (np.isin(Y[:,t],[1,3])) & ~np.isnan(P)
    if m31.sum()>20 and len(set((Y[m31,t]==3).astype(int)))>1:
        auc31=roc_auc_score((Y[m31,t]==3).astype(int),P[m31])
    else: auc31=float('nan')
    flag='  <-- signal' if (auc31==auc31 and auc31>0.60) else ''
    print("  T%02d  %4d  %.2f %.2f %.2f |  %.3f    %.3f      %.3f%s"%(t,act.sum(),b1,b2,b3,modal_acc,auc31,auc,flag))
    if auc31==auc31 and auc31>0.60: gain_targets.append(t)
print("targets with predictable high-bin (AUC>0.60):",gain_targets)
