"""Do target-target co-expression correlations let a 2nd-stage model beat the 0.65 AUC ceiling?
Stage-2: predict target t from [source PCs + OTHER targets' stage-1 OOF prob-active].
Grouped OOF. If AUC rises, co-expression carries exploitable ranking signal."""
import numpy as np, pandas as pd, srlib as L
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
import lightgbm as lgb, warnings; warnings.filterwarnings('ignore')

train=pd.read_csv('../dataset/train.csv'); N=len(train)
Y=np.load('Y.npy'); ACT=(Y>0).astype(int); gid,gkeys=L.make_groups(train)
OV=np.zeros((N,80))
for i,s in enumerate(train.source_sequence):
    ov,_,_,_,_=L.parse_source(s); OV[i]=ov
lg=np.log1p(OV); Z=(lg-lg.mean(0))/(lg.std(0)+1e-6)
PC=PCA(30,random_state=0).fit_transform(Z)

# correlation of target activity
C=np.corrcoef(ACT.T)
offdiag=C[~np.eye(16,dtype=bool)]
print("target-target activity corr: mean|r|=%.3f max|r|=%.3f  (frac |r|>0.15: %.2f)"%(np.abs(offdiag).mean(),np.abs(offdiag).max(),(np.abs(offdiag)>0.15).mean()))

Os=[np.load(f) for f in ['oof_gbdt_grouped.npy','oof_lr_grouped.npy','oof_mlp_grouped.npy','oof_tlin.npy','oof_tmlp.npy']]
ens=np.mean(Os,0); P1=1-ens[:,:,0]   # stage-1 prob active (already grouped OOF)
folds=list(GroupKFold(5).split(np.arange(N),groups=gid))

# stage-1 AUC ref
auc1=np.mean([roc_auc_score(ACT[:,t],P1[:,t]) for t in range(16)])
print("stage-1 ensemble mean-AUC=%.4f"%auc1)

# stage-2: for target t use PCs + other 15 targets' P1
P2=np.zeros((N,16))
for t in range(16):
    others=[j for j in range(16) if j!=t]
    X=np.hstack([PC, P1[:,others], np.log(P1[:,[t]]+1e-6)])  # include own stage1 as anchor
    for tr,va in folds:
        m=lgb.LGBMClassifier(n_estimators=200,learning_rate=0.03,num_leaves=15,min_child_samples=30,
            subsample=0.8,colsample_bytree=0.7,reg_lambda=1.0,verbose=-1)
        m.fit(X[tr],ACT[tr,t]); P2[va,t]=m.predict_proba(X[va])[:,1]
auc2=np.mean([roc_auc_score(ACT[:,t],P2[:,t]) for t in range(16)])
print("stage-2 (PCs + other-target probs) mean-AUC=%.4f  (delta %+.4f)"%(auc2,auc2-auc1))
# blend
for w in [0.3,0.5,0.7]:
    bl=(1-w)*P1+w*P2
    a=np.mean([roc_auc_score(ACT[:,t],bl[:,t]) for t in range(16)])
    print("  blend w=%.1f AUC=%.4f"%(w,a))
# per-target delta
d=[roc_auc_score(ACT[:,t],P2[:,t])-roc_auc_score(ACT[:,t],P1[:,t]) for t in range(16)]
print("per-target AUC delta:", ' '.join('%+.2f'%x for x in d))
