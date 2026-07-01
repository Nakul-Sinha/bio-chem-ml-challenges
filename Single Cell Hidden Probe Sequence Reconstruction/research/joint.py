"""Does JOINT structure exist? Upper-bound the value of modeling target-target dependence.
If LR(source + OTHER TRUE targets) >> LR(source alone), joint modeling is the breakthrough."""
import numpy as np, pandas as pd, srlib as L
from sklearn.model_selection import GroupKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
D="../dataset/"; train=pd.read_csv(D+"train.csv"); N=len(train)
def parse_all(df):
    OV=np.zeros((len(df),80),np.float32);TOT=np.zeros(len(df),np.float32);NZ=np.zeros(len(df),np.float32)
    for i,s in enumerate(df['source_sequence'].values):
        ov,tot,nz,panel,meta=L.parse_source(s); OV[i]=ov;TOT[i]=tot;NZ[i]=nz
    return OV,TOT,NZ
OV,TOT,NZ=parse_all(train)
Y=np.stack([L.parse_target(s) for s in train['target_sequence']]).astype(np.int64)
active=(Y>0).astype(int); N=len(train)
gid,_=L.make_groups(train)
mu=OV.mean(0);sd=OV.std(0)+1e-6
Xs=np.concatenate([(OV-mu)/sd,(OV>0).astype(np.float32),(TOT/63.)[:,None],(NZ/60.)[:,None]],axis=1).astype(np.float32)

# 1) target-target correlation of active indicators
print("Target-target ACTIVE correlation (off-diagonal |corr| summary):")
C=np.corrcoef(active.T)
off=C[~np.eye(16,dtype=bool)]
print(f"  mean|corr|={np.abs(off).mean():.3f} max|corr|={np.abs(off).max():.3f}  frac|corr|>0.2={ (np.abs(off)>0.2).mean():.2f}")
# also correlation with total active count
tc=active.sum(1)
print("  corr(target_active, row_total_count): mean=%.3f"%np.mean([np.corrcoef(active[:,t],tc)[0,1] for t in range(16)]))

# 2) AUC: source alone vs source + OTHER TRUE targets (oracle upper bound)
def auc_source_only():
    pact=np.zeros((N,16))
    for tr,va in GroupKFold(5).split(Xs,groups=gid):
        for t in range(16):
            c=LogisticRegression(max_iter=300,C=1.0); c.fit(Xs[tr],active[tr,t]); pact[va,t]=c.predict_proba(Xs[va])[:,1]
    return np.array([roc_auc_score(active[:,t],pact[:,t]) for t in range(16)])
def auc_source_plus_others():
    pact=np.zeros((N,16))
    for tr,va in GroupKFold(5).split(Xs,groups=gid):
        for t in range(16):
            others=np.delete(active,t,axis=1).astype(np.float32)
            Xtr=np.concatenate([Xs[tr],others[tr]],1); Xva=np.concatenate([Xs[va],others[va]],1)
            c=LogisticRegression(max_iter=300,C=1.0); c.fit(Xtr,active[tr,t]); pact[va,t]=c.predict_proba(Xva)[:,1]
    return np.array([roc_auc_score(active[:,t],pact[:,t]) for t in range(16)])
a0=auc_source_only(); a1=auc_source_plus_others()
print("\nAUC source-only vs source+OTHER-TRUE-targets (oracle):")
for t in range(16):
    print(f"  T{t:02d}: source={a0[t]:.3f}  +others={a1[t]:.3f}  lift={a1[t]-a0[t]:+.3f}")
print(f"  MEAN: source={a0.mean():.3f}  +others={a1.mean():.3f}  lift={a1.mean()-a0.mean():+.3f}")
print("\n=> If lift is large, targets carry mutual info beyond source (joint modeling headroom).")
print("   If lift ~0, targets are conditionally independent given source; 0.64 is the ceiling.")
print("DONE")
