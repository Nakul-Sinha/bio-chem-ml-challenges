"""Test whether unsupervised expression-module features (NMF/PCA on train+test) lift active AUC.
Uses test.csv expression too (unlabeled -> allowed)."""
import numpy as np, pandas as pd, srlib as L
from sklearn.model_selection import GroupKFold
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import NMF, PCA
from sklearn.metrics import roc_auc_score
D="../dataset/"; train=pd.read_csv(D+"train.csv"); test=pd.read_csv(D+"test.csv")
def O(df):
    OV=np.zeros((len(df),80),np.float32)
    for i,s in enumerate(df.source_sequence.values):
        ov,_,_,_,_=L.parse_source(s); OV[i]=ov
    return OV
OVtr=O(train); OVte=O(test); N=len(train)
Y=np.stack([L.parse_target(s) for s in train.target_sequence]).astype(np.int64); active=(Y>0).astype(int)
gid,_=L.make_groups(train)
mu=OVtr.mean(0);sd=OVtr.std(0)+1e-6
base=np.concatenate([(OVtr-mu)/sd,(OVtr>0).astype(np.float32)],1)

# module features from train+test
allO=np.vstack([OVtr,OVte]).astype(np.float32)
def auc_of(X):
    pact=np.zeros((N,16))
    for tr,va in GroupKFold(5).split(X,groups=gid):
        for t in range(16):
            c=LogisticRegression(max_iter=300,C=1.0); c.fit(X[tr],active[tr,t]); pact[va,t]=c.predict_proba(X[va])[:,1]
    return np.array([roc_auc_score(active[:,t],pact[:,t]) for t in range(16)])

print("base meanAUC=%.3f"%auc_of(base).mean())
for k in [16,32]:
    nmf=NMF(n_components=k,init='nndsvda',max_iter=300,random_state=0)
    W=nmf.fit_transform(np.log1p(allO)); Wtr=W[:N]
    Wz=(Wtr-Wtr.mean(0))/(Wtr.std(0)+1e-6)
    a=auc_of(np.concatenate([base,Wz],1))
    print(f"NMF k={k}: meanAUC={a.mean():.3f}")
    pca=PCA(n_components=k,random_state=0); P=pca.fit_transform((np.log1p(allO)-np.log1p(allO).mean(0)))
    Ptr=P[:N]; Pz=(Ptr-Ptr.mean(0))/(Ptr.std(0)+1e-6)
    a=auc_of(np.concatenate([base,Pz],1))
    print(f"PCA k={k}: meanAUC={a.mean():.3f}")
print("DONE")
