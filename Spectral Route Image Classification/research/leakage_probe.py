"""Quantify the same-source leakage: the spec says train = MULTIPLE degraded views per
source, test = HELD-OUT sources (one view). So random/stratified CV puts same-source views
on both sides -> inflated. Here: frozen-feature logreg, random CV vs source-grouped CV
(single-linkage on cosine sim at several thresholds), to see how far OOF falls toward the
real ~0.537.
"""
import os, numpy as np, pandas as pd, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from metric import final_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, GroupKFold, cross_val_predict
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
HERE=os.path.dirname(os.path.abspath(__file__)); DS=os.path.abspath(os.path.join(HERE,"..","dataset"))
tr=pd.read_csv(os.path.join(DS,"train.csv")); y=tr["target_id"].values
stress=(tr["sensor_noise_score"].values>=0.5757).astype(int)
E=np.load(os.path.join(HERE,"emb_train.npy")); E=E/(np.linalg.norm(E,axis=1,keepdims=True)+1e-8)
S=E@E.T
def logreg_oof(splitter, groups=None):
    P=cross_val_predict(LogisticRegression(max_iter=3000,class_weight="balanced"),E,y,
                        cv=splitter,groups=groups,method="predict_proba",n_jobs=-1)
    return final_score(y,P.argmax(1),stress)["Final"]
print("random 5-fold Final       : %.4f"%logreg_oof(StratifiedKFold(5,shuffle=True,random_state=0)))
for thr in [0.97,0.93,0.90,0.87,0.84,0.80]:
    n,lab=connected_components(csr_matrix(S>=thr),directed=False)
    ng=len(np.unique(lab)); big=pd.Series(lab).value_counts().iloc[0]
    if ng<5: print(f"grouped thr {thr}: only {ng} groups (skip)"); continue
    f=logreg_oof(GroupKFold(5),groups=lab)
    print(f"grouped thr {thr}: {ng:4d} groups (largest {big:3d}) -> Final %.4f"%f)
print("\nreal pre-submission ~0.537 ; public LB best to beat was 0.517")
