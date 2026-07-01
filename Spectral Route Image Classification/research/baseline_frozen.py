"""Honest baseline on FROZEN convnext_tiny.fb_in22k embeddings (no training/GPU).
Logistic regression, exact 4-component Final on stratified 5-fold OOF,
image-only vs image+metadata, raw vs OOF-calibrated decision rule.
"""
import os, numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
import sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from metric import final_score, print_scores

HERE=os.path.dirname(os.path.abspath(__file__)); DS=os.path.abspath(os.path.join(HERE,"..","dataset"))
tr=pd.read_csv(os.path.join(DS,"train.csv")); y=tr["target_id"].values
stress=(tr["sensor_noise_score"].values>=0.5757).astype(int)
E=np.load(os.path.join(HERE,"emb_train.npy")); E=E/(np.linalg.norm(E,axis=1,keepdims=True)+1e-8)
meta_cols=[c for c in tr.columns if c.endswith("_score")]
M=StandardScaler().fit_transform(tr[meta_cols].values)

def oof(X):
    skf=StratifiedKFold(5,shuffle=True,random_state=0); P=np.zeros((len(y),6))
    for a,b in skf.split(X,y):
        clf=LogisticRegression(max_iter=3000,C=1.0,class_weight="balanced")
        clf.fit(X[a],y[a]); P[b]=clf.predict_proba(X[b])
    return P

def calibrate(P):
    w=np.ones(6); best=final_score(y,P.argmax(1),stress)["Final"]
    for _ in range(50):
        imp=False
        for c in range(6):
            for m in [0.7,0.85,0.93,1.07,1.15,1.3]:
                w2=w.copy(); w2[c]*=m; s=final_score(y,(P*w2).argmax(1),stress)["Final"]
                if s>best+1e-6: w,best,imp=w2,s,True
        if not imp: break
    return w

for name,X in [("frozen img-only",E),("frozen img+meta",np.hstack([E,M])),("meta-only",M)]:
    P=oof(X)
    d=final_score(y,P.argmax(1),stress); print_scores(d,name+" raw")
    w=calibrate(P); dc=final_score(y,(P*w).argmax(1),stress); print_scores(dc,name+" calib")
    print(f"   vs baseline 0.5368: raw {d['Final']-0.5368:+.4f}  calib {dc['Final']-0.5368:+.4f}\n")
print("NOTE: frozen convnext_tiny features; fine-tuning (Kaggle) should exceed this.")
