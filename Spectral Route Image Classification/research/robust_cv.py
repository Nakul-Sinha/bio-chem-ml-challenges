"""Leakage-safe source-grouped CV + signal-robustness check.
Grouping (via frozen features) is a DIAGNOSTIC/CV tool only — NOT part of the submitted model.
Compares, on a source-grouped split (held-out sources) vs a leaky random split:
  - metadata-only (allowed, from-scratch-compatible, source-agnostic content stats)
  - frozen-image reference (upper-bound-ish for image signal; pretrained, so reference only)
  - fusion
Saves group_labels.npy (aligned to train.csv row order) for the from-scratch kernel's GroupKFold.
"""
import os, numpy as np, pandas as pd, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from metric import final_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, GroupKFold, cross_val_predict
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
HERE=os.path.dirname(os.path.abspath(__file__)); DS=os.path.abspath(os.path.join(HERE,"..","dataset"))
tr=pd.read_csv(os.path.join(DS,"train.csv")); y=tr["target_id"].values
stress=(tr["sensor_noise_score"].values>=0.5757).astype(int)
meta_cols=[c for c in tr.columns if c.endswith("_score")]
M=StandardScaler().fit_transform(tr[meta_cols].values)
E=np.load(os.path.join(HERE,"emb_train.npy")); E=E/(np.linalg.norm(E,axis=1,keepdims=True)+1e-8)
S=E@E.T

def gf(P): return final_score(y,P.argmax(1),stress)["Final"]
def oof(X,clf,splitter,groups=None):
    P=cross_val_predict(clf,X,y,cv=splitter,groups=groups,method="predict_proba",n_jobs=-1)
    return gf(P)
def LR(): return LogisticRegression(max_iter=3000,class_weight="balanced")
def GB(): return HistGradientBoostingClassifier(max_iter=400,learning_rate=0.06,l2_regularization=1.0,random_state=0)

# choose grouping threshold: aim for a non-leaky (source-like) split
best=None
for thr in [0.86,0.84,0.82,0.80]:
    n,lab=connected_components(csr_matrix(S>=thr),directed=False)
    ng=len(np.unique(lab)); big=pd.Series(lab).value_counts().iloc[0]
    fig=oof(E,LR(),GroupKFold(5),lab)
    print(f"thr {thr}: {ng} groups (largest {big}) frozen-img grouped Final {fig:.4f}")
    if best is None or abs(fig-0.537)<abs(best[1]-0.537): best=(thr,fig,lab,ng)
THR,figref,lab,ng=best
print(f"\n==> using grouping thr {THR} ({ng} groups), frozen-img grouped Final {figref:.4f} (target ~0.537)")
np.save(os.path.join(HERE,"group_labels.npy"), lab.astype(np.int32))

rand=StratifiedKFold(5,shuffle=True,random_state=0); grp=GroupKFold(5)
print("\nsignal                 random-CV   grouped-CV(honest)")
for name,X,clf in [("metadata-only (GBM)",M,GB()),("metadata-only (LR)",M,LR()),
                   ("frozen-image (ref)",E,LR()),("fusion img+meta",np.hstack([E,M]),LR())]:
    r=oof(X,clf,rand); g=oof(X,clf,grp,lab)
    print(f"{name:22s}   {r:.4f}      {g:.4f}")
print("\n(metadata-only grouped-CV = the robust signal a from-scratch model can build on;")
print(" frozen-image is pretrained -> reference only, not for the submitted model.)")
