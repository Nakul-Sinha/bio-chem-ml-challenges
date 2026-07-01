"""Reverse-engineer how stress_flag is defined from PUBLIC metadata using the test set
(test.csv has metadata; sample_submission.csv gives stress_flag for the same ids),
then replicate an honest stress proxy on train for the StressMacroF1 CV component.
"""
import os, numpy as np, pandas as pd
HERE=os.path.dirname(os.path.abspath(__file__)); DS=os.path.abspath(os.path.join(HERE,"..","dataset"))
test=pd.read_csv(os.path.join(DS,"test.csv")); ss=pd.read_csv(os.path.join(DS,"sample_submission.csv"))
train=pd.read_csv(os.path.join(DS,"train.csv"))
meta=[c for c in test.columns if c.endswith('_score')]
df=test.merge(ss[['id','stress_flag']],on='id')
print("test stress rate:", df['stress_flag'].mean(), "| n=",len(df))

# 1) univariate separation: AUC of each feature vs stress_flag
from sklearn.metrics import roc_auc_score
aucs={}
for c in meta:
    try: aucs[c]=roc_auc_score(df['stress_flag'], df[c])
    except Exception: aucs[c]=np.nan
order=sorted(aucs.items(), key=lambda kv:-abs(kv[1]-0.5))
print("\ntop features by |AUC-0.5| vs stress_flag:")
for c,a in order[:10]: print(f"  {c:30s} AUC {a:.3f}")

# 2) best single-threshold accuracy on the strongest feature(s)
def best_threshold(x, yv):
    xs=np.sort(np.unique(x)); best=(0,None,None)
    for t in xs:
        for sign in (1,-1):
            pred=( (x>=t) if sign>0 else (x<t) ).astype(int)
            acc=(pred==yv).mean()
            if acc>best[0]: best=(acc,t,sign)
    return best
for c,_ in order[:3]:
    acc,t,sign=best_threshold(df[c].values, df['stress_flag'].values)
    print(f"\nsingle-threshold on {c}: acc {acc:.3f} rule: {c} {'>=' if sign>0 else '<'} {t:.4f}")

# 3) full metadata -> stress_flag (is it deterministic from public metadata?)
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
X=df[meta].values; yv=df['stress_flag'].values
skf=StratifiedKFold(5,shuffle=True,random_state=0)
for name,clf in [("logreg",make_pipeline(StandardScaler(),LogisticRegression(max_iter=2000))),
                 ("hgb",HistGradientBoostingClassifier(max_iter=300,random_state=0))]:
    p=cross_val_predict(clf,X,yv,cv=skf)
    print(f"predict stress_flag from metadata [{name}]: acc {(p==yv).mean():.3f}")

# 4) Build train stress proxy.
# Fit the best available rule on ALL test (metadata->stress), apply to train.
clf=HistGradientBoostingClassifier(max_iter=400,random_state=0).fit(X,yv)
train_stress=clf.predict(train[meta].values)
np.save(os.path.join(HERE,"train_stress.npy"), train_stress.astype(np.int8))
print("\ntrain stress proxy rate (hgb rule):", float(train_stress.mean()))
# also a simple burden-threshold proxy matched to test rate, as a robustness alt
burden='artifact_burden_score'
thr=np.quantile(train[burden], 1-df['stress_flag'].mean())
alt=(train[burden].values>=thr).astype(np.int8)
np.save(os.path.join(HERE,"train_stress_burden.npy"), alt)
print(f"alt burden-threshold proxy: {burden} >= {thr:.4f} -> rate {alt.mean():.3f}")
print("agreement hgb-vs-burden proxy:", (train_stress==alt).mean())
print("STRESS DONE.")
