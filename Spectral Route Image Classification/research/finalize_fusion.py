"""Ensemble the leakage-safe fusion models on the honest grouped OOF; write candidate submission."""
import os, sys, itertools, numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from metric import final_score, print_scores
HERE=os.path.dirname(os.path.abspath(__file__)); ROOT=os.path.abspath(os.path.join(HERE,".."))
DS=os.path.join(ROOT,"dataset"); R=os.environ.get("RES","C:/srf/results")
tr=pd.read_csv(os.path.join(DS,"train.csv")); ss=pd.read_csv(os.path.join(DS,"sample_submission.csv"))
y=tr["target_id"].values; stress=(tr["sensor_noise_score"].values>=0.5757).astype(int)
ID2LAB={0:'route-aphelion',1:'route-borealis',2:'route-cygnus',3:'route-driftwood',4:'route-equinox',5:'route-fjord'}
def norm(P): return P/(P.sum(1,keepdims=True)+1e-9)
runs={}
for f in sorted(os.listdir(R)):
    if f.endswith("_oof.npy"):
        n=f[:-8]; t=os.path.join(R,n+"_test.npy")
        if os.path.exists(t): runs[n]=(norm(np.load(os.path.join(R,f))),norm(np.load(t)))
print("runs:",list(runs))
def calib(P):
    w=np.ones(6);b=final_score(y,P.argmax(1),stress)["Final"]
    for _ in range(60):
        imp=False
        for c in range(6):
            for m in [0.7,0.8,0.88,0.94,1.06,1.13,1.25,1.4]:
                w2=w.copy();w2[c]*=m;s=final_score(y,(P*w2).argmax(1),stress)["Final"]
                if s>b+1e-6:w,b,imp=w2,s,True
        if not imp:break
    return w,b
cands={n:(o,t) for n,(o,t) in runs.items()}
names=list(runs)
for r in range(2,len(names)+1):
    for c in itertools.combinations(names,r):
        cands["+".join(c)]=(norm(np.mean([runs[n][0] for n in c],0)),norm(np.mean([runs[n][1] for n in c],0)))
best=None
for name,(O,T) in cands.items():
    raw=final_score(y,O.argmax(1),stress); w,cal=calib(O)
    print(f"{name:28s} grouped-OOF raw {raw['Final']:.4f}  cal {cal:.4f}")
    if best is None or cal>best[0]: best=(cal,name,O,T,w)
cal,name,O,T,w=best
print(f"\nBEST: {name}  grouped-OOF Final cal {cal:.4f}")
sub=ss.copy(); sub["target"]=[ID2LAB[int(i)] for i in (T*w).argmax(1)]; sub=sub[["id","target","stress_flag"]]
out=os.path.join(ROOT,"working"); os.makedirs(out,exist_ok=True); sub.to_csv(os.path.join(out,"submission.csv"),index=False)
print("wrote working/submission.csv | dist",sub["target"].value_counts().to_dict())
