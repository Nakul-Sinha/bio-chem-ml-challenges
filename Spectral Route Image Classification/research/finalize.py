"""Lock the base+swin ensemble: calibrate on OOF, write ./working/submission.csv."""
import os, sys, numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from metric import final_score, print_scores
HERE=os.path.dirname(os.path.abspath(__file__)); ROOT=os.path.abspath(os.path.join(HERE,".."))
DS=os.path.join(ROOT,"dataset"); R=os.environ.get("RES","C:/srk/results")
tr=pd.read_csv(os.path.join(DS,"train.csv")); ss=pd.read_csv(os.path.join(DS,"sample_submission.csv"))
y=tr["target_id"].values; stress=(tr["sensor_noise_score"].values>=0.5757).astype(int)
ID2LAB={0:'route-aphelion',1:'route-borealis',2:'route-cygnus',3:'route-driftwood',4:'route-equinox',5:'route-fjord'}
def norm(P): return P/(P.sum(1,keepdims=True)+1e-9)
Ob,Os=norm(np.load(f"{R}/base_oof.npy")),norm(np.load(f"{R}/swin_oof.npy"))
Tb,Ts=norm(np.load(f"{R}/base_test.npy")),norm(np.load(f"{R}/swin_test.npy"))
O=(Ob+Os)/2; T=(Tb+Ts)/2
def calibrate(P):
    w=np.ones(6); best=final_score(y,P.argmax(1),stress)["Final"]
    for _ in range(60):
        imp=False
        for c in range(6):
            for m in [0.7,0.8,0.88,0.94,1.06,1.13,1.25,1.4]:
                w2=w.copy(); w2[c]*=m; s=final_score(y,(P*w2).argmax(1),stress)["Final"]
                if s>best+1e-6: w,best,imp=w2,s,True
        if not imp: break
    return w
print_scores(final_score(y,O.argmax(1),stress),"base+swin raw")
w=calibrate(O); print_scores(final_score(y,(O*w).argmax(1),stress),"base+swin CAL")
print("calib weights",np.round(w,3))
labels=(T*w).argmax(1)
out=os.path.join(ROOT,"working"); os.makedirs(out,exist_ok=True)
sub=ss.copy(); sub["target"]=[ID2LAB[int(i)] for i in labels]; sub=sub[["id","target","stress_flag"]]
sub.to_csv(os.path.join(out,"submission.csv"),index=False)
print("wrote working/submission.csv | dist",sub["target"].value_counts().to_dict())
print("anchor predicted %.1f%% (train prior 14.0%%)"%(100*(sub['target']=='route-aphelion').mean()))
