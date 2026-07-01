"""Compare single models and their prob-average ensembles on the EXACT Final (OOF),
raw vs OOF-calibrated. Reads saved OOF/test probs from /c/srk/results. Picks the best
by calibrated OOF Final and writes the corresponding calibrated submission.
"""
import os, sys, glob, itertools, numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from metric import final_score, print_scores
HERE=os.path.dirname(os.path.abspath(__file__)); DS=os.path.abspath(os.path.join(HERE,"..","dataset"))
R=os.environ.get("RES","/c/srk/results")
tr=pd.read_csv(os.path.join(DS,"train.csv")); ss=pd.read_csv(os.path.join(DS,"sample_submission.csv"))
y=tr["target_id"].values; stress=(tr["sensor_noise_score"].values>=0.5757).astype(int)
ID2LAB={0:'route-aphelion',1:'route-borealis',2:'route-cygnus',3:'route-driftwood',4:'route-equinox',5:'route-fjord'}

runs={}
for op in sorted(glob.glob(os.path.join(R,"*_oof.npy"))):
    name=os.path.basename(op)[:-8]; tp=os.path.join(R,name+"_test.npy")
    if os.path.exists(tp): runs[name]=(np.load(op), np.load(tp))
print("runs:", list(runs))

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

def norm(P): return P/ (P.sum(1,keepdims=True)+1e-9)
cands={n:(o,t) for n,(o,t) in runs.items()}
names=list(runs)
for r in range(2,len(names)+1):
    for combo in itertools.combinations(names,r):
        O=norm(np.mean([runs[n][0] for n in combo],0)); T=norm(np.mean([runs[n][1] for n in combo],0))
        cands["+".join(combo)]=(O,T)

results=[]
for name,(O,T) in cands.items():
    raw=final_score(y,O.argmax(1),stress); w=calibrate(O); cal=final_score(y,(O*w).argmax(1),stress)
    print_scores(raw,name+" raw"); print_scores(cal,name+" CAL")
    results.append((cal["Final"],name,O,T,w))
results.sort(reverse=True, key=lambda x:x[0])
best=results[0]; print("\nBEST:",best[1],"OOF cal Final %.4f (+%.4f vs 0.5368)"%(best[0],best[0]-0.5368))
O,T,w=best[2],best[3],best[4]
sub=ss.copy(); sub["target"]=[ID2LAB[i] for i in (T*w).argmax(1)]; sub=sub[["id","target","stress_flag"]]
out=os.path.join(HERE,"best_submission.csv"); sub.to_csv(out,index=False); print("wrote",out,"| dist",sub["target"].value_counts().to_dict())
np.save(os.path.join(HERE,"best_calw.npy"),w); print("best config:",best[1])
