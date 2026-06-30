"""Tune decode params (incl NONE bias) on saved OOF (oof_mlp2.npz). No retrain."""
import sys, numpy as np, pandas as pd
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from metric import composite, make_masks, TEMPS, TIMES
DS=Path(__file__).resolve().parent.parent/"dataset"; RES=Path(__file__).resolve().parent
train=pd.read_csv(DS/"train.csv"); vocab=pd.read_csv(DS/"solvent_vocabulary.csv")
SOLV=[s for s in vocab["solvent_label"] if s!="NONE"]; NS=len(SOLV)
def parse_solv(s):
    s=str(s); return [] if s in ("nan","NONE","") else s.split("|")
solv_lists=[parse_solv(s) for s in train["solvent_labels"]]
Ytemp=train["temp_bin"].map({t:i for i,t in enumerate(TEMPS)}).values
Ytime=train["time_bin"].map({t:i for i,t in enumerate(TIMES)}).values
Ycat=train["catalyst_present"].values.astype(np.float32)
d=np.load(RES/"oof_mlp2.npz"); oof={k.split("_",1)[1]:d[k] for k in d.files if k.startswith("oof_")}
rare,shift=make_masks(train)
true={"solv":solv_lists,"temp":[TEMPS[i] for i in Ytemp],"time":[TIMES[i] for i in Ytime],"cat":Ycat}
prior_t=np.bincount(Ytemp,minlength=5)/len(Ytemp); prior_tm=np.bincount(Ytime,minlength=6)/len(Ytime)

def decode(p,a_t,a_tm,cat_thr,sec_thr,none_boost):
    tp=(p["temp"]/(prior_t**a_t)).argmax(1); tmp=(p["time"]/(prior_tm**a_tm)).argmax(1)
    cat=(p["cat"]>cat_thr).astype(int)
    prim=p["prim"].copy(); prim[:,0]=prim[:,0]*none_boost; pa=prim.argmax(1); solv=[]
    for i in range(len(pa)):
        if pa[i]==0: solv.append([]); continue
        s=[SOLV[pa[i]-1]]
        for j in np.argsort(p["solv"][i])[::-1][:3]:
            if SOLV[j]!=s[0] and p["solv"][i,j]>sec_thr: s.append(SOLV[j])
        solv.append(s)
    return {"solv":solv,"temp":[TEMPS[i] for i in tp],"time":[TIMES[i] for i in tmp],"cat":cat}

best=(-1,None)
for a_t in [0.5,0.7,1.0]:
  for a_tm in [0,0.3,0.5]:
    for cat_thr in [0.4,0.45,0.5]:
      for sec_thr in [0.25,0.3,0.4]:
        for nb in [0.7,1.0,1.3,1.6,2.0,2.5]:
            sc=composite(decode(oof,a_t,a_tm,cat_thr,sec_thr,nb),true,rare,shift)
            if sc>best[0]: best=(sc,(a_t,a_tm,cat_thr,sec_thr,nb))
print(f"BEST composite={best[0]:.4f} params(a_t,a_tm,cat_thr,sec_thr,none_boost)={best[1]}")
composite(decode(oof,*best[1]),true,rare,shift,debug=True)
# NONE recall at best
prim=oof["prim"].copy(); prim[:,0]*=best[1][4]; pa=prim.argmax(1)
none_idx=[i for i in range(len(solv_lists)) if not solv_lists[i]]
print("NONE recall at best none_boost:",np.mean([pa[i]==0 for i in none_idx]).round(3))
