"""Multi-task MLP on reaction FPs -> 4 condition targets. OOF CV + post-hoc tuning to the
composite metric (prior-adjust alpha for temp/time balanced-acc, thresholds for cat/solvent)."""
import sys, time, numpy as np, pandas as pd, torch, torch.nn as nn
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
sys.path.insert(0,str(Path(__file__).resolve().parent))
from feat import featurize
from metric import composite, make_masks, TEMPS, TIMES, set_f1_row

DS=Path(__file__).resolve().parent.parent/"dataset"
dev="cuda" if torch.cuda.is_available() else "cpu"
SEED=42
np.random.seed(SEED); torch.manual_seed(SEED)

train=pd.read_csv(DS/"train.csv"); test=pd.read_csv(DS/"test.csv")
vocab=pd.read_csv(DS/"solvent_vocabulary.csv")
SOLV=[s for s in vocab["solvent_label"] if s!="NONE"]   # 81 labels incl OTHER
S2I={s:i for i,s in enumerate(SOLV)}; NS=len(SOLV)
def parse_solv(s):
    s=str(s); return [] if s in ("nan","NONE","") else s.split("|")
def solv_multihot(lst):
    v=np.zeros(NS,dtype=np.float32)
    for x in lst:
        if x in S2I: v[S2I[x]]=1.0
    return v

Ytemp=train["temp_bin"].map({t:i for i,t in enumerate(TEMPS)}).values
Ytime=train["time_bin"].map({t:i for i,t in enumerate(TIMES)}).values
Ycat=train["catalyst_present"].values.astype(np.float32)
solv_lists=[parse_solv(s) for s in train["solvent_labels"]]
Ysolv=np.vstack([solv_multihot(l) for l in solv_lists])

Xtr=featurize(train,"train"); Xte=featurize(test,"test")
# standardize last 49 descriptor cols
nd=49; mu=Xtr[:,-nd:].mean(0); sd=Xtr[:,-nd:].std(0)+1e-6
Xtr=Xtr.copy(); Xte=Xte.copy()
Xtr[:,-nd:]=(Xtr[:,-nd:]-mu)/sd; Xte[:,-nd:]=(Xte[:,-nd:]-mu)/sd
D=Xtr.shape[1]

class Net(nn.Module):
    def __init__(s,d):
        super().__init__()
        s.bb=nn.Sequential(nn.Linear(d,1024),nn.BatchNorm1d(1024),nn.ReLU(),nn.Dropout(0.4),
                           nn.Linear(1024,512),nn.BatchNorm1d(512),nn.ReLU(),nn.Dropout(0.3))
        s.hs=nn.Linear(512,NS); s.ht=nn.Linear(512,5); s.htm=nn.Linear(512,6); s.hc=nn.Linear(512,1)
    def forward(s,x):
        z=s.bb(x); return s.hs(z),s.ht(z),s.htm(z),s.hc(z).squeeze(-1)

def predict(net,Xb):
    net.eval()
    with torch.no_grad():
        outs={"solv":[],"temp":[],"time":[],"cat":[]}; Xb_t=torch.tensor(Xb)
        for i in range(0,len(Xb),1024):
            xb=Xb_t[i:i+1024].to(dev)
            ls,lt,ltm,lc=net(xb)
            outs["solv"].append(torch.sigmoid(ls).cpu().numpy())
            outs["temp"].append(torch.softmax(lt,1).cpu().numpy())
            outs["time"].append(torch.softmax(ltm,1).cpu().numpy())
            outs["cat"].append(torch.sigmoid(lc).cpu().numpy())
    return {k:np.concatenate(v) for k,v in outs.items()}

def train_fold(Xa,ya,Xvals,EPOCHS=12,bs=256,lr=1e-3):
    Xa_t=torch.tensor(Xa)
    yt=torch.tensor(ya["temp"]); ytm=torch.tensor(ya["time"]); yc=torch.tensor(ya["cat"]); ysv=torch.tensor(ya["solv"])
    net=Net(D).to(dev); opt=torch.optim.AdamW(net.parameters(),lr=lr,weight_decay=1e-5)
    n=len(Xa); bce=nn.BCEWithLogitsLoss(); ce=nn.CrossEntropyLoss()
    sched=torch.optim.lr_scheduler.OneCycleLR(opt,max_lr=lr,total_steps=EPOCHS*((n+bs-1)//bs))
    for ep in range(EPOCHS):
        net.train(); perm=torch.randperm(n)
        for i in range(0,n,bs):
            idx=perm[i:i+bs]; xb=Xa_t[idx].to(dev)
            ls,lt,ltm,lc=net(xb)
            loss=(bce(ls,ysv[idx].to(dev))+ce(lt,yt[idx].to(dev))+ce(ltm,ytm[idx].to(dev))+bce(lc,yc[idx].to(dev)))
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
    return [predict(net,X) for X in Xvals]

# OOF
skf=StratifiedKFold(5,shuffle=True,random_state=SEED)
oof={"solv":np.zeros((len(train),NS)),"temp":np.zeros((len(train),5)),"time":np.zeros((len(train),6)),"cat":np.zeros(len(train))}
testp={"solv":np.zeros((len(test),NS)),"temp":np.zeros((len(test),5)),"time":np.zeros((len(test),6)),"cat":np.zeros(len(test))}
t0=time.time()
for fold,(tri,vai) in enumerate(skf.split(Xtr,Ytemp)):
    ya={"temp":Ytemp[tri],"time":Ytime[tri],"cat":Ycat[tri],"solv":Ysolv[tri]}
    pv,pt=train_fold(Xtr[tri],ya,[Xtr[vai],Xte])
    for k in oof: oof[k][vai]=pv[k]
    for k in testp: testp[k]+=pt[k]/5
    print(f"fold {fold} done [{time.time()-t0:.0f}s]")

# ---- post-hoc tuning on OOF ----
rare,shift=make_masks(train)
true={"solv":solv_lists,"temp":[TEMPS[i] for i in Ytemp],"time":[TIMES[i] for i in Ytime],"cat":Ycat}
prior_t=np.bincount(Ytemp,minlength=5)/len(Ytemp); prior_tm=np.bincount(Ytime,minlength=6)/len(Ytime)
def decode(probs, a_t,a_tm,cat_thr,solv_thr):
    tp=(probs["temp"]/ (prior_t**a_t)).argmax(1)
    tmp=(probs["time"]/ (prior_tm**a_tm)).argmax(1)
    cat=(probs["cat"]>cat_thr).astype(int)
    solv=[]
    for i in range(len(probs["cat"])):
        labs=[SOLV[j] for j in range(NS) if probs["solv"][i,j]>solv_thr]
        solv.append(labs)  # empty -> NONE
    return {"solv":solv,"temp":[TEMPS[i] for i in tp],"time":[TIMES[i] for i in tmp],"cat":cat}

best=(-1,None)
for a_t in [0,0.3,0.5,0.7,1.0]:
  for a_tm in [0,0.3,0.5,0.7,1.0]:
    for cat_thr in [0.3,0.4,0.5]:
      for solv_thr in [0.2,0.3,0.4,0.5]:
        pred=decode(oof,a_t,a_tm,cat_thr,solv_thr)
        sc=composite(pred,true,rare,shift)
        if sc>best[0]: best=(sc,(a_t,a_tm,cat_thr,solv_thr))
print(f"\nBEST OOF composite={best[0]:.4f} params(a_t,a_tm,cat_thr,solv_thr)={best[1]}")
a_t,a_tm,cat_thr,solv_thr=best[1]
composite(decode(oof,*best[1]),true,rare,shift,debug=True)

# write submission from test probs
pred=decode(testp,a_t,a_tm,cat_thr,solv_thr)
sub=pd.DataFrame({"reaction_id":test["reaction_id"],
    "pred_solvents":["|".join(s) if s else "NONE" for s in pred["solv"]],
    "pred_temp_bin":pred["temp"],"pred_time_bin":pred["time"],"pred_catalyst_present":pred["cat"]})
Path(__file__).resolve().parent.parent.joinpath("working").mkdir(exist_ok=True)
sub.to_csv(Path(__file__).resolve().parent/"sub_mlp.csv",index=False)
print("wrote sub_mlp.csv",sub.shape); print(sub.head())
np.savez(Path(__file__).resolve().parent/"oof_mlp.npz", **{f"oof_{k}":v for k,v in oof.items()}, **{f"test_{k}":v for k,v in testp.items()})
