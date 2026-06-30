"""v2: multi-task MLP with softmax PRIMARY-solvent head (NONE+81) + sigmoid multilabel head
for secondaries. Optimizes solvent top-1 (77% singleton, 11% NONE). OOF CV + post-hoc tuning."""
import sys, time, collections, numpy as np, pandas as pd, torch, torch.nn as nn
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
sys.path.insert(0,str(Path(__file__).resolve().parent))
from feat import featurize
from metric import composite, make_masks, TEMPS, TIMES, set_f1_row

DS=Path(__file__).resolve().parent.parent/"dataset"
dev="cuda" if torch.cuda.is_available() else "cpu"
SEED=42; np.random.seed(SEED); torch.manual_seed(SEED)
train=pd.read_csv(DS/"train.csv"); test=pd.read_csv(DS/"test.csv")
vocab=pd.read_csv(DS/"solvent_vocabulary.csv")
SOLV=[s for s in vocab["solvent_label"] if s!="NONE"]; S2I={s:i for i,s in enumerate(SOLV)}; NS=len(SOLV)
def parse_solv(s):
    s=str(s); return [] if s in ("nan","NONE","") else s.split("|")
def solv_multihot(lst):
    v=np.zeros(NS,dtype=np.float32)
    for x in lst:
        if x in S2I: v[S2I[x]]=1.0
    return v
solv_lists=[parse_solv(s) for s in train["solvent_labels"]]
Ysolv=np.vstack([solv_multihot(l) for l in solv_lists])
solv_freq=collections.Counter(x for l in solv_lists for x in l)
def primary_idx(l):
    if not l: return 0
    best=max(l,key=lambda x: solv_freq.get(x,0))
    return (S2I[best]+1) if best in S2I else 0
Yprim=np.array([primary_idx(l) for l in solv_lists])
Ytemp=train["temp_bin"].map({t:i for i,t in enumerate(TEMPS)}).values
Ytime=train["time_bin"].map({t:i for i,t in enumerate(TIMES)}).values
Ycat=train["catalyst_present"].values.astype(np.float32)

Xtr=featurize(train,"train"); Xte=featurize(test,"test")
nd=49; mu=Xtr[:,-nd:].mean(0); sd=Xtr[:,-nd:].std(0)+1e-6
Xtr=Xtr.copy(); Xte=Xte.copy(); Xtr[:,-nd:]=(Xtr[:,-nd:]-mu)/sd; Xte[:,-nd:]=(Xte[:,-nd:]-mu)/sd
D=Xtr.shape[1]

class Net(nn.Module):
    def __init__(s,d):
        super().__init__()
        s.bb=nn.Sequential(nn.Linear(d,1024),nn.BatchNorm1d(1024),nn.ReLU(),nn.Dropout(0.4),
                           nn.Linear(1024,512),nn.BatchNorm1d(512),nn.ReLU(),nn.Dropout(0.3))
        s.hs=nn.Linear(512,NS); s.hp=nn.Linear(512,NS+1)
        s.ht=nn.Linear(512,5); s.htm=nn.Linear(512,6); s.hc=nn.Linear(512,1)
    def forward(s,x):
        z=s.bb(x); return s.hs(z),s.hp(z),s.ht(z),s.htm(z),s.hc(z).squeeze(-1)

def predict(net,Xb):
    net.eval()
    with torch.no_grad():
        o={"solv":[],"prim":[],"temp":[],"time":[],"cat":[]}; Xb_t=torch.tensor(Xb)
        for i in range(0,len(Xb),1024):
            xb=Xb_t[i:i+1024].to(dev); ls,lp,lt,ltm,lc=net(xb)
            o["solv"].append(torch.sigmoid(ls).cpu().numpy())
            o["prim"].append(torch.softmax(lp,1).cpu().numpy())
            o["temp"].append(torch.softmax(lt,1).cpu().numpy())
            o["time"].append(torch.softmax(ltm,1).cpu().numpy())
            o["cat"].append(torch.sigmoid(lc).cpu().numpy())
    return {k:np.concatenate(v) for k,v in o.items()}

def train_fold(Xa,ya,Xvals,EPOCHS=14,bs=256,lr=1e-3):
    Xa_t=torch.tensor(Xa)
    yt=torch.tensor(ya["temp"]); ytm=torch.tensor(ya["time"]); yc=torch.tensor(ya["cat"])
    ysv=torch.tensor(ya["solv"]); yp=torch.tensor(ya["prim"])
    net=Net(D).to(dev); opt=torch.optim.AdamW(net.parameters(),lr=lr,weight_decay=1e-5)
    n=len(Xa); bce=nn.BCEWithLogitsLoss(); ce=nn.CrossEntropyLoss()
    sched=torch.optim.lr_scheduler.OneCycleLR(opt,max_lr=lr,total_steps=EPOCHS*((n+bs-1)//bs))
    for ep in range(EPOCHS):
        net.train(); perm=torch.randperm(n)
        for i in range(0,n,bs):
            idx=perm[i:i+bs]; xb=Xa_t[idx].to(dev); ls,lp,lt,ltm,lc=net(xb)
            loss=(bce(ls,ysv[idx].to(dev))+ce(lp,yp[idx].to(dev))+ce(lt,yt[idx].to(dev))
                  +ce(ltm,ytm[idx].to(dev))+bce(lc,yc[idx].to(dev)))
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
    return [predict(net,X) for X in Xvals]

skf=StratifiedKFold(5,shuffle=True,random_state=SEED)
keys=["solv","prim","temp","time","cat"]; dims={"solv":NS,"prim":NS+1,"temp":5,"time":6,"cat":0}
oof={k:(np.zeros((len(train),dims[k])) if dims[k] else np.zeros(len(train))) for k in keys}
testp={k:(np.zeros((len(test),dims[k])) if dims[k] else np.zeros(len(test))) for k in keys}
t0=time.time()
for fold,(tri,vai) in enumerate(skf.split(Xtr,Ytemp)):
    ya={"temp":Ytemp[tri],"time":Ytime[tri],"cat":Ycat[tri],"solv":Ysolv[tri],"prim":Yprim[tri]}
    pv,pt=train_fold(Xtr[tri],ya,[Xtr[vai],Xte])
    for k in keys: oof[k][vai]=pv[k]; testp[k]+=pt[k]/5
    print(f"fold {fold} done [{time.time()-t0:.0f}s]")

rare,shift=make_masks(train)
true={"solv":solv_lists,"temp":[TEMPS[i] for i in Ytemp],"time":[TIMES[i] for i in Ytime],"cat":Ycat}
prior_t=np.bincount(Ytemp,minlength=5)/len(Ytemp); prior_tm=np.bincount(Ytime,minlength=6)/len(Ytime)
def decode(p,a_t,a_tm,cat_thr,sec_thr):
    tp=(p["temp"]/(prior_t**a_t)).argmax(1); tmp=(p["time"]/(prior_tm**a_tm)).argmax(1)
    cat=(p["cat"]>cat_thr).astype(int); solv=[]; prim=p["prim"].argmax(1)
    for i in range(len(prim)):
        if prim[i]==0: solv.append([]); continue
        s=[SOLV[prim[i]-1]]
        for j in np.argsort(p["solv"][i])[::-1][:3]:
            if SOLV[j]!=s[0] and p["solv"][i,j]>sec_thr: s.append(SOLV[j])
        solv.append(s)
    return {"solv":solv,"temp":[TEMPS[i] for i in tp],"time":[TIMES[i] for i in tmp],"cat":cat}

prim_oof=oof["prim"].argmax(1)
single=[i for i in range(len(solv_lists)) if len(solv_lists[i])==1]
acc1=np.mean([(prim_oof[i]>0 and SOLV[prim_oof[i]-1]==solv_lists[i][0]) for i in single])
print(f"softmax primary top-1 (singletons): {acc1:.4f}")
none_idx=[i for i in range(len(solv_lists)) if len(solv_lists[i])==0]
print(f"NONE recall: {np.mean([prim_oof[i]==0 for i in none_idx]):.4f}")

best=(-1,None)
for a_t in [0,0.3,0.5,0.7]:
  for a_tm in [0,0.3,0.5,0.7]:
    for cat_thr in [0.35,0.45,0.5]:
      for sec_thr in [0.3,0.4,0.5,0.6]:
        sc=composite(decode(oof,a_t,a_tm,cat_thr,sec_thr),true,rare,shift)
        if sc>best[0]: best=(sc,(a_t,a_tm,cat_thr,sec_thr))
print(f"\nBEST OOF composite={best[0]:.4f} params={best[1]}")
composite(decode(oof,*best[1]),true,rare,shift,debug=True)
pred=decode(testp,*best[1])
sub=pd.DataFrame({"reaction_id":test["reaction_id"],
    "pred_solvents":["|".join(s) if s else "NONE" for s in pred["solv"]],
    "pred_temp_bin":pred["temp"],"pred_time_bin":pred["time"],"pred_catalyst_present":pred["cat"]})
sub.to_csv(Path(__file__).resolve().parent/"sub_mlp2.csv",index=False)
np.savez(Path(__file__).resolve().parent/"oof_mlp2.npz",**{f"oof_{k}":oof[k] for k in keys},**{f"test_{k}":testp[k] for k in keys})
print("wrote sub_mlp2.csv",sub.shape)
