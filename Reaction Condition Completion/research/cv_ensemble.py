"""Ensemble CV: Morgan-MLP + n-gram-MLP (softmax primary head). 5-fold OOF, blend, tune decode."""
import sys, time, collections, numpy as np, pandas as pd, torch, torch.nn as nn
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
sys.path.insert(0,str(Path(__file__).resolve().parent))
import feat, feat_str
from metric import composite, make_masks, TEMPS, TIMES
DS=Path(__file__).resolve().parent.parent/"dataset"
dev="cuda" if torch.cuda.is_available() else "cpu"; SEED=42; np.random.seed(SEED); torch.manual_seed(SEED)
train=pd.read_csv(DS/"train.csv"); test=pd.read_csv(DS/"test.csv"); vocab=pd.read_csv(DS/"solvent_vocabulary.csv")
SOLV=[s for s in vocab["solvent_label"] if s!="NONE"]; S2I={s:i for i,s in enumerate(SOLV)}; NS=len(SOLV)
def parse_solv(s):
    s=str(s); return [] if s in ("nan","NONE","") else s.split("|")
solv_lists=[parse_solv(s) for s in train["solvent_labels"]]; freq=collections.Counter(x for l in solv_lists for x in l)
def multihot(l):
    v=np.zeros(NS,dtype=np.float32)
    for x in l:
        if x in S2I: v[S2I[x]]=1.0
    return v
def prim(l):
    if not l: return 0
    b=max(l,key=lambda x:freq.get(x,0)); return (S2I[b]+1) if b in S2I else 0
Ysolv=np.vstack([multihot(l) for l in solv_lists]); Yprim=np.array([prim(l) for l in solv_lists])
Ytemp=train["temp_bin"].map({t:i for i,t in enumerate(TEMPS)}).values
Ytime=train["time_bin"].map({t:i for i,t in enumerate(TIMES)}).values
Ycat=train["catalyst_present"].values.astype(np.float32)
feats={"morgan":feat.featurize(train,"train"),"ngram":feat_str.featurize(train,"train")}
for k in feats:  # standardize last 49 desc
    X=feats[k]; nd=49; mu=X[:,-nd:].mean(0); sd=X[:,-nd:].std(0)+1e-6; X[:,-nd:]=(X[:,-nd:]-mu)/sd
class Net(nn.Module):
    def __init__(s,d):
        super().__init__()
        s.bb=nn.Sequential(nn.Linear(d,1024),nn.BatchNorm1d(1024),nn.ReLU(),nn.Dropout(0.4),
                           nn.Linear(1024,512),nn.BatchNorm1d(512),nn.ReLU(),nn.Dropout(0.3))
        s.hs=nn.Linear(512,NS); s.hp=nn.Linear(512,NS+1); s.ht=nn.Linear(512,5); s.htm=nn.Linear(512,6); s.hc=nn.Linear(512,1)
    def forward(s,x):
        z=s.bb(x); return s.hs(z),s.hp(z),s.ht(z),s.htm(z),s.hc(z).squeeze(-1)
def run_fold(X,tri,vai,seed=0,EP=14,BS=256,LR=1e-3):
    torch.manual_seed(seed)
    Xa=torch.tensor(X[tri]); yt=torch.tensor(Ytemp[tri]); ytm=torch.tensor(Ytime[tri]); yc=torch.tensor(Ycat[tri])
    ysv=torch.tensor(Ysolv[tri]); yp=torch.tensor(Yprim[tri]); net=Net(X.shape[1]).to(dev)
    opt=torch.optim.AdamW(net.parameters(),lr=LR,weight_decay=1e-5); n=len(tri); bce=nn.BCEWithLogitsLoss(); ce=nn.CrossEntropyLoss()
    sch=torch.optim.lr_scheduler.OneCycleLR(opt,LR,total_steps=EP*((n+BS-1)//BS))
    for ep in range(EP):
        net.train(); perm=torch.randperm(n)
        for i in range(0,n,BS):
            idx=perm[i:i+BS]; xb=Xa[idx].to(dev); ls,lp,lt,ltm,lc=net(xb)
            loss=bce(ls,ysv[idx].to(dev))+ce(lp,yp[idx].to(dev))+ce(lt,yt[idx].to(dev))+ce(ltm,ytm[idx].to(dev))+bce(lc,yc[idx].to(dev))
            opt.zero_grad(); loss.backward(); opt.step(); sch.step()
    net.eval(); o={"solv":[],"prim":[],"temp":[],"time":[],"cat":[]}; Xv=torch.tensor(X[vai])
    with torch.no_grad():
        for i in range(0,len(vai),1024):
            xb=Xv[i:i+1024].to(dev); ls,lp,lt,ltm,lc=net(xb)
            o["solv"].append(torch.sigmoid(ls).cpu().numpy()); o["prim"].append(torch.softmax(lp,1).cpu().numpy())
            o["temp"].append(torch.softmax(lt,1).cpu().numpy()); o["time"].append(torch.softmax(ltm,1).cpu().numpy()); o["cat"].append(torch.sigmoid(lc).cpu().numpy())
    return {k:np.concatenate(v) for k,v in o.items()}
skf=StratifiedKFold(5,shuffle=True,random_state=SEED)
oof={m:{k:np.zeros((len(train),d)) if d else np.zeros(len(train)) for k,d in [("solv",NS),("prim",NS+1),("temp",5),("time",6),("cat",0)]} for m in feats}
t0=time.time()
for fi,(tri,vai) in enumerate(skf.split(train,Ytemp)):
    for m in feats:
        pv=run_fold(feats[m],tri,vai)
        for k in oof[m]: oof[m][k][vai]=pv[k]
    print(f"fold {fi} done [{time.time()-t0:.0f}s]")
rare,shift=make_masks(train)
true={"solv":solv_lists,"temp":[TEMPS[i] for i in Ytemp],"time":[TIMES[i] for i in Ytime],"cat":Ycat}
prior_t=np.bincount(Ytemp,minlength=5)/len(Ytemp); prior_tm=np.bincount(Ytime,minlength=6)/len(Ytime)
def decode(p,a_t,a_tm,cat_thr,sec_thr,nb):
    tp=(p["temp"]/(prior_t**a_t)).argmax(1); tmp=(p["time"]/(prior_tm**a_tm)).argmax(1)
    cat=(p["cat"]>cat_thr).astype(int); pr=p["prim"].copy(); pr[:,0]*=nb; pa=pr.argmax(1); solv=[]
    for i in range(len(pa)):
        if pa[i]==0: solv.append([]); continue
        s=[SOLV[pa[i]-1]]
        for j in np.argsort(p["solv"][i])[::-1][:3]:
            if SOLV[j]!=s[0] and p["solv"][i,j]>sec_thr: s.append(SOLV[j])
        solv.append(s)
    return {"solv":solv,"temp":[TEMPS[i] for i in tp],"time":[TIMES[i] for i in tmp],"cat":cat}
def blend(w):  # w = weight on morgan
    return {k:(w*oof["morgan"][k]+(1-w)*oof["ngram"][k]) for k in oof["morgan"]}
def besttune(p):
    b=(-1,None)
    for a_t in [0,0.3,0.5,0.7]:
      for cat_thr in [0.4,0.45,0.5]:
        for sec_thr in [0.3,0.4]:
          for nb in [1.0,1.5,2.0]:
            sc=composite(decode(p,a_t,0,cat_thr,sec_thr,nb),true,rare,shift)
            if sc>b[0]: b=(sc,(a_t,0,cat_thr,sec_thr,nb))
    return b
print("morgan only:", round(besttune(oof["morgan"])[0],4))
print("ngram only:", round(besttune(oof["ngram"])[0],4))
for w in [0.3,0.5,0.6,0.7]:
    print(f"ensemble w_morgan={w}:", round(besttune(blend(w))[0],4))
