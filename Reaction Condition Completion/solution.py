import os, sys, time, collections, zlib
from pathlib import Path
import numpy as np, pandas as pd, torch, torch.nn as nn

SEED=42
N_SEEDS=3
EPOCHS=14
BS=256
LR=1e-3
NBITS=2048
NGRAMS=(2,3,4,5)
A_T=0.3
A_TM=0.0
CAT_THR=0.5
SEC_THR=0.3
NONE_BOOST=1.5
TEMPS=["cryogenic","cold","room","warm","hot"]
TIMES=["very_short","short","medium","long","very_long","overnight"]

def find_data_dir():
    here=Path(__file__).resolve().parent
    for c in [here/"dataset"/"public",here/"dataset",Path("dataset/public"),Path("dataset"),here,Path(".")]:
        if (c/"train.csv").exists() and (c/"test.csv").exists(): return c
    raise FileNotFoundError("train.csv/test.csv not found")

REAGENT_PATS={"boron":"B","Pd":"[Pd","Cu":"[Cu","Ni":"[Ni","Pt":"[Pt","Fe":"[Fe","Zn":"[Zn",
 "Li":"[Li","Mg":"[Mg","Na":"[Na","K_ion":"[K+","Cs":"[Cs","Al":"[Al","Cl_anion":"[Cl-]",
 "carbonate":"C([O-])([O-])=O","hydroxide":"[OH-]","hydride":"[H-]","amine_base":"CCN(CC)CC",
 "DIPEA":"CCN(C(C)C)C(C)C","pyridine":"c1ccncc1","TFA":"OC(=O)C(F)(F)F","acid_Cl":"C(=O)Cl",
 "azide":"[N-]=[N+]=[N-]","BOC":"OC(=O)","phosphine":"P(c1ccccc1)","fluoride":"[F-]",
 "tosyl":"S(=O)(=O)","nitro":"[N+](=O)[O-]"}
def ngram_vec(s):
    v=np.zeros(NBITS,dtype=np.float32); s=str(s)
    for n in NGRAMS:
        for i in range(len(s)-n+1): v[zlib.crc32(s[i:i+n].encode())%NBITS]=1.0
    return v
def desc(smi):
    left,_,right=str(smi).partition(">>"); f=[left.count(".")+1,right.count(".")+1,len(left),len(right),len(str(smi))]
    for ch in ["Cl","Br","F","N","O","S","P","B","c","=","#","+","-","[","@","/"]: f.append(str(smi).count(ch))
    for k,pat in REAGENT_PATS.items(): f.append(left.count(pat))
    return np.array(f,dtype=np.float32)
def featurize(df):
    R=np.zeros((len(df),NBITS),dtype=np.float32); P=np.zeros((len(df),NBITS),dtype=np.float32); Dd=[]
    for i,smi in enumerate(df["reaction_smiles"].astype(str)):
        left,_,right=smi.partition(">>"); R[i]=ngram_vec(left); P[i]=ngram_vec(right); Dd.append(desc(smi))
    return np.hstack([R,P,P-R,np.vstack(Dd)]).astype(np.float32)
def parse_solv(s):
    s=str(s); return [] if s in ("nan","NONE","") else s.split("|")

class Net(nn.Module):
    def __init__(s,d,ns):
        super().__init__()
        s.bb=nn.Sequential(nn.Linear(d,1024),nn.BatchNorm1d(1024),nn.ReLU(),nn.Dropout(0.4),
                           nn.Linear(1024,512),nn.BatchNorm1d(512),nn.ReLU(),nn.Dropout(0.3))
        s.hs=nn.Linear(512,ns); s.hp=nn.Linear(512,ns+1); s.ht=nn.Linear(512,5); s.htm=nn.Linear(512,6); s.hc=nn.Linear(512,1)
    def forward(s,x):
        z=s.bb(x); return s.hs(z),s.hp(z),s.ht(z),s.htm(z),s.hc(z).squeeze(-1)

def main():
    np.random.seed(SEED); torch.manual_seed(SEED)
    dev="cuda" if torch.cuda.is_available() else "cpu"
    DATA=find_data_dir(); print("data dir:",DATA,"device:",dev)
    train=pd.read_csv(DATA/"train.csv"); test=pd.read_csv(DATA/"test.csv"); vocab=pd.read_csv(DATA/"solvent_vocabulary.csv")
    SOLV=[s for s in vocab["solvent_label"] if s!="NONE"]; S2I={s:i for i,s in enumerate(SOLV)}; NS=len(SOLV)
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
    prior_t=np.bincount(Ytemp,minlength=5)/len(Ytemp); prior_tm=np.bincount(Ytime,minlength=6)/len(Ytime)

    print("featurizing..."); t0=time.time()
    Xtr=featurize(train); Xte=featurize(test); print("feat done",Xtr.shape,f"[{time.time()-t0:.0f}s]")
    nd=49; mu=Xtr[:,-nd:].mean(0); sd=Xtr[:,-nd:].std(0)+1e-6
    Xtr[:,-nd:]=(Xtr[:,-nd:]-mu)/sd; Xte[:,-nd:]=(Xte[:,-nd:]-mu)/sd; D=Xtr.shape[1]

    def train_one(seed):
        torch.manual_seed(seed); np.random.seed(seed)
        Xa=torch.tensor(Xtr); yt=torch.tensor(Ytemp); ytm=torch.tensor(Ytime); yc=torch.tensor(Ycat); ysv=torch.tensor(Ysolv); yp=torch.tensor(Yprim)
        net=Net(D,NS).to(dev); opt=torch.optim.AdamW(net.parameters(),lr=LR,weight_decay=1e-5)
        n=len(Xtr); bce=nn.BCEWithLogitsLoss(); ce=nn.CrossEntropyLoss()
        sch=torch.optim.lr_scheduler.OneCycleLR(opt,LR,total_steps=EPOCHS*((n+BS-1)//BS))
        for ep in range(EPOCHS):
            net.train(); perm=torch.randperm(n)
            for i in range(0,n,BS):
                idx=perm[i:i+BS]; xb=Xa[idx].to(dev); ls,lp,lt,ltm,lc=net(xb)
                loss=bce(ls,ysv[idx].to(dev))+ce(lp,yp[idx].to(dev))+ce(lt,yt[idx].to(dev))+ce(ltm,ytm[idx].to(dev))+bce(lc,yc[idx].to(dev))
                opt.zero_grad(); loss.backward(); opt.step(); sch.step()
        net.eval()
        with torch.no_grad():
            o={"solv":[],"prim":[],"temp":[],"time":[],"cat":[]}; Xb=torch.tensor(Xte)
            for i in range(0,len(Xte),1024):
                xb=Xb[i:i+1024].to(dev); ls,lp,lt,ltm,lc=net(xb)
                o["solv"].append(torch.sigmoid(ls).cpu().numpy()); o["prim"].append(torch.softmax(lp,1).cpu().numpy())
                o["temp"].append(torch.softmax(lt,1).cpu().numpy()); o["time"].append(torch.softmax(ltm,1).cpu().numpy()); o["cat"].append(torch.sigmoid(lc).cpu().numpy())
        return {k:np.concatenate(v) for k,v in o.items()}

    acc={k:0 for k in ["solv","prim","temp","time","cat"]}
    for s in range(N_SEEDS):
        p=train_one(SEED+s)
        for k in acc: acc[k]=acc[k]+p[k]/N_SEEDS
        print(f"seed {s} done [{time.time()-t0:.0f}s]")
    p=acc
    tp=(p["temp"]/(prior_t**A_T)).argmax(1); tmp=(p["time"]/(prior_tm**A_TM)).argmax(1)
    cat=(p["cat"]>CAT_THR).astype(int); pr=p["prim"].copy(); pr[:,0]*=NONE_BOOST; pa=pr.argmax(1); solv=[]
    for i in range(len(pa)):
        if pa[i]==0: solv.append([]); continue
        s=[SOLV[pa[i]-1]]
        for j in np.argsort(p["solv"][i])[::-1][:3]:
            if SOLV[j]!=s[0] and p["solv"][i,j]>SEC_THR: s.append(SOLV[j])
        solv.append(s)
    sub=pd.DataFrame({"reaction_id":test["reaction_id"],
        "pred_solvents":["|".join(s) if s else "NONE" for s in solv],
        "pred_temp_bin":[TEMPS[i] for i in tp],"pred_time_bin":[TIMES[i] for i in tmp],"pred_catalyst_present":cat})
    assert list(sub.columns)==["reaction_id","pred_solvents","pred_temp_bin","pred_time_bin","pred_catalyst_present"]
    assert len(sub)==len(test) and sub["reaction_id"].is_unique
    assert sub["pred_temp_bin"].isin(TEMPS).all() and sub["pred_time_bin"].isin(TIMES).all()
    assert sub["pred_catalyst_present"].isin([0,1]).all() and not sub.isna().any().any()
    Path("working").mkdir(exist_ok=True)
    sub.to_csv("working/submission.csv",index=False); sub.to_csv("submission.csv",index=False)
    print("wrote submission.csv & working/submission.csv:",sub.shape); print(sub.head())

if __name__=="__main__":
    main()
