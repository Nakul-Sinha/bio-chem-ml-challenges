"""Fine-tune ChemBERTa (multi-task) on reaction SMILES. Single stratified holdout, report components.
Saves OOF + test probs for ensembling with the n-gram MLP."""
import sys, time, collections, numpy as np, pandas as pd, torch, torch.nn as nn
from pathlib import Path
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModel
sys.path.insert(0,str(Path(__file__).resolve().parent))
from metric import composite, make_masks, TEMPS, TIMES, set_f1_row
DS=Path(__file__).resolve().parent.parent/"dataset"
dev="cuda" if torch.cuda.is_available() else "cpu"
SEED=42; np.random.seed(SEED); torch.manual_seed(SEED)
MODEL=sys.argv[1] if len(sys.argv)>1 else "DeepChem/ChemBERTa-77M-MLM"
MAXLEN=192; BS=32; EPOCHS=8; LR=5e-5

train=pd.read_csv(DS/"train.csv"); test=pd.read_csv(DS/"test.csv")
vocab=pd.read_csv(DS/"solvent_vocabulary.csv")
SOLV=[s for s in vocab["solvent_label"] if s!="NONE"]; S2I={s:i for i,s in enumerate(SOLV)}; NS=len(SOLV)
def parse_solv(s):
    s=str(s); return [] if s in ("nan","NONE","") else s.split("|")
solv_lists=[parse_solv(s) for s in train["solvent_labels"]]
freq=collections.Counter(x for l in solv_lists for x in l)
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

tok=AutoTokenizer.from_pretrained(MODEL)
def encode(smis):
    return tok(list(smis),truncation=True,max_length=MAXLEN,padding="max_length",return_tensors="pt")

class Net(nn.Module):
    def __init__(s):
        super().__init__(); s.bert=AutoModel.from_pretrained(MODEL); h=s.bert.config.hidden_size
        s.drop=nn.Dropout(0.2)
        s.hs=nn.Linear(h,NS); s.hp=nn.Linear(h,NS+1); s.ht=nn.Linear(h,5); s.htm=nn.Linear(h,6); s.hc=nn.Linear(h,1)
    def forward(s,ids,mask):
        h=s.bert(input_ids=ids,attention_mask=mask).last_hidden_state
        m=mask.unsqueeze(-1).float(); o=(h*m).sum(1)/m.sum(1).clamp(min=1)
        z=s.drop(o); return s.hs(z),s.hp(z),s.ht(z),s.htm(z),s.hc(z).squeeze(-1)

tri,vai=train_test_split(np.arange(len(train)),test_size=0.15,random_state=SEED,stratify=Ytemp)
enc_all=encode(train["reaction_smiles"]); ids_all=enc_all["input_ids"]; mask_all=enc_all["attention_mask"]
net=Net().to(dev); opt=torch.optim.AdamW(net.parameters(),lr=LR,weight_decay=0.01)
bce=nn.BCEWithLogitsLoss(); ce=nn.CrossEntropyLoss()
steps=EPOCHS*((len(tri)+BS-1)//BS); sch=torch.optim.lr_scheduler.OneCycleLR(opt,LR,total_steps=steps,pct_start=0.1)
bf16=torch.cuda.is_available(); t0=time.time()
for ep in range(EPOCHS):
    net.train(); perm=np.random.permutation(tri)
    tot=0
    for i in range(0,len(perm),BS):
        idx=perm[i:i+BS]
        ii=ids_all[idx].to(dev); mm=mask_all[idx].to(dev)
        with torch.autocast("cuda",dtype=torch.bfloat16,enabled=bf16):
            ls,lp,lt,ltm,lc=net(ii,mm)
            loss=(bce(ls,torch.tensor(Ysolv[idx]).to(dev))+ce(lp,torch.tensor(Yprim[idx]).to(dev))
                  +ce(lt,torch.tensor(Ytemp[idx]).to(dev))+ce(ltm,torch.tensor(Ytime[idx]).to(dev))
                  +bce(lc,torch.tensor(Ycat[idx]).to(dev)))
        opt.zero_grad(); loss.backward(); opt.step(); sch.step(); tot+=loss.item()
    print(f"epoch {ep+1}/{EPOCHS} loss {tot/(len(perm)//BS):.4f} [{time.time()-t0:.0f}s]")

def predict(idxs):
    net.eval(); o={"solv":[],"prim":[],"temp":[],"time":[],"cat":[]}
    with torch.no_grad():
        for i in range(0,len(idxs),64):
            idx=idxs[i:i+64]; ii=ids_all[idx].to(dev); mm=mask_all[idx].to(dev)
            with torch.autocast("cuda",dtype=torch.bfloat16,enabled=bf16):
                ls,lp,lt,ltm,lc=net(ii,mm)
            o["solv"].append(torch.sigmoid(ls).float().cpu().numpy()); o["prim"].append(torch.softmax(lp,1).float().cpu().numpy())
            o["temp"].append(torch.softmax(lt,1).float().cpu().numpy()); o["time"].append(torch.softmax(ltm,1).float().cpu().numpy())
            o["cat"].append(torch.sigmoid(lc).float().cpu().numpy())
    return {k:np.concatenate(v) for k,v in o.items()}
pv=predict(vai)

# decode + composite on holdout
vt=train.iloc[vai].reset_index(drop=True); rare,shift=make_masks(vt)
truev={"solv":[solv_lists[i] for i in vai],"temp":[TEMPS[Ytemp[i]] for i in vai],"time":[TIMES[Ytime[i]] for i in vai],"cat":Ycat[vai]}
prior_t=np.bincount(Ytemp[tri],minlength=5)/len(tri); prior_tm=np.bincount(Ytime[tri],minlength=6)/len(tri)
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
single=[i for i in range(len(vai)) if len(truev["solv"][i])==1]
acc1=np.mean([pv["prim"].argmax(1)[i]>0 and SOLV[pv["prim"].argmax(1)[i]-1]==truev["solv"][i][0] for i in single])
print(f"\nChemBERTa ({MODEL}) solvent top-1 (singletons): {acc1:.4f}")
best=(-1,None)
for a_t in [0,0.3,0.5,0.7]:
  for cat_thr in [0.4,0.45,0.5]:
    for sec_thr in [0.3,0.4]:
      for nb in [1.0,1.5,2.0]:
        sc=composite(decode(pv,a_t,0,cat_thr,sec_thr,nb),truev,rare,shift)
        if sc>best[0]: best=(sc,(a_t,0,cat_thr,sec_thr,nb))
print(f"BEST holdout composite={best[0]:.4f} params={best[1]}")
composite(decode(pv,*best[1]),truev,rare,shift,debug=True)
np.savez(Path(__file__).resolve().parent/"chemberta_holdout.npz",**{f"v_{k}":pv[k] for k in pv}, vai=vai)
