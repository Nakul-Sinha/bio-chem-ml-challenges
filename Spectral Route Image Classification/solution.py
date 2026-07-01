#!/usr/bin/env python
"""Spectral Route Image Classification — self-contained, self-timing solution.
Runs on a single A10G (24GB) in <30 min. Fine-tunes pretrained backbone(s) with
degradation-simulating augmentation, hflip-TTA, and OOF decision-rule calibration
targeting the EXACT competition Final. Writes ./working/submission.csv (contract-checked).

No external data, no metadata fusion (it hurt the true metric), no private answers.
Stress proxy (sensor_noise_score>=0.5757) is used ONLY for the internal CV read, never as a feature.
"""
import os, sys, time, json, math, glob, zipfile
import numpy as np, pandas as pd, cv2
from PIL import Image
import torch, torch.nn as nn, torch.nn.functional as F
import timm
cv2.setNumThreads(0); T0=time.time()
def env(k,d): return os.environ.get(k,d)
TIME_BUDGET=float(env("TIME_BUDGET","1620"))   # 27 min; hard wall is 30
SEED=int(env("SEED","0")); FOLDS=int(env("FOLDS","5")); NW=int(env("NW","4"))
# Ensemble spec (finalized from Kaggle OOF). Trained in priority order; self-timing drops
# later (model,fold) jobs if the budget would be exceeded — a valid submission is always produced.
MODELS=json.loads(env("MODELS", json.dumps([
    # base first (strongest single, ~0.82 OOF cal): if the budget cuts swin, base alone still ships.
    {"name":"convnext_base.fb_in22k","img":224,"bs":24,"epochs":16,"sev":0.85},
    {"name":"swin_small_patch4_window7_224.ms_in22k","img":224,"bs":24,"epochs":16,"sev":0.85},
])))
dev='cuda' if torch.cuda.is_available() else 'cpu'
torch.manual_seed(SEED); np.random.seed(SEED); torch.backends.cudnn.benchmark=True
print(f"dev {dev} budget {TIME_BUDGET}s models {[m['name'] for m in MODELS]}", flush=True)

# ---------------- data ----------------
def _valid(d): return all(os.path.exists(os.path.join(d,f)) for f in ["train.csv","test.csv","sample_submission.csv"]) and os.path.isdir(os.path.join(d,"images"))
def find_data():
    for c in ["./dataset/public","dataset/public","./dataset","dataset","./public","public","."]:
        if os.path.isdir(c) and _valid(c): return c
    for p in glob.glob("**/train.csv", recursive=True)+glob.glob("/kaggle/input/**/train.csv", recursive=True):
        if _valid(os.path.dirname(p)): return os.path.dirname(p)
    raise FileNotFoundError("dataset (train.csv/test.csv/sample_submission.csv + images/) not found")
DS=find_data(); OUT="./working"; os.makedirs(OUT,exist_ok=True); print("DATA",DS,flush=True)
train_df=pd.read_csv(os.path.join(DS,"train.csv")); test_df=pd.read_csv(os.path.join(DS,"test.csv"))
ss=pd.read_csv(os.path.join(DS,"sample_submission.csv"))
ID2LAB={0:'route-aphelion',1:'route-borealis',2:'route-cygnus',3:'route-driftwood',4:'route-equinox',5:'route-fjord'}
y=train_df["target_id"].values.astype(np.int64)
stress=(train_df["sensor_noise_score"].values>=0.5757).astype(np.int8)

# ---------------- exact metric (for internal calibration only) ----------------
C=list(range(6))
def macro_f1(yt,yp):
    f=[]
    for c in C:
        tp=int(((yt==c)&(yp==c)).sum());fp=int(((yt!=c)&(yp==c)).sum());fn=int(((yt==c)&(yp!=c)).sum())
        d=2*tp+fp+fn;f.append(0.0 if d==0 else 2*tp/d)
    return float(np.mean(f))
def bal_acc(yt,yp):
    r=[]
    for c in C:
        p=int((yt==c).sum());r.append(0.0 if p==0 else int(((yt==c)&(yp==c)).sum())/p)
    return float(np.mean(r))
def gate_f1(yt,yp,a=0):
    tp=int(((yt==a)&(yp==a)).sum());fp=int(((yt!=a)&(yp==a)).sum());fn=int(((yt==a)&(yp!=a)).sum())
    d=2*tp+fp+fn;return 0.0 if d==0 else 2*tp/d
def stress_mf1(yt,yp,m):
    m=m.astype(bool)
    if m.sum()<2 or len(np.unique(yt[m]))<2: return macro_f1(yt,yp)
    return macro_f1(yt[m],yp[m])
def FINAL(yt,yp,m): return 0.40*macro_f1(yt,yp)+0.35*bal_acc(yt,yp)+0.10*gate_f1(yt,yp)+0.15*stress_mf1(yt,yp,m)

# ---------------- degradation aug ----------------
def R(a,b): return np.random.uniform(a,b)
def degrade(img,s):
    h,w=img.shape[:2]
    if np.random.rand()<0.9*s:
        img=img.astype(np.float32)*R(1-0.4*s,1+0.4*s)+R(-30*s,30*s);img=np.clip(img,0,255)
        img=np.clip(255*np.power(np.clip(img/255,0,1),R(1-0.5*s,1+0.6*s)),0,255).astype(np.uint8)
    if np.random.rand()<0.6*s: img=np.clip(img.astype(np.float32)*np.random.uniform(1-0.5*s,1+0.5*s,3),0,255).astype(np.uint8)
    if np.random.rand()<0.15*s: img=img.copy();img[:,:,np.random.randint(3)]=0
    if np.random.rand()<0.6*s: k=int(R(1,3+4*s))*2+1;img=cv2.GaussianBlur(img,(k,k),0)
    if np.random.rand()<0.6*s: img=np.clip(img.astype(np.float32)+np.random.randn(h,w,3)*R(3,35*s),0,255).astype(np.uint8)
    if np.random.rand()<0.25*s: m=np.random.rand(h,w)<0.02*s;img=img.copy();img[m]=np.random.choice([0,255])
    if np.random.rand()<0.4*s:
        yy,xx=np.ogrid[:h,:w];d=np.sqrt(((yy-h/2)/(h/2))**2+((xx-w/2)/(w/2))**2)
        img=np.clip(img.astype(np.float32)*np.clip(1-R(0.2,0.8*s)*d,0.2,1)[...,None],0,255).astype(np.uint8)
    if np.random.rand()<0.2*s: img=img.copy();st=np.random.randint(2,5);img[::st]=(img[::st]*R(0.3,0.7)).astype(np.uint8)
    if np.random.rand()<0.7*s: q=int(R(25,90));img=cv2.imdecode(cv2.imencode(".jpg",img,[cv2.IMWRITE_JPEG_QUALITY,q])[1],cv2.IMREAD_COLOR)
    return img
class Ds(torch.utils.data.Dataset):
    def __init__(s,df,train,img,sev,mean,std):
        s.p=[os.path.join(DS,str(x)) for x in df["image_path"]];s.tr=train;s.S=img;s.sev=sev;s.m=mean;s.sd=std
        s.y=df["target_id"].values.astype(np.int64) if "target_id" in df else np.zeros(len(df),np.int64)
    def __len__(s): return len(s.p)
    def __getitem__(s,i):
        im=Image.open(s.p[i]).convert("RGB")
        if s.tr:
            sc=R(0.6,1.0);rr=R(0.8,1.25);W,H=im.size
            cw=min(W,int(round(math.sqrt(sc*W*H*rr))));ch=min(H,int(round(math.sqrt(sc*W*H/rr))))
            x0=np.random.randint(0,W-cw+1);y0=np.random.randint(0,H-ch+1)
            im=im.crop((x0,y0,x0+cw,y0+ch)).resize((s.S,s.S),Image.BILINEAR)
            if np.random.rand()<0.5: im=im.transpose(Image.FLIP_LEFT_RIGHT)
            a=degrade(np.asarray(im),s.sev)
        else:
            a=np.asarray(im.resize((s.S,s.S),Image.BILINEAR))
        x=torch.from_numpy(((a.astype(np.float32)/255.-s.m)/s.sd)).permute(2,0,1).float()
        if s.tr and np.random.rand()<0.4*s.sev:
            eh=int(s.S*R(0.05,0.25));ew=int(s.S*R(0.05,0.25));yy=np.random.randint(0,s.S-eh+1);xx=np.random.randint(0,s.S-ew+1);x[:,yy:yy+eh,xx:xx+ew]=0
        return x,s.y[i]

def build(name):
    try: return timm.create_model(name,pretrained=True,num_classes=6).to(dev)
    except Exception as e:
        for wf in glob.glob("./weights/*"+name.split('.')[0]+"*"):
            m=timm.create_model(name,pretrained=False,num_classes=6)
            sd=torch.load(wf,map_location='cpu'); m.load_state_dict(sd,strict=False); print("loaded local weights",wf); return m.to(dev)
        raise
@torch.no_grad()
def predict(model,ds,bs):
    model.eval();dl=torch.utils.data.DataLoader(ds,batch_size=bs,shuffle=False,num_workers=NW,pin_memory=True);out=[]
    for x,_ in dl:
        x=x.to(dev)
        with torch.autocast('cuda',dtype=torch.float16,enabled=(dev=='cuda')):
            p=F.softmax(model(x),1)+F.softmax(model(torch.flip(x,[3])),1)
        out.append((p/2).float().cpu().numpy())
    return np.concatenate(out)
def train_fold(mc,tr_idx,va_idx,cw):
    m=timm.data.resolve_data_config({},model=timm.create_model(mc["name"],pretrained=False))
    mean=np.array(m["mean"],np.float32);std=np.array(m["std"],np.float32)
    model=build(mc["name"])
    dl=torch.utils.data.DataLoader(Ds(train_df.iloc[tr_idx],True,mc["img"],mc["sev"],mean,std),batch_size=mc["bs"],
                                   shuffle=True,num_workers=NW,pin_memory=True,drop_last=True,persistent_workers=(NW>0))
    opt=torch.optim.AdamW(model.parameters(),lr=2e-4,weight_decay=0.05)
    steps=mc["epochs"]*max(1,len(dl));sch=torch.optim.lr_scheduler.OneCycleLR(opt,2e-4,total_steps=steps,pct_start=0.15)
    sc=torch.cuda.amp.GradScaler(enabled=(dev=='cuda'));W=torch.tensor(cw,dtype=torch.float32,device=dev)
    for ep in range(mc["epochs"]):
        model.train()
        for x,yb in dl:
            x=x.to(dev,non_blocking=True);yb=yb.to(dev,non_blocking=True);opt.zero_grad(set_to_none=True)
            with torch.autocast('cuda',dtype=torch.float16,enabled=(dev=='cuda')):
                loss=F.cross_entropy(model(x),yb,weight=W,label_smoothing=0.1)
            sc.scale(loss).backward();sc.step(opt);sc.update();sch.step()
    va=predict(model,Ds(train_df.iloc[va_idx],False,mc["img"],0,mean,std),mc["bs"])
    te=predict(model,Ds(test_df,False,mc["img"],0,mean,std),mc["bs"])
    del model; torch.cuda.empty_cache() if dev=='cuda' else None
    return va,te

# ---------------- train (self-timing) ----------------
from sklearn.model_selection import StratifiedKFold
cnt=np.bincount(y,minlength=6);cw=(cnt.sum()/(6*cnt));cw/=cw.mean()
folds=list(StratifiedKFold(FOLDS,shuffle=True,random_state=SEED).split(y,y))
test_sum=np.zeros((len(test_df),6));test_n=0;oof_full=[];fold_t=[]
for mc in MODELS:
    oof_m=np.zeros((len(train_df),6));done=0
    for f,(tr,va) in enumerate(folds):
        est=(np.mean(fold_t) if fold_t else 0)
        if fold_t and (time.time()-T0)+est*1.15>TIME_BUDGET:
            print(f"[budget] stop before {mc['name']} fold{f}; elapsed {time.time()-T0:.0f}s",flush=True);break
        t=time.time();pv,pt=train_fold(mc,tr,va,cw);fold_t.append(time.time()-t)
        oof_m[va]=pv;test_sum+=pt;test_n+=1;done+=1
        print(f"{mc['name']} fold{f} done {fold_t[-1]:.0f}s elapsed {time.time()-T0:.0f}s",flush=True)
    if done==FOLDS: oof_full.append(oof_m)
    if fold_t and (time.time()-T0)+np.mean(fold_t)*1.15>TIME_BUDGET:
        print("[budget] stop before next model",flush=True);break
test=test_sum/max(1,test_n)

# ---------------- OOF decision-rule calibration ----------------
w=np.ones(6)
if oof_full:
    oof=np.mean(oof_full,0)
    best=FINAL(y,oof.argmax(1),stress)
    for _ in range(60):
        imp=False
        for c in range(6):
            for mm in [0.7,0.8,0.88,0.94,1.06,1.13,1.25,1.4]:
                w2=w.copy();w2[c]*=mm;s=FINAL(y,(oof*w2).argmax(1),stress)
                if s>best+1e-6: w,best,imp=w2,s,True
        if not imp: break
    print(f"OOF calibrated Final {best:.4f} (models fully trained: {len(oof_full)})",flush=True)
else:
    print("WARN: no fully-trained model for calibration; using raw argmax",flush=True)

# ---------------- write + verify submission ----------------
labels=(test*w).argmax(1)
sub=ss.copy(); sub["target"]=[ID2LAB[int(i)] for i in labels]; sub=sub[["id","target","stress_flag"]]
path=os.path.join(OUT,"submission.csv"); sub.to_csv(path,index=False)

def check(sub,ss):
    assert list(sub.columns)==["id","target","stress_flag"], "columns"
    assert len(sub)==len(ss), "row count"
    assert (sub["id"].astype(str).values==ss["id"].astype(str).values).all(), "id order"
    assert not sub["id"].duplicated().any(), "dup id"
    assert sub["target"].isin(set(ID2LAB.values())).all(), "target labels"
    assert (sub["stress_flag"].astype(str).values==ss["stress_flag"].astype(str).values).all(), "stress passthrough"
check(sub,ss)
print("SUBMISSION OK ->",path,"| dist",sub["target"].value_counts().to_dict(),"| total",round(time.time()-T0),"s",flush=True)
