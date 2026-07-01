"""Kaggle GPU kernel — Spectral Route Image Classification.
5-fold fine-tune of a pretrained backbone with degradation-simulating augmentation,
hflip TTA, and OOF decision-rule calibration. Computes the EXACT 4-component Final
(stress proxy: sensor_noise_score>=0.5757) + a re-degraded 'test-sim' stress read.
Writes OOF/test probs, cv summary, and submission.csv to /kaggle/working.
"""
import os, sys, time, json, math, glob, zipfile, subprocess
# Kaggle's API-assigned GPU is a P100 (sm_60); Kaggle's prebuilt torch dropped sm_60.
# Install the official torch build (still ships sm_60) so the P100 works.
if os.environ.get("REINSTALL_TORCH", "1") == "1":
    print("installing sm_60-compatible torch ...", flush=True)
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--force-reinstall", "--no-deps",
                    "torch==2.4.1", "torchvision==0.19.1",
                    "--index-url", "https://download.pytorch.org/whl/cu121"], check=False)
import numpy as np, pandas as pd, cv2
from PIL import Image
import torch, torch.nn as nn, torch.nn.functional as F
import timm
cv2.setNumThreads(0)
if torch.cuda.is_available():
    try:
        _v=(torch.randn(128,128,device='cuda')@torch.randn(128,128,device='cuda')).sum().item()
        print("GPU OK", torch.cuda.get_device_name(0), "cap", torch.cuda.get_device_capability(0), "torch", torch.__version__, flush=True)
    except Exception as e:
        print("GPU SANITY FAILED:", e, flush=True); sys.exit(3)
else:
    print("WARNING: CUDA not available", flush=True)
T0=time.time()
def env(k,d): return os.environ.get(k,d)
CFG=dict(MODEL=env("MODEL","swin_small_patch4_window7_224.ms_in22k"), IMG=int(env("IMG","224")),
         FOLDS=int(env("FOLDS","5")), EPOCHS=int(env("EPOCHS","16")), BS=int(env("BS","24")),
         LR=float(env("LR","2e-4")), WD=float(env("WD","0.05")), SEV=float(env("SEV","0.85")),
         SEED=int(env("SEED","0")), NW=int(env("NW","2")), TAG=env("TAG","sw224"))
dev='cuda' if torch.cuda.is_available() else 'cpu'
print("CFG",json.dumps(CFG),"dev",dev, torch.cuda.get_device_name(0) if dev=='cuda' else "")
torch.manual_seed(CFG["SEED"]); np.random.seed(CFG["SEED"])
torch.backends.cudnn.benchmark=True

# ---------- locate data ----------
def _valid(d):
    return os.path.exists(os.path.join(d,"train.csv")) and os.path.exists(os.path.join(d,"sample_submission.csv")) and os.path.isdir(os.path.join(d,"images"))
def find_data():
    for p in glob.glob("/kaggle/input/**/train.csv", recursive=True):
        d=os.path.dirname(p)
        if _valid(d): return d
    for z in glob.glob("/kaggle/input/**/*.zip", recursive=True):
        ex="/kaggle/tmp/data"; os.makedirs(ex,exist_ok=True); zipfile.ZipFile(z).extractall(ex)
        for p in glob.glob(ex+"/**/train.csv", recursive=True):
            d=os.path.dirname(p)
            if _valid(d): return d
    raise FileNotFoundError("valid train.csv (with sample_submission.csv + images/) not found under /kaggle/input")
DS=find_data(); print("DATA_DIR",DS)
train_df=pd.read_csv(os.path.join(DS,"train.csv")); test_df=pd.read_csv(os.path.join(DS,"test.csv"))
ss=pd.read_csv(os.path.join(DS,"sample_submission.csv"))
ID2LAB={0:'route-aphelion',1:'route-borealis',2:'route-cygnus',3:'route-driftwood',4:'route-equinox',5:'route-fjord'}
y=train_df["target_id"].values.astype(np.int64)
STRESS_THR=0.5757
tr_stress=(train_df["sensor_noise_score"].values>=STRESS_THR).astype(np.int8)
print("train stress rate %.3f"%tr_stress.mean())

# ---------- exact metric ----------
C=list(range(6)); ANCHOR=0
def macro_f1(yt,yp,lab=C):
    yt=np.asarray(yt);yp=np.asarray(yp);f=[]
    for c in lab:
        tp=int(((yt==c)&(yp==c)).sum());fp=int(((yt!=c)&(yp==c)).sum());fn=int(((yt==c)&(yp!=c)).sum())
        d=2*tp+fp+fn;f.append(0.0 if d==0 else 2*tp/d)
    return float(np.mean(f))
def bal_acc(yt,yp,lab=C):
    yt=np.asarray(yt);yp=np.asarray(yp);r=[]
    for c in lab:
        p=int((yt==c).sum()); r.append(0.0 if p==0 else int(((yt==c)&(yp==c)).sum())/p)
    return float(np.mean(r))
def gate_f1(yt,yp,a=ANCHOR):
    yt=np.asarray(yt);yp=np.asarray(yp)
    tp=int(((yt==a)&(yp==a)).sum());fp=int(((yt!=a)&(yp==a)).sum());fn=int(((yt==a)&(yp!=a)).sum())
    d=2*tp+fp+fn;return 0.0 if d==0 else 2*tp/d
def stress_mf1(yt,yp,m):
    yt=np.asarray(yt);yp=np.asarray(yp);m=np.asarray(m).astype(bool)
    if m.sum()<2 or len(np.unique(yt[m]))<2: return macro_f1(yt,yp)
    return macro_f1(yt[m],yp[m])
def final_score(yt,yp,m):
    mf=macro_f1(yt,yp);ba=bal_acc(yt,yp);gf=gate_f1(yt,yp);sf=stress_mf1(yt,yp,m)
    return dict(Final=0.40*mf+0.35*ba+0.10*gf+0.15*sf,MacroF1=mf,BalAcc=ba,Gate=gf,Stress=sf)
def pr(d,t=""): print(f"{t:16s} Final {d['Final']:.4f} | MF1 {d['MacroF1']:.4f} BA {d['BalAcc']:.4f} Gate {d['Gate']:.4f} Str {d['Stress']:.4f}")

# ---------- degradation aug ----------
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

_m=timm.create_model(CFG["MODEL"],pretrained=False);dc=timm.data.resolve_data_config({},model=_m)
MEAN=np.array(dc["mean"],np.float32);STD=np.array(dc["std"],np.float32);del _m
class Ds(torch.utils.data.Dataset):
    def __init__(s,df,train,testsim=False):
        s.p=[os.path.join(DS,str(x)) for x in df["image_path"]];s.tr=train;s.sim=testsim
        s.y=df["target_id"].values.astype(np.int64) if "target_id" in df else np.zeros(len(df),np.int64);s.S=CFG["IMG"]
    def __len__(s): return len(s.p)
    def __getitem__(s,i):
        im=Image.open(s.p[i]).convert("RGB")
        if s.tr:
            sc=R(0.6,1.0);rr=R(0.8,1.25);W,H=im.size
            cw=min(W,int(round(math.sqrt(sc*W*H*rr))));ch=min(H,int(round(math.sqrt(sc*W*H/rr))))
            x0=np.random.randint(0,W-cw+1);y0=np.random.randint(0,H-ch+1)
            im=im.crop((x0,y0,x0+cw,y0+ch)).resize((s.S,s.S),Image.BILINEAR)
            if np.random.rand()<0.5: im=im.transpose(Image.FLIP_LEFT_RIGHT)
            a=degrade(np.asarray(im),CFG["SEV"])
        else:
            im=im.resize((s.S,s.S),Image.BILINEAR);a=np.asarray(im)
            if s.sim: a=degrade(a,1.0)
        x=torch.from_numpy(((a.astype(np.float32)/255.-MEAN)/STD)).permute(2,0,1).float()
        if s.tr and np.random.rand()<0.4*CFG["SEV"]:
            eh=int(s.S*R(0.05,0.25));ew=int(s.S*R(0.05,0.25));yy=np.random.randint(0,s.S-eh+1);xx=np.random.randint(0,s.S-ew+1);x[:,yy:yy+eh,xx:xx+ew]=0
        return x,s.y[i]

def loader(ds,sh): return torch.utils.data.DataLoader(ds,batch_size=CFG["BS"],shuffle=sh,num_workers=CFG["NW"],pin_memory=True,drop_last=sh,persistent_workers=(CFG["NW"]>0))
@torch.no_grad()
def predict(model,ds,tta=True):
    model.eval();out=[]
    for x,_ in loader(ds,False):
        x=x.to(dev)
        with torch.autocast('cuda',dtype=torch.float16):
            p=F.softmax(model(x),1)
            if tta: p=p+F.softmax(model(torch.flip(x,[3])),1)
        out.append((p/(2 if tta else 1)).float().cpu().numpy())
    return np.concatenate(out)

def run_fold(f,tr,va,cw):
    model=timm.create_model(CFG["MODEL"],pretrained=True,num_classes=6).to(dev)
    opt=torch.optim.AdamW(model.parameters(),lr=CFG["LR"],weight_decay=CFG["WD"])
    tl=loader(Ds(train_df.iloc[tr],True),True)
    steps=CFG["EPOCHS"]*max(1,len(tl));sch=torch.optim.lr_scheduler.OneCycleLR(opt,CFG["LR"],total_steps=steps,pct_start=0.15)
    sc=torch.cuda.amp.GradScaler();W=torch.tensor(cw,dtype=torch.float32,device=dev)
    for ep in range(CFG["EPOCHS"]):
        model.train();t=time.time()
        for x,yb in tl:
            x=x.to(dev,non_blocking=True);yb=yb.to(dev,non_blocking=True);opt.zero_grad(set_to_none=True)
            with torch.autocast('cuda',dtype=torch.float16):
                loss=F.cross_entropy(model(x),yb,weight=W,label_smoothing=0.1)
            sc.scale(loss).backward();sc.step(opt);sc.update();sch.step()
        print(f"  f{f} ep{ep+1}/{CFG['EPOCHS']} loss {loss.item():.3f} {time.time()-t:.0f}s elapsed {time.time()-T0:.0f}s",flush=True)
    return predict(model,Ds(train_df.iloc[va],False)),predict(model,Ds(train_df.iloc[va],False,True)),predict(model,Ds(test_df,False))

# ---------- CV ----------
from sklearn.model_selection import StratifiedKFold
cnt=np.bincount(y,minlength=6);cw=(cnt.sum()/(6*cnt));cw/=cw.mean()
skf=StratifiedKFold(CFG["FOLDS"],shuffle=True,random_state=CFG["SEED"])
oof=np.zeros((len(train_df),6));oofs=np.zeros((len(train_df),6));tep=np.zeros((len(test_df),6));rows=[]
for f,(tr,va) in enumerate(skf.split(y,y)):
    pv,ps,pt=run_fold(f,tr,va,cw);oof[va]=pv;oofs[va]=ps;tep+=pt/CFG["FOLDS"]
    d=final_score(y[va],pv.argmax(1),tr_stress[va]);pr(d,f"fold{f}");rows.append(d)
d=final_score(y,oof.argmax(1),tr_stress);pr(d,"OOF raw")
dsim=final_score(y,oofs.argmax(1),np.ones(len(y)));pr(dsim,"OOF testsim")

# ---------- OOF decision-rule calibration (coordinate ascent on per-class prob weights) ----------
def scored(weights):
    yp=(oof*weights).argmax(1);return final_score(y,yp,tr_stress)["Final"]
w=np.ones(6);best=scored(w)
for it in range(40):
    improved=False
    for c in range(6):
        for mult in [0.8,0.9,0.95,1.05,1.1,1.25]:
            w2=w.copy();w2[c]*=mult;s=scored(w2)
            if s>best+1e-6: w=w2;best=s;improved=True
    if not improved: break
dc_=final_score(y,(oof*w).argmax(1),tr_stress);pr(dc_,"OOF calibrated");print("cal weights",np.round(w,3))

# ---------- submissions ----------
sub=ss.copy();sub["target"]=[ID2LAB[i] for i in (tep*w).argmax(1)]
sub=sub[["id","target","stress_flag"]];sub.to_csv("/kaggle/working/submission.csv",index=False)
sub_raw=ss.copy();sub_raw["target"]=[ID2LAB[i] for i in tep.argmax(1)];sub_raw[["id","target","stress_flag"]].to_csv("/kaggle/working/submission_raw.csv",index=False)
np.save("/kaggle/working/oof.npy",oof);np.save("/kaggle/working/test.npy",tep);np.save("/kaggle/working/calw.npy",w)
summary=dict(cfg=CFG,oof_raw=final_score(y,oof.argmax(1),tr_stress),oof_cal=dc_,oof_testsim=dsim,
             baseline=0.5368,elapsed=time.time()-T0)
json.dump(summary,open("/kaggle/working/summary.json","w"),indent=2,default=float)
print("\n=== SUMMARY ===");print(json.dumps(summary,indent=2,default=float))
print("beats_baseline(cal):",dc_["Final"]>0.5368,"margin",round(dc_["Final"]-0.5368,4))
print("TOTAL",round(time.time()-T0),"s")
