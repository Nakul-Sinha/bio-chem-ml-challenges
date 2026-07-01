"""Kaggle kernel — Spectral Route, leakage-safe fusion.
Fine-tune a pretrained backbone + metadata fusion, validated on a SOURCE-GROUPED split
(group_labels.npy) so nothing rewards source memorization. Strong regularization
(stochastic depth, dropout, mixup, weight decay, heavy degradation aug). Pick ONE epoch
budget on the aggregate honest OOF. Reports grouped OOF Final (raw + calibrated).
"""
import os, sys, time, json, math, glob, subprocess
if os.environ.get("REINSTALL_TORCH","1")=="1":
    subprocess.run([sys.executable,"-m","pip","install","-q","--force-reinstall","--no-deps",
                    "torch==2.4.1","torchvision==0.19.1","--index-url","https://download.pytorch.org/whl/cu121"],check=False)
import numpy as np, pandas as pd, cv2
from PIL import Image
import torch, torch.nn as nn, torch.nn.functional as F
import timm
cv2.setNumThreads(0); T0=time.time()
def env(k,d): return os.environ.get(k,d)
CFG=dict(MODEL=env("MODEL","convnext_small.fb_in22k"), IMG=int(env("IMG","224")), BS=int(env("BS","32")),
         EPOCHS=int(env("EPOCHS","22")), LR=float(env("LR","1.5e-4")), WD=float(env("WD","0.05")),
         SEV=float(env("SEV","0.9")), DROP=float(env("DROP","0.4")), DROP_PATH=float(env("DROP_PATH","0.2")),
         MIXUP=float(env("MIXUP","0.2")), META=int(env("META","1")), SEED=int(env("SEED","0")), NW=int(env("NW","2")))
dev='cuda' if torch.cuda.is_available() else 'cpu'
print("CFG",json.dumps(CFG),"dev",dev, torch.cuda.get_device_name(0) if dev=='cuda' else "",flush=True)
torch.manual_seed(CFG["SEED"]); np.random.seed(CFG["SEED"]); torch.backends.cudnn.benchmark=True

def find(name):
    for p in glob.glob("/kaggle/input/**/"+name, recursive=True):
        return p
    return None
DS=os.path.dirname(find("train.csv")); print("DATA",DS,flush=True)
train_df=pd.read_csv(os.path.join(DS,"train.csv")); test_df=pd.read_csv(os.path.join(DS,"test.csv")); ss=pd.read_csv(os.path.join(DS,"sample_submission.csv"))
groups=np.load(find("group_labels.npy")); print("groups",len(np.unique(groups)),flush=True)
ID2LAB={0:'route-aphelion',1:'route-borealis',2:'route-cygnus',3:'route-driftwood',4:'route-equinox',5:'route-fjord'}
y=train_df["target_id"].values.astype(np.int64)
stress=(train_df["sensor_noise_score"].values>=0.5757).astype(np.int8)
META=[c for c in train_df.columns if c.endswith("_score")]
mu=train_df[META].mean().values; sd=train_df[META].std().values+1e-6
Mtr=((train_df[META].values-mu)/sd).astype(np.float32); Mte=((test_df[META].values-mu)/sd).astype(np.float32)
np.save("/kaggle/working/meta_mu.npy",mu); np.save("/kaggle/working/meta_sd.npy",sd)

C=list(range(6))
def mf1(yt,yp):
    f=[]
    for c in C:
        tp=int(((yt==c)&(yp==c)).sum());fp=int(((yt!=c)&(yp==c)).sum());fn=int(((yt==c)&(yp!=c)).sum());d=2*tp+fp+fn;f.append(0.0 if d==0 else 2*tp/d)
    return float(np.mean(f))
def ba(yt,yp):
    r=[]
    for c in C:
        p=int((yt==c).sum());r.append(0.0 if p==0 else int(((yt==c)&(yp==c)).sum())/p)
    return float(np.mean(r))
def gate(yt,yp):
    tp=int(((yt==0)&(yp==0)).sum());fp=int(((yt!=0)&(yp==0)).sum());fn=int(((yt==0)&(yp!=0)).sum());d=2*tp+fp+fn;return 0.0 if d==0 else 2*tp/d
def smf(yt,yp,m):
    m=m.astype(bool)
    return mf1(yt,yp) if (m.sum()<2 or len(np.unique(yt[m]))<2) else mf1(yt[m],yp[m])
def FINAL(yt,yp,m): return 0.40*mf1(yt,yp)+0.35*ba(yt,yp)+0.10*gate(yt,yp)+0.15*smf(yt,yp,m)

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
        yy,xx=np.ogrid[:h,:w];d=np.sqrt(((yy-h/2)/(h/2))**2+((xx-w/2)/(w/2))**2);img=np.clip(img.astype(np.float32)*np.clip(1-R(0.2,0.8*s)*d,0.2,1)[...,None],0,255).astype(np.uint8)
    if np.random.rand()<0.2*s: img=img.copy();st=np.random.randint(2,5);img[::st]=(img[::st]*R(0.3,0.7)).astype(np.uint8)
    if np.random.rand()<0.7*s: q=int(R(25,90));img=cv2.imdecode(cv2.imencode(".jpg",img,[cv2.IMWRITE_JPEG_QUALITY,q])[1],cv2.IMREAD_COLOR)
    return img
_m=timm.create_model(CFG["MODEL"],pretrained=False);dc=timm.data.resolve_data_config({},model=_m)
MEAN=np.array(dc["mean"],np.float32);STD=np.array(dc["std"],np.float32);del _m
class Ds(torch.utils.data.Dataset):
    def __init__(s,df,Mx,train):
        s.p=[os.path.join(DS,str(x)) for x in df["image_path"]];s.M=Mx;s.tr=train;s.S=CFG["IMG"]
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
            a=degrade(np.asarray(im),CFG["SEV"])
        else: a=np.asarray(im.resize((s.S,s.S),Image.BILINEAR))
        x=torch.from_numpy(((a.astype(np.float32)/255.-MEAN)/STD)).permute(2,0,1).float()
        if s.tr and np.random.rand()<0.4*CFG["SEV"]:
            eh=int(s.S*R(0.05,0.25));ew=int(s.S*R(0.05,0.25));yy=np.random.randint(0,s.S-eh+1);xx=np.random.randint(0,s.S-ew+1);x[:,yy:yy+eh,xx:xx+ew]=0
        return x, torch.from_numpy(s.M[i]), s.y[i]
class Fusion(nn.Module):
    def __init__(s):
        super().__init__()
        s.bb=timm.create_model(CFG["MODEL"],pretrained=True,num_classes=0,drop_path_rate=CFG["DROP_PATH"])
        d=s.bb.num_features; s.use=CFG["META"]==1
        s.mlp=nn.Sequential(nn.Linear(len(META),96),nn.BatchNorm1d(96),nn.ReLU(),nn.Dropout(CFG["DROP"]),nn.Linear(96,96),nn.ReLU()) if s.use else None
        s.head=nn.Sequential(nn.Dropout(CFG["DROP"]),nn.Linear(d+(96 if s.use else 0),6))
    def forward(s,x,m):
        f=s.bb(x)
        if s.use: f=torch.cat([f,s.mlp(m)],1)
        return s.head(f)
def dl(ds,sh): return torch.utils.data.DataLoader(ds,batch_size=CFG["BS"],shuffle=sh,num_workers=CFG["NW"],pin_memory=True,drop_last=sh,persistent_workers=(CFG["NW"]>0))
@torch.no_grad()
def predict(model,ds):
    model.eval();out=[]
    for x,m,_ in dl(ds,False):
        x=x.to(dev);m=m.to(dev)
        with torch.autocast('cuda',dtype=torch.float16,enabled=(dev=='cuda')):
            p=F.softmax(model(x,m),1)+F.softmax(model(torch.flip(x,[3]),m),1)
        out.append((p/2).float().cpu().numpy())
    return np.concatenate(out)

from sklearn.model_selection import GroupKFold
cnt=np.bincount(y,minlength=6);cw=torch.tensor((cnt.sum()/(6*cnt))/ (cnt.sum()/(6*cnt)).mean(),dtype=torch.float32,device=dev)
E=CFG["EPOCHS"]
oof_ep=np.zeros((E,len(train_df),6),np.float32); test_ep=np.zeros((E,len(test_df),6),np.float32)
for f,(tr,va) in enumerate(GroupKFold(5).split(y,y,groups)):
    model=Fusion().to(dev)
    opt=torch.optim.AdamW(model.parameters(),lr=CFG["LR"],weight_decay=CFG["WD"])
    tl=dl(Ds(train_df.iloc[tr],Mtr[tr],True),True)
    sch=torch.optim.lr_scheduler.OneCycleLR(opt,CFG["LR"],total_steps=E*max(1,len(tl)),pct_start=0.1)
    sc=torch.cuda.amp.GradScaler(enabled=(dev=='cuda'))
    vds=Ds(train_df.iloc[va],Mtr[va],False); tds=Ds(test_df,Mte,False)
    for ep in range(E):
        model.train()
        for x,m,yb in tl:
            x=x.to(dev,non_blocking=True);m=m.to(dev,non_blocking=True);yb=yb.to(dev,non_blocking=True)
            if CFG["MIXUP"]>0 and np.random.rand()<0.5:
                lam=float(np.random.beta(CFG["MIXUP"],CFG["MIXUP"]));perm=torch.randperm(x.size(0),device=dev)
                x=lam*x+(1-lam)*x[perm];m=lam*m+(1-lam)*m[perm]
                with torch.autocast('cuda',dtype=torch.float16,enabled=(dev=='cuda')):
                    o=model(x,m);loss=lam*F.cross_entropy(o,yb,weight=cw,label_smoothing=0.1)+(1-lam)*F.cross_entropy(o,yb[perm],weight=cw,label_smoothing=0.1)
            else:
                with torch.autocast('cuda',dtype=torch.float16,enabled=(dev=='cuda')):
                    loss=F.cross_entropy(model(x,m),yb,weight=cw,label_smoothing=0.1)
            opt.zero_grad(set_to_none=True);sc.scale(loss).backward();sc.step(opt);sc.update();sch.step()
        oof_ep[ep,va]=predict(model,vds); test_ep[ep]+=predict(model,tds)/5.0
    print(f"fold{f} done elapsed {time.time()-T0:.0f}s",flush=True)
# pick single epoch budget on aggregate honest OOF
scores=[FINAL(y,oof_ep[ep].argmax(1),stress) for ep in range(E)]
best=int(np.argmax(scores)); oof=oof_ep[best]; test=test_ep[best]
print("per-epoch grouped OOF Final:",[f"{s:.3f}" for s in scores],flush=True)
print(f"best epoch {best+1}/{E} grouped OOF Final {scores[best]:.4f}",flush=True)
# calibrate on grouped OOF
w=np.ones(6);bfin=FINAL(y,oof.argmax(1),stress)
for _ in range(60):
    imp=False
    for c in range(6):
        for mm in [0.7,0.8,0.88,0.94,1.06,1.13,1.25,1.4]:
            w2=w.copy();w2[c]*=mm;s=FINAL(y,(oof*w2).argmax(1),stress)
            if s>bfin+1e-6:w,bfin,imp=w2,s,True
    if not imp:break
print(f"grouped OOF Final calibrated {bfin:.4f}  (was raw {scores[best]:.4f})",flush=True)
sub=ss.copy();sub["target"]=[ID2LAB[int(i)] for i in (test*w).argmax(1)];sub[["id","target","stress_flag"]].to_csv("/kaggle/working/submission.csv",index=False)
np.save("/kaggle/working/oof.npy",oof);np.save("/kaggle/working/test.npy",test);np.save("/kaggle/working/calw.npy",w)
json.dump(dict(cfg=CFG,oof_raw=scores[best],oof_cal=bfin,best_epoch=best+1,elapsed=time.time()-T0),open("/kaggle/working/summary.json","w"),indent=2,default=float)
print("beats 0.6:",bfin>0.6," | TOTAL",round(time.time()-T0),"s",flush=True)
