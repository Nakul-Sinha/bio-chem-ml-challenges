"""Spectral Route — training/CV workhorse. Runs locally and on Kaggle unchanged.
Computes the EXACT 4-component Final on stratified CV + a re-degraded 'test-sim' stress read.
Config via env vars (see CFG). Saves OOF probs, test probs, and per-fold CV rows.
"""
import os, sys, time, json, math, io
import numpy as np, pandas as pd
import cv2
from PIL import Image
import torch, torch.nn as nn, torch.nn.functional as F
import timm
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from metric import final_score, print_scores

cv2.setNumThreads(0)
def env(k, d): return os.environ.get(k, d)
CFG = dict(
    DATA=env("DATA_DIR", os.path.abspath(os.path.join(os.path.dirname(__file__),"..","dataset"))),
    OUT =env("OUT_DIR", os.path.dirname(os.path.abspath(__file__))),
    MODEL=env("MODEL","convnext_tiny.fb_in22k"), IMG=int(env("IMG","224")),
    FOLDS=int(env("FOLDS","5")), ONLY_FOLD=env("ONLY_FOLD",""),  # "" = all folds
    EPOCHS=int(env("EPOCHS","12")), BS=int(env("BS","32")), LR=float(env("LR","2e-4")),
    WD=float(env("WD","0.05")), AUG=env("AUG","med"), SEED=int(env("SEED","0")),
    NW=int(env("NW","0")), PRETRAINED=int(env("PRETRAINED","1")), TAG=env("TAG","run"),
)
dev='cuda' if torch.cuda.is_available() else 'cpu'
print("CFG:", json.dumps(CFG), "| dev", dev)
torch.manual_seed(CFG["SEED"]); np.random.seed(CFG["SEED"])

DS=CFG["DATA"]
train_df=pd.read_csv(os.path.join(DS,"train.csv"))
test_df =pd.read_csv(os.path.join(DS,"test.csv"))
STRESS_THR=0.5757  # stress_flag == sensor_noise_score >= thr (100% acc on test)
train_stress=(train_df["sensor_noise_score"].values>=STRESS_THR).astype(np.int8)
y=train_df["target_id"].values.astype(np.int64)
print(f"train stress rate {train_stress.mean():.3f} (n_stress={train_stress.sum()})")

_cfg=timm.data.resolve_data_config({}, model=timm.create_model(CFG["MODEL"],pretrained=False))
MEAN=np.array(_cfg["mean"],dtype=np.float32); STD=np.array(_cfg["std"],dtype=np.float32)

# ---------------- degradation augmentation (mirrors the description's op list) ----------------
def _rand(a,b): return np.random.uniform(a,b)
def degrade(img, sev):
    """img uint8 HxWx3 RGB. sev in [0,1] severity/probability scaler. returns uint8."""
    h,w=img.shape[:2]
    if np.random.rand()<0.9*sev:  # brightness/contrast/gamma
        img=img.astype(np.float32)
        img=img*_rand(1-0.4*sev,1+0.4*sev)+_rand(-30*sev,30*sev)         # bright/contrast
        img=np.clip(img,0,255); g=_rand(1-0.5*sev,1+0.6*sev)
        img=255*np.power(np.clip(img/255,0,1),g)
        img=np.clip(img,0,255).astype(np.uint8)
    if np.random.rand()<0.6*sev:  # per-channel scale
        img=np.clip(img.astype(np.float32)*np.random.uniform(1-0.5*sev,1+0.5*sev,3),0,255).astype(np.uint8)
    if np.random.rand()<0.15*sev:  # channel dropout
        img=img.copy(); img[:,:,np.random.randint(3)]=0
    if np.random.rand()<0.6*sev:  # gaussian blur
        k=int(_rand(1,3+4*sev))*2+1; img=cv2.GaussianBlur(img,(k,k),0)
    if np.random.rand()<0.6*sev:  # gaussian noise
        img=np.clip(img.astype(np.float32)+np.random.randn(h,w,3)*_rand(3,35*sev),0,255).astype(np.uint8)
    if np.random.rand()<0.25*sev:  # salt & pepper
        m=np.random.rand(h,w)<0.02*sev; img=img.copy(); img[m]=np.random.choice([0,255])
    if np.random.rand()<0.4*sev:  # vignette
        yy,xx=np.ogrid[:h,:w]; cy,cx=h/2,w/2
        d=np.sqrt(((yy-cy)/cy)**2+((xx-cx)/cx)**2); v=np.clip(1-_rand(0.2,0.8*sev)*d,0.2,1)[...,None]
        img=np.clip(img.astype(np.float32)*v,0,255).astype(np.uint8)
    if np.random.rand()<0.2*sev:  # scanlines
        img=img.copy(); step=np.random.randint(2,5); img[::step]= (img[::step]*_rand(0.3,0.7)).astype(np.uint8)
    if np.random.rand()<0.7*sev:  # jpeg compression
        q=int(_rand(25,90)); enc=cv2.imencode(".jpg",img,[cv2.IMWRITE_JPEG_QUALITY,q])[1]
        img=cv2.imdecode(enc,cv2.IMREAD_COLOR)
    return img

SEV={"none":0.0,"light":0.5,"med":0.85,"heavy":1.0}
class DsImg(torch.utils.data.Dataset):
    def __init__(self, df, train=True, sev_key="med", test_sim=False):
        self.paths=[os.path.join(DS,str(p)) for p in df["image_path"]]
        self.train=train; self.sev=SEV[sev_key]; self.test_sim=test_sim
        self.y=df["target_id"].values.astype(np.int64) if "target_id" in df else np.zeros(len(df),np.int64)
        self.S=int(CFG["IMG"])
    def __len__(self): return len(self.paths)
    def __getitem__(self,i):
        im=Image.open(self.paths[i]).convert("RGB")
        if self.train:
            # random resized crop + hflip
            s=_rand(0.6,1.0); r=_rand(0.8,1.25); W,H=im.size
            cw=min(W,int(round(math.sqrt(s*W*H*r)))); ch=min(H,int(round(math.sqrt(s*W*H/r))))
            x0=np.random.randint(0,W-cw+1); y0=np.random.randint(0,H-ch+1)
            im=im.crop((x0,y0,x0+cw,y0+ch)).resize((self.S,self.S),Image.BILINEAR)
            if np.random.rand()<0.5: im=im.transpose(Image.FLIP_LEFT_RIGHT)
            a=np.asarray(im); a=degrade(a,self.sev)
        else:
            im=im.resize((self.S,self.S),Image.BILINEAR); a=np.asarray(im)
            if self.test_sim: a=degrade(a, SEV["heavy"])
        x=(a.astype(np.float32)/255.-MEAN)/STD
        x=torch.from_numpy(x).permute(2,0,1).float()
        if self.train and np.random.rand()<0.4*self.sev:  # random erasing
            eh=int(self.S*_rand(0.05,0.25)); ew=int(self.S*_rand(0.05,0.25))
            yy=np.random.randint(0,self.S-eh+1); xx=np.random.randint(0,self.S-ew+1); x[:,yy:yy+eh,xx:xx+ew]=0
        return x, self.y[i]

def make_model():
    m=timm.create_model(CFG["MODEL"], pretrained=bool(CFG["PRETRAINED"]), num_classes=6)
    return m.to(dev)

def run_fold(fold, tr_idx, va_idx, class_w):
    tr=DsImg(train_df.iloc[tr_idx], train=True, sev_key=CFG["AUG"])
    va=DsImg(train_df.iloc[va_idx], train=False)
    va_sim=DsImg(train_df.iloc[va_idx], train=False, test_sim=True)
    dl=lambda d,sh: torch.utils.data.DataLoader(d,batch_size=CFG["BS"],shuffle=sh,num_workers=CFG["NW"],
                                                pin_memory=(dev=='cuda'),drop_last=sh)
    model=make_model()
    opt=torch.optim.AdamW(model.parameters(),lr=CFG["LR"],weight_decay=CFG["WD"])
    steps=CFG["EPOCHS"]*max(1,len(tr)//CFG["BS"]); sched=torch.optim.lr_scheduler.OneCycleLR(
        opt,max_lr=CFG["LR"],total_steps=steps,pct_start=0.15)
    scaler=torch.cuda.amp.GradScaler(enabled=(dev=='cuda'))
    cw=torch.tensor(class_w,dtype=torch.float32,device=dev)
    tl=dl(tr,True)
    for ep in range(CFG["EPOCHS"]):
        model.train(); t0=time.time(); tot=0
        for x,yb in tl:
            x=x.to(dev,non_blocking=True); yb=yb.to(dev,non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.autocast('cuda',dtype=torch.float16,enabled=(dev=='cuda')):
                out=model(x); loss=F.cross_entropy(out,yb,weight=cw,label_smoothing=0.1)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); sched.step(); tot+=1
        print(f"  fold{fold} ep{ep+1}/{CFG['EPOCHS']} loss {loss.item():.3f} {time.time()-t0:.0f}s")
    # predict val (clean) + val (test-sim heavy) + test
    def predict(ds):
        model.eval(); ps=[]
        with torch.no_grad():
            for x,_ in dl(ds,False):
                x=x.to(dev)
                with torch.autocast('cuda',dtype=torch.float16,enabled=(dev=='cuda')):
                    ps.append(F.softmax(model(x),1).float().cpu().numpy())
        return np.concatenate(ps)
    return predict(va), predict(va_sim), predict(DsImg(test_df,train=False))

def main():
    from sklearn.model_selection import StratifiedKFold
    skf=StratifiedKFold(CFG["FOLDS"],shuffle=True,random_state=CFG["SEED"])
    cnt=np.bincount(y,minlength=6); class_w=(cnt.sum()/(6*cnt)); class_w/=class_w.mean()
    oof=np.zeros((len(train_df),6)); oof_sim=np.zeros((len(train_df),6))
    test_acc=np.zeros((len(test_df),6)); nfold=0; rows=[]
    for f,(tr,va) in enumerate(skf.split(y,y)):
        if CFG["ONLY_FOLD"]!="" and str(f)!=CFG["ONLY_FOLD"]: continue
        t0=time.time(); pv,psim,pte=run_fold(f,tr,va,class_w)
        oof[va]=pv; oof_sim[va]=psim; test_acc+=pte; nfold+=1
        d=final_score(y[va], pv.argmax(1), train_stress[va]); print_scores(d,f"fold{f} clean")
        dsim=final_score(y[va], psim.argmax(1), np.ones(len(va))); print_scores(dsim,f"fold{f} testsim")
        rows.append(dict(fold=f,**{k:d[k] for k in d},sim_MacroF1=dsim["MacroF1"],sec=time.time()-t0))
    if nfold>0:
        mask=oof.sum(1)>0
        d=final_score(y[mask], oof[mask].argmax(1), train_stress[mask]); print_scores(d,"OOF ALL clean")
        dsim=final_score(y[mask], oof_sim[mask].argmax(1), np.ones(mask.sum())); print_scores(dsim,"OOF ALL testsim")
        np.save(os.path.join(CFG["OUT"],f"oof_{CFG['TAG']}.npy"),oof)
        np.save(os.path.join(CFG["OUT"],f"oofsim_{CFG['TAG']}.npy"),oof_sim)
        np.save(os.path.join(CFG["OUT"],f"test_{CFG['TAG']}.npy"),test_acc/nfold)
        pd.DataFrame(rows).to_csv(os.path.join(CFG["OUT"],f"cv_{CFG['TAG']}.csv"),index=False)
        print("saved oof/test/cv for tag", CFG["TAG"])
if __name__=="__main__": main()
