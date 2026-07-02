import os, sys, time
from pathlib import Path
import numpy as np, pandas as pd, cv2
from PIL import Image
import torch, torch.nn as nn, torch.nn.functional as F
import torchvision

T0=time.time()
TIME_BUDGET=float(os.environ.get("PPMC_BUDGET","1700"))
EPOCHS=int(os.environ.get("PPMC_EPOCHS","45"))
WX=0.85
WY=0.65
T_GRID=0.03
CONFIGS=[("resnet18",101,96),("resnet34",303,96),("resnet50",42,96),
         ("resnet50",42,128),("resnet50",202,128),("resnet18",202,96),("resnet50",303,128)]

def find_data_dir():
    here=Path(__file__).resolve().parent
    for c in [here/"dataset"/"public",here/"dataset",Path("dataset/public"),Path("dataset"),here,Path(".")]:
        if (c/"train.csv").exists() and (c/"test.csv").exists(): return c
    raise FileNotFoundError("train.csv/test.csv not found")
DATA=find_data_dir(); WORK=Path("working"); WORK.mkdir(exist_ok=True,parents=True)

def xb_of(dx): return 0 if dx<-30 else 1 if dx<-22 else 2 if dx<-14 else 3 if dx<-6 else 4
def yb_of(dy): return 0 if dy<-2 else 1 if dy<0 else 2 if dy<2 else 3
def load_pair(path):
    im=np.array(Image.open(DATA/path).convert('RGB')); return im[:,0:96],im[:,104:200]

_CL=cv2.createCLAHE(clipLimit=2.0,tileGridSize=(8,8))
def red_mask(p):
    r,g,b=p[...,0].astype(int),p[...,1].astype(int),p[...,2].astype(int)
    return ((r>90)&(r-g>35)&(r-b>35)).astype(np.uint8)
def chan(im,c):
    if c=='gray': return cv2.cvtColor(im,cv2.COLOR_RGB2GRAY)
    if c=='green': return im[...,1]
    return _CL.apply(cv2.cvtColor(im,cv2.COLOR_RGB2GRAY))
DXLO,DXHI,DYLO,DYHI=-42,8,-9,9
GW=DXHI-DXLO+1; GH=DYHI-DYLO+1
CFG=[(h,c) for h in (10,12,14,16) for c in ('gray','clahe','green')]
_xcol=np.array([xb_of(dx) for dx in range(DXLO,DXHI+1)])
_yrow=np.array([yb_of(dy) for dy in range(DYLO,DYHI+1)])
def score_grid(L,R,half,c):
    cy,cx=48,48; tpl=L[cy-half:cy+half+1,cx-half:cx+half+1]
    m=red_mask(tpl); tg=chan(tpl,c).astype(np.float32); mask=(1-m).astype(np.float32)
    Rg=chan(R,c).astype(np.float32)
    res=cv2.matchTemplate(Rg,tg,cv2.TM_CCOEFF_NORMED,mask=mask)
    res=np.nan_to_num(res,nan=-1e9,posinf=-1e9,neginf=-1e9); H,W=res.shape
    grid=np.full((GH,GW),np.nan,np.float32)
    for gy,dy in enumerate(range(DYLO,DYHI+1)):
        ly=dy-half+cy
        if 0<=ly<H:
            for gx,dx in enumerate(range(DXLO,DXHI+1)):
                lx=dx-half+cx
                if 0<=lx<W: grid[gy,gx]=res[ly,lx]
    return grid
def fused_grid(L,R):
    acc=np.zeros((GH,GW),np.float32); cnt=np.zeros((GH,GW),np.float32)
    for h,c in CFG:
        g=score_grid(L,R,h,c); v=~np.isnan(g); acc[v]+=g[v]; cnt[v]+=1
    return np.where(cnt>0,acc/np.maximum(cnt,1),-1e9)
def classical_probs(df):
    N=len(df); Px=np.zeros((N,5)); Py=np.zeros((N,4))
    for i,p in enumerate(df['image_path'].values):
        L,R=load_pair(p); g=fused_grid(L,R).astype(np.float64); g[g<-1e8]=-np.inf
        w=np.exp((g-np.nanmax(g[np.isfinite(g)]))/T_GRID); w[~np.isfinite(w)]=0; w/=w.sum()+1e-12
        for b in range(5): Px[i,b]=w[:,_xcol==b].sum()
        for b in range(4): Py[i,b]=w[_yrow==b].sum()
    return Px,Py

dev='cuda' if torch.cuda.is_available() else 'cpu'
def load6(df,res):
    X=np.zeros((len(df),6,res,res),np.float32)
    for i,p in enumerate(df['image_path'].values):
        L,R=load_pair(p)
        if res!=96: L=cv2.resize(L,(res,res)); R=cv2.resize(R,(res,res))
        X[i,:3]=L.transpose(2,0,1)/255.; X[i,3:]=R.transpose(2,0,1)/255.
    return X
def make_bb(name):
    if name=='resnet50':
        m=torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V1)
    elif name=='resnet34':
        m=torchvision.models.resnet34(weights=torchvision.models.ResNet34_Weights.IMAGENET1K_V1)
    else:
        m=torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
    w=m.conv1.weight.data; c1=nn.Conv2d(6,64,7,2,3,bias=False)
    with torch.no_grad(): c1.weight[:]=torch.cat([w,w],1)*0.5
    m.conv1=c1; nf=m.fc.in_features; m.fc=nn.Identity(); return m,nf
class Net(nn.Module):
    def __init__(self,arch):
        super().__init__(); self.bb,nf=make_bb(arch); self.drop=nn.Dropout(0.4)
        self.hx=nn.Linear(nf+1,5); self.hy=nn.Linear(nf+1,4)
    def forward(self,x,h):
        f=self.drop(self.bb(x)); f=torch.cat([f,h[:,None]],1); return self.hx(f),self.hy(f)
def aug(x):
    B=x.size(0)
    if torch.rand(1).item()<0.6: x=x*(0.8+0.4*torch.rand(B,1,1,1,device=dev))
    if torch.rand(1).item()<0.5: x=x+0.05*torch.randn_like(x)
    if torch.rand(1).item()<0.4:
        m=x.mean((2,3),keepdim=True); x=(x-m)*(0.8+0.4*torch.rand(B,1,1,1,device=dev))+m
    return x
def train_one(X,h,xb,yb,arch,seed,res,epochs):
    torch.manual_seed(seed); np.random.seed(seed)
    net=Net(arch).to(dev); opt=torch.optim.AdamW(net.parameters(),lr=8e-4,weight_decay=1e-3)
    sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt,epochs); n=len(X)
    bs=24 if res<=96 else 16
    X=torch.tensor(X); h=torch.tensor(h); xb=torch.tensor(xb); yb=torch.tensor(yb)
    for ep in range(epochs):
        net.train(); perm=torch.randperm(n)
        for i in range(0,n,bs):
            b=perm[i:i+bs]; xba=aug(X[b].to(dev)); hba=h[b].to(dev)
            ybb=yb[b].to(dev); flip=torch.rand(len(b),device=dev)<0.5
            xba=torch.where(flip[:,None,None,None],xba.flip(2),xba); ybb=torch.where(flip,3-ybb,ybb)
            ox,oy=net(xba,hba); loss=F.cross_entropy(ox,xb[b].to(dev))+F.cross_entropy(oy,ybb)
            opt.zero_grad(); loss.backward(); opt.step()
        sch.step()
    return net
@torch.no_grad()
def predict(net,X,h,res):
    net.eval(); bs=64 if res<=96 else 32; ox=[]; oy=[]
    for i in range(0,len(X),bs):
        xb_=torch.tensor(X[i:i+bs]).to(dev); hb=torch.tensor(h[i:i+bs]).to(dev)
        px,py=net(xb_,hb); px=px.softmax(1); py=py.softmax(1)
        px2,py2=net(xb_.flip(2),hb); px=(px+px2.softmax(1))/2; py=(py+py2.softmax(1).flip(1))/2
        ox.append(px.cpu().numpy()); oy.append(py.cpu().numpy())
    return np.concatenate(ox),np.concatenate(oy)

def main():
    tr=pd.read_csv(DATA/'train.csv'); te=pd.read_csv(DATA/'test.csv')
    samp=pd.read_csv(DATA/'sample_submission.csv')
    y=tr['motion_class'].values; xb=(y%5).astype(np.int64); yb=(y//5).astype(np.int64)
    htr=(tr['horizon'].values.astype(np.float32)-3.0); hte=(te['horizon'].values.astype(np.float32)-3.0)

    print("[classical] matching test...",flush=True); tcx=time.time()
    CTx,CTy=classical_probs(te); print("  done %.0fs"%(time.time()-tcx),flush=True)

    print("[cnn] loading images...",flush=True)
    need=sorted({r for _,_,r in CONFIGS})
    Xtr={}; Xte={}
    for r in need:
        a=load6(tr,r); b=load6(te,r)
        mu=a.mean((0,2,3),keepdims=True); sd=a.std((0,2,3),keepdims=True)+1e-6
        Xtr[r]=(a-mu)/sd; Xte[r]=(b-mu)/sd

    NTx=np.zeros((len(te),5)); NTy=np.zeros((len(te),4)); nm=0; per={}
    for arch,seed,res in CONFIGS:
        est=per.get(res)
        if nm>=1 and est is not None and time.time()-T0+est*1.15>TIME_BUDGET:
            print("  budget: stop after %d models"%nm,flush=True); break
        tm=time.time(); net=train_one(Xtr[res],htr,xb,yb,arch,seed,res,EPOCHS)
        px,py=predict(net,Xte[res],hte,res); NTx+=px; NTy+=py; nm+=1
        per[res]=time.time()-tm; del net; torch.cuda.empty_cache()
        print("  model %d (%s s%d r%d) %.0fs elapsed=%.0fs"%(nm,arch,seed,res,per[res],time.time()-T0),flush=True)
    NTx/=max(nm,1); NTy/=max(nm,1)

    eps=1e-6
    bx=(WX*np.log(NTx+eps)+(1-WX)*np.log(CTx+eps)).argmax(1)
    by=(WY*np.log(NTy+eps)+(1-WY)*np.log(CTy+eps)).argmax(1)
    pred=(5*by+bx).astype(int)

    sub=pd.DataFrame({'sample_id':te['sample_id'].values,'motion_class':pred})
    sub=sub.set_index('sample_id').loc[samp['sample_id'].values].reset_index()
    assert list(sub.columns)==['sample_id','motion_class']
    assert sub['motion_class'].between(0,19).all() and sub['sample_id'].is_unique
    assert list(sub['sample_id'])==list(samp['sample_id'])
    sub.to_csv(WORK/'submission.csv',index=False)
    print("wrote",WORK/'submission.csv',sub.shape,"models=%d elapsed=%.0fs"%(nm,time.time()-T0),flush=True)
    print("pred dist:",np.bincount(pred,minlength=20),flush=True)

if __name__=='__main__': main()
