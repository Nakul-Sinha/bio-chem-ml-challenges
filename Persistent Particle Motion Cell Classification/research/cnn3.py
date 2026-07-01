import numpy as np, pandas as pd, os, time, sys, random
import torch, torch.nn as nn, torch.nn.functional as F, torchvision
import common as C
# args: ARCH SEED TAG EPOCHS HYBRID(0/1) TRANS(0/1)
ARCH=sys.argv[1] if len(sys.argv)>1 else 'small'
SEED=int(sys.argv[2]) if len(sys.argv)>2 else 42
TAG=sys.argv[3] if len(sys.argv)>3 else 'fs'
EPOCHS=int(sys.argv[4]) if len(sys.argv)>4 else 60
HYBRID=int(sys.argv[5]) if len(sys.argv)>5 else 0
TRANS=int(sys.argv[6]) if len(sys.argv)>6 else 0
OOF_ONLY=int(os.environ.get('PPMC_OOF_ONLY','1'))
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.benchmark=True
dev='cuda'
tr=pd.read_csv(os.path.join(C.ROOT,'train.csv')); te=pd.read_csv(os.path.join(C.ROOT,'test.csv'))
y=tr['motion_class'].values; xb=(y%5).astype(np.int64); yb=(y//5).astype(np.int64)
hor_tr=(tr['horizon'].values.astype(np.float32)-3.0); hor_te=(te['horizon'].values.astype(np.float32)-3.0)

def load6(df):
    X=np.zeros((len(df),6,96,96),np.float32)
    for i,p in enumerate(df['image_path'].values):
        L,R=C.load_pair(p); X[i,:3]=L.transpose(2,0,1)/255.; X[i,3:]=R.transpose(2,0,1)/255.
    return X
Xtr=load6(tr); Xte=load6(te)
MEAN=Xtr.mean((0,2,3),keepdims=True); STD=Xtr.std((0,2,3),keepdims=True)+1e-6
Xtr=(Xtr-MEAN)/STD; Xte=(Xte-MEAN)/STD

# classical aux features from cached grids (per-image, label-free)
def classical_feats(tag):
    d=np.load(f'research/cache/oof_{tag}.npz',allow_pickle=True)
    grids=d['grids']; dx=d['dx']; dy=d['dy']; pk=d['peak']
    DXs=np.arange(C.DXLO,C.DXHI+1); DYs=np.arange(C.DYLO,C.DYHI+1)
    xcol=np.array([C.xb_of(v) for v in DXs]); yrow=np.array([C.yb_of(v) for v in DYs])
    N=len(dx); Px=np.zeros((N,5),np.float32); Py=np.zeros((N,4),np.float32)
    for i in range(N):
        g=grids[i].astype(np.float64).copy(); g[g<-1e8]=-np.inf
        w=np.exp((g-np.nanmax(g[np.isfinite(g)]))/0.03); w[~np.isfinite(w)]=0; w/=w.sum()+1e-12
        for b in range(5): Px[i,b]=w[:,xcol==b].sum()
        for b in range(4): Py[i,b]=w[yrow==b].sum()
    aux=np.concatenate([ (dx/20.)[:,None],(dy/5.)[:,None],pk[:,None],Px,Py ],1).astype(np.float32)  # 12 dims
    return aux
AUXtr=classical_feats('train') if HYBRID else None
AUXte=classical_feats('test') if HYBRID else None
AUXDIM=(AUXtr.shape[1] if HYBRID else 0)

class SmallCNN(nn.Module):
    def __init__(self):
        super().__init__()
        def blk(i,o,p=True):
            L=[nn.Conv2d(i,o,3,1,1),nn.BatchNorm2d(o),nn.ReLU(inplace=True),
               nn.Conv2d(o,o,3,1,1),nn.BatchNorm2d(o),nn.ReLU(inplace=True)]
            if p: L.append(nn.MaxPool2d(2))
            return nn.Sequential(*L)
        self.f=nn.Sequential(blk(6,32),blk(32,64),blk(64,128),blk(128,128,False))
        self.gap=nn.AdaptiveAvgPool2d(1); self.nf=128
    def forward(self,x): return self.gap(self.f(x)).flatten(1)

def backbone(name):
    if name=='small': return SmallCNN(),128
    m=getattr(torchvision.models,name)(weights=None)  # FROM SCRATCH
    w=m.conv1.weight.data; c1=nn.Conv2d(6,64,7,2,3,bias=False); c1.weight.data[:]=torch.cat([w,w],1)*0.5
    m.conv1=c1; nf=m.fc.in_features; m.fc=nn.Identity(); return m,nf

class Net(nn.Module):
    def __init__(self):
        super().__init__(); self.bb,nf=backbone(ARCH); self.drop=nn.Dropout(0.4)
        extra=1  # horizon
        if HYBRID:
            self.aux=nn.Sequential(nn.Linear(AUXDIM,32),nn.ReLU(inplace=True),nn.Linear(32,32),nn.ReLU(inplace=True)); extra+=32
        self.hx=nn.Linear(nf+extra,5); self.hy=nn.Linear(nf+extra,4)
    def forward(self,x,h,a=None):
        f=self.drop(self.bb(x)); parts=[f,h[:,None]]
        if HYBRID: parts.append(self.aux(a))
        f=torch.cat(parts,1); return self.hx(f),self.hy(f)

def aug(x):
    B=x.size(0)
    if random.random()<0.6: x=x*(0.8+0.4*torch.rand(B,1,1,1,device=dev))
    if random.random()<0.5: x=x+0.06*torch.randn_like(x)
    if random.random()<0.4:
        m=x.mean((2,3),keepdim=True); x=(x-m)*(0.8+0.4*torch.rand(B,1,1,1,device=dev))+m
    if TRANS and random.random()<0.5:
        tx,ty=random.randint(-6,6),random.randint(-4,4)
        x=torch.roll(x,shifts=(ty,tx),dims=(2,3))
    return x

@torch.no_grad()
def predict(net,X,h,a):
    net.eval(); X=torch.tensor(X).to(dev); h=torch.tensor(h).to(dev)
    A=torch.tensor(a).to(dev) if HYBRID else None
    px,py=net(X,h,A); px=px.softmax(1); py=py.softmax(1)
    px2,py2=net(X.flip(2),h,A); px=(px+px2.softmax(1))/2; py=(py+py2.softmax(1).flip(1))/2
    return px.cpu().numpy(),py.cpu().numpy()

def train_model(idx,seed,epochs):
    torch.manual_seed(seed)
    Xt=torch.tensor(Xtr[idx]); ht=torch.tensor(hor_tr[idx]); xbt=torch.tensor(xb[idx]); ybt=torch.tensor(yb[idx])
    At=torch.tensor(AUXtr[idx]) if HYBRID else None
    net=Net().to(dev); opt=torch.optim.AdamW(net.parameters(),lr=1.2e-3,weight_decay=2e-3)
    sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt,epochs); n=len(Xt)
    for ep in range(epochs):
        net.train(); perm=torch.randperm(n)
        for i in range(0,n,32):
            b=perm[i:i+32]; xba=aug(Xt[b].to(dev)); hba=ht[b].to(dev)
            aba=At[b].to(dev) if HYBRID else None
            ybb=ybt[b].to(dev); flip=torch.rand(len(b),device=dev)<0.5
            xba=torch.where(flip[:,None,None,None],xba.flip(2),xba); ybb=torch.where(flip,3-ybb,ybb)
            ox,oy=net(xba,hba,aba); loss=F.cross_entropy(ox,xbt[b].to(dev))+F.cross_entropy(oy,ybb)
            opt.zero_grad(); loss.backward(); opt.step()
        sch.step()
    return net

def main():
    folds=C.folds(y); OFx=np.zeros((len(y),5),np.float32); OFy=np.zeros((len(y),4),np.float32)
    for k in range(5):
        t=time.time(); idx=np.where(folds!=k)[0]; va=np.where(folds==k)[0]
        net=train_model(idx,SEED+k,EPOCHS)
        px,py=predict(net,Xtr[va],hor_tr[va],AUXtr[va] if HYBRID else None)
        OFx[va]=px; OFy[va]=py
        print(f'fold {k}: exact={(5*py.argmax(1)+px.argmax(1)==y[va]).mean():.4f} ({time.time()-t:.0f}s)',flush=True)
    pred=5*OFy.argmax(1)+OFx.argmax(1)
    print(f'[{ARCH} s{SEED} {TAG} hyb{HYBRID} tr{TRANS}] OOF exact={(pred==y).mean():.4f} xb={(OFx.argmax(1)==xb).mean():.4f} yb={(OFy.argmax(1)==yb).mean():.4f}',flush=True)
    np.savez_compressed(f'research/cache/cnn_oof_{TAG}.npz',ofx=OFx,ofy=OFy)
    if OOF_ONLY: print('OOF only'); return
    TEx=np.zeros((len(te),5)); TEy=np.zeros((len(te),4))
    for s in range(2):
        net=train_model(np.arange(len(y)),300+s,EPOCHS)
        px,py=predict(net,Xte,hor_te,AUXte if HYBRID else None); TEx+=px/2; TEy+=py/2
    np.savez_compressed(f'research/cache/cnn_test_{TAG}.npz',tex=TEx,tey=TEy,sample_id=te['sample_id'].values)

if __name__=='__main__': main()
