import numpy as np, pandas as pd, os, time, sys, random
import torch, torch.nn as nn, torch.nn.functional as F
import common as C
import torchvision
SEED=42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.benchmark=True
dev='cuda' if torch.cuda.is_available() else 'cpu'

tr=pd.read_csv(os.path.join(C.ROOT,'train.csv')); te=pd.read_csv(os.path.join(C.ROOT,'test.csv'))
y=tr['motion_class'].values; xb=(y%5).astype(np.int64); yb=(y//5).astype(np.int64)
hor_tr=tr['horizon'].values.astype(np.float32); hor_te=te['horizon'].values.astype(np.float32)

def load6(df):
    X=np.zeros((len(df),6,96,96),np.float32)
    for i,p in enumerate(df['image_path'].values):
        L,R=C.load_pair(p)
        X[i,:3]=L.transpose(2,0,1)/255.; X[i,3:]=R.transpose(2,0,1)/255.
    return X
print('loading images...'); t=time.time()
Xtr=load6(tr); Xte=load6(te); print('loaded',time.time()-t)
MEAN=Xtr.mean((0,2,3),keepdims=True); STD=Xtr.std((0,2,3),keepdims=True)+1e-6
Xtr=(Xtr-MEAN)/STD; Xte=(Xte-MEAN)/STD

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        m=torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
        w=m.conv1.weight.data  # (64,3,7,7)
        c1=nn.Conv2d(6,64,7,2,3,bias=False)
        c1.weight.data[:,:3]=w; c1.weight.data[:,3:]=w
        m.conv1=c1
        m.fc=nn.Identity()
        self.bb=m; self.drop=nn.Dropout(0.3)
        self.hx=nn.Linear(512+1,5); self.hy=nn.Linear(512+1,4)
    def forward(self,x,h):
        f=self.bb(x); f=self.drop(f)
        f=torch.cat([f,h[:,None]],1)
        return self.hx(f),self.hy(f)

def augment(x):
    # photometric only, label-safe. x: (B,6,96,96) tensor on dev
    B=x.size(0)
    if random.random()<0.5:
        g=(0.8+0.4*torch.rand(B,1,1,1,device=dev)); x=x*g
    if random.random()<0.5:
        x=x+0.05*torch.randn_like(x)
    if random.random()<0.3:  # cutout away from center
        for _ in range(2):
            cy,cx=random.randint(0,95),random.randint(0,95)
            if abs(cy-48)<14 and abs(cx-48)<14: continue
            s=random.randint(6,16)
            x[:,:,max(0,cy-s):cy+s,max(0,cx-s):cx+s]=0
    return x

def run_fold(k,folds,epochs=40,bs=32):
    trm=folds!=k; vam=folds==k
    Xt=torch.tensor(Xtr[trm]); ht=torch.tensor(hor_tr[trm])
    xbt=torch.tensor(xb[trm]); ybt=torch.tensor(yb[trm])
    Xv=torch.tensor(Xtr[vam]).to(dev); hv=torch.tensor(hor_tr[vam]).to(dev)
    net=Net().to(dev)
    opt=torch.optim.AdamW(net.parameters(),lr=1e-3,weight_decay=1e-3)
    sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt,epochs)
    n=len(Xt); best=0; best_prob=None
    for ep in range(epochs):
        net.train(); perm=torch.randperm(n)
        for i in range(0,n,bs):
            idx=perm[i:i+bs]
            xbatch=Xt[idx].to(dev); hbatch=ht[idx].to(dev)
            xbatch=augment(xbatch)
            ox,oy=net(xbatch,hbatch)
            loss=F.cross_entropy(ox,xbt[idx].to(dev))+F.cross_entropy(oy,ybt[idx].to(dev))
            opt.zero_grad(); loss.backward(); opt.step()
        sch.step()
        net.eval()
        with torch.no_grad():
            px,py=net(Xv,hv); px=px.softmax(1).cpu().numpy(); py=py.softmax(1).cpu().numpy()
        pred=5*py.argmax(1)+px.argmax(1)
        e=(pred==y[vam]).mean()
        if e>=best: best=e; best_prob=(px,py)
    return vam,best_prob,best

def main():
    folds=C.folds(y)
    OFx=np.zeros((len(y),5),np.float32); OFy=np.zeros((len(y),4),np.float32)
    for k in range(5):
        t=time.time(); vam,(px,py),be=run_fold(k,folds)
        OFx[vam]=px; OFy[vam]=py
        print(f'fold {k}: best_val_exact={be:.4f} ({time.time()-t:.0f}s)')
    pred=5*OFy.argmax(1)+OFx.argmax(1)
    print(f'\nOOF exact={ (pred==y).mean():.4f} xband={(OFx.argmax(1)==xb).mean():.4f} yband={(OFy.argmax(1)==yb).mean():.4f}')
    np.savez_compressed('research/cache/cnn_oof.npz',ofx=OFx,ofy=OFy)
    # train full for test preds (2 seeds avg light)
    print('training full model for test...')
    Xall=torch.tensor(Xtr); hall=torch.tensor(hor_tr)
    Xtet=torch.tensor(Xte).to(dev); htet=torch.tensor(hor_te).to(dev)
    xba=torch.tensor(xb); yba=torch.tensor(yb)
    TEx=np.zeros((len(te),5),np.float32); TEy=np.zeros((len(te),4),np.float32)
    nseed=2
    for s in range(nseed):
        torch.manual_seed(100+s)
        net=Net().to(dev); opt=torch.optim.AdamW(net.parameters(),lr=1e-3,weight_decay=1e-3)
        sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt,40); n=len(Xall)
        for ep in range(40):
            net.train(); perm=torch.randperm(n)
            for i in range(0,n,32):
                idx=perm[i:i+32]; xbatch=augment(Xall[idx].to(dev)); hbatch=hall[idx].to(dev)
                ox,oy=net(xbatch,hbatch)
                loss=F.cross_entropy(ox,xba[idx].to(dev))+F.cross_entropy(oy,yba[idx].to(dev))
                opt.zero_grad(); loss.backward(); opt.step()
            sch.step()
        net.eval()
        with torch.no_grad():
            px,py=net(Xtet,htet); TEx+=px.softmax(1).cpu().numpy()/nseed; TEy+=py.softmax(1).cpu().numpy()/nseed
    np.savez_compressed('research/cache/cnn_test.npz',tex=TEx,tey=TEy,sample_id=te['sample_id'].values)
    print('saved cnn test preds')

if __name__=='__main__':
    if len(sys.argv)>1 and sys.argv[1]=='smoke':
        folds=C.folds(y); t=time.time()
        vam,(px,py),be=run_fold(0,folds,epochs=3)
        print('smoke fold0 3ep best_exact=%.3f time=%.0fs'%(be,time.time()-t))
    else: main()
