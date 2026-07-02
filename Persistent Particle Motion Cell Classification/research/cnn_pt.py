import numpy as np, pandas as pd, os, time, sys, random
import torch, torch.nn as nn, torch.nn.functional as F, torchvision as tv
import common as C
ARCH=sys.argv[1] if len(sys.argv)>1 else 'resnet50'
SEED=int(sys.argv[2]) if len(sys.argv)>2 else 42
TAG=sys.argv[3] if len(sys.argv)>3 else 'pt'
EPOCHS=int(sys.argv[4]) if len(sys.argv)>4 else 45
RES=int(sys.argv[5]) if len(sys.argv)>5 else 96
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.benchmark=True
dev='cuda'
tr=pd.read_csv(os.path.join(C.ROOT,'train.csv')); te=pd.read_csv(os.path.join(C.ROOT,'test.csv'))
y=tr['motion_class'].values; xb=(y%5).astype(np.int64); yb=(y//5).astype(np.int64)
hor=(tr['horizon'].values.astype(np.float32)-3.0)

def load6(df):
    X=np.zeros((len(df),6,RES,RES),np.float32)
    import cv2
    for i,p in enumerate(df['image_path'].values):
        L,R=C.load_pair(p)
        if RES!=96: L=cv2.resize(L,(RES,RES)); R=cv2.resize(R,(RES,RES))
        X[i,:3]=L.transpose(2,0,1)/255.; X[i,3:]=R.transpose(2,0,1)/255.
    return X
Xtr=load6(tr)
MEAN=Xtr.mean((0,2,3),keepdims=True); STD=Xtr.std((0,2,3),keepdims=True)+1e-6
Xtr=(Xtr-MEAN)/STD

def backbone(name):
    if name in ('resnet18','resnet34','resnet50'):
        w={'resnet18':tv.models.ResNet18_Weights,'resnet34':tv.models.ResNet34_Weights,'resnet50':tv.models.ResNet50_Weights}[name].IMAGENET1K_V1
        m=getattr(tv.models,name)(weights=w); w0=m.conv1.weight.data
        c1=nn.Conv2d(6,64,7,2,3,bias=False)
        with torch.no_grad(): c1.weight[:]=torch.cat([ w0,w0 ],1)*0.5
        m.conv1=c1; nf=m.fc.in_features; m.fc=nn.Identity(); return m,nf
    if name=='convnext_tiny':
        m=tv.models.convnext_tiny(weights=tv.models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1)
        old=m.features[0][0]; w0=old.weight.data
        c=nn.Conv2d(6,old.out_channels,kernel_size=4,stride=4)
        with torch.no_grad(): c.weight[:]=torch.cat([w0,w0],1)*0.5; c.bias[:]=old.bias
        m.features[0][0]=c; nf=m.classifier[2].in_features; m.classifier[2]=nn.Identity(); return m,nf
    if name=='efficientnet_b0':
        m=tv.models.efficientnet_b0(weights=tv.models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        old=m.features[0][0]; w0=old.weight.data
        c=nn.Conv2d(6,old.out_channels,3,2,1,bias=False)
        with torch.no_grad(): c.weight[:]=torch.cat([w0,w0],1)*0.5
        m.features[0][0]=c; nf=m.classifier[1].in_features; m.classifier=nn.Identity(); return m,nf
    raise ValueError(name)

class Net(nn.Module):
    def __init__(self):
        super().__init__(); self.bb,nf=backbone(ARCH); self.drop=nn.Dropout(0.4)
        self.hx=nn.Linear(nf+1,5); self.hy=nn.Linear(nf+1,4)
    def forward(self,x,h):
        f=self.drop(self.bb(x)); f=torch.cat([f,h[:,None]],1); return self.hx(f),self.hy(f)
def aug(x):
    B=x.size(0)
    if random.random()<0.6: x=x*(0.8+0.4*torch.rand(B,1,1,1,device=dev))
    if random.random()<0.5: x=x+0.05*torch.randn_like(x)
    if random.random()<0.4:
        m=x.mean((2,3),keepdim=True); x=(x-m)*(0.8+0.4*torch.rand(B,1,1,1,device=dev))+m
    return x
@torch.no_grad()
def predict(net,X,h):
    net.eval(); out_x=[]; out_y=[]
    for i in range(0,len(X),64):
        xb_=torch.tensor(X[i:i+64]).to(dev); hb=torch.tensor(h[i:i+64]).to(dev)
        px,py=net(xb_,hb); px=px.softmax(1); py=py.softmax(1)
        px2,py2=net(xb_.flip(2),hb); px=(px+px2.softmax(1))/2; py=(py+py2.softmax(1).flip(1))/2
        out_x.append(px.cpu().numpy()); out_y.append(py.cpu().numpy())
    return np.concatenate(out_x),np.concatenate(out_y)
def train(idx,seed,epochs):
    torch.manual_seed(seed)
    Xt=torch.tensor(Xtr[idx]); ht=torch.tensor(hor[idx]); xbt=torch.tensor(xb[idx]); ybt=torch.tensor(yb[idx])
    net=Net().to(dev); opt=torch.optim.AdamW(net.parameters(),lr=8e-4,weight_decay=1e-3)
    sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt,epochs); n=len(Xt)
    for ep in range(epochs):
        net.train(); perm=torch.randperm(n)
        for i in range(0,n,24):
            b=perm[i:i+24]; xba=aug(Xt[b].to(dev)); hba=ht[b].to(dev)
            ybb=ybt[b].to(dev); flip=torch.rand(len(b),device=dev)<0.5
            xba=torch.where(flip[:,None,None,None],xba.flip(2),xba); ybb=torch.where(flip,3-ybb,ybb)
            ox,oy=net(xba,hba); loss=F.cross_entropy(ox,xbt[b].to(dev))+F.cross_entropy(oy,ybb)
            opt.zero_grad(); loss.backward(); opt.step()
        sch.step()
    return net
def main():
    folds=C.folds(y); OFx=np.zeros((len(y),5),np.float32); OFy=np.zeros((len(y),4),np.float32)
    for k in range(5):
        t=time.time(); idx=np.where(folds!=k)[0]; va=np.where(folds==k)[0]
        net=train(idx,SEED+k,EPOCHS)
        px,py=predict(net,Xtr[va],hor[va]); OFx[va]=px; OFy[va]=py; del net; torch.cuda.empty_cache()
        print(f'fold {k}: exact={(5*py.argmax(1)+px.argmax(1)==y[va]).mean():.4f} ({time.time()-t:.0f}s)',flush=True)
    pred=5*OFy.argmax(1)+OFx.argmax(1)
    print(f'[{ARCH} s{SEED} {TAG} res{RES}] OOF exact={(pred==y).mean():.4f} xb={(OFx.argmax(1)==xb).mean():.4f} yb={(OFy.argmax(1)==yb).mean():.4f}',flush=True)
    np.savez_compressed(f'research/cache/cnn_oof_{TAG}.npz',ofx=OFx,ofy=OFy)
if __name__=='__main__': main()
