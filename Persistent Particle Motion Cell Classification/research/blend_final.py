import numpy as np, pandas as pd, os, glob
import common as C
tr=pd.read_csv(os.path.join(C.ROOT,'train.csv'))
y=tr['motion_class'].values; xb=y%5; yb=y//5
f=C.folds(y)
oo=np.load('research/cache/oof_train.npz',allow_pickle=True)
grids=oo['grids']; DX,DY=oo['dx'],oo['dy']
DXs=np.arange(C.DXLO,C.DXHI+1); DYs=np.arange(C.DYLO,C.DYHI+1)
xcol=np.array([C.xb_of(dx) for dx in DXs]); yrow=np.array([C.yb_of(dy) for dy in DYs])

def grid_marg(T):
    N=grids.shape[0]; Px=np.zeros((N,5)); Py=np.zeros((N,4))
    for i in range(N):
        g=grids[i].astype(np.float64).copy(); g[g<-1e8]=-np.inf
        mx=np.nanmax(g[np.isfinite(g)]); w=np.exp((g-mx)/T); w[~np.isfinite(w)]=0; w/=w.sum()+1e-12
        for b in range(5): Px[i,b]=w[:,xcol==b].sum()
        for b in range(4): Py[i,b]=w[yrow==b].sum()
    return Px,Py

# CNN probs (avg all)
cnn=sorted(glob.glob('research/cache/cnn_oof*.npz'))
NPx=np.zeros((len(y),5)); NPy=np.zeros((len(y),4))
for fp in cnn: d=np.load(fp); NPx+=d['ofx']; NPy+=d['ofy']
NPx/=len(cnn); NPy/=len(cnn)
print('CNN files:',[os.path.basename(x) for x in cnn])
print('CNN ens: exact=%.4f xb=%.4f yb=%.4f'%((5*NPy.argmax(1)+NPx.argmax(1)==y).mean(),(NPx.argmax(1)==xb).mean(),(NPy.argmax(1)==yb).mean()))

T=0.03; CPx,CPy=grid_marg(T)
eps=1e-6
def blended(wx,wy,CPx,CPy):
    bx=(wx*np.log(NPx+eps)+(1-wx)*np.log(CPx+eps)).argmax(1)
    by=(wy*np.log(NPy+eps)+(1-wy)*np.log(CPy+eps)).argmax(1)
    return 5*by+bx

# naive (tune on all)
best=(-1,)
for wx in np.arange(0,1.001,0.05):
    for wy in np.arange(0,1.001,0.05):
        e=(blended(wx,wy,CPx,CPy)==y).mean()
        if e>best[0]: best=(e,wx,wy)
print('naive best-on-all: exact=%.4f wx=%.2f wy=%.2f'%best)

# nested CV: tune weights on train-folds, apply to held-out fold
pred=np.zeros(len(y),int)
for k in range(5):
    trm=f!=k; vam=f==k; bb=(-1,)
    for wx in np.arange(0,1.001,0.05):
        for wy in np.arange(0,1.001,0.05):
            bxk=(wx*np.log(NPx[trm]+eps)+(1-wx)*np.log(CPx[trm]+eps)).argmax(1)
            byk=(wy*np.log(NPy[trm]+eps)+(1-wy)*np.log(CPy[trm]+eps)).argmax(1)
            e=(5*byk+bxk==y[trm]).mean()
            if e>bb[0]: bb=(e,wx,wy)
    wx,wy=bb[1],bb[2]
    bxv=(wx*np.log(NPx[vam]+eps)+(1-wx)*np.log(CPx[vam]+eps)).argmax(1)
    byv=(wy*np.log(NPy[vam]+eps)+(1-wy)*np.log(CPy[vam]+eps)).argmax(1)
    pred[vam]=5*byv+bxv
print('NESTED-CV honest: exact=%.4f'%((pred==y).mean()))
# report component honest baselines
print('  CNN-only exact=%.4f  classical-argmax exact=%.4f'%((5*NPy.argmax(1)+NPx.argmax(1)==y).mean(),(C.cls_arr(DX,DY)==y).mean()))
