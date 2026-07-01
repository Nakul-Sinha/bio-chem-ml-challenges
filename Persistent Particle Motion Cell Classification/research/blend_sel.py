import numpy as np, pandas as pd, os, sys
import common as C
tr=pd.read_csv(os.path.join(C.ROOT,'train.csv'))
y=tr['motion_class'].values; xb=y%5; yb=y//5; f=C.folds(y)
oo=np.load('research/cache/oof_train.npz',allow_pickle=True); grids=oo['grids']; DX,DY=oo['dx'],oo['dy']
DXs=np.arange(C.DXLO,C.DXHI+1); DYs=np.arange(C.DYLO,C.DYHI+1)
xcol=np.array([C.xb_of(dx) for dx in DXs]); yrow=np.array([C.yb_of(dy) for dy in DYs])
def grid_marg(T=0.03):
    N=grids.shape[0]; Px=np.zeros((N,5)); Py=np.zeros((N,4))
    for i in range(N):
        g=grids[i].astype(np.float64).copy(); g[g<-1e8]=-np.inf
        mx=np.nanmax(g[np.isfinite(g)]); w=np.exp((g-mx)/T); w[~np.isfinite(w)]=0; w/=w.sum()+1e-12
        for b in range(5): Px[i,b]=w[:,xcol==b].sum()
        for b in range(4): Py[i,b]=w[yrow==b].sum()
    return Px,Py
CPx,CPy=grid_marg()
# select CNN files by tag: 'v1'->cnn_oof.npz else cnn_oof_<tag>.npz
tags=sys.argv[1].split(',') if len(sys.argv)>1 else ['v2','r34']
NPx=np.zeros((len(y),5)); NPy=np.zeros((len(y),4))
for t in tags:
    fp='research/cache/cnn_oof.npz' if t=='v1' else f'research/cache/cnn_oof_{t}.npz'
    d=np.load(fp); NPx+=d['ofx']; NPy+=d['ofy']
    print('  %-5s exact=%.4f xb=%.4f yb=%.4f'%(t,(5*d['ofy'].argmax(1)+d['ofx'].argmax(1)==y).mean(),(d['ofx'].argmax(1)==xb).mean(),(d['ofy'].argmax(1)==yb).mean()))
NPx/=len(tags); NPy/=len(tags)
print('CNN ens %s: exact=%.4f xb=%.4f yb=%.4f'%(tags,(5*NPy.argmax(1)+NPx.argmax(1)==y).mean(),(NPx.argmax(1)==xb).mean(),(NPy.argmax(1)==yb).mean()))
eps=1e-6
def blended(wx,wy,idx=slice(None)):
    bx=(wx*np.log(NPx[idx]+eps)+(1-wx)*np.log(CPx[idx]+eps)).argmax(1)
    by=(wy*np.log(NPy[idx]+eps)+(1-wy)*np.log(CPy[idx]+eps)).argmax(1)
    return 5*by+bx
best=(-1,)
for wx in np.arange(0,1.001,0.05):
    for wy in np.arange(0,1.001,0.05):
        e=(blended(wx,wy)==y).mean()
        if e>best[0]: best=(e,round(wx,2),round(wy,2))
print('naive best-on-all: exact=%.4f wx=%.2f wy=%.2f'%best)
pred=np.zeros(len(y),int)
for k in range(5):
    trm=f!=k; vam=f==k; bb=(-1,)
    for wx in np.arange(0,1.001,0.05):
        for wy in np.arange(0,1.001,0.05):
            bx=(wx*np.log(NPx[trm]+eps)+(1-wx)*np.log(CPx[trm]+eps)).argmax(1)
            by=(wy*np.log(NPy[trm]+eps)+(1-wy)*np.log(CPy[trm]+eps)).argmax(1)
            e=(5*by+bx==y[trm]).mean()
            if e>bb[0]: bb=(e,wx,wy)
    pred[vam]=blended(bb[1],bb[2],vam)
print('NESTED-CV honest: exact=%.4f'%((pred==y).mean()))
