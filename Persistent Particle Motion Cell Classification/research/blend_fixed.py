import numpy as np, pandas as pd, os, sys
import common as C
tr=pd.read_csv(os.path.join(C.ROOT,'train.csv'))
y=tr['motion_class'].values; xb=y%5; yb=y//5; f=C.folds(y)
oo=np.load('research/cache/oof_train.npz',allow_pickle=True); grids=oo['grids']
DXs=np.arange(C.DXLO,C.DXHI+1); DYs=np.arange(C.DYLO,C.DYHI+1)
xcol=np.array([C.xb_of(dx) for dx in DXs]); yrow=np.array([C.yb_of(dy) for dy in DYs])
CPx=np.zeros((len(y),5));CPy=np.zeros((len(y),4))
for i in range(len(y)):
    g=grids[i].astype(np.float64).copy(); g[g<-1e8]=-np.inf
    w=np.exp((g-np.nanmax(g[np.isfinite(g)]))/0.03); w[~np.isfinite(w)]=0; w/=w.sum()+1e-12
    for b in range(5): CPx[i,b]=w[:,xcol==b].sum()
    for b in range(4): CPy[i,b]=w[yrow==b].sum()
tags=sys.argv[1].split(',')
NPx=np.zeros((len(y),5)); NPy=np.zeros((len(y),4))
for t in tags:
    fp='research/cache/cnn_oof.npz' if t=='v1' else f'research/cache/cnn_oof_{t}.npz'
    d=np.load(fp); NPx+=d['ofx']; NPy+=d['ofy']
NPx/=len(tags); NPy/=len(tags); eps=1e-6
print('tags=%s  CNN-ens exact=%.4f xb=%.4f yb=%.4f  classical=%.4f'%(
    tags,(5*NPy.argmax(1)+NPx.argmax(1)==y).mean(),(NPx.argmax(1)==xb).mean(),(NPy.argmax(1)==yb).mean(),
    (C.cls_arr(oo['dx'],oo['dy'])==y).mean()))
def blend(wx,wy):
    bx=(wx*np.log(NPx+eps)+(1-wx)*np.log(CPx+eps)).argmax(1)
    by=(wy*np.log(NPy+eps)+(1-wy)*np.log(CPy+eps)).argmax(1)
    return 5*by+bx
print('  fixed-weight blends (full-data, honest for fixed constants):')
for wx,wy in [(0.80,0.55),(0.85,0.60),(0.90,0.60),(0.90,0.65),(0.95,0.65),(1.0,0.60)]:
    print('    WX=%.2f WY=%.2f -> exact=%.4f'%(wx,wy,(blend(wx,wy)==y).mean()))
# nested for reference
pred=np.zeros(len(y),int)
for k in range(5):
    trm=f!=k; va=f==k; bb=(-1,)
    for wx in np.arange(0.5,1.001,0.05):
        for wy in np.arange(0.3,1.001,0.05):
            bx=(wx*np.log(NPx[trm]+eps)+(1-wx)*np.log(CPx[trm]+eps)).argmax(1)
            by=(wy*np.log(NPy[trm]+eps)+(1-wy)*np.log(CPy[trm]+eps)).argmax(1)
            e=(5*by+bx==y[trm]).mean()
            if e>bb[0]: bb=(e,wx,wy)
    bx=(bb[1]*np.log(NPx[va]+eps)+(1-bb[1])*np.log(CPx[va]+eps)).argmax(1)
    by=(bb[2]*np.log(NPy[va]+eps)+(1-bb[2])*np.log(CPy[va]+eps)).argmax(1)
    pred[va]=5*by+bx
print('  nested-CV honest=%.4f'%((pred==y).mean()))
