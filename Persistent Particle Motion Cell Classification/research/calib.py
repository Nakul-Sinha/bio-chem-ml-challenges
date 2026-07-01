import numpy as np, pandas as pd, os
import common as C
tr=pd.read_csv(os.path.join(C.ROOT,'train.csv'))
d=np.load('research/cache/oof_train.npz',allow_pickle=True)
DX,DY,PK=d['dx'],d['dy'],d['peak']
y=tr['motion_class'].values; xb=y%5; yb=y//5; hor=tr['horizon'].values
f=C.folds(y)

def acc(dx,dy):
    P=C.cls_arr(dx,dy); return (P==y).mean(),((P%5)==xb).mean(),((P//5)==yb).mean()

print('baseline:', [round(a,4) for a in acc(DX,DY)])

# global additive offset scan
best=(-1,0,0)
for ox in np.arange(-3,3.01,0.25):
    for oy in np.arange(-3,3.01,0.25):
        e=(C.cls_arr(DX+ox,DY+oy)==y).mean()
        if e>best[0]: best=(e,ox,oy)
print('best global additive offset:', round(best[0],4), 'ox=%.2f oy=%.2f'%(best[1],best[2]))

# additive + multiplicative (scale) on dx,dy
best2=(-1,0,0,1,1)
for sx in [0.9,0.95,1.0,1.05,1.1,1.15]:
    for sy in [0.9,1.0,1.1,1.2,1.3]:
        for ox in np.arange(-2,2.01,0.5):
            for oy in np.arange(-2,2.01,0.25):
                e=(C.cls_arr(DX*sx+ox,DY*sy+oy)==y).mean()
                if e>best2[0]: best2=(e,ox,oy,sx,sy)
print('best affine:', round(best2[0],4),'ox=%.2f oy=%.2f sx=%.2f sy=%.2f'%best2[1:])

# per-horizon additive offset
print('\nper-horizon additive offset (fit on all, applied per horizon):')
DXc=DX.copy(); DYc=DY.copy()
for h in [2,3,4]:
    m=hor==h; bb=(-1,0,0)
    for ox in np.arange(-3,3.01,0.25):
        for oy in np.arange(-3,3.01,0.25):
            e=(C.cls_arr(DX[m]+ox,DY[m]+oy)==y[m]).mean()
            if e>bb[0]: bb=(e,ox,oy)
    print(f'  h={h}: exact={bb[0]:.4f} ox={bb[1]:.2f} oy={bb[2]:.2f}')
    DXc[m]=DX[m]+bb[1]; DYc[m]=DY[m]+bb[2]
print('per-horizon applied (in-sample):', [round(a,4) for a in acc(DXc,DYc)])

# HONEST CV: fit offset on train folds, apply to val fold
print('\n=== HONEST 5-fold CV of calibration ===')
def cv_offset(mode='global'):
    Pcv=np.zeros(len(y),int)
    for k in range(5):
        trm=f!=k; vam=f==k
        if mode=='global':
            bb=(-1,0,0)
            for ox in np.arange(-3,3.01,0.25):
                for oy in np.arange(-3,3.01,0.25):
                    e=(C.cls_arr(DX[trm]+ox,DY[trm]+oy)==y[trm]).mean()
                    if e>bb[0]: bb=(e,ox,oy)
            Pcv[vam]=C.cls_arr(DX[vam]+bb[1],DY[vam]+bb[2])
        elif mode=='perh':
            for h in [2,3,4]:
                trh=trm&(hor==h); vah=vam&(hor==h); bb=(-1,0,0)
                for ox in np.arange(-3,3.01,0.25):
                    for oy in np.arange(-3,3.01,0.25):
                        e=(C.cls_arr(DX[trh]+ox,DY[trh]+oy)==y[trh]).mean()
                        if e>bb[0]: bb=(e,ox,oy)
                Pcv[vah]=C.cls_arr(DX[vah]+bb[1],DY[vah]+bb[2])
    return (Pcv==y).mean(),((Pcv%5)==xb).mean(),((Pcv//5)==yb).mean()
print('no calib CV      :', round((C.cls_arr(DX,DY)==y).mean(),4))
print('global offset CV :', [round(a,4) for a in cv_offset('global')])
print('per-horizon CV   :', [round(a,4) for a in cv_offset('perh')])
