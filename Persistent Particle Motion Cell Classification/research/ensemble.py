import numpy as np, pandas as pd, os
import common as C
tr=pd.read_csv(os.path.join(C.ROOT,'train.csv'))
y=tr['motion_class'].values; xb=y%5; yb=y//5

# classical grids -> marginal xband/yband probs
oo=np.load('research/cache/oof_train.npz',allow_pickle=True)
grids=oo['grids']  # (N,GH,GW)
DXs=np.arange(C.DXLO,C.DXHI+1); DYs=np.arange(C.DYLO,C.DYHI+1)
xb_of_col=np.array([C.xb_of(dx) for dx in DXs])   # (GW,)
yb_of_row=np.array([C.yb_of(dy) for dy in DYs])   # (GH,)

def classical_probs(T=0.06):
    N=grids.shape[0]
    Px=np.zeros((N,5)); Py=np.zeros((N,4))
    for i in range(N):
        g=grids[i].astype(np.float64).copy()
        g[g<-1e8]=-np.inf
        w=np.exp((g-np.nanmax(g[np.isfinite(g)]))/T)
        w[~np.isfinite(w)]=0
        w/=w.sum()+1e-12
        for b in range(5): Px[i,b]=w[:,xb_of_col==b].sum()
        for b in range(4): Py[i,b]=w[yb_of_row==b,:].sum()
    return Px,Py

def score(px_argmax,py_argmax):
    pred=5*py_argmax+px_argmax
    return (pred==y).mean(),(px_argmax==xb).mean(),(py_argmax==yb).mean()

def main():
    # classical marginals: tune T
    print('=== classical marginal probs, T sweep ===')
    bestT=None;bestv=-1
    for T in [0.03,0.05,0.06,0.08,0.1,0.15]:
        Px,Py=classical_probs(T); e,xx,yy=score(Px.argmax(1),Py.argmax(1))
        print(f'  T={T}: exact={e:.4f} xband={xx:.4f} yband={yy:.4f}')
        if e>bestv: bestv=e;bestT=T
    CPx,CPy=classical_probs(bestT)
    print(f'chosen T={bestT}, classical exact={bestv:.4f}')

    if not os.path.exists('research/cache/cnn_oof.npz'):
        print('\n[cnn_oof.npz not ready yet]'); return
    cc=np.load('research/cache/cnn_oof.npz'); NPx,NPy=cc['ofx'],cc['ofy']
    print('\n=== CNN alone ===')
    print('  ',score(NPx.argmax(1),NPy.argmax(1)))
    print('\n=== mix-and-match (argmax) ===')
    print('  classical-x + CNN-y   :', score(CPx.argmax(1),NPy.argmax(1)))
    print('  CNN-x + classical-y   :', score(NPx.argmax(1),CPy.argmax(1)))
    print('  classical-x + classical-y:', score(CPx.argmax(1),CPy.argmax(1)))
    print('  CNN-x + CNN-y         :', score(NPx.argmax(1),NPy.argmax(1)))

    print('\n=== per-band probabilistic blend (weight on CNN) sweep ===')
    best=(-1,)
    for wx in np.arange(0,1.01,0.1):
        for wy in np.arange(0,1.01,0.1):
            bx=(wx*NPx+(1-wx)*CPx).argmax(1); by=(wy*NPy+(1-wy)*CPy).argmax(1)
            e=(5*by+bx==y).mean()
            if e>best[0]: best=(e,wx,wy)
    print(f'  best blend exact={best[0]:.4f} at wx(cnn)={best[1]:.2f} wy(cnn)={best[2]:.2f}')
    # geometric (log) blend
    bestg=(-1,)
    eps=1e-6
    for wx in np.arange(0,1.01,0.1):
        for wy in np.arange(0,1.01,0.1):
            bx=(wx*np.log(NPx+eps)+(1-wx)*np.log(CPx+eps)).argmax(1)
            by=(wy*np.log(NPy+eps)+(1-wy)*np.log(CPy+eps)).argmax(1)
            e=(5*by+bx==y).mean()
            if e>bestg[0]: bestg=(e,wx,wy)
    print(f'  best LOG blend exact={bestg[0]:.4f} at wx(cnn)={bestg[1]:.2f} wy(cnn)={bestg[2]:.2f}')

if __name__=='__main__': main()
