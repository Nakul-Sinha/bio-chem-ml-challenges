import numpy as np, pandas as pd, os, glob
import common as C
from scipy.stats import norm
tr=pd.read_csv(os.path.join(C.ROOT,'train.csv'))
y=tr['motion_class'].values; xb=y%5; yb=y//5
oo=np.load('research/cache/oof_train.npz',allow_pickle=True)
DX,DY,PK=oo['dx'],oo['dy'],oo['peak']
XEDGES=[-30,-22,-14,-6]  # xband boundaries
YEDGES=[-2,0,2]

def gauss_bandprobs(vals, edges, nb, sigma):
    # p(band) from a Gaussian centered at each val with std sigma, integrated over band intervals
    edges=[-1e9]+list(edges)+[1e9]
    N=len(vals); P=np.zeros((N,nb))
    for b in range(nb):
        lo,hi=edges[b],edges[b+1]
        P[:,b]=norm.cdf((hi-vals)/sigma)-norm.cdf((lo-vals)/sigma)
    P/=P.sum(1,keepdims=True)+1e-12
    return P

def classical_probs(sx,sy):
    return gauss_bandprobs(DX,XEDGES,5,sx), gauss_bandprobs(DY,YEDGES,4,sy)

def sc(px,py):
    pred=5*py.argmax(1)+px.argmax(1); return (pred==y).mean()

def main():
    print('=== tune classical Gaussian sigma ===')
    best=(-1,)
    for sx in [2,3,4,5,6,8]:
        for sy in [0.5,0.8,1.0,1.3,1.6,2.0,2.5]:
            CPx,CPy=classical_probs(sx,sy); e=sc(CPx,CPy)
            if e>best[0]: best=(e,sx,sy)
    print(f'  best classical-prob exact={best[0]:.4f} sx={best[1]} sy={best[2]}')
    CPx,CPy=classical_probs(best[1],best[2])

    cnn_files=sorted(glob.glob('research/cache/cnn_oof*.npz'))
    print('CNN OOF files:',cnn_files)
    NPx=np.zeros((len(y),5)); NPy=np.zeros((len(y),4)); nc=0
    per={}
    for f in cnn_files:
        d=np.load(f); px,py=d['ofx'],d['ofy']; per[os.path.basename(f)]=(5*py.argmax(1)+px.argmax(1)==y).mean()
        NPx+=px; NPy+=py; nc+=1
    NPx/=nc; NPy/=nc
    for k,v in per.items(): print(f'  {k}: exact={v:.4f}')
    print(f'CNN ensemble ({nc}): exact={sc(NPx,NPy):.4f} xband={(NPx.argmax(1)==xb).mean():.4f} yband={(NPy.argmax(1)==yb).mean():.4f}')
    print(f'classical argmax exact={ (C.cls_arr(DX,DY)==y).mean():.4f}')

    print('\n=== LOG blend weight sweep (w=weight on CNN) ===')
    eps=1e-6; bestb=(-1,)
    for wx in np.arange(0,1.001,0.05):
        for wy in np.arange(0,1.001,0.05):
            bx=(wx*np.log(NPx+eps)+(1-wx)*np.log(CPx+eps)).argmax(1)
            by=(wy*np.log(NPy+eps)+(1-wy)*np.log(CPy+eps)).argmax(1)
            e=(5*by+bx==y).mean()
            if e>bestb[0]: bestb=(e,wx,wy)
    print(f'  best LOG blend exact={bestb[0]:.4f} wx(cnn)={bestb[1]:.2f} wy(cnn)={bestb[2]:.2f}')
    print('\n=== linear blend ===')
    bestl=(-1,)
    for wx in np.arange(0,1.001,0.05):
        for wy in np.arange(0,1.001,0.05):
            bx=(wx*NPx+(1-wx)*CPx).argmax(1); by=(wy*NPy+(1-wy)*CPy).argmax(1)
            e=(5*by+bx==y).mean()
            if e>bestl[0]: bestl=(e,wx,wy)
    print(f'  best LIN blend exact={bestl[0]:.4f} wx(cnn)={bestl[1]:.2f} wy(cnn)={bestl[2]:.2f}')

if __name__=='__main__': main()
