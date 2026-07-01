import numpy as np, pandas as pd, os, cv2
from PIL import Image
ROOT='dataset'

def xb_of(dx): return 0 if dx<-30 else 1 if dx<-22 else 2 if dx<-14 else 3 if dx<-6 else 4
def yb_of(dy): return 0 if dy<-2 else 1 if dy<0 else 2 if dy<2 else 3
def to_class(dx,dy): return 5*yb_of(dy)+xb_of(dx)
def xb_arr(dx): return np.select([dx<-30,dx<-22,dx<-14,dx<-6],[0,1,2,3],4).astype(int)
def yb_arr(dy): return np.select([dy<-2,dy<0,dy<2],[0,1,2],3).astype(int)
def cls_arr(dx,dy): return 5*yb_arr(dy)+xb_arr(dx)

def load_pair(path):
    im=np.array(Image.open(os.path.join(ROOT,path)).convert('RGB')); return im[:,0:96],im[:,104:200]

_CL=cv2.createCLAHE(clipLimit=2.0,tileGridSize=(8,8))
def red_mask(patch):
    r,g,b=patch[...,0].astype(int),patch[...,1].astype(int),patch[...,2].astype(int)
    return ((r>90)&(r-g>35)&(r-b>35)).astype(np.uint8)
def chan(im,c):
    if c=='gray': return cv2.cvtColor(im,cv2.COLOR_RGB2GRAY)
    if c=='green': return im[...,1]
    if c=='clahe': return _CL.apply(cv2.cvtColor(im,cv2.COLOR_RGB2GRAY))

DXLO,DXHI,DYLO,DYHI=-42,8,-9,9
GW=DXHI-DXLO+1; GH=DYHI-DYLO+1
DEFAULT_CFG=[(h,c) for h in (10,12,14,16) for c in ('gray','clahe','green')]

def score_grid(L,R,half,c):
    cy,cx=48,48
    tpl=L[cy-half:cy+half+1,cx-half:cx+half+1]
    m=red_mask(tpl); tg=chan(tpl,c).astype(np.float32); mask=(1-m).astype(np.float32)
    Rg=chan(R,c).astype(np.float32)
    res=cv2.matchTemplate(Rg,tg,cv2.TM_CCOEFF_NORMED,mask=mask)
    res=np.nan_to_num(res,nan=-1e9,posinf=-1e9,neginf=-1e9)
    H,W=res.shape
    grid=np.full((GH,GW),np.nan,np.float32)
    ys=np.arange(DYLO,DYHI+1); xs=np.arange(DXLO,DXHI+1)
    for gy,dy in enumerate(ys):
        ly=dy-half+cy
        if ly<0 or ly>=H: continue
        for gx,dx in enumerate(xs):
            lx=dx-half+cx
            if 0<=lx<W: grid[gy,gx]=res[ly,lx]
    return grid

def fused_grid(L,R,configs=DEFAULT_CFG):
    acc=np.zeros((GH,GW),np.float32); cnt=np.zeros((GH,GW),np.float32)
    for h,c in configs:
        g=score_grid(L,R,h,c)
        valid=~np.isnan(g)
        acc[valid]+=g[valid]; cnt[valid]+=1
    return np.where(cnt>0,acc/np.maximum(cnt,1),-1e9)

def subpix_grid(grid,gx,gy,up=10):
    H,W=grid.shape
    x0,x1=max(0,gx-2),min(W,gx+3); y0,y1=max(0,gy-2),min(H,gy+3)
    patch=np.where(grid[y0:y1,x0:x1]<-1e8,-1.0,grid[y0:y1,x0:x1]).astype(np.float32)
    if patch.shape[0]<3 or patch.shape[1]<3: return 0.0,0.0
    big=cv2.resize(patch,((patch.shape[1]-1)*up+1,(patch.shape[0]-1)*up+1),interpolation=cv2.INTER_CUBIC)
    iy,ix=np.unravel_index(np.argmax(big),big.shape)
    return (x0+ix/up)-gx,(y0+iy/up)-gy

def disp_from_grid(grid):
    gy,gx=np.unravel_index(np.argmax(grid),grid.shape)
    sx,sy=subpix_grid(grid,gx,gy)
    peak=float(grid[gy,gx])
    return DXLO+gx+sx, DYLO+gy+sy, peak

def match(L,R,configs=DEFAULT_CFG):
    return disp_from_grid(fused_grid(L,R,configs))

def folds(y,n=5,seed=42):
    from sklearn.model_selection import StratifiedKFold
    skf=StratifiedKFold(n_splits=n,shuffle=True,random_state=seed)
    f=np.zeros(len(y),int)
    for i,(_,va) in enumerate(skf.split(np.zeros(len(y)),y)): f[va]=i
    return f
