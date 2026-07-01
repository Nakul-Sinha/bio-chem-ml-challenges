import numpy as np, pandas as pd, os, cv2, time
from PIL import Image
np.set_printoptions(suppress=True, linewidth=200)
ROOT='dataset'
tr = pd.read_csv(os.path.join(ROOT,'train.csv'))
def xb_of(dx): return 0 if dx<-30 else 1 if dx<-22 else 2 if dx<-14 else 3 if dx<-6 else 4
def yb_of(dy): return 0 if dy<-2 else 1 if dy<0 else 2 if dy<2 else 3
def to_class(dx,dy): return 5*yb_of(dy)+xb_of(dx)
def load(path):
    im=np.array(Image.open(os.path.join(ROOT,path)).convert('RGB')); return im[:,0:96],im[:,104:200]
CACHE={sid:load(p) for sid,p in zip(tr['sample_id'],tr['image_path'])}
CL=cv2.createCLAHE(clipLimit=2.0,tileGridSize=(8,8))
def red_mask(patch):
    r,g,b=patch[...,0].astype(int),patch[...,1].astype(int),patch[...,2].astype(int)
    return ((r>90)&(r-g>35)&(r-b>35)).astype(np.uint8)
def chan(im,c):
    if c=='gray': return cv2.cvtColor(im,cv2.COLOR_RGB2GRAY)
    if c=='green': return im[...,1]
    if c=='clahe': return CL.apply(cv2.cvtColor(im,cv2.COLOR_RGB2GRAY))

# Grid of displacements
DXLO,DXHI,DYLO,DYHI=-42,8,-9,9
GW=DXHI-DXLO+1; GH=DYHI-DYLO+1

def score_grid(L,R, half, c):
    cy,cx=48,48
    tpl=L[cy-half:cy+half+1,cx-half:cx+half+1]
    m=red_mask(tpl); tg=chan(tpl,c).astype(np.float32); mask=(1-m).astype(np.float32)
    Rg=chan(R,c).astype(np.float32)
    res=cv2.matchTemplate(Rg,tg,cv2.TM_CCOEFF_NORMED,mask=mask)
    res=np.nan_to_num(res,nan=-1e9,posinf=-1e9,neginf=-1e9)
    H,W=res.shape
    grid=np.full((GH,GW),np.nan,np.float32)
    for gy,dy in enumerate(range(DYLO,DYHI+1)):
        ly=dy-half+cy
        if ly<0 or ly>=H: continue
        for gx,dx in enumerate(range(DXLO,DXHI+1)):
            lx=dx-half+cx
            if lx<0 or lx>=W: continue
            grid[gy,gx]=res[ly,lx]
    return grid

def subpix_grid(grid,gx,gy,up=10):
    H,W=grid.shape
    x0,x1=max(0,gx-2),min(W,gx+3); y0,y1=max(0,gy-2),min(H,gy+3)
    patch=np.nan_to_num(grid[y0:y1,x0:x1],nan=-1.0).astype(np.float32)
    if patch.shape[0]<3 or patch.shape[1]<3: return 0.0,0.0
    big=cv2.resize(patch,((patch.shape[1]-1)*up+1,(patch.shape[0]-1)*up+1),interpolation=cv2.INTER_CUBIC)
    iy,ix=np.unravel_index(np.argmax(big),big.shape)
    return (x0+ix/up)-gx,(y0+iy/up)-gy

def fused_disp(L,R,configs):
    grids=[score_grid(L,R,h,c) for (h,c) in configs]
    # normalize each grid to 0..1 over valid then average (nan-> -1)
    acc=np.zeros((GH,GW),np.float32); cnt=np.zeros((GH,GW),np.float32)
    for g in grids:
        gg=np.where(np.isnan(g),np.nan,g)
        acc=np.where(np.isnan(gg),acc,acc+np.nan_to_num(gg))
        cnt=np.where(np.isnan(gg),cnt,cnt+1)
    fused=np.where(cnt>0,acc/np.maximum(cnt,1),-1e9)
    gy,gx=np.unravel_index(np.argmax(fused),fused.shape)
    sx,sy=subpix_grid(fused,gx,gy)
    dx=DXLO+gx+sx; dy=DYLO+gy+sy
    return dx,dy

def evaluate_fused(configs):
    P=[]
    for sid in tr['sample_id']:
        L,R=CACHE[sid]; dx,dy=fused_disp(L,R,configs); P.append(to_class(dx,dy))
    P=np.array(P); y=tr['motion_class'].values
    return dict(exact=(P==y).mean(),xband=((P%5)==(y%5)).mean(),yband=((P//5)==(y//5)).mean())

if __name__=='__main__':
    tests={
     'gray12': [(12,'gray')],
     'clahe12':[(12,'clahe')],
     'gray[10,12,14]':[(10,'gray'),(12,'gray'),(14,'gray')],
     'clahe[10,12,14]':[(10,'clahe'),(12,'clahe'),(14,'clahe')],
     'clahe[12,14,16]':[(12,'clahe'),(14,'clahe'),(16,'clahe')],
     'gray+clahe[10,12,14]':[(10,'gray'),(12,'gray'),(14,'gray'),(10,'clahe'),(12,'clahe'),(14,'clahe')],
     'gray+clahe[12,14,16]':[(12,'gray'),(14,'gray'),(16,'gray'),(12,'clahe'),(14,'clahe'),(16,'clahe')],
     'all[10..16]':[(h,c) for h in (10,12,14,16) for c in ('gray','clahe','green')],
    }
    for name,cfg in tests.items():
        t=time.time(); m=evaluate_fused(cfg)
        print(f'{name:24s} exact={m["exact"]:.4f} xband={m["xband"]:.4f} yband={m["yband"]:.4f} ({time.time()-t:.1f}s)')
