import numpy as np, pandas as pd, os, cv2, time
from PIL import Image
np.set_printoptions(suppress=True, linewidth=200)
ROOT='dataset'
tr = pd.read_csv(os.path.join(ROOT,'train.csv'))

def xb_of(dx):
    return 0 if dx<-30 else 1 if dx<-22 else 2 if dx<-14 else 3 if dx<-6 else 4
def yb_of(dy):
    return 0 if dy<-2 else 1 if dy<0 else 2 if dy<2 else 3
def to_class(dx,dy): return 5*yb_of(dy)+xb_of(dx)

# preload all grayscale panels + red masks
def load(path):
    im = np.array(Image.open(os.path.join(ROOT,path)).convert('RGB'))
    L = im[:,0:96]; R = im[:,104:200]
    return im,L,R
CACHE={}
for _,r in tr.iterrows():
    CACHE[r['sample_id']] = load(r['image_path'])

def red_mask(patch):
    r,g,b = patch[...,0].astype(int),patch[...,1].astype(int),patch[...,2].astype(int)
    return ((r>90)&(r-g>35)&(r-b>35)).astype(np.uint8)
def gray(im): return cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)

def subpix(res, x, y):
    sx=sy=0.0
    if 0<x<res.shape[1]-1:
        l,c,rr=res[y,x-1],res[y,x],res[y,x+1]; d=(l-2*c+rr)
        if d!=0: sx=0.5*(l-rr)/d
    if 0<y<res.shape[0]-1:
        u,c,dn=res[y-1,x],res[y,x],res[y+1,x]; d=(u-2*c+dn)
        if d!=0: sy=0.5*(u-dn)/d
    return np.clip(sx,-1,1),np.clip(sy,-1,1)

def match_disp(sid, half=14, dxlo=-50,dxhi=12,dylo=-12,dyhi=12, method=cv2.TM_CCOEFF_NORMED):
    im,L,R = CACHE[sid]
    cy,cx=48,48
    tplc = L[cy-half:cy+half+1, cx-half:cx+half+1]
    m = red_mask(tplc)
    tg = gray(tplc).astype(np.float32)
    Rg = gray(R).astype(np.float32)
    mask=(1-m).astype(np.float32)
    res = cv2.matchTemplate(Rg, tg, method, mask=mask)  # shape (96-2half, 96-2half)
    res = np.nan_to_num(res, nan=-1e9, posinf=-1e9, neginf=-1e9)
    H,W = res.shape
    # valid loc range: loc_x in [cx+dxlo-half, cx+dxhi-half]
    xs = np.arange(W); ys=np.arange(H)
    validx = (xs>=cx+dxlo-half)&(xs<=cx+dxhi-half)
    validy = (ys>=cy+dylo-half)&(ys<=cy+dyhi-half)
    mask2 = np.ones_like(res)*(-1e9)
    yy,xx = np.ix_(validy,validx)
    mask2[yy,xx]=res[yy,xx]
    idx = np.argmax(mask2)
    py,px = np.unravel_index(idx, res.shape)
    sx,sy = subpix(res,px,py)
    fx = px+half+sx; fy = py+half+sy
    return fx-cx, fy-cy

def evaluate(**kw):
    P=[];DX=[];DY=[]
    for sid in tr['sample_id']:
        dx,dy=match_disp(sid,**kw); DX.append(dx);DY.append(dy);P.append(to_class(dx,dy))
    P=np.array(P); y=tr['motion_class'].values
    xb=y%5; yb=y//5; pxb=P%5; pyb=P//5
    return dict(exact=(P==y).mean(), xband=(pxb==xb).mean(), yband=(pyb==yb).mean()), np.array(DX),np.array(DY)

if __name__=='__main__':
    print('half sweep (CCOEFF_NORMED, window dx[-50,12] dy[-12,12], subpix):')
    for half in [10,12,14,16,18,20,22]:
        t=time.time(); m,dx,dy=evaluate(half=half)
        print(f'  half={half:2d} exact={m["exact"]:.4f} xband={m["xband"]:.4f} yband={m["yband"]:.4f}  ({time.time()-t:.1f}s)')
    print('\nmethod compare at half=16:')
    for name,meth in [('CCOEFF_NORMED',cv2.TM_CCOEFF_NORMED),('CCORR_NORMED',cv2.TM_CCORR_NORMED),('SQDIFF_NORMED',cv2.TM_SQDIFF_NORMED)]:
        # SQDIFF: min is best -> need different handling; skip for now if not CCOEFF/CCORR
        if meth==cv2.TM_SQDIFF_NORMED: continue
        m,_,_=evaluate(half=16,method=meth)
        print(f'  {name:16s} exact={m["exact"]:.4f} xband={m["xband"]:.4f} yband={m["yband"]:.4f}')
    print('\nwindow tightness at half=16:')
    for dxlo,dxhi,dylo,dyhi in [(-50,12,-12,12),(-45,8,-8,8),(-42,6,-6,6),(-40,4,-5,5)]:
        m,dx,dy=evaluate(half=16,dxlo=dxlo,dxhi=dxhi,dylo=dylo,dyhi=dyhi)
        print(f'  dx[{dxlo},{dxhi}] dy[{dylo},{dyhi}] exact={m["exact"]:.4f} xband={m["xband"]:.4f} yband={m["yband"]:.4f}')
