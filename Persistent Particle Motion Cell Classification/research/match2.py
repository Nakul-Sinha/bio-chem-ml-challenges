import numpy as np, pandas as pd, os, cv2, time
from PIL import Image
np.set_printoptions(suppress=True, linewidth=200)
ROOT='dataset'
tr = pd.read_csv(os.path.join(ROOT,'train.csv'))

def xb_of(dx): return 0 if dx<-30 else 1 if dx<-22 else 2 if dx<-14 else 3 if dx<-6 else 4
def yb_of(dy): return 0 if dy<-2 else 1 if dy<0 else 2 if dy<2 else 3
def to_class(dx,dy): return 5*yb_of(dy)+xb_of(dx)

def load(path):
    im = np.array(Image.open(os.path.join(ROOT,path)).convert('RGB'))
    return im[:,0:96], im[:,104:200]
CACHE={sid:load(p) for sid,p in zip(tr['sample_id'],tr['image_path'])}

CLAHE=cv2.createCLAHE(clipLimit=2.0,tileGridSize=(8,8))
def red_mask(patch):
    r,g,b=patch[...,0].astype(int),patch[...,1].astype(int),patch[...,2].astype(int)
    return ((r>90)&(r-g>35)&(r-b>35)).astype(np.uint8)
def to_chan(im,chan):
    if chan=='gray': return cv2.cvtColor(im,cv2.COLOR_RGB2GRAY)
    if chan=='green': return im[...,1]
    if chan=='clahe': return CLAHE.apply(cv2.cvtColor(im,cv2.COLOR_RGB2GRAY))
    if chan=='sobel':
        g=cv2.cvtColor(im,cv2.COLOR_RGB2GRAY).astype(np.float32)
        gx=cv2.Sobel(g,cv2.CV_32F,1,0,ksize=3); gy=cv2.Sobel(g,cv2.CV_32F,0,1,ksize=3)
        m=np.sqrt(gx*gx+gy*gy); return np.clip(m,0,255).astype(np.uint8)

def subpix_fine(res,x,y,up=10):
    # upsample 5x5 neighborhood for finer subpixel
    H,W=res.shape
    x0,x1=max(0,x-2),min(W,x+3); y0,y1=max(0,y-2),min(H,y+3)
    patch=res[y0:y1,x0:x1].astype(np.float32)
    if patch.shape[0]<3 or patch.shape[1]<3: return 0.0,0.0
    big=cv2.resize(patch,((patch.shape[1]-1)*up+1,(patch.shape[0]-1)*up+1),interpolation=cv2.INTER_CUBIC)
    iy,ix=np.unravel_index(np.argmax(big),big.shape)
    fx=x0+ix/up; fy=y0+iy/up
    return fx-x, fy-y

def match_one(L,R, half=10, chan='gray', dxlo=-42,dxhi=6,dylo=-6,dyhi=6, inpaint=False, fine=True):
    cy,cx=48,48
    tpl_rgb=L[cy-half:cy+half+1,cx-half:cx+half+1]
    m=red_mask(tpl_rgb)
    if inpaint:
        Lc=cv2.inpaint(L, (red_mask(L)*255).astype(np.uint8),3,cv2.INPAINT_TELEA)
        tg=to_chan(Lc,chan)[cy-half:cy+half+1,cx-half:cx+half+1].astype(np.float32)
        mask=np.ones_like(tg)
    else:
        tg=to_chan(tpl_rgb,chan).astype(np.float32)
        mask=(1-m).astype(np.float32)
    Rg=to_chan(R,chan).astype(np.float32)
    res=cv2.matchTemplate(Rg,tg,cv2.TM_CCOEFF_NORMED,mask=mask)
    res=np.nan_to_num(res,nan=-1e9,posinf=-1e9,neginf=-1e9)
    H,W=res.shape
    xs=np.arange(W); ys=np.arange(H)
    vx=(xs>=cx+dxlo-half)&(xs<=cx+dxhi-half); vy=(ys>=cy+dylo-half)&(ys<=cy+dyhi-half)
    masked=np.full_like(res,-1e9); yy,xx=np.ix_(vy,vx); masked[yy,xx]=res[yy,xx]
    py,px=np.unravel_index(np.argmax(masked),res.shape)
    if fine: sx,sy=subpix_fine(res,px,py)
    else:
        sx=sy=0.0
    return (px+half+sx)-cx,(py+half+sy)-cy

def evaluate(**kw):
    P=[];DX=[];DY=[]
    for sid in tr['sample_id']:
        L,R=CACHE[sid]; dx,dy=match_one(L,R,**kw); DX.append(dx);DY.append(dy);P.append(to_class(dx,dy))
    P=np.array(P); y=tr['motion_class'].values
    return dict(exact=(P==y).mean(),xband=((P%5)==(y%5)).mean(),yband=((P//5)==(y//5)).mean()),np.array(DX),np.array(DY)

def match_multi(L,R,halves=(8,10,12),**kw):
    ds=[match_one(L,R,half=h,**kw) for h in halves]
    dx=np.median([d[0] for d in ds]); dy=np.median([d[1] for d in ds])
    return dx,dy
def evaluate_multi(halves=(8,10,12),**kw):
    P=[]
    for sid in tr['sample_id']:
        L,R=CACHE[sid]; dx,dy=match_multi(L,R,halves=halves,**kw); P.append(to_class(dx,dy))
    P=np.array(P); y=tr['motion_class'].values
    return dict(exact=(P==y).mean(),xband=((P%5)==(y%5)).mean(),yband=((P//5)==(y//5)).mean())

if __name__=='__main__':
    print('channel x small-half sweep (tight window, fine subpix):')
    for chan in ['gray','green','clahe','sobel']:
        for half in [6,8,10,12]:
            m,_,_=evaluate(half=half,chan=chan)
            print(f'  chan={chan:6s} half={half:2d} exact={m["exact"]:.4f} xband={m["xband"]:.4f} yband={m["yband"]:.4f}')
    print('\ninpaint vs mask (gray, half=10):')
    for ip in [False,True]:
        m,_,_=evaluate(half=10,chan='gray',inpaint=ip)
        print(f'  inpaint={ip} exact={m["exact"]:.4f} xband={m["xband"]:.4f} yband={m["yband"]:.4f}')
    print('\nfine subpix on/off (gray, half=10):')
    for f in [False,True]:
        m,_,_=evaluate(half=10,chan='gray',fine=f)
        print(f'  fine={f} exact={m["exact"]:.4f} yband={m["yband"]:.4f}')
    print('\nmulti-scale median:')
    for halves in [(8,10,12),(6,8,10),(8,10,12,14)]:
        for chan in ['gray','clahe']:
            m=evaluate_multi(halves=halves,chan=chan)
            print(f'  halves={halves} chan={chan:6s} exact={m["exact"]:.4f} xband={m["xband"]:.4f} yband={m["yband"]:.4f}')
