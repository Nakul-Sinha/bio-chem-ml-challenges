import numpy as np, pandas as pd, os, cv2
import common as C
tr=pd.read_csv(os.path.join(C.ROOT,'train.csv'))
y=tr['motion_class'].values; xb=y%5; yb=y//5
oo=np.load('research/cache/oof_train.npz',allow_pickle=True)
DX,DY=oo['dx'].copy(),oo['dy'].copy()

# baseline
P=C.cls_arr(DX,DY)
print('baseline: exact=%.4f xband=%.4f yband=%.4f'%((P==y).mean(),((P%5)==xb).mean(),((P//5)==yb).mean()))

def refine(sign=+1, P=28, win=True):
    rdx=DX.copy(); rdy=DY.copy()
    han=cv2.createHanningWindow((P,P),cv2.CV_32F) if win else np.ones((P,P),np.float32)
    for i,p in enumerate(tr['image_path'].values):
        L,R=C.load_pair(p)
        # inpaint marker in left
        Lg=cv2.cvtColor(L,cv2.COLOR_RGB2GRAY)
        mk=(C.red_mask(L)*255).astype(np.uint8)
        Lg=cv2.inpaint(cv2.cvtColor(L,cv2.COLOR_RGB2GRAY)[:,:,None].repeat(3,2) if False else L, mk,3,cv2.INPAINT_TELEA)
        Lg=cv2.cvtColor(Lg,cv2.COLOR_RGB2GRAY).astype(np.float32)
        Rg=cv2.cvtColor(R,cv2.COLOR_RGB2GRAY).astype(np.float32)
        cx=cy=48; h=P//2
        cdx=int(round(DX[i])); cdy=int(round(DY[i]))
        lx0,ly0=cx-h,cy-h
        rx0,ry0=cx+cdx-h, cy+cdy-h
        if lx0<0 or ly0<0 or lx0+P>96 or ly0+P>96: continue
        if rx0<0 or ry0<0 or rx0+P>96 or ry0+P>96: continue
        a=Lg[ly0:ly0+P,lx0:lx0+P]; b=Rg[ry0:ry0+P,rx0:rx0+P]
        try:
            (sx,sy),resp=cv2.phaseCorrelate(a,b,han)
        except Exception:
            continue
        if abs(sx)>4 or abs(sy)>4: continue  # reject wild
        rdx[i]=cdx+sign*sx; rdy[i]=cdy+sign*sy
    Pr=C.cls_arr(rdx,rdy)
    return (Pr==y).mean(),((Pr%5)==xb).mean(),((Pr//5)==yb).mean()

for sign in [+1,-1]:
    for Pp in [20,28,36]:
        e,xx,yy=refine(sign=sign,P=Pp)
        print(f'phasecorr sign={sign:+d} P={Pp}: exact={e:.4f} xband={xx:.4f} yband={yy:.4f}')
