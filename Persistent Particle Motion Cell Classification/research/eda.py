import numpy as np, pandas as pd, os, cv2
from PIL import Image
np.set_printoptions(suppress=True, linewidth=200)
ROOT = 'dataset'
tr = pd.read_csv(os.path.join(ROOT,'train.csv'))
te = pd.read_csv(os.path.join(ROOT,'test.csv'))

def xb_of(dx):
    if dx < -30: return 0
    if dx < -22: return 1
    if dx < -14: return 2
    if dx < -6: return 3
    return 4
def yb_of(dy):
    if dy < -2: return 0
    if dy < 0: return 1
    if dy < 2: return 2
    return 3
def to_class(dx,dy): return 5*yb_of(dy)+xb_of(dx)

tr['xb']=tr['motion_class']%5
tr['yb']=tr['motion_class']//5
print('=== TRAIN class distribution (0..19) ===')
vc = tr['motion_class'].value_counts().reindex(range(20),fill_value=0)
print(vc.values)
print('most common class:', int(vc.idxmax()), 'freq', vc.max(), f'({vc.max()/len(tr):.3f})')
print('\n=== x_band marginal (0..4) ===')
print(tr['xb'].value_counts().reindex(range(5),fill_value=0).values)
print('=== y_band marginal (0..3) ===')
print(tr['yb'].value_counts().reindex(range(4),fill_value=0).values)
print('\n=== horizon distribution ===')
print('train:', tr['horizon'].value_counts().sort_index().to_dict())
print('test :', te['horizon'].value_counts().sort_index().to_dict())
print('\n=== class x horizon crosstab (rows=horizon) ===')
print(pd.crosstab(tr['horizon'], tr['motion_class']).reindex(columns=range(20),fill_value=0))
print('\n=== xb x horizon (rows=horizon) ===')
print(pd.crosstab(tr['horizon'], tr['xb']))
print('=== yb x horizon (rows=horizon) ===')
print(pd.crosstab(tr['horizon'], tr['yb']))

# Baselines
print('\n=== PRIOR baselines (in-sample) ===')
maj = int(vc.idxmax())
print('global majority acc:', (tr['motion_class']==maj).mean().round(4))
# horizon-conditioned majority
hmaj = tr.groupby('horizon')['motion_class'].agg(lambda s: s.value_counts().idxmax())
acc = tr.apply(lambda r: r['motion_class']==hmaj[r['horizon']], axis=1).mean()
print('horizon-cond majority acc:', round(acc,4), 'map', hmaj.to_dict())
# independent xb,yb majority
xbmaj=int(tr['xb'].value_counts().idxmax()); ybmaj=int(tr['yb'].value_counts().idxmax())
print('xb-maj acc', (tr['xb']==xbmaj).mean().round(4), 'yb-maj acc', (tr['yb']==ybmaj).mean().round(4))

# ---- Classical template matcher v1 ----
def load(path):
    im = np.array(Image.open(os.path.join(ROOT,path)).convert('RGB'))
    return im
def split_panels(im):
    L = im[:, 0:96]; R = im[:, 104:200]
    return L,R
def red_mask(patch):
    r,g,b = patch[...,0].astype(int),patch[...,1].astype(int),patch[...,2].astype(int)
    return ((r>90)&(r-g>35)&(r-b>35)).astype(np.uint8)
def gray(im): return cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)

def match_disp(im, half=14, method=cv2.TM_CCOEFF_NORMED):
    L,R = split_panels(im)
    cy,cx = 48,48
    tpl = L[cy-half:cy+half+1, cx-half:cx+half+1]
    m = red_mask(tpl)
    tg = gray(tpl).astype(np.float32)
    Rg = gray(R).astype(np.float32)
    mask = (1-m).astype(np.float32)  # 1 where valid
    # masked template matching
    res = cv2.matchTemplate(Rg, tg, method, mask=mask)
    _,_,minloc,maxloc = cv2.minMaxLoc(res)
    loc = maxloc  # for CCOEFF/CCORR normed, max is best
    px = loc[0]+half; py = loc[1]+half  # center of matched patch in right panel
    dx = px - cx; dy = py - cy
    return dx,dy

# evaluate on all train (in-sample; classical has no fit)
preds=[]; dxs=[]; dys=[]
for _,r in tr.iterrows():
    im = load(r['image_path'])
    dx,dy = match_disp(im)
    dxs.append(dx); dys.append(dy)
    preds.append(to_class(dx,dy))
tr['pred']=preds; tr['pdx']=dxs; tr['pdy']=dys
tr['pxb']=tr['pred']%5; tr['pyb']=tr['pred']//5
print('\n=== Classical matcher v1 (half=14, CCOEFF_NORMED, masked) ===')
print('exact acc :', (tr['pred']==tr['motion_class']).mean().round(4))
print('xband acc :', (tr['pxb']==tr['xb']).mean().round(4))
print('yband acc :', (tr['pyb']==tr['yb']).mean().round(4))
print('pred dx range', np.percentile(dxs,[1,50,99]).round(1), 'dy range', np.percentile(dys,[1,50,99]).round(1))
