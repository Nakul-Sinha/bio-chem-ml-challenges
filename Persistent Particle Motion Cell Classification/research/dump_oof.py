import numpy as np, pandas as pd, os, time
import common as C
os.makedirs('research/cache',exist_ok=True)

def process(df, tag):
    N=len(df); DX=np.zeros(N);DY=np.zeros(N);PK=np.zeros(N); grids=np.zeros((N,C.GH,C.GW),np.float32)
    t=time.time()
    for i,p in enumerate(df['image_path'].values):
        L,R=C.load_pair(p); g=C.fused_grid(L,R); grids[i]=g
        dx,dy,pk=C.disp_from_grid(g); DX[i]=dx;DY[i]=dy;PK[i]=pk
    np.savez_compressed(f'research/cache/oof_{tag}.npz', dx=DX,dy=DY,peak=PK,grids=grids,
                        sample_id=df['sample_id'].values)
    print(f'{tag}: {N} imgs in {time.time()-t:.1f}s')
    return DX,DY,PK

if __name__=='__main__':
    tr=pd.read_csv(os.path.join(C.ROOT,'train.csv')); te=pd.read_csv(os.path.join(C.ROOT,'test.csv'))
    DX,DY,PK=process(tr,'train'); process(te,'test')
    y=tr['motion_class'].values; xb=y%5; yb=y//5
    P=C.cls_arr(DX,DY); pxb=P%5; pyb=P//5
    print('\n=== ACCURACY ===')
    print(f'exact={ (P==y).mean():.4f}  xband={(pxb==xb).mean():.4f}  yband={(pyb==yb).mean():.4f}')
    print('\n=== x-band confusion (rows=true 0..4, cols=pred) ===')
    print(pd.crosstab(xb,pxb).reindex(index=range(5),columns=range(5),fill_value=0).values)
    print('=== y-band confusion (rows=true 0..3, cols=pred) ===')
    print(pd.crosstab(yb,pyb).reindex(index=range(4),columns=range(4),fill_value=0).values)
    err = P!=y
    xoff=np.abs(pxb-xb); yoff=np.abs(pyb-yb)
    print('\n=== error decomposition (among all N) ===')
    print(f'correct           : {(~err).sum()}')
    print(f'xband off-by-1 only (y ok): {((xoff==1)&(yoff==0)).sum()}')
    print(f'yband off-by-1 only (x ok): {((xoff==0)&(yoff==1)).sum()}')
    print(f'both off-by-1     : {((xoff==1)&(yoff==1)).sum()}')
    print(f'gross (x off>=2)  : {(xoff>=2).sum()}   (y off>=2): {(yoff>=2).sum()}')
    print('\n=== accuracy by horizon ===')
    for h in sorted(tr['horizon'].unique()):
        mask=tr['horizon'].values==h
        print(f'  h={h}: n={mask.sum():3d} exact={(P[mask]==y[mask]).mean():.3f} xband={(pxb[mask]==xb[mask]).mean():.3f} yband={(pyb[mask]==yb[mask]).mean():.3f}')
    print('\n=== accuracy by peak-confidence quartile ===')
    order=np.argsort(PK)
    for q in range(4):
        idx=order[q*len(order)//4:(q+1)*len(order)//4]
        print(f'  Q{q} peak[{PK[idx].min():.2f},{PK[idx].max():.2f}] exact={(P[idx]==y[idx]).mean():.3f}')
    print('\n=== accuracy by true y-band (where y errors concentrate) ===')
    for b in range(4):
        m=yb==b; print(f'  true yb={b}: n={m.sum():3d} yband_acc={(pyb[m]==b).mean():.3f}')
    print('=== accuracy by true x-band ===')
    for b in range(5):
        m=xb==b; print(f'  true xb={b}: n={m.sum():3d} xband_acc={(pxb[m]==b).mean():.3f}')
