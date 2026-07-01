import numpy as np, pandas as pd, os, cv2
from PIL import Image
ROOT='dataset'
tr=pd.read_csv(os.path.join(ROOT,'train.csv')); te=pd.read_csv(os.path.join(ROOT,'test.csv'))
def sig(path, panel='full'):
    im=np.array(Image.open(os.path.join(ROOT,path)).convert('L'))
    if panel=='left': im=im[:,0:96]
    elif panel=='right': im=im[:,104:200]
    s=cv2.resize(im,(32,16)).astype(np.float32).ravel()
    s=(s-s.mean())/(s.std()+1e-6); return s
for panel in ['full','right','left']:
    Xtr=np.stack([sig(p,panel) for p in tr['image_path']]); Xte=np.stack([sig(p,panel) for p in te['image_path']])
    Xtr/= (np.linalg.norm(Xtr,axis=1,keepdims=True)+1e-9); Xte/=(np.linalg.norm(Xte,axis=1,keepdims=True)+1e-9)
    S=Xtr@Xtr.T; np.fill_diagonal(S,-1)
    nn=S.max(1); nn_idx=S.argmax(1)
    y=tr['motion_class'].values
    print(f'\n=== panel={panel} : within-train nearest-neighbor cosine ===')
    print('  pctiles [50,90,95,99,max]:', np.percentile(nn,[50,90,95,99]).round(3), round(nn.max(),3))
    for thr in [0.90,0.95,0.98,0.99]:
        pairs=np.where(nn>=thr)[0]
        if len(pairs):
            same=np.mean([y[i]==y[nn_idx[i]] for i in pairs])
            print(f'  n with NN>= {thr}: {len(pairs):4d}   same-class among them: {same:.3f}')
        else:
            print(f'  n with NN>= {thr}: 0')
    # cross train->test
    Sc=Xte@Xtr.T; nnc=Sc.max(1)
    print(f'  test->train nearest cosine pctiles [50,90,99,max]:', np.percentile(nnc,[50,90,99]).round(3), round(nnc.max(),3))
    print(f'  n test with NN>=0.98: {(nnc>=0.98).sum()}  >=0.99: {(nnc>=0.99).sum()}')
