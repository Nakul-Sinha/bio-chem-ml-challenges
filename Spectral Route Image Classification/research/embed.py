"""Embedding-based interrogation: recover same-source (near-duplicate) structure,
measure the random-vs-grouped CV leakage gap, and gauge image-signal separability.
Runs locally (GPU if available). Caches embeddings for reuse.
"""
import os, sys, time
import numpy as np, pandas as pd
import torch, timm
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
DS = os.path.abspath(os.path.join(HERE, "..", "dataset"))
dev = 'cuda' if torch.cuda.is_available() else 'cpu'
print("device:", dev, "| torch", torch.__version__)

train = pd.read_csv(os.path.join(DS, "train.csv"))
test = pd.read_csv(os.path.join(DS, "test.csv"))
def resolve(p): return os.path.join(DS, str(p))

# ---- model ----
MODEL = os.environ.get("EMB_MODEL", "convnext_tiny.fb_in22k")
try:
    model = timm.create_model(MODEL, pretrained=True, num_classes=0).eval().to(dev)
    print("loaded", MODEL)
except Exception as e:
    print("failed", MODEL, "->", e, "; falling back to resnet50")
    MODEL = "resnet50.a1_in1k"
    model = timm.create_model(MODEL, pretrained=True, num_classes=0).eval().to(dev)
cfg = timm.data.resolve_data_config({}, model=model)
mean = torch.tensor(cfg['mean']).view(1,3,1,1); std = torch.tensor(cfg['std']).view(1,3,1,1)
print("input cfg:", cfg['input_size'], cfg['mean'], cfg['std'])

def load_batch(paths):
    xs=[]
    for p in paths:
        im=Image.open(resolve(p)).convert('RGB').resize((224,224), Image.BILINEAR)
        xs.append(torch.from_numpy(np.asarray(im)).permute(2,0,1).float()/255.)
    x=torch.stack(xs)
    return (x-mean)/std

@torch.no_grad()
def embed(df, tag):
    cache=os.path.join(HERE, f"emb_{tag}.npy")
    if os.path.exists(cache):
        print("cached", tag); return np.load(cache)
    feats=[]; t0=time.time(); B=32
    for i in range(0,len(df),B):
        x=load_batch(df['image_path'].iloc[i:i+B].tolist()).to(dev)
        if dev=='cuda':
            with torch.autocast('cuda', dtype=torch.float16):
                f=model(x)
        else:
            f=model(x)
        feats.append(f.float().cpu().numpy())
    E=np.concatenate(feats); np.save(cache, E)
    print(f"{tag}: embedded {len(df)} in {time.time()-t0:.1f}s shape {E.shape}")
    return E

Etr=embed(train,"train"); Ete=embed(test,"test")
# L2 normalize
Etr = Etr/ (np.linalg.norm(Etr,axis=1,keepdims=True)+1e-8)
Ete = Ete/ (np.linalg.norm(Ete,axis=1,keepdims=True)+1e-8)
y = train['target_id'].values

def hdr(s): print("\n"+"="*70+f"\n{s}\n"+"="*70)

hdr("TRAIN near-duplicate / same-source structure")
S = Etr @ Etr.T
np.fill_diagonal(S, -1)
top1 = S.max(1); nn = S.argmax(1)
for q in [50,75,90,95,99]:
    print(f"  top1-cos p{q}: {np.percentile(top1,q):.3f}")
# nearest-neighbour same-class rate as a function of threshold
for thr in [0.90,0.93,0.95,0.97,0.99]:
    m = top1>=thr
    if m.sum()==0: print(f"  thr {thr}: 0 imgs"); continue
    same = (y[nn][m]==y[m]).mean()
    print(f"  thr {thr}: {m.sum():4d} imgs have a NN>=thr | same-class rate {same:.3f}")

# connected-components clustering at a threshold -> candidate source groups
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
def cluster(thr):
    A=(S>=thr)
    n,lab=connected_components(csr_matrix(A), directed=False)
    sizes=pd.Series(lab).value_counts()
    multi=sizes[sizes>1]
    # purity of multi-clusters
    pure=0
    for c in multi.index:
        idx=np.where(lab==c)[0]
        pure+= int(len(set(y[idx]))==1)
    print(f"  thr {thr}: {n} components | multi(>1) {len(multi)} covering {int(multi.sum())} imgs | "
          f"max size {sizes.max()} | single-class multi-clusters {pure}/{len(multi)}")
    return lab
print("connected-components (group = near-dup source candidate):")
for thr in [0.97,0.95,0.93,0.90]:
    lab=cluster(thr)
grp = cluster(0.93)  # pick a working threshold for the leakage test

hdr("TRAIN<->TEST similarity (is test held-out-source?)")
St = Ete @ Etr.T
tmax = St.max(1)
for q in [50,75,90,95,99]:
    print(f"  test->train max-cos p{q}: {np.percentile(tmax,q):.3f}")
print("  (if test max-cos << train near-dup mode, test sources are genuinely held out)")

hdr("IMAGE-SIGNAL separability: kNN on embeddings, random vs grouped CV")
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedKFold, GroupKFold, cross_val_predict
from sklearn.metrics import f1_score, balanced_accuracy_score
def cvscore(splitter, groups=None, name=""):
    pred=cross_val_predict(KNeighborsClassifier(n_neighbors=15, metric='cosine'),
                           Etr, y, cv=splitter, groups=groups, n_jobs=-1)
    print(f"  {name:22s} macroF1 {f1_score(y,pred,average='macro'):.4f} balAcc {balanced_accuracy_score(y,pred):.4f}")
cvscore(StratifiedKFold(5,shuffle=True,random_state=0), None, "random 5-fold")
# group by near-dup cluster label
gk = GroupKFold(5)
cvscore(gk, grp, "grouped-by-nn-cluster")
print("\n(model used:", MODEL, ") EMBED DONE.")
