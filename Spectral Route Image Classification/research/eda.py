"""Interrogation / EDA for Spectral Route Image Classification.
CPU-only. Answers: schema, class balance, anchor freq, artifact_signature grouping,
metadata leakiness (does metadata alone predict class?), stress proxy, image integrity,
and a first look at same-source (near-duplicate) structure that would break random CV.
"""
import os, sys, hashlib
import numpy as np, pandas as pd
from collections import Counter, defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
DS = os.path.abspath(os.path.join(HERE, "..", "dataset"))

train = pd.read_csv(os.path.join(DS, "train.csv"))
test = pd.read_csv(os.path.join(DS, "test.csv"))
ss = pd.read_csv(os.path.join(DS, "sample_submission.csv"))

def hdr(s): print("\n" + "="*70 + f"\n{s}\n" + "="*70)

hdr("SHAPES / COLUMNS")
print("train", train.shape, "| test", test.shape, "| sample_sub", ss.shape)
print("\ntrain cols:", list(train.columns))
print("\ntest cols :", list(test.columns))
print("\nss cols   :", list(ss.columns))
print("\nsample_submission head:\n", ss.head(3).to_string())

hdr("CLASS BALANCE")
print(train['target'].value_counts().to_string())
if 'target_id' in train.columns:
    chk = train.groupby('target')['target_id'].agg(['min','max','nunique'])
    print("\ntarget<->target_id consistency (nunique should be 1 each):\n", chk.to_string())
ANCHOR = 'route-aphelion'
print(f"\nANCHOR {ANCHOR}: {(train['target']==ANCHOR).sum()}/{len(train)} = {(train['target']==ANCHOR).mean():.4f}")
print("balanced-chance macro-F1 ~= 1/6 = 0.167")

hdr("IDS")
print("train id unique:", train['id'].is_unique, "| test id unique:", test['id'].is_unique)
print("train/test id overlap:", len(set(train['id']) & set(test['id'])))
if len(ss)==len(test):
    print("ss id order == test id order:", bool((ss['id'].values==test['id'].values).all()))
print("id length sample:", train['id'].iloc[0], "len", len(str(train['id'].iloc[0])))

hdr("ARTIFACT_SIGNATURE")
for nm, df in [('train',train),('test',test)]:
    if 'artifact_signature' in df.columns:
        u = df['artifact_signature'].nunique()
        print(f"{nm}: {u} unique / {len(df)} rows (unique ratio {u/len(df):.3f})")
        print(f"   examples: {list(df['artifact_signature'].unique()[:5])}")
if 'artifact_signature' in train.columns:
    sizes = train['artifact_signature'].value_counts()
    purity = train.groupby('artifact_signature')['target'].nunique()
    multi = sizes[sizes>1].index
    print(f"\nsignature group sizes: max {sizes.max()}, mean {sizes.mean():.2f}, #groups>1 {(sizes>1).sum()}")
    if len(multi):
        print(f"among {len(multi)} multi-row signatures: pure(1 class) {(purity[multi]==1).sum()}, impure {(purity[multi]>1).sum()}")
    ov = len(set(train['artifact_signature']) & set(test['artifact_signature']))
    print("train/test signature overlap:", ov)

hdr("METADATA FEATURES: ranges / NaN")
meta = [c for c in train.columns if c.endswith('_score')]
print(f"{len(meta)} score features")
desc = train[meta].describe().T[['min','max','mean','std']]
print(desc.to_string())
print("total NaNs in meta:", int(train[meta].isna().sum().sum()))
print("stress_flag in train?", 'stress_flag' in train.columns)
if 'stress_flag' in ss.columns:
    print("ss stress_flag dist:", ss['stress_flag'].value_counts().to_dict())

hdr("METADATA LEAKINESS / SIGNAL  (metadata-only classifier)")
# Does metadata ALONE predict the class? high macro-F1 => leak or real signal.
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import f1_score, balanced_accuracy_score
X = train[meta].values.astype(np.float32)
y = train['target_id'].values if 'target_id' in train.columns else pd.factorize(train['target'])[0]
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
try:
    pred = cross_val_predict(HistGradientBoostingClassifier(max_iter=200, learning_rate=0.08,
                             max_depth=None, random_state=0), X, y, cv=skf, n_jobs=-1)
    print("metadata-only (random 5-fold): macroF1 %.4f  balAcc %.4f" %
          (f1_score(y,pred,average='macro'), balanced_accuracy_score(y,pred)))
    print("  -> if ~0.17 metadata is non-leaky/uninformative; if high, it carries class info")
except Exception as e:
    print("meta GBM failed:", e)

hdr("IMAGE PATHS / INTEGRITY")
print("example train image_path:", train['image_path'].iloc[0])
print("example test  image_path:", test['image_path'].iloc[0])
def resolve(p):
    cands = [os.path.join(DS, p),
             os.path.join(DS, str(p).replace('./dataset/public/','').replace('dataset/public/','')),
             os.path.join(DS, os.path.basename(os.path.dirname(str(p))), os.path.basename(str(p)))]
    for c in cands:
        if os.path.exists(c): return c
    # last resort: search under dataset/images
    return None
root_ok = resolve(train['image_path'].iloc[0])
print("resolved first train image to:", root_ok)
from PIL import Image
def check_sample(df, n=40):
    sizes=Counter(); modes=Counter(); missing=0; ok=0
    idx=np.linspace(0,len(df)-1,min(n,len(df))).astype(int)
    for i in idx:
        rp=resolve(df['image_path'].iloc[i])
        if rp is None: missing+=1; continue
        try:
            im=Image.open(rp); sizes[im.size]+=1; modes[im.mode]+=1; ok+=1
        except Exception: missing+=1
    return ok,missing,sizes,modes
for nm,df in [('train',train),('test',test)]:
    ok,miss,sizes,modes=check_sample(df)
    print(f"{nm}: sampled ok {ok} missing {miss} | sizes {dict(sizes)} | modes {dict(modes)}")

hdr("SAME-SOURCE / NEAR-DUP STRUCTURE (cheap dhash on downscaled gray)")
# If train has multiple degraded views per source, random CV leaks. First cheap probe:
# 8x8 dhash of a strongly-downscaled, blurred grayscale -> exact-collision clusters.
def dhash(path, size=8):
    try:
        im = Image.open(path).convert('L').resize((size+1, size), Image.BILINEAR)
        a = np.asarray(im, dtype=np.int16)
        diff = a[:,1:] > a[:,:-1]
        return diff.tobytes()
    except Exception:
        return None
paths = [resolve(p) for p in train['image_path']]
valid = [(i,p) for i,p in enumerate(paths) if p]
print(f"hashing {len(valid)} train images...")
buckets=defaultdict(list)
for i,p in valid:
    h=dhash(p)
    if h is not None: buckets[h].append(i)
clusters=[v for v in buckets.values() if len(v)>1]
print(f"exact-dhash clusters (size>1): {len(clusters)}; images in clusters: {sum(len(c) for c in clusters)}")
if clusters:
    szc=Counter(len(c) for c in clusters)
    print("cluster-size distribution:", dict(sorted(szc.items())))
    # are clustered images same class? (they should be if same source)
    same=0; tot=0
    for c in clusters:
        labs=train['target_id'].iloc[c].values if 'target_id' in train.columns else pd.factorize(train['target'])[0][c]
        tot+=1; same+= int(len(set(labs))==1)
    print(f"of {tot} clusters, {same} are single-class (same source signature)")
print("\nNOTE: dhash is a weak probe under heavy degradation; will confirm via CNN embeddings on GPU.")
print("\nEDA DONE.")
