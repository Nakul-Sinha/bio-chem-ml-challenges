"""Two questions:
(1) Is AUC 0.64 a data ceiling or overfitting? -> train vs val AUC.
(2) Which 'normalized_edit_similarity' / 'order_lcs' convention is the grader likely using?
    Test variants on prior + a decent decode; see which yields ~0.30 for a reasonable model."""
import numpy as np, pandas as pd, srlib as L, difflib
from sklearn.model_selection import GroupKFold
import lightgbm as lgb
D="../dataset/"; train=pd.read_csv(D+"train.csv"); N=len(train)
Y=np.load("Y.npy"); oof=np.load("oof_gbdt_grouped.npy")
active=(Y>0).astype(int)

# ---------- (1) train vs val AUC (GBDT) ----------
from sklearn.metrics import roc_auc_score
def parse_all(df):
    OV=np.zeros((len(df),80),np.float32);TOT=np.zeros(len(df),np.float32);NZ=np.zeros(len(df),np.float32)
    for i,s in enumerate(df['source_sequence'].values):
        ov,tot,nz,panel,meta=L.parse_source(s); OV[i]=ov;TOT[i]=tot;NZ[i]=nz
    return OV,TOT,NZ
OV,TOT,NZ=parse_all(train)
X=np.concatenate([OV,(OV>0).astype(np.float32),TOT[:,None],NZ[:,None]],1).astype(np.float32)
gid,gkeys=L.make_groups(train)
tr_auc=[];va_auc=[]
P=dict(objective='binary',learning_rate=0.05,num_leaves=31,min_child_samples=20,
       subsample=0.8,colsample_bytree=0.6,reg_lambda=1.0,n_estimators=200,verbose=-1)
for t in [0,7,13,14]:  # sample weak + strong targets
    trA=[];vaA=[]
    for tr,va in GroupKFold(5).split(X,groups=gid):
        m=lgb.LGBMClassifier(**P); m.fit(X[tr],active[tr,t])
        trA.append(roc_auc_score(active[tr,t],m.predict_proba(X[tr])[:,1]))
        vaA.append(roc_auc_score(active[va,t],m.predict_proba(X[va])[:,1]))
    print(f"  T{t:02d}: train_auc={np.mean(trA):.3f}  val_auc={np.mean(vaA):.3f}  gap={np.mean(trA)-np.mean(vaA):+.3f}")

# ---------- (2) metric variants ----------
def to_str(bins):
    s=L.bins_to_seq(bins); return s if s else ['NONE']
def tokens(bins):
    t=L.bins_to_seq(bins); return t if t else ['NONE']

# map every possible token to a unique single char for string-based Levenshtein
ALLTOK=['NONE']+[f"T{ti:02d}_B{b}" for ti in range(16) for b in (1,2,3)]
CH={tk:chr(33+i) for i,tk in enumerate(ALLTOK)}
def as_str(toklist): return ''.join(CH[x] for x in toklist)

def score_variant(pred_bins, variant):
    tot=0.0
    for i in range(N):
        p=tokens(pred_bins[i]); t=tokens(Y[i])
        la,lb=len(p),len(t); m=max(la,lb)
        ps,ts=set(p),set(t); inter=len(ps&ts)
        prec=inter/la if la else 0; rec=inter/lb if lb else 0
        f1=2*prec*rec/(prec+rec) if prec+rec>0 else 0
        lev=L._lev(p,t)          # exact token-level Levenshtein, substitution cost 1
        lcs=inter                # canonical => LCS=intersection
        if variant=='max':      es=1-lev/m; ls=lcs/m
        elif variant=='sum':    es=1-lev/(la+lb); ls=lcs/m
        elif variant=='ratio':  es=2*lcs/(la+lb); ls=2*lcs/(la+lb)   # difflib / Lev.ratio style
        elif variant=='mix':    es=1-lev/m; ls=2*lcs/(la+lb)
        tot+=0.5*es+0.3*f1+0.2*ls
    return tot/N

# prior top-15 all B1
ac=active.sum(0); order=np.argsort(-ac)
prior=np.zeros((N,16),int); prior[:,order[:15]]=1
# decent decode: threshold
pact=1-oof[:,:,0]; dec=np.where(pact>=0.06,1,0)
for v in ['max','sum','ratio','mix']:
    try:
        print(f"  variant={v:9s}  prior(all-B1)={score_variant(prior,v):.4f}   gbdt_thresh={score_variant(dec,v):.4f}")
    except Exception as e:
        print("  variant",v,"err",e)
print("DONE")
