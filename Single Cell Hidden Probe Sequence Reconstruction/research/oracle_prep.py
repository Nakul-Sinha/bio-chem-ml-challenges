"""(1) Oracle decomposition: where is score lost - selection vs bins?
   (2) Preprocessing sweep: does library-size norm / log / rank lift active AUC?"""
import numpy as np, pandas as pd, srlib as L
from sklearn.model_selection import GroupKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
D="../dataset/"; train=pd.read_csv(D+"train.csv"); N=len(train)

def parse_all(df):
    OV=np.zeros((len(df),80),np.float32);TOT=np.zeros(len(df),np.float32);NZ=np.zeros(len(df),np.float32)
    for i,s in enumerate(df['source_sequence'].values):
        ov,tot,nz,panel,meta=L.parse_source(s); OV[i]=ov;TOT[i]=tot;NZ[i]=nz
    return OV,TOT,NZ
OV,TOT,NZ=parse_all(train)
Y=np.stack([L.parse_target(s) for s in train['target_sequence']]).astype(np.int64)
active=(Y>0).astype(int)
gid,gkeys=L.make_groups(train)
gsize=pd.Series(gkeys).value_counts(); small=set(gsize[gsize<=gsize.quantile(0.35)].index)
rare_tokens,_=L.rare_token_set(train,160)
flags=L.subset_flags(train,np.arange(N),rare_tokens,small,gkeys)
true_seqs=[L.target_tokens(s) for s in train['target_sequence']]
def score(pb):
    ps=[L.bins_to_seq(pb[i]) for i in range(N)]; return L.weighted_score(ps,true_seqs,flags)['final']

# modal bin per target
modal=np.zeros(16,int)
for t in range(16):
    v=Y[:,t][Y[:,t]>0]; modal[t]=np.bincount(v).argmax()

print("="*60);print("ORACLE DECOMPOSITION (where is the score lost?)")
# O1: true active set, true bins -> perfect
print("  O1 true set + true bins (perfect):", round(score(Y),4))
# O2: true active set, modal bins
pb=np.zeros((N,16),int)
for t in range(16): pb[:,t]=np.where(active[:,t]>0,modal[t],0)
print("  O2 true set + MODAL bins:        ", round(score(pb),4))
# O3: true active set, all-B1
pb=np.where(active>0,1,0)
print("  O3 true set + all-B1 bins:       ", round(score(pb),4))
# Prior: top-15 targets all-B1
ac=active.sum(0); order=np.argsort(-ac)
pb=np.zeros((N,16),int); pb[:,order[:15]]=1
print("  PRIOR top-15 all-B1:             ", round(score(pb),4))
# O4: oracle COUNT per row (know how many active), pick top-count by a weak model P(active), all-B1
# use grouped LR pact
mu=OV.mean(0);sd=OV.std(0)+1e-6
X=np.concatenate([(OV-mu)/sd,(OV>0).astype(np.float32),(TOT/63.)[:,None],(NZ/60.)[:,None]],axis=1).astype(np.float32)
pact=np.zeros((N,16))
for tr,va in GroupKFold(5).split(X,groups=gid):
    for t in range(16):
        yt=active[tr,t]
        if len(np.unique(yt))<2: pact[va,t]=yt.mean(); continue
        c=LogisticRegression(max_iter=200,C=1.0); c.fit(X[tr],yt); pact[va,t]=c.predict_proba(X[va])[:,1]
order_p=np.argsort(-pact,axis=1)
truecount=active.sum(1)
pb=np.zeros((N,16),int)
for i in range(N):
    pb[i,order_p[i,:truecount[i]]]=1
print("  O4 ORACLE-count top-k by LR pact, all-B1:", round(score(pb),4))
# O5: model selection (best fixed decode) already ~0.25

print("="*60);print("PREPROCESSING SWEEP: active AUC (grouped LR)")
def auc_for(Xf):
    pact=np.zeros((N,16))
    for tr,va in GroupKFold(5).split(Xf,groups=gid):
        for t in range(16):
            yt=active[tr,t]
            if len(np.unique(yt))<2: pact[va,t]=yt.mean(); continue
            c=LogisticRegression(max_iter=300,C=1.0); c.fit(Xf[tr],yt); pact[va,t]=c.predict_proba(Xf[va])[:,1]
    return np.array([roc_auc_score(active[:,t],pact[:,t]) for t in range(16)])

reps={}
reps['raw_z']=(OV-mu)/sd
# library-size normalize: divide by row total (use actual sum of OV), then z
lib=OV/np.clip(OV.sum(1,keepdims=True),1e-6,None)
reps['libnorm_z']=(lib-lib.mean(0))/(lib.std(0)+1e-6)
# log1p of raw
lg=np.log1p(OV); reps['log_z']=(lg-lg.mean(0))/(lg.std(0)+1e-6)
# log of libnorm *1e4 (scRNA standard)
lln=np.log1p(lib*1e4); reps['log_libnorm_z']=(lln-lln.mean(0))/(lln.std(0)+1e-6)
# rank per row (composition rank)
rank=np.argsort(np.argsort(OV,axis=1),axis=1).astype(np.float32)/79.0
reps['rowrank']=(rank-rank.mean(0))/(rank.std(0)+1e-6)
# presence only
reps['presence']=(OV>0).astype(np.float32)
for name,Xf in reps.items():
    Xf=np.concatenate([Xf,(OV>0).astype(np.float32)],axis=1).astype(np.float32)
    a=auc_for(Xf)
    print(f"  {name:16s} meanAUC={a.mean():.3f}  T13={a[13]:.3f} T14={a[14]:.3f}  weak-mean(excl 13,14)={np.delete(a,[13,14]).mean():.3f}")
print("DONE")
