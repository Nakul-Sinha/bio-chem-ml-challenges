"""LightGBM per-target ceiling probe on honest grouped CV. Tells us achievable signal."""
import numpy as np, pandas as pd, time, srlib as L
import lightgbm as lgb
from sklearn.model_selection import GroupKFold, KFold

D="../dataset/"; train=pd.read_csv(D+"train.csv"); test=pd.read_csv(D+"test.csv")
N=len(train)

def build_X(df):
    OV=np.zeros((len(df),80),np.float32); TOT=np.zeros(len(df),np.float32); NZ=np.zeros(len(df),np.float32)
    DMG=np.full(len(df),-1,np.int64); COND=[];SEX=[];STAGE=[]
    for i,s in enumerate(df['source_sequence'].values):
        ov,tot,nz,panel,meta=L.parse_source(s)
        OV[i]=ov;TOT[i]=tot;NZ[i]=nz;DMG[i]=L.panel_damage_group(panel)
        COND.append(meta.get('COND','?'));SEX.append(meta.get('SEX','?'));STAGE.append(meta.get('STAGE','?'))
    cond=np.array([0 if c=='COND_004' else 1 for c in COND],np.float32)
    sex=np.array([1 if s=='SEX_006' else 0 for s in SEX],np.float32)
    st_u=sorted(set(STAGE)); st_map={v:i for i,v in enumerate(st_u)}
    stg=np.array([st_map[s] for s in STAGE],np.float32)
    X=np.concatenate([OV,(OV>0).astype(np.float32),TOT[:,None],NZ[:,None],
                      DMG[:,None].astype(np.float32),cond[:,None],sex[:,None],stg[:,None]],axis=1)
    return X.astype(np.float32)

X=build_X(train)
Y=np.stack([L.parse_target(s) for s in train['target_sequence']])
print("X",X.shape)

gid,gkeys=L.make_groups(train)
gsize=pd.Series(gkeys).value_counts()
small=set(gsize[gsize<=gsize.quantile(0.35)].index)
rare_tokens,_=L.rare_token_set(train,160)
flags=L.subset_flags(train,np.arange(N),rare_tokens,small,gkeys)
true_seqs=[L.target_tokens(s) for s in train['target_sequence']]

def score_bins(pred_bins):
    ps=[L.bins_to_seq(pred_bins[i]) for i in range(N)]
    return L.weighted_score(ps,true_seqs,flags)

LGB=dict(objective='multiclass',num_class=4,learning_rate=0.05,num_leaves=31,
         min_child_samples=20,subsample=0.8,subsample_freq=1,colsample_bytree=0.6,
         reg_lambda=1.0,n_estimators=300,verbose=-1,n_jobs=-1)

def cv_oof(splitter, groups):
    oof=np.zeros((N,16,4),np.float32)
    for tr,va in splitter.split(X,groups=groups):
        for t in range(16):
            yt=Y[tr,t]
            m=lgb.LGBMClassifier(**LGB)
            m.fit(X[tr],yt)
            pr=m.predict_proba(X[va])
            for ci,c in enumerate(m.classes_): oof[va,t,c]=pr[:,ci]
    return oof

t0=time.time()
gkf=GroupKFold(n_splits=5)
oof=cv_oof(gkf,gid)
print("trained grouped in %.1fs"%(time.time()-t0))
np.save("oof_gbdt_grouped.npy",oof)

# decode: global threshold sweep
def decode_global(oof,tau):
    pact=1-oof[:,:,0]; binc=oof[:,:,1:].argmax(2)+1
    return np.where(pact>=tau,binc,0)
print("GBDT grouped, global threshold:")
best=(-1,None,None)
for tau in np.arange(0.15,0.55,0.05):
    r=score_bins(decode_global(oof,tau))
    print(f"  tau={tau:.2f} FINAL={r['final']:.4f} all={r['all']:.4f} sh={r['shifted']:.4f} dm={r['damaged']:.4f} ra={r['rare']:.4f}")
    if r['final']>best[0]: best=(r['final'],tau,r)
print(f"  >> best tau={best[1]:.2f} FINAL={best[0]:.4f}")

# argmax reference
r=score_bins(oof.argmax(2)); print(f"  argmax ref FINAL={r['final']:.4f}")
print("DONE gbdt")
