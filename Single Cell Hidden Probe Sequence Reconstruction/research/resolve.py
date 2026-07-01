"""Resolve the CV contradiction: random vs grouped AUC, and a kNN retrieval baseline."""
import numpy as np, pandas as pd, srlib as L
from sklearn.model_selection import GroupKFold, KFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
D="../dataset/"; train=pd.read_csv(D+"train.csv"); N=len(train)
obs=pd.read_csv(D+"observed_panel.csv"); DG=obs.set_index('observed_index').damage_group.values

def parse_all(df):
    OV=np.zeros((len(df),80),np.float32);TOT=np.zeros(len(df),np.float32);NZ=np.zeros(len(df),np.float32)
    DMG=np.full(len(df),-1,np.int64);COND=[];SEX=[];STAGE=[]
    for i,s in enumerate(df['source_sequence'].values):
        ov,tot,nz,panel,meta=L.parse_source(s)
        OV[i]=ov;TOT[i]=tot;NZ[i]=nz;DMG[i]=L.panel_damage_group(panel)
        COND.append(0 if meta.get('COND')=='COND_004' else 1)
        SEX.append(1 if meta.get('SEX')=='SEX_006' else 0)
        STAGE.append(meta.get('STAGE','?'))
    st_u=sorted(set(STAGE));st_map={v:i for i,v in enumerate(st_u)}
    STG=np.array([st_map[s] for s in STAGE],np.int64)
    return OV,TOT,NZ,DMG,np.array(COND,np.float32),np.array(SEX,np.float32),STG,len(st_u)
OV,TOT,NZ,DMG,COND,SEX,STG,nst=parse_all(train)
Y=np.stack([L.parse_target(s) for s in train['target_sequence']]).astype(np.int64)
mu=OV.mean(0);sd=OV.std(0)+1e-6
X=np.concatenate([(OV-mu)/sd,(OV>0).astype(np.float32),(TOT/63.)[:,None],(NZ/60.)[:,None],
                  COND[:,None],SEX[:,None],np.eye(nst,dtype=np.float32)[STG]],axis=1).astype(np.float32)
gid,gkeys=L.make_groups(train)
active=(Y>0).astype(int)

def auc_cv(splitter,groups=None):
    pact=np.zeros((N,16))
    for tr,va in splitter.split(X,groups=groups):
        for t in range(16):
            clf=LogisticRegression(max_iter=300,C=1.0)
            yt=active[tr,t]
            if len(np.unique(yt))<2: pact[va,t]=yt.mean(); continue
            clf.fit(X[tr],yt); pact[va,t]=clf.predict_proba(X[va])[:,1]
    return np.array([roc_auc_score(active[:,t],pact[:,t]) for t in range(16)])

ag=auc_cv(GroupKFold(5),gid)
ar=auc_cv(KFold(5,shuffle=True,random_state=0))
print("per-target active AUC:  grouped  random  gap")
for t in range(16):
    print(f"  T{t:02d}: {ag[t]:.3f}   {ar[t]:.3f}   {ar[t]-ag[t]:+.3f}")
print(f"MEAN: grouped={ag.mean():.3f} random={ar.mean():.3f} gap={ar.mean()-ag.mean():+.3f}")

# ---- kNN retrieval baseline (grouped) ----
print("\nkNN RETRIEVAL (cosine on normalized presence+value), grouped CV")
gsize=pd.Series(gkeys).value_counts(); small=set(gsize[gsize<=gsize.quantile(0.35)].index)
rare_tokens,_=L.rare_token_set(train,160)
flags=L.subset_flags(train,np.arange(N),rare_tokens,small,gkeys)
true_seqs=[L.target_tokens(s) for s in train['target_sequence']]
def score(pb):
    ps=[L.bins_to_seq(pb[i]) for i in range(N)]; return L.weighted_score(ps,true_seqs,flags)
Xv=np.concatenate([OV.astype(np.float32),(OV>0).astype(np.float32)*3.0],axis=1)
Xn=Xv/np.clip(np.linalg.norm(Xv,axis=1,keepdims=True),1e-9,None)
oof_knn=np.zeros((N,16,4))
gkf=GroupKFold(5)
for tr,va in gkf.split(X,groups=gid):
    sim=Xn[va]@Xn[tr].T   # (nva, ntr)
    for k,vi in enumerate(va):
        nn=np.argpartition(-sim[k],30)[:30]
        wt=np.clip(sim[k,nn],0,None)+1e-6
        for t in range(16):
            for b in range(4):
                oof_knn[vi,t,b]=wt[Y[tr[nn],t]==b].sum()
            oof_knn[vi,t]/=oof_knn[vi,t].sum()+1e-9
pact=1-oof_knn[:,:,0]; argbin=oof_knn[:,:,1:].argmax(2)+1
bestk=(-1,None,None)
for tau in np.arange(0.04,0.5,0.03):
    for mode in ['B1','argmax']:
        bins=np.ones((N,16),int) if mode=='B1' else argbin
        pb=np.where(pact>=tau,bins,0)
        f=score(pb)['final']
        if f>bestk[0]: bestk=(f,tau,mode)
print(f"  >> best kNN FINAL={bestk[0]:.4f} tau={bestk[1]:.2f} mode={bestk[2]}")
print("DONE")
