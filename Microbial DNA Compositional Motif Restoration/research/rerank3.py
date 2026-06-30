"""Reranker v3: LOGO-honest, richer features (neighbor co-occurrence, sibling fraction, gcond rank),
CF ensemble. Reports raw CV + test-calibrated MRR@10."""
import sys, time, numpy as np, collections
from pathlib import Path
import lightgbm as lgb
sys.path.insert(0,str(Path(__file__).resolve().parent))
from common import load, genome_key, group_folds, VOCAB
tr,te=load()
keys=[genome_key(g) for g in tr["gc"]]; folds=group_folds(keys,n=5,seed=0)
KS=[np.array(r,dtype=np.int64) for r in tr["kseq"]]
GCS=[set(int(x) for x in r) for r in tr["gc"]]
by=collections.defaultdict(list)
for i,k in enumerate(keys): by[k].append(i)
MASKPOS=np.array(range(4,16)); TOPK=80; POWER=4.0
TOT_COOC=np.zeros((VOCAB+1,VOCAB+1)); TOT_TOK=np.zeros(VOCAB+1); TOT_DF=np.zeros(VOCAB+1)
for k in KS:
    for t in set(int(x) for x in k): TOT_DF[t]+=1
    for a in k:
        TOT_TOK[a]+=1
        for b in k:
            if a!=b: TOT_COOC[a,b]+=1
NDOC=len(KS)

def gen_features(rng):
    rows=[];labels=[];groups=[];gin=[];gfold=[];gsize=[]
    for gk,idxs in by.items():
        m=len(idxs)
        cg=np.zeros((VOCAB+1,VOCAB+1)); tg=np.zeros(VOCAB+1); dg=np.zeros(VOCAB+1)
        for j in idxs:
            k=KS[j]
            for t in set(int(x) for x in k): dg[t]+=1
            for a in k:
                tg[a]+=1
                for b in k:
                    if a!=b: cg[a,b]+=1
        cooc=TOT_COOC-cg; tok=TOT_TOK-tg; dfo=TOT_DF-dg
        LOGCOND=np.log((cooc+0.1)/(tok[:,None]+0.1*VOCAB)); GLOB=np.log((dfo+0.5)/(dfo.sum()+0.5*VOCAB))
        IDFo=np.log((NDOC-m+1)/(dfo+1))+1.0
        mp=rng.choice(MASKPOS,size=m)
        Vb=np.zeros((m,VOCAB+1),dtype=np.float32); gold=np.zeros(m,dtype=np.int64); pos=np.full((m,VOCAB+1),-1,np.int64)
        seq=[]
        for a,j in enumerate(idxs):
            k=KS[j]; p=mp[a]; gold[a]=k[p]; seq.append(k)
            for q in range(16):
                if q!=p: Vb[a,k[q]]=1.0; pos[a,k[q]]=q
        Vidf=Vb*IDFo; sim=Vidf@Vb.T; np.fill_diagonal(sim,0.0); simp=sim**POWER
        cf=simp@Vb; C=Vb.T@Vb; gcond=Vb@C
        nsib=Vb.sum(0); ps=np.where(pos>=0,pos,0).sum(0); pc=(pos>=0).sum(0); avgpos=np.where(pc>0,ps/np.maximum(pc,1),8.0)
        f=folds[idxs[0]]
        for a in range(m):
            k=seq[a]; p=mp[a]
            nb=[k[q] for q in (p-2,p-1,p+1,p+2) if 0<=q<16 and q!=p]
            cocb=LOGCOND[nb].mean(0) if nb else np.zeros(VOCAB+1)
            co=LOGCOND[np.where(Vb[a]>0)[0]].mean(0)
            s=cf[a].copy(); s[0]=-1e9; s[Vb[a]>0]=-1e9
            cand=np.argpartition(-s,TOPK)[:TOPK]; cand=cand[np.argsort(-s[cand])]
            gcr={t:r for r,t in enumerate(np.argsort(-gcond[a]))}
            g=gold[a]; nf=0
            for rk,t in enumerate(cand):
                rows.append([cf[a,t],rk,np.log1p(gcond[a,t]),gcr.get(t,999),co[t],cocb[t],
                             float(t in GCS[idxs[a]]),IDFo[t],GLOB[t],nsib[t],nsib[t]/m,avgpos[t]-p,p,m,s[t]])
                labels.append(1 if t==g else 0); nf+=1
            groups.append(nf); gin.append(int(g in cand)); gfold.append(f); gsize.append(m)
    return (np.array(rows,dtype=np.float32),np.array(labels),np.array(groups),np.array(gfold),np.array(gsize),np.mean(gin))

t0=time.time(); rng=np.random.default_rng(100)
X,y,grp,gfold,gsize,recall=gen_features(rng)
print(f"features {X.shape} recall@{TOPK}={recall:.4f} [{time.time()-t0:.0f}s]")
offs=np.concatenate([[0],np.cumsum(grp)])
CFRANKcol=1
def per_group_mrr(scores):
    rr=np.zeros(len(grp))
    for g in range(len(grp)):
        sl=slice(offs[g],offs[g+1]); order=np.argsort(-scores[sl])[:10]; lab=y[sl]
        rank=next((r+1 for r,o in enumerate(order) if lab[o]==1),None); rr[g]=1.0/rank if rank else 0.0
    return rr
scores=np.zeros(len(y))
for f in range(5):
    trg=np.where(gfold!=f)[0]; vag=np.where(gfold==f)[0]
    trr=np.concatenate([np.arange(offs[g],offs[g+1]) for g in trg]); var=np.concatenate([np.arange(offs[g],offs[g+1]) for g in vag])
    d=lgb.Dataset(X[trr],label=y[trr],group=grp[trg])
    p=dict(objective="lambdarank",metric="ndcg",ndcg_eval_at=[10],learning_rate=0.04,num_leaves=31,
           min_data_in_leaf=80,feature_fraction=0.8,bagging_fraction=0.8,bagging_freq=1,verbose=-1,lambda_l2=1.0)
    mdl=lgb.train(p,d,num_boost_round=350); scores[var]=mdl.predict(X[var])
print(f"reranker built [{time.time()-t0:.0f}s]")
# ensemble: reranker + w*(-cfrank)  (cf order)
def calib(rr_per_group):
    buckets=[(1,3),(4,8),(9,20),(21,60),(61,150),(151,10**9)]
    tek=[genome_key(g) for g in te["gc"]]; tec=collections.Counter(tek); test_w=collections.Counter()
    for k in tek:
        sz=tec[k]
        for lo,hi in buckets:
            if lo<=sz<=hi: test_w[(lo,hi)]+=1; break
    tot=sum(test_w.values()); bm={}
    for lo,hi in buckets:
        mask=(gsize>=lo)&(gsize<=hi)
        if mask.sum(): bm[(lo,hi)]=rr_per_group[mask].mean()
    return sum(bm.get(b,0)*test_w.get(b,0) for b in buckets)/tot, bm
rrr=per_group_mrr(scores)
print(f"RAW reranker CV MRR={rrr.mean():.4f}  test-calibrated={calib(rrr)[0]:.4f}")
# blend reranker with CF (use -cfrank as CF score proxy within candidates)
best=(-1,None)
for w in [0,0.5,1,2,4,8]:
    sc=scores - w*(X[:,CFRANKcol])   # lower cfrank = better
    rr=per_group_mrr(sc); c=calib(rr)[0]
    if c>best[0]: best=(c,w,rr.mean())
print(f"BEST ensemble test-calibrated={best[0]:.4f} (w={best[1]}, raw={best[2]:.4f})")
print("importance:",dict(zip(['cf','cfrank','gcond','gcrank','cooc','coocnb','gcmem','idf','glob','nsib','sibfrac','posdiff','p','m','cfscore'],mdl.feature_importance())))
