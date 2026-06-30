"""Honest reranker: leave-one-GENOME-out global co-occurrence/idf (mirrors disjoint test).
Within-genome CF/gcond are transductive & leak-free. Group-CV, MRR@10."""
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
MASKPOS=np.array(range(4,16)); TOPK=60; POWER=4.0

# global totals (full kseq) over all train
TOT_COOC=np.zeros((VOCAB+1,VOCAB+1)); TOT_TOK=np.zeros(VOCAB+1); TOT_DF=np.zeros(VOCAB+1)
for k in KS:
    for t in set(int(x) for x in k): TOT_DF[t]+=1
    for a in k:
        TOT_TOK[a]+=1
        for b in k:
            if a!=b: TOT_COOC[a,b]+=1
NDOC=len(KS)

def gen_features(rng):
    rows=[];labels=[];groups=[];gold_in=[];grp_fold=[]
    for gk,idxs in by.items():
        m=len(idxs)
        # ---- leave-one-genome-out globals ----
        cg=np.zeros((VOCAB+1,VOCAB+1)); tg=np.zeros(VOCAB+1); dg=np.zeros(VOCAB+1)
        for j in idxs:
            k=KS[j]
            for t in set(int(x) for x in k): dg[t]+=1
            for a in k:
                tg[a]+=1
                for b in k:
                    if a!=b: cg[a,b]+=1
        cooc=TOT_COOC-cg; tok=TOT_TOK-tg; dfo=TOT_DF-dg
        LOGCOND=np.log((cooc+0.1)/(tok[:,None]+0.1*VOCAB))
        GLOBLOG=np.log((dfo+0.5)/(dfo.sum()+0.5*VOCAB))
        IDFo=np.log((NDOC-m+1)/(dfo+1))+1.0
        # ---- within-genome transductive (visible) ----
        mp=rng.choice(MASKPOS,size=m)
        Vb=np.zeros((m,VOCAB+1),dtype=np.float32); gold=np.zeros(m,dtype=np.int64); posmat=np.full((m,VOCAB+1),-1,dtype=np.int64)
        for a,j in enumerate(idxs):
            k=KS[j]; p=mp[a]; gold[a]=k[p]
            for q in range(16):
                if q!=p: Vb[a,k[q]]=1.0; posmat[a,k[q]]=q
        Vidf=Vb*IDFo; sim=Vidf@Vb.T; np.fill_diagonal(sim,0.0); simp=sim**POWER
        cf=simp@Vb; C=Vb.T@Vb; gcond=Vb@C
        nsib=Vb.sum(0); possum=np.where(posmat>=0,posmat,0).sum(0); poscnt=(posmat>=0).sum(0)
        avgpos=np.where(poscnt>0,possum/np.maximum(poscnt,1),8.0)
        f=folds[idxs[0]]
        for a in range(m):
            s=cf[a].copy(); s[0]=-1e9; s[Vb[a]>0]=-1e9
            cand=np.argpartition(-s,TOPK)[:TOPK]; cand=cand[np.argsort(-s[cand])]
            vis=np.where(Vb[a]>0)[0]; co=LOGCOND[vis].mean(0); g=gold[a]; nf=0
            for rk,t in enumerate(cand):
                rows.append([cf[a,t],rk,np.log1p(gcond[a,t]),co[t],float(t in GCS[idxs[a]]),
                             IDFo[t],GLOBLOG[t],nsib[t],avgpos[t]-mp[a],mp[a],m,s[t]])
                labels.append(1 if t==g else 0); nf+=1
            groups.append(nf); gold_in.append(int(g in cand)); grp_fold.append(f)
    return (np.array(rows,dtype=np.float32),np.array(labels),np.array(groups),
            np.array(grp_fold),np.mean(gold_in))

t0=time.time(); rng=np.random.default_rng(100)
X,y,grp,grp_fold,recall=gen_features(rng)
print(f"features {X.shape} recall@{TOPK}={recall:.4f} [{time.time()-t0:.0f}s]")
offs=np.concatenate([[0],np.cumsum(grp)])
def mrr(scores,vg):
    rr=[]
    for g in vg:
        sl=slice(offs[g],offs[g+1]); order=np.argsort(-scores[sl])[:10]; lab=y[sl]
        rank=next((r+1 for r,o in enumerate(order) if lab[o]==1),None); rr.append(1.0/rank if rank else 0.0)
    return np.mean(rr)
scores=np.zeros(len(y))
for f in range(5):
    trg=np.where(grp_fold!=f)[0]; vag=np.where(grp_fold==f)[0]
    trr=np.concatenate([np.arange(offs[g],offs[g+1]) for g in trg])
    var=np.concatenate([np.arange(offs[g],offs[g+1]) for g in vag])
    d=lgb.Dataset(X[trr],label=y[trr],group=grp[trg])
    p=dict(objective="lambdarank",metric="ndcg",ndcg_eval_at=[10],learning_rate=0.05,num_leaves=31,
           min_data_in_leaf=50,feature_fraction=0.9,bagging_fraction=0.8,bagging_freq=1,verbose=-1)
    mdl=lgb.train(p,d,num_boost_round=300); scores[var]=mdl.predict(X[var]); print(f"fold{f} [{time.time()-t0:.0f}s]")
print(f"\nHONEST RERANKER CV MRR@10 = {mrr(scores,np.arange(len(grp))):.4f}")
print("importance:",dict(zip(['cf','cfrank','gcond','cooc','gcmem','idf','globlog','nsib','posdiff','p','m','cfscore'],mdl.feature_importance())))
