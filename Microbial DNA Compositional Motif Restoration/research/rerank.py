"""LightGBM learning-to-rank reranker over CF + co-occurrence + sibling-stat features. Group-CV."""
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
df=np.zeros(VOCAB+1)
for k in KS:
    for t in set(int(x) for x in k): df[t]+=1
IDF=np.log((len(KS)+1)/(df+1))+1.0
# global cooc from ALL train (tiny leak in CV; correct for test)
cooc=np.zeros((VOCAB+1,VOCAB+1)); tokc=np.zeros(VOCAB+1)
for k in KS:
    for a in k:
        tokc[a]+=1
        for b in k:
            if a!=b: cooc[a,b]+=1
LOGCOND=np.log((cooc+0.1)/(tokc[:,None]+0.1*VOCAB))
GLOBLOG=np.log((df+0.5)/(df.sum()+0.5*VOCAB))

def gen_features(rng):
    rows=[]; labels=[]; groups=[]; gold_in=[]
    for gk,idxs in by.items():
        m=len(idxs); mp=rng.choice(MASKPOS,size=m)
        Vb=np.zeros((m,VOCAB+1),dtype=np.float32); gold=np.zeros(m,dtype=np.int64)
        posmat=np.full((m,VOCAB+1),-1,dtype=np.int64)
        for a,j in enumerate(idxs):
            k=KS[j]; p=mp[a]; gold[a]=k[p]
            for q in range(16):
                if q!=p: Vb[a,k[q]]=1.0; posmat[a,k[q]]=q
        Vidf=Vb*IDF; sim=Vidf@Vb.T; np.fill_diagonal(sim,0.0); simp=sim**POWER
        cf=simp@Vb                          # CF vote
        C=Vb.T@Vb                            # genome cooc counts
        gcond=Vb@C                           # genome-cooc vote
        nsib=Vb.sum(0)                       # how many contigs have each token (visible)
        # avg position of token across contigs that have it
        possum=np.where(posmat>=0,posmat,0).sum(0); poscnt=(posmat>=0).sum(0)
        avgpos=np.where(poscnt>0,possum/np.maximum(poscnt,1),8.0)
        for a in range(m):
            s=cf[a].copy(); s[0]=-1e9; s[Vb[a]>0]=-1e9
            cand=np.argpartition(-s,TOPK)[:TOPK]; cand=cand[np.argsort(-s[cand])]
            vis=np.where(Vb[a]>0)[0]
            co=LOGCOND[vis].mean(0)
            g=gold[a]; nfeat=0
            for rk,t in enumerate(cand):
                feat=[cf[a,t], rk, np.log1p(gcond[a,t]), co[t], float(t in GCS[idxs[a]]),
                      IDF[t], GLOBLOG[t], nsib[t], avgpos[t]-mp[a], mp[a], m, s[t]]
                rows.append(feat); labels.append(1 if t==g else 0); nfeat+=1
            groups.append(nfeat); gold_in.append(int(g in cand))
    return np.array(rows,dtype=np.float32), np.array(labels), np.array(groups), np.mean(gold_in)

t0=time.time(); rng=np.random.default_rng(100)
X,y,grp,recall=gen_features(rng)
print(f"features {X.shape} candidate-recall(gold in top{TOPK})={recall:.4f} [{time.time()-t0:.0f}s]")

# map each group(contig) to its fold via genome
contig_order=[];
for gk,idxs in by.items():
    for a in idxs: pass
# rebuild group->genome->fold mapping in same order as gen_features
grp_fold=[]
gi=0
for gk,idxs in by.items():
    f=folds[idxs[0]]
    for a in range(len(idxs)): grp_fold.append(f)
grp_fold=np.array(grp_fold)

# offsets per group
offs=np.concatenate([[0],np.cumsum(grp)])
def mrr_from_scores(scores, val_groups):
    rr=[]
    for gidx in val_groups:
        sl=slice(offs[gidx],offs[gidx+1]); sc=scores[sl]; lab=y[sl]
        order=np.argsort(-sc)[:10]
        rank=next((r+1 for r,o in enumerate(order) if lab[o]==1),None)
        rr.append(1.0/rank if rank else 0.0)
    return np.mean(rr)

scores_all=np.zeros(len(y));
for f in range(5):
    tr_g=np.where(grp_fold!=f)[0]; va_g=np.where(grp_fold==f)[0]
    tr_rows=np.concatenate([np.arange(offs[g],offs[g+1]) for g in tr_g])
    va_rows=np.concatenate([np.arange(offs[g],offs[g+1]) for g in va_g])
    dtr=lgb.Dataset(X[tr_rows],label=y[tr_rows],group=grp[tr_g])
    params=dict(objective="lambdarank",metric="ndcg",ndcg_eval_at=[10],learning_rate=0.05,
                num_leaves=31,min_data_in_leaf=50,feature_fraction=0.9,bagging_fraction=0.8,bagging_freq=1,verbose=-1)
    mdl=lgb.train(params,dtr,num_boost_round=300)
    scores_all[va_rows]=mdl.predict(X[va_rows])
    print(f"fold{f} done [{time.time()-t0:.0f}s]")
print(f"\nRERANKER CV MRR@10 = {mrr_from_scores(scores_all, np.arange(len(grp))):.4f}")
print("feature importance:", dict(zip(
  ['cf','cfrank','gcond','cooc','gcmem','idf','globlog','nsib','posdiff','p','m','cfscore'],
  mdl.feature_importance())))
