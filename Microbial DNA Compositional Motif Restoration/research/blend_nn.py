"""Blend PMI(genome-pool) + co-occurrence + NN masked-LM. Full group-CV, tune weights, report MRR@10."""
import sys, time, numpy as np, collections
from pathlib import Path
import torch, torch.nn.functional as F
sys.path.insert(0,str(Path(__file__).resolve().parent))
from common import load, genome_key, mrr10, group_folds, VOCAB
from nn_mlm import MLM, train_model, predict, MASKPOS, K, G, keys, folds

tr,te=load()
KS=[r for r in tr["kseq"]]; GC=[r for r in tr["gc"]]
N=len(KS)

def build_global(idxs):
    cooc=np.zeros((VOCAB+1,VOCAB+1)); tokcnt=np.zeros(VOCAB+1); glob=np.zeros(VOCAB+1)
    for i in idxs:
        k=KS[i]
        for t in k: glob[t]+=1
        for a in k:
            tokcnt[a]+=1
            for b in k:
                if a!=b: cooc[a,b]+=1
    logcond=np.log((cooc+0.1)/(tokcnt[:,None]+0.1*VOCAB))
    logglob=np.log((glob+0.5)/(glob.sum()+0.5*VOCAB))
    return logcond,logglob
def gtotals(idxs):
    tot=collections.defaultdict(lambda: np.zeros(VOCAB+1))
    for i in idxs:
        for t in KS[i]: tot[keys[i]][t]+=1
    return tot

t0=time.time()
all_pmi=[]; all_cond=[]; all_nn=[]; all_gold=[]; all_vis=[]
for f in range(5):
    tri=np.where(folds!=f)[0]; vai=np.where(folds==f)[0]
    logcond,logglob=build_global(tri); tot=gtotals(vai)
    net=train_model(tri,seed=0)
    nn_lg={p:predict(net,K[vai],G[vai],np.full(len(vai),p)) for p in MASKPOS}  # (nval,1024) for tokens 1..1024
    for pidx,p in enumerate(MASKPOS):
        nn_log=torch.log_softmax(torch.tensor(nn_lg[p]),dim=1).numpy()  # (nval,1024)
        for j,idx in enumerate(vai):
            k=KS[idx]; gk=keys[idx]; g=k[p]; vis=[k[i] for i in range(16) if i!=p]
            prof=tot[gk].copy(); prof[g]-=1
            logprof=np.log((prof+0.3)/(prof.sum()+0.3*VOCAB))
            pmi=(logprof-logglob)[1:]                 # drop index0 (pad) -> 1024 for tokens 1..1024
            cond=logcond[vis].mean(0)[1:]
            all_pmi.append(pmi.astype(np.float32)); all_cond.append(cond.astype(np.float32))
            all_nn.append(nn_log[j]); all_gold.append(g); all_vis.append(vis)
    print(f"fold{f} components built [{time.time()-t0:.0f}s]")
PMI=np.array(all_pmi); COND=np.array(all_cond); NN=np.array(all_nn); GOLD=np.array(all_gold)
print("rows",PMI.shape)
# visible mask
VIS=np.full((len(all_vis),1024),0.0,dtype=np.float32)
for r,vis in enumerate(all_vis):
    for v in vis:
        if 1<=v<=VOCAB: VIS[r,v-1]=-1e9

def score_mrr(wpmi,wcond,wnn):
    S=wpmi*PMI+wcond*COND+wnn*NN+VIS
    # rank gold (token g -> col g-1)
    rr=0.0
    # vectorized: rank of gold = count of cols with score > gold_score
    goldcol=GOLD-1
    gs=S[np.arange(len(S)),goldcol]
    greater=(S> gs[:,None]).sum(1)   # number strictly greater => rank-1
    rank=greater+1
    rr=np.where(rank<=10,1.0/rank,0.0).mean()
    return rr

best=(-1,None)
for wpmi in [0,0.5,1,1.5,2]:
  for wcond in [0,0.5,1,1.5,2]:
    for wnn in [0,0.5,1,1.5,2,3]:
        m=score_mrr(wpmi,wcond,wnn)
        if m>best[0]: best=(m,(wpmi,wcond,wnn))
print(f"\nBEST blend CV MRR@10={best[0]:.4f} (wpmi,wcond,wnn)={best[1]}")
print("pmi+cond only:",round(score_mrr(1,1,0),4)," nn only:",round(score_mrr(0,0,1),4))
np.savez(Path(__file__).resolve().parent/"blend_components.npz",
         PMI=PMI,COND=COND,NN=NN,GOLD=GOLD,VIS=VIS)
