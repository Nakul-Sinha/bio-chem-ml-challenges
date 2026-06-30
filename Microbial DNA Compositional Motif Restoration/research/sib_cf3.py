"""CF v3 (fully faithful: siblings use VISIBLE tokens): IDF similarity^power voting
+ rank-position weighting + global co-occurrence blend. Faithful group-CV, MRR@10."""
import sys, time, numpy as np, collections
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from common import load, genome_key, group_folds, VOCAB
tr,te=load()
keys=[genome_key(g) for g in tr["gc"]]; folds=group_folds(keys,n=5,seed=0)
KS=[np.array(r) for r in tr["kseq"]]
by=collections.defaultdict(list)
for i,k in enumerate(keys): by[k].append(i)
MASKPOS=np.array(range(4,16))
df=np.zeros(VOCAB+1)
for k in KS:
    for t in set(int(x) for x in k): df[t]+=1
IDF=np.log((len(KS)+1)/(df+1))+1.0

def build_cond(idxs):
    cooc=np.zeros((VOCAB+1,VOCAB+1)); tokcnt=np.zeros(VOCAB+1)
    for i in idxs:
        for a in KS[i]:
            tokcnt[a]+=1
            for b in KS[i]:
                if a!=b: cooc[a,b]+=1
    return np.log((cooc+0.1)/(tokcnt[:,None]+0.1*VOCAB))

def eval3(power=6.0, pos_w=0.0, tau=2.0, cooc_w=0.0, rep_seed=100):
    rng=np.random.default_rng(rep_seed); rr=[]
    logcond=None
    if cooc_w:
        # global cond from TRAIN folds only -> need per val fold; approximate: build per-fold
        pass
    for f in range(5):
        vai=set(np.where(folds==f)[0].tolist())
        tri=np.where(folds!=f)[0]
        lc=build_cond(tri) if cooc_w else None
        # genomes that belong to this val fold
        fold_genomes=[gk for gk,idxs in by.items() if idxs[0] in vai]
        for gk in fold_genomes:
            idxs=by[gk]
            mp={j:int(rng.choice(MASKPOS)) for j in idxs}
            vis=[ [int(KS[j][q]) for q in range(16) if q!=mp[j]] for j in idxs ]
            visset=[set(v) for v in vis]
            posof=[{int(KS[j][q]):q for q in range(16) if q!=mp[j]} for j in idxs]
            for a in range(len(idxs)):
                i=idxs[a]; p=mp[i]; g=int(KS[i][p]); V=visset[a]
                score=np.zeros(VOCAB+1)
                for b in range(len(idxs)):
                    if b==a: continue
                    inter=V & visset[b]
                    if not inter: continue
                    ov=sum(IDF[t] for t in inter); w=ov**power
                    if pos_w:
                        for t,q in posof[b].items():
                            score[t]+=w*(1.0+pos_w*np.exp(-abs(q-p)/tau))
                    else:
                        for t in visset[b]: score[t]+=w
                if cooc_w:
                    score[1:]+=cooc_w*lc[list(V)].mean(0)[1:]*1.0
                for v in V: score[v]=-1e9
                score[0]=-1e9
                top=np.argpartition(-score,10)[:10]; top=top[np.argsort(-score[top])]
                rank=next((r+1 for r,t in enumerate(top) if t==g),None)
                rr.append(1.0/rank if rank else 0.0)
    return np.mean(rr)

t0=time.time()
print("faithful CF power=6 (siblings visible):", round(eval3(power=6),4), f"[{time.time()-t0:.0f}s]")
for pw in [0.5,1,2,4]:
    print(f"  +pos_w={pw} tau=2:", round(eval3(power=6,pos_w=pw,tau=2),4), f"[{time.time()-t0:.0f}s]")
for pw in [2,4]:
  for tau in [1,3]:
    print(f"  +pos_w={pw} tau={tau}:", round(eval3(power=6,pos_w=pw,tau=tau),4), f"[{time.time()-t0:.0f}s]")
