"""Collaborative-filtering ranker: similarity-weighted sibling voting (+ rank-position term).
Faithful group-CV (one mask/contig, no leak). MRR@10."""
import sys, time, numpy as np, collections
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from common import load, genome_key, group_folds, VOCAB
tr,te=load()
keys=[genome_key(g) for g in tr["gc"]]
folds=group_folds(keys,n=5,seed=0)
KS=[np.array(r) for r in tr["kseq"]]
by=collections.defaultdict(list)
for i,k in enumerate(keys): by[k].append(i)
MASKPOS=np.array(range(4,16))

def eval_cf(power=2.0, use_pos=False, pos_delta=2, pos_w=0.0, rep_seed=100):
    rng=np.random.default_rng(rep_seed)
    rr=[]
    for gk,idxs in by.items():
        # token sets + position maps per sibling
        toks=[set(int(x) for x in KS[j]) for j in idxs]
        posof=[{int(KS[j][q]):q for q in range(16)} for j in idxs]
        # assign one mask per contig
        mp={j:int(rng.choice(MASKPOS)) for j in idxs}
        # visible sets (exclude mask) for similarity
        vis=[set(int(KS[j][q]) for q in range(16) if q!=mp[j]) for j in idxs]
        for a in range(len(idxs)):
            i=idxs[a]; p=mp[i]; g=int(KS[i][p]); V=vis[a]
            score=np.zeros(VOCAB+1)
            for b in range(len(idxs)):
                if b==a: continue
                ov=len(V & toks[b])
                if ov==0: continue
                w=ov**power
                for t in toks[b]:
                    score[t]+=w
                if use_pos:
                    for t,q in posof[b].items():
                        if abs(q-p)<=pos_delta: score[t]+=pos_w*w
            for v in V: score[v]=-1e9
            score[0]=-1e9
            top=np.argpartition(-score,10)[:10]; top=top[np.argsort(-score[top])]
            rank=None
            for r,t in enumerate(top):
                if t==g: rank=r+1; break
            rr.append(1.0/rank if rank else 0.0)
    return np.mean(rr)

t0=time.time()
for power in [0.5,1,2,3,4]:
    print(f"power={power}: MRR@10={eval_cf(power=power):.4f} [{time.time()-t0:.0f}s]")
print("--- with position term ---")
for pw in [0.5,1,2]:
    print(f"pos_w={pw} delta2: MRR@10={eval_cf(power=2,use_pos=True,pos_delta=2,pos_w=pw):.4f} [{time.time()-t0:.0f}s]")
