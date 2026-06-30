"""CF ranker v2: IDF-weighted contig similarity + high power + position term. Faithful CV."""
import sys, time, numpy as np, collections
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from common import load, genome_key, group_folds, VOCAB
tr,te=load()
keys=[genome_key(g) for g in tr["gc"]]
KS=[np.array(r) for r in tr["kseq"]]
by=collections.defaultdict(list)
for i,k in enumerate(keys): by[k].append(i)
MASKPOS=np.array(range(4,16))
# global IDF over contigs
df=np.zeros(VOCAB+1)
for k in KS:
    for t in set(int(x) for x in k): df[t]+=1
IDF=np.log((len(KS)+1)/(df+1))+1.0

def eval_cf(power=4.0, use_idf=True, pos_w=0.0, pos_delta=2, cosine=False, rep_seed=100):
    rng=np.random.default_rng(rep_seed); rr=[]
    for gk,idxs in by.items():
        toks=[set(int(x) for x in KS[j]) for j in idxs]
        posof=[{int(KS[j][q]):q for q in range(16)} for j in idxs]
        mp={j:int(rng.choice(MASKPOS)) for j in idxs}
        vis=[set(int(KS[j][q]) for q in range(16) if q!=mp[j]) for j in idxs]
        # sibling token-weight vectors for cosine norm
        if cosine:
            snorm=[np.sqrt(sum((IDF[t]**2 if use_idf else 1) for t in toks[b])) for b in range(len(idxs))]
        for a in range(len(idxs)):
            i=idxs[a]; p=mp[i]; g=int(KS[i][p]); V=vis[a]
            vnorm=np.sqrt(sum((IDF[t]**2 if use_idf else 1) for t in V)) if cosine else 1.0
            score=np.zeros(VOCAB+1)
            for b in range(len(idxs)):
                if b==a: continue
                inter=V & toks[b]
                if not inter: continue
                ov=sum(IDF[t] for t in inter) if use_idf else len(inter)
                if cosine: ov=ov/(vnorm*snorm[b]+1e-9)
                w=ov**power
                for t in toks[b]: score[t]+=w
                if pos_w:
                    for t,q in posof[b].items():
                        if abs(q-p)<=pos_delta: score[t]+=pos_w*w
            for v in V: score[v]=-1e9
            score[0]=-1e9
            top=np.argpartition(-score,10)[:10]; top=top[np.argsort(-score[top])]
            rank=next((r+1 for r,t in enumerate(top) if t==g),None)
            rr.append(1.0/rank if rank else 0.0)
    return np.mean(rr)

t0=time.time()
print("=== IDF overlap, power sweep ===")
for power in [3,4,6,8,10]:
    print(f"idf power={power}: {eval_cf(power=power,use_idf=True):.4f} [{time.time()-t0:.0f}s]")
print("=== IDF cosine, power sweep ===")
for power in [3,5,8,12]:
    print(f"idf cosine power={power}: {eval_cf(power=power,use_idf=True,cosine=True):.4f} [{time.time()-t0:.0f}s]")
