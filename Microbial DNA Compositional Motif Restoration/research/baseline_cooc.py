"""Co-occurrence + position-prior + genome-context ranker. Group-CV by genome. MRR@10."""
import sys, numpy as np, pandas as pd, collections
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from common import load, genome_key, mrr10, group_folds, VOCAB

tr,te=load()
keys=[genome_key(g) for g in tr["gc"]]
folds=group_folds(keys,n=5,seed=0)
MASKPOS=list(range(4,16))  # test masks positions 4..15

def build_stats(rows):
    pos_cnt=np.zeros((16,VOCAB+1)); cooc=np.zeros((VOCAB+1,VOCAB+1)); tokcnt=np.zeros(VOCAB+1)
    for k in rows["kseq"]:
        for pos,t in enumerate(k): pos_cnt[pos,t]+=1
        for a in k:
            tokcnt[a]+=1
            for b in k:
                if a!=b: cooc[a,b]+=1
    return pos_cnt,cooc,tokcnt

def rank_scores(pos_cnt,cooc,tokcnt,w):
    wp,wc,wg,wgc=w
    logpos=np.log((pos_cnt+0.5)/(pos_cnt.sum(1,keepdims=True)+0.5*VOCAB))
    # P(t|v) ~ cooc[v,t]/tokcnt[v]
    Pcond=(cooc+0.1)/(tokcnt[:,None]+0.1*VOCAB)
    logcond=np.log(Pcond)
    def score(kseq,gc,pos):
        vis=[t for i,t in enumerate(kseq) if i!=pos and t is not None]
        s=wp*logpos[pos].copy()
        if vis: s+=wc*logcond[vis].mean(0)
        if gc: s+=wgc*logcond[[g for g in gc]].mean(0)
        gcv=np.zeros(VOCAB+1);
        for g in set(gc): gcv[g]=1.0
        s+=wg*gcv
        s[0]=-1e9
        for v in vis: s[v]=-1e9   # masked token != visible tokens
        return s
    return score

def eval_fold(tr_rows,va_rows,w):
    pos_cnt,cooc,tokcnt=build_stats(tr_rows)
    score=rank_scores(pos_cnt,cooc,tokcnt,w)
    gold=[]; ranked=[]
    for _,r in va_rows.iterrows():
        for pos in MASKPOS:
            if pos>=len(r["kseq"]): continue
            g=r["kseq"][pos]; sc=score(r["kseq"],r["gc"],pos)
            top=np.argpartition(-sc,10)[:10]; top=top[np.argsort(-sc[top])]
            gold.append(g); ranked.append(list(top))
    return mrr10(gold,ranked)

# tune weights on fold 0 quickly
va=tr[folds==0]; trn=tr[folds!=0]
best=(-1,None)
for wp in [0.5,1.0,1.5]:
  for wc in [1.0,2.0,3.0]:
    for wg in [0,2.0,4.0]:
      for wgc in [0,0.5,1.0]:
        m=eval_fold(trn,va,(wp,wc,wg,wgc))
        if m>best[0]: best=(m,(wp,wc,wg,wgc))
print("fold0 best:",best)
# full CV with best weights
w=best[1]; scores=[]
for f in range(5):
    m=eval_fold(tr[folds!=f],tr[folds==f],w); scores.append(m); print(f"fold{f} MRR@10={m:.4f}")
print(f"\nCV MRR@10 (cooc baseline) = {np.mean(scores):.4f}  weights={w}")
