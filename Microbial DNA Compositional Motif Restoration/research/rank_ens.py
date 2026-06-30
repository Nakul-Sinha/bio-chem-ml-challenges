"""Rank-fusion ensemble: z-scored blend of disc-CF + gcond + cooc. Faithful, test-calibrated."""
import sys, numpy as np, collections
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from common import load, genome_key, VOCAB
tr,te=load()
keys=[genome_key(g) for g in tr["gc"]]; KS=[np.array(r,dtype=np.int64) for r in tr["kseq"]]
by=collections.defaultdict(list)
for i,k in enumerate(keys): by[k].append(i)
MASKPOS=np.array(range(4,16))
TOT_COOC=np.zeros((VOCAB+1,VOCAB+1)); TOT_TOK=np.zeros(VOCAB+1); TOT_DF=np.zeros(VOCAB+1)
for k in KS:
    for t in set(int(x) for x in k): TOT_DF[t]+=1
    for a in k:
        TOT_TOK[a]+=1
        for b in k:
            if a!=b: TOT_COOC[a,b]+=1
NDOC=len(KS)
buckets=[(1,3),(4,8),(9,20),(21,60),(61,150),(151,10**9)]
tek=[genome_key(g) for g in te["gc"]]; tec=collections.Counter(tek); test_w=collections.Counter()
for k in tek:
    sz=tec[k]
    for lo,hi in buckets:
        if lo<=sz<=hi: test_w[(lo,hi)]+=1; break
TOT=sum(test_w.values())

def zrow(x):
    mu=x.mean(1,keepdims=True); sd=x.std(1,keepdims=True)+1e-9; return (x-mu)/sd

def run(weights, power=3, gamma=0.5, rng_seed=100):
    rng=np.random.default_rng(rng_seed); recs=[]
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
        LOGCOND=np.log((cooc+0.1)/(tok[:,None]+0.1*VOCAB)); IDFo=np.log((NDOC-m+1)/(dfo+1))+1.0
        mp=rng.choice(MASKPOS,size=m); Vb=np.zeros((m,VOCAB+1),dtype=np.float32); gold=np.zeros(m,dtype=np.int64)
        for a,j in enumerate(idxs):
            k=KS[j]; p=mp[a]; gold[a]=k[p]
            for q in range(16):
                if q!=p: Vb[a,k[q]]=1.0
        Vidf=Vb*IDFo; sim=Vidf@Vb.T; np.fill_diagonal(sim,0.0); simp=sim**power
        cf=simp@Vb; sibfrac=Vb.sum(0)/m; disc=cf/((sibfrac+0.05)**gamma)
        C=Vb.T@Vb; gcond=Vb@C
        co=np.zeros((m,VOCAB+1),dtype=np.float32)
        for a in range(m): co[a]=LOGCOND[np.where(Vb[a]>0)[0]].mean(0)
        comps=[zrow(np.log1p(disc)[:,1:]), zrow(np.log1p(gcond)[:,1:]), zrow(co[:,1:])]
        S=sum(w*c for w,c in zip(weights,comps))
        Vm=Vb[:,1:]>0; S[Vm]=-1e9
        for a in range(m):
            s=S[a]; g=gold[a]-1; top=np.argpartition(-s,10)[:10]; top=top[np.argsort(-s[top])]
            rank=next((r+1 for r,t in enumerate(top) if t==g),None)
            recs.append((m,1.0/rank if rank else 0.0))
    recs=np.array(recs); raw=recs[:,1].mean(); bm={}
    for lo,hi in buckets:
        mask=(recs[:,0]>=lo)&(recs[:,0]<=hi)
        if mask.sum(): bm[(lo,hi)]=recs[mask,1].mean()
    cal=sum(bm.get(b,0)*test_w.get(b,0) for b in buckets)/TOT
    return raw,cal

print("disc-CF only:", [round(x,4) for x in run((1,0,0))])
print("cf+gcond:", [round(x,4) for x in run((1,1,0))])
print("cf+gcond+cooc:", [round(x,4) for x in run((1,1,0.5))])
print("cf+0.5gcond:", [round(x,4) for x in run((1,0.5,0))])
print("cf+0.5gcond+0.3cooc:", [round(x,4) for x in run((1,0.5,0.3))])
print("0.7cf+0.5gcond+0.3cooc:", [round(x,4) for x in run((0.7,0.5,0.3))])
