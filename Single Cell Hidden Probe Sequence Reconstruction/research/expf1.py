"""Expected-F1(=Dice=ratio) optimal decode. For each row: pick best bin-token per target
(argmax specific-token prob), rank by prob, choose top-k maximizing plug-in E[F1]
= 2*cumsum(q)/(k + expected_true_count). Adaptive per-row count. Compare vs threshold decode
on HONEST nested grouped CV, under ratio AND max. Blend = ensemble + kNN."""
import numpy as np, pandas as pd, srlib as L
from fastexact import FastScorer
from sklearn.model_selection import GroupKFold
import warnings; warnings.filterwarnings('ignore')

train=pd.read_csv('../dataset/train.csv'); N=len(train)
Y=np.load('Y.npy'); fs=FastScorer(Y); gid,gkeys=L.make_groups(train)
gsize=pd.Series(gkeys).value_counts(); small=set(gsize[gsize<=gsize.quantile(0.35)].index)
rare_tokens,_=L.rare_token_set(train,160); FL=L.subset_flags(train,np.arange(N),rare_tokens,small,gkeys)
Os=[np.load(f) for f in ['oof_gbdt_grouped.npy','oof_lr_grouped.npy','oof_mlp_grouped.npy','oof_tlin.npy','oof_tmlp.npy']]
allowed=np.zeros((16,4),bool)
for t in range(16):
    allowed[t,0]=True
    for b in [1,2,3]:
        if (Y[:,t]==b).any(): allowed[t,b]=True
def npb(o):
    o=o.copy()
    for t in range(16): o[:,t,~allowed[t]]=0
    return o/(o.sum(2,keepdims=True)+1e-9)
ens=npb(np.mean(Os,0))
# kNN blend on prob-active AND redistribute bins from ensemble
OV=np.zeros((N,80))
for i,s in enumerate(train.source_sequence):
    ov,_,_,_,_=L.parse_source(s); OV[i]=ov
lg=np.log1p(OV); Z=(lg-lg.mean(0))/(lg.std(0)+1e-6); Zn=Z/(np.linalg.norm(Z,axis=1,keepdims=True)+1e-9)
S=Zn@Zn.T; np.fill_diagonal(S,-1); ACT=(Y>0).astype(float)
folds=list(GroupKFold(5).split(np.arange(N),groups=gid))
def knn_oof(K=80,temp=0.1):
    P=np.zeros((N,16))
    for tr,va in folds:
        m=np.zeros(N,bool); m[tr]=True
        for i in va:
            sims=np.where(m,S[i],-1); idx=np.argpartition(-sims,K)[:K]
            w=np.exp(sims[idx]/temp); w/=w.sum()+1e-9; P[i]=w@ACT[idx]
    return P
Pk=knn_oof()
# blended prob-active; keep ensemble bin-conditional shape
pact_ens=1-ens[:,:,0]
pact=0.5*pact_ens+0.5*Pk
# specific-token prob q[t,b] = pact * P(bin b | active) from ensemble
bincond=ens[:,:,1:]/(ens[:,:,1:].sum(2,keepdims=True)+1e-9)   # (N,16,3)
qtok=pact[:,:,None]*bincond                                    # (N,16,3) prob of exact token
best_b=qtok.argmax(2)+1; best_q=qtok.max(2)                    # best specific token per target

def wf(rs,idx):
    s=rs[idx]
    def mn(m): m=np.asarray(m,bool)[idx]; return float(rs[idx][m].mean()) if m.any() else float(s.mean())
    return 0.45*s.mean()+0.25*mn(FL['shifted'])+0.20*mn(FL['damaged'])+0.10*mn(FL['rare'])

def decode_expf1(scale):
    exp_true=pact.sum(1)  # expected true-token count per row
    order=np.argsort(-best_q,axis=1)
    bins=np.zeros((N,16),int)
    for i in range(N):
        q=best_q[i,order[i]]; cs=np.cumsum(q)
        ks=np.arange(1,17)
        ef1=2*cs/(ks+scale*exp_true[i]+1e-9)
        k=int(np.argmax(ef1))+1
        # also allow k=0 (NONE) if even best token prob tiny
        if best_q[i].max()<0.08: k=0
        for j in range(k):
            t=order[i,j]; bins[i,t]=best_b[i,t]
    return bins
def decode_thresh(tau):
    bins=np.where(pact>=tau,1,0)
    for t in range(16): bins[:,t]=np.where(pact[:,t]>=tau,best_b[:,t],0)
    return bins

ALL=np.arange(N)
# nested tune of scale (expf1) and tau (thresh) on mean(max,ratio)
def obj(pb,idx): return 0.5*(wf(fs.rows(pb,'max'),idx)+wf(fs.rows(pb,'ratio'),idx))
def nested(kind):
    hb=np.zeros((N,16),int)
    for tr,va in folds:
        if kind=='expf1':
            best=(-1,1.0)
            for sc in np.arange(0.5,2.0,0.1):
                f=obj(decode_expf1(sc),tr)
                if f>best[0]: best=(f,sc)
            hb[va]=decode_expf1(best[1])[va]
        else:
            best=(-1,0.2)
            for tau in np.arange(0.05,0.55,0.01):
                f=obj(decode_thresh(tau),tr)
                if f>best[0]: best=(f,tau)
            hb[va]=decode_thresh(best[1])[va]
    return hb
for kind in ['thresh','expf1']:
    hb=nested(kind)
    print("%-8s nested: max=%.4f ratio=%.4f tok/row=%.2f"%(kind,wf(fs.rows(hb,'max'),ALL),wf(fs.rows(hb,'ratio'),ALL),(hb>0).sum(1).mean()))
print("ref: prior improved decode ratio~0.333")
