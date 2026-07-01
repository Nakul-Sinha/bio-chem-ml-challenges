"""Test smarter decode strategies on saved GBDT grouped OOF.
Insight: bins should default to B1; only deviate where signal exists. Select active by threshold."""
import numpy as np, pandas as pd, srlib as L
D="../dataset/"; train=pd.read_csv(D+"train.csv"); N=len(train)
Y=np.load("Y.npy"); oof=np.load("oof_gbdt_grouped.npy")
gid,gkeys=L.make_groups(train)
gsize=pd.Series(gkeys).value_counts(); small=set(gsize[gsize<=gsize.quantile(0.35)].index)
rare_tokens,_=L.rare_token_set(train,160)
flags=L.subset_flags(train,np.arange(N),rare_tokens,small,gkeys)
true_seqs=[L.target_tokens(s) for s in train['target_sequence']]
def score(pb):
    ps=[L.bins_to_seq(pb[i]) for i in range(N)]; return L.weighted_score(ps,true_seqs,flags)
def rep(tag,pb):
    r=score(pb); print(f"  [{tag}] FINAL={r['final']:.4f} all={r['all']:.4f} sh={r['shifted']:.4f} dm={r['damaged']:.4f} ra={r['rare']:.4f}"); return r['final']

pact=1-oof[:,:,0]        # (N,16)
argbin=oof[:,:,1:].argmax(2)+1

print("A) always-B1 bins, global threshold sweep")
bestA=(-1,None)
for tau in np.arange(0.04,0.40,0.02):
    pb=np.where(pact>=tau,1,0)
    f=rep(f"B1 tau={tau:.2f}",pb) if tau in (0.10,0.16,0.20,0.24) else score(np.where(pact>=tau,1,0))['final']
    if f>bestA[0]: bestA=(f,tau)
print(f"  >> best always-B1 tau={bestA[1]:.2f} FINAL={bestA[0]:.4f}")

print("B) argmax bins, lower threshold sweep")
bestB=(-1,None)
for tau in np.arange(0.04,0.40,0.02):
    pb=np.where(pact>=tau,argbin,0)
    f=score(pb)['final']
    if f>bestB[0]: bestB=(f,tau)
print(f"  >> best argmax tau={bestB[1]:.2f} FINAL={bestB[0]:.4f}")

print("C) hybrid: B1 default, argmax only for strong-bin targets {5,6,12,13,14}")
strong=[5,6,12,13,14]
bestC=(-1,None)
for tau in np.arange(0.04,0.40,0.02):
    bins=np.ones((N,16),int)
    for t in strong: bins[:,t]=argbin[:,t]
    pb=np.where(pact>=tau,bins,0)
    f=score(pb)['final']
    if f>bestC[0]: bestC=(f,tau)
print(f"  >> best hybrid tau={bestC[1]:.2f} FINAL={bestC[0]:.4f}")

print("D) per-row top-k by P(active), always B1")
bestD=(-1,None)
order=np.argsort(-pact,axis=1)
for k in range(3,13):
    pb=np.zeros((N,16),int)
    for i in range(N):
        pb[i,order[i,:k]]=1
    f=score(pb)['final']
    if f>bestD[0]: bestD=(f,k)
print(f"  >> best top-k k={bestD[1]} FINAL={bestD[0]:.4f}")

print("E) per-target threshold (greedy), always B1")
tau=np.full(16,bestA[1])
def dec(tau):
    return np.where(pact>=tau[None,:],1,0)
cur=score(dec(tau))['final']
for _ in range(3):
    for t in range(16):
        bt,bv=tau[t],cur
        for c in np.arange(0.04,0.60,0.02):
            tau[t]=c; v=score(dec(tau))['final']
            if v>bv: bv,bt=v,c
        tau[t]=bt; cur=bv
print(f"  >> per-target B1 FINAL={cur:.4f}  tau={np.round(tau,2)}")

print("F) per-target threshold + strong-target argmax bins")
bins=np.ones((N,16),int)
for t in strong: bins[:,t]=argbin[:,t]
def dec2(tau):
    return np.where(pact>=tau[None,:],bins,0)
tau2=np.full(16,bestA[1]); cur=score(dec2(tau2))['final']
for _ in range(3):
    for t in range(16):
        bt,bv=tau2[t],cur
        for c in np.arange(0.04,0.60,0.02):
            tau2[t]=c; v=score(dec2(tau2))['final']
            if v>bv: bv,bt=v,c
        tau2[t]=bt; cur=bv
print(f"  >> per-target+strongbins FINAL={cur:.4f}")
print("DONE")
