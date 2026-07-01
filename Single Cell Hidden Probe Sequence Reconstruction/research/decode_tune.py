"""Show the decoding lever: threshold-tune the LR OOF probs on the weighted metric."""
import numpy as np, pandas as pd, srlib as L
D="../dataset/"; train=pd.read_csv(D+"train.csv")
oof=np.load("oof_lr_grouped.npy"); Y=np.load("Y.npy"); N=len(train)
gid,gkeys=L.make_groups(train)
gsize=pd.Series(gkeys).value_counts()
small=set(gsize[gsize<=gsize.quantile(0.35)].index)
rare_tokens,_=L.rare_token_set(train,160)
flags=L.subset_flags(train,np.arange(N),rare_tokens,small,gkeys)
true_seqs=[L.target_tokens(s) for s in train['target_sequence']]

def score(pred_bins):
    ps=[L.bins_to_seq(pred_bins[i]) for i in range(N)]
    return L.weighted_score(ps,true_seqs,flags)

def decode_global(oof,tau):
    pact=1-oof[:,:,0]
    binc=oof[:,:,1:].argmax(2)+1
    return np.where(pact>=tau,binc,0)

print("GLOBAL THRESHOLD SWEEP (LR grouped OOF)")
best=(-1,None)
for tau in np.arange(0.15,0.55,0.025):
    r=score(decode_global(oof,tau))
    if r['final']>best[0]: best=(r['final'],tau)
    print(f"  tau={tau:.3f} FINAL={r['final']:.4f} all={r['all']:.4f} sh={r['shifted']:.4f} dm={r['damaged']:.4f} ra={r['rare']:.4f}")
print(f"  >> best global tau={best[1]:.3f} FINAL={best[0]:.4f}")

# per-target threshold (greedy coordinate ascent on OOF)
print("PER-TARGET THRESHOLD (greedy)")
tau=np.full(16,best[1])
def decode_pt(oof,tau):
    pact=1-oof[:,:,0]; binc=oof[:,:,1:].argmax(2)+1
    return np.where(pact>=tau[None,:],binc,0)
cur=score(decode_pt(oof,tau))['final']
for _ in range(3):
    for t in range(16):
        bt,bv=tau[t],cur
        for cand in np.arange(0.10,0.60,0.02):
            tau[t]=cand; v=score(decode_pt(oof,tau))['final']
            if v>bv: bv,bt=v,cand
        tau[t]=bt; cur=bv
r=score(decode_pt(oof,tau))
print(f"  per-target FINAL={r['final']:.4f} all={r['all']:.4f} sh={r['shifted']:.4f} dm={r['damaged']:.4f} ra={r['rare']:.4f}")
print("  tau:",np.round(tau,2))
print("DONE")
