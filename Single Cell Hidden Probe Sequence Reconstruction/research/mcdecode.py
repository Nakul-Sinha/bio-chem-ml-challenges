"""Expected-metric (Bayes) decode: targets independent -> MC-sample truth, greedily build the
prediction set maximizing expected row score. Compare to threshold decode on the EXACT metric."""
import numpy as np, pandas as pd, srlib as L, time
D="../dataset/"; train=pd.read_csv(D+"train.csv"); N=len(train)
Y=np.load("Y.npy"); oof=np.load("oof_gbdt_grouped.npy")   # (N,16,4)
gid,gkeys=L.make_groups(train)
gsize=pd.Series(gkeys).value_counts(); small=set(gsize[gsize<=gsize.quantile(0.35)].index)
rare_tokens,_=L.rare_token_set(train,160)
flags=L.subset_flags(train,np.arange(N),rare_tokens,small,gkeys)
true_seqs=[L.target_tokens(s) for s in train['target_sequence']]
def score_final(pb):
    ps=[L.bins_to_seq(pb[i]) for i in range(N)]; return L.weighted_score(ps,true_seqs,flags)

# modal bin per target (default bin when we choose to include a target)
modal=np.zeros(16,int)
for t in range(16):
    v=Y[:,t][Y[:,t]>0]; modal[t]=np.bincount(v).argmax()

rng=np.random.default_rng(0)
def sample_truth(pt, K):
    """pt: (16,4) probs -> (K,16) sampled bins."""
    cum=np.cumsum(pt,axis=1)                # (16,4)
    r=rng.random((K,16))
    smp=(r[:,:,None]>cum[None,:,:]).sum(2)  # count thresholds passed -> bin index 0..3
    return smp                              # (K,16)

def fixed_bins(pt):
    """chosen bin per target if included: argmax over B1..B3, but default modal for stability."""
    ab=pt[:,1:].argmax(1)+1
    return ab

def expected_score_of_pred(dbins, Tsmp):
    """dbins:(16,) pred bins; Tsmp:(K,16) truth. Return mean EXACT-ish (F1+LCS+edit_proxy)."""
    dp=(dbins>0)
    lp=dp.sum()
    match=((dbins[None,:]==Tsmp)&(dbins[None,:]>0)).sum(1)   # (K,) overlap
    lt=(Tsmp>0).sum(1)                                       # (K,)
    m=np.maximum(np.maximum(lp,lt),1)
    # handle NONE both-empty -> score 1
    f1=np.where((lp+lt)>0, 2*match/np.maximum(lp+lt,1), 1.0)
    lcs=match/m
    # edit proxy: substitutions cheap. edit = max(lp,lt) - match - min(extra_p,extra_t_wrongbin)...
    # use lower-bound-ish: edit ~ m - match (ins/del) but allow sub savings for wrong-bin same-target
    wrongbin=((dbins[None,:]>0)&(Tsmp>0)&(dbins[None,:]!=Tsmp)).sum(1)  # substitutable pairs at same target
    edit=m-match-np.minimum(wrongbin, np.minimum(lp-match, lt-match))
    editsim=1-edit/m
    both_empty=(lp==0)&(lt==0)
    s=0.5*editsim+0.3*f1+0.2*lcs
    s=np.where(both_empty,1.0, np.where((lp==0)|(lt==0), np.where((lp==0)&(lt==0),1.0, 0.0*match + s*((lp>0)&(lt>0))), s))
    return s.mean()

def mc_greedy(pt, K=64):
    Tsmp=sample_truth(pt,K)
    fb=fixed_bins(pt)
    d=np.zeros(16,int)
    cur=expected_score_of_pred(d,Tsmp)
    # candidate order by P(active) desc for efficiency
    cand=list(np.argsort(-(1-pt[:,0])))
    improved=True
    while improved:
        improved=False; best_t=-1; best_v=cur
        for t in cand:
            if d[t]>0: continue
            d[t]=fb[t]; v=expected_score_of_pred(d,Tsmp); d[t]=0
            if v>best_v+1e-6: best_v=v; best_t=t
        if best_t>=0:
            d[best_t]=fb[best_t]; cur=best_v; improved=True
    return d

t0=time.time()
pred=np.zeros((N,16),int)
for i in range(N):
    pred[i]=mc_greedy(oof[i],K=64)
print("MC-greedy decoded in %.0fs"%(time.time()-t0))
r=score_final(pred)
print(f"  [MC-greedy expected] FINAL={r['final']:.4f} all={r['all']:.4f} sh={r['shifted']:.4f} dm={r['damaged']:.4f} ra={r['rare']:.4f}")
print("  avg pred set size:", (pred>0).sum(1).mean(), " true avg:", (Y>0).sum(1).mean())
# reference: best threshold decode
pact=1-oof[:,:,0]
best=(-1,None)
for tau in np.arange(0.04,0.4,0.02):
    pb=np.where(pact>=tau,1,0); f=score_final(pb)['final']
    if f>best[0]: best=(f,tau)
print(f"  [threshold ref] best FINAL={best[0]:.4f} tau={best[1]:.2f}")
print("DONE")
