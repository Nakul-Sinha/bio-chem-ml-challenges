"""Diagnose signal: per-target active AUC (grouped OOF), bin accuracy, random-vs-grouped gap."""
import numpy as np, pandas as pd, srlib as L
from sklearn.metrics import roc_auc_score
D="../dataset/"; train=pd.read_csv(D+"train.csv"); N=len(train)
Y=np.load("Y.npy")
oof_g=np.load("oof_gbdt_grouped.npy")
oof_l=np.load("oof_lr_grouped.npy")

print("Per-target ACTIVE AUC (grouped OOF):  gbdt / lr")
pact_g=1-oof_g[:,:,0]; pact_l=1-oof_l[:,:,0]
active=(Y>0).astype(int)
aucs_g=[];aucs_l=[]
for t in range(16):
    a=roc_auc_score(active[:,t],pact_g[:,t]); b=roc_auc_score(active[:,t],pact_l[:,t])
    aucs_g.append(a);aucs_l.append(b)
    print(f"  T{t:02d} active_rate={active[:,t].mean():.2f}  gbdt_auc={a:.3f}  lr_auc={b:.3f}")
print(f"  MEAN active AUC: gbdt={np.mean(aucs_g):.3f}  lr={np.mean(aucs_l):.3f}")

print("\nBin accuracy GIVEN truly active (gbdt argmax of B1/B2/B3):")
for t in range(16):
    mask=Y[:,t]>0
    if mask.sum()==0: continue
    pred_bin=oof_g[mask,t,1:].argmax(1)+1
    acc=(pred_bin==Y[mask,t]).mean()
    # modal-bin baseline
    modal=np.bincount(Y[mask,t]).argmax()
    macc=(Y[mask,t]==modal).mean()
    tag="(B2)" if t in L.B2_TARGETS else ""
    print(f"  T{t:02d}{tag}: n_active={mask.sum()} bin_acc={acc:.3f}  modal_bin_acc={macc:.3f}  modal={modal}")

# how sharp is P(active)? distribution
print("\nP(active) distribution gbdt: p10/p50/p90 =",
      np.round(np.percentile(pact_g,[10,50,90]),3))
print("mean P(active) on truly-active vs inactive:",
      round(pact_g[active==1].mean(),3), round(pact_g[active==0].mean(),3))
print("DONE")
