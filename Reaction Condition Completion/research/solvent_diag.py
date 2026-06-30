"""Diagnose solvent prediction: decode strategies on MLP OOF + kNN similarity baseline."""
import sys, numpy as np, pandas as pd
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from metric import set_f1_row
DS=Path(__file__).resolve().parent.parent/"dataset"
RES=Path(__file__).resolve().parent
train=pd.read_csv(DS/"train.csv"); vocab=pd.read_csv(DS/"solvent_vocabulary.csv")
SOLV=[s for s in vocab["solvent_label"] if s!="NONE"]; NS=len(SOLV)
def parse_solv(s):
    s=str(s); return [] if s in ("nan","NONE","") else s.split("|")
truth=[parse_solv(s) for s in train["solvent_labels"]]
d=np.load(RES/"oof_mlp.npz"); P=d["oof_solv"]   # (N,NS)
print("solvent prob matrix",P.shape)

def score(pred): return np.mean([set_f1_row(pred[i],truth[i]) for i in range(len(truth))])

# strategy A: threshold
for thr in [0.15,0.2,0.25,0.3,0.4,0.5]:
    pred=[[SOLV[j] for j in range(NS) if P[i,j]>thr] for i in range(len(P))]
    print(f"thr={thr}: setF1={score(pred):.4f}  avg|set|={np.mean([len(p) for p in pred]):.2f}")

# strategy B: top-1 always
pred=[[SOLV[P[i].argmax()]] for i in range(len(P))]
print(f"top1 always: setF1={score(pred):.4f}")

# strategy C: top-1 unless max<thr -> NONE
for thr in [0.1,0.2,0.3,0.4]:
    pred=[([SOLV[P[i].argmax()]] if P[i].max()>thr else []) for i in range(len(P))]
    print(f"top1 if max>{thr} else NONE: setF1={score(pred):.4f}")

# strategy D: top-1 + add 2nd if close
for gap in [0.2,0.3,0.5]:
  for nthr in [0.1,0.2]:
    pred=[]
    for i in range(len(P)):
        order=P[i].argsort()[::-1]
        if P[i,order[0]]<nthr: pred.append([]); continue
        s=[SOLV[order[0]]]
        if P[i,order[1]]>P[i,order[0]]*gap and P[i,order[1]]>0.25: s.append(SOLV[order[1]])
        pred.append(s)
    print(f"top1+2nd(gap>{gap},>0.25) noneif<{nthr}: setF1={score(pred):.4f}")

# fraction of truth that is NONE / singleton / multi
card=np.array([len(t) for t in truth])
print(f"\ntruth NONE={np.mean(card==0):.3f} singleton={np.mean(card==1):.3f} multi={np.mean(card>=2):.3f}")
# how often is the true (singleton) solvent the model's argmax?
single=[i for i in range(len(truth)) if len(truth[i])==1]
hit=np.mean([truth[i][0]==SOLV[P[i].argmax()] for i in single])
print(f"singleton top-1 accuracy: {hit:.4f} (n={len(single)})")
top3=np.mean([truth[i][0] in [SOLV[j] for j in P[i].argsort()[::-1][:3]] for i in single])
print(f"singleton top-3 accuracy: {top3:.4f}")
top5=np.mean([truth[i][0] in [SOLV[j] for j in P[i].argsort()[::-1][:5]] for i in single])
print(f"singleton top-5 accuracy: {top5:.4f}")
