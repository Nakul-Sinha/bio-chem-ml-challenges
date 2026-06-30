"""Estimate achievable score: oracle-decode visible slots + conditional inference of hidden.
Mirrors test: show 3 note slots + 1 correction slot, hide the rest. CV over train."""
import pandas as pd, numpy as np, re, collections, json
from pathlib import Path
rng=np.random.default_rng(0)
DS = Path(__file__).resolve().parent.parent / "dataset"
train = pd.read_csv(DS / "train.csv")
SLOTS = ["prep","activation","order","control","quench","workup"]
W = {"prep":2.20,"activation":0.85,"order":0.60,"control":3.00,"quench":4.00,"workup":0.25}; WSUM=sum(W.values())
def parse_seq(s):
    d={}
    for p in str(s).split(";"):
        if "=" in p: k,v=p.split("=",1); d[k.strip()]=v.strip()
    return d
def get_family(p):
    f=str(p).split("\n")[0].lower()
    for fam in ["imine reduction","resin exchange","cross coupling","carbonate closure",
                "salt metathesis","benzylic oxidation","acyl transfer","photoredox capture"]:
        if fam in f: return fam
    return "?"
train["family"]=train["prompt"].apply(get_family)
P=pd.DataFrame(list(train["repaired_sequence"].apply(parse_seq))); P["family"]=train["family"].values

def row_score(pred,true):
    return sum(W[s]*(pred[s]==true[s]) for s in SLOTS)/WSUM

def build_cond(trainP):
    """conditional tables: for each target slot, dict keyed by (family, tuple(sorted visible (slot,val))) -> mode.
       We approximate by using ALL visible slots as conditioning. Backoff chain."""
    # We'll store full joint rows for kNN-style lookup; plus family-only mode.
    fam_mode={}
    for sl in SLOTS:
        fam_mode[sl]=trainP.groupby("family")[sl].agg(lambda s:s.value_counts().index[0]).to_dict()
    glob_mode={sl:trainP[sl].value_counts().index[0] for sl in SLOTS}
    return trainP, fam_mode, glob_mode

def predict_hidden(target, fam, visible, trainP, fam_mode, glob_mode):
    """visible: dict slot->val (the known slots). Predict target by conditional mode given family+visible."""
    sub=trainP[trainP["family"]==fam]
    # filter by visible matches, progressively relaxing (drop least informative if empty)
    vis_items=list(visible.items())
    # try full match first, then drop slots one at a time (keep most: prefer keeping prep/control/quench links)
    order_keep=sorted(vis_items, key=lambda kv: -W[kv[0]])  # keep high-weight visible slots longest
    for k in range(len(order_keep),-1,-1):
        cond=order_keep[:k] if k>0 else []
        m=sub
        for s,v in cond: m=m[m[s]==v]
        if len(m)>=3:
            return m[target].value_counts().index[0]
    return fam_mode[target].get(fam, glob_mode[target])

# CV
from sklearn.model_selection import StratifiedKFold
strat=(train["family"]).values
skf=StratifiedKFold(5,shuffle=True,random_state=1)
scores=[]; baseline_scores=[]
n_hidden_dist=collections.Counter()
for tri,vai in skf.split(train,strat):
    trP=P.iloc[tri].reset_index(drop=True); vaP=P.iloc[vai].reset_index(drop=True)
    trainP,fam_mode,glob_mode=build_cond(trP)
    for _,row in vaP.iterrows():
        true={s:row[s] for s in SLOTS}; fam=row["family"]
        # simulate test: choose corrected slot (random), then choose 3 note slots (random of 6)
        cslot=rng.choice(SLOTS)
        note_slots=set(rng.choice(SLOTS,size=3,replace=False))
        visible_slots=note_slots|{cslot}
        n_hidden_dist[6-len(visible_slots)]+=1
        visible={s:true[s] for s in visible_slots}  # oracle-decoded (correct)
        pred=dict(visible)
        for s in SLOTS:
            if s not in pred:
                pred[s]=predict_hidden(s,fam,visible,trainP,fam_mode,glob_mode)
        scores.append(row_score(pred,true))
        # baseline: family mode for hidden
        predb=dict(visible)
        for s in SLOTS:
            if s not in predb: predb[s]=fam_mode[s].get(fam,glob_mode[s])
        baseline_scores.append(row_score(predb,true))
print("hidden-slot count distribution:",dict(n_hidden_dist))
print(f"CEILING (oracle-decode + conditional infer): {np.mean(scores):.4f}")
print(f"  (oracle-decode + family-mode infer):      {np.mean(baseline_scores):.4f}")

# Also: what if we decode visible PERFECTLY and predict hidden as family-mode -- per-slot acc
# And the pure 'all family-mode' (no decoding) baseline
allmode=[]
for _,row in P.iterrows():
    true={s:row[s] for s in SLOTS}
    pred={s:P[P["family"]==row["family"]][s].value_counts().index[0] for s in SLOTS}
    allmode.append(row_score(pred,true))
print(f"  pure family-mode (no decode at all):       {np.mean(allmode):.4f}")
