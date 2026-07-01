"""Oracle ceiling: if every SHOWN slot + the correction slot were decoded perfectly and
HIDDEN slots got the family mode, what weighted CV results? Bounds the achievable score.
Also reports per-slot: P(shown), hidden-only family-mode accuracy (the irreducible floor).
"""
import collections, numpy as np, pandas as pd
from pathlib import Path
from aug import SLOTS, parse_train_row, make_example

W={"prep":2.20,"activation":0.85,"order":0.60,"control":3.00,"quench":4.00,"workup":0.25}; WSUM=sum(W.values())
DS=Path(__file__).resolve().parent.parent/"dataset"; VAL_FRAC=0.15; VAL_REPS=3
train=pd.read_csv(DS/"train.csv"); recs=[parse_train_row(r) for _,r in train.iterrows()]
fams=np.array([r["family"] for r in recs]); rng=np.random.default_rng(42); val_idx=[]
for f in sorted(set(fams)):
    idx=np.where(fams==f)[0]; rng.shuffle(idx); val_idx+=list(idx[:int(len(idx)*VAL_FRAC)])
val_idx=set(int(x) for x in val_idx); tr=[r for i,r in enumerate(recs) if i not in val_idx]; va=[r for i,r in enumerate(recs) if i in val_idx]
famtbl=collections.defaultdict(lambda:collections.defaultdict(collections.Counter))
for r in tr:
    for s in SLOTS: famtbl[r["family"]][s][r["truth"][s]]+=1
glob={s:collections.Counter([r["truth"][s] for r in tr]).most_common(1)[0][0] for s in SLOTS}
def fmode(fam,s):
    c=famtbl.get(fam,{}).get(s); return c.most_common(1)[0][0] if c else glob[s]

valrng=np.random.default_rng(12345); examples=[]
for rep in range(VAL_REPS):
    for r in va:
        inp,tgt,show,cslot=make_example(r,valrng,n_show=3,return_show=True)
        examples.append((r,show,cslot))

n=len(examples)
shown_cnt={s:0 for s in SLOTS}; known_cnt={s:0 for s in SLOTS}
oracle_acc={s:0 for s in SLOTS}; fam_hidden_acc={s:0 for s in SLOTS}; fam_hidden_n={s:0 for s in SLOTS}
woracle=0; wfamall=0
for r,show,cslot in examples:
    known=set(show);
    if cslot: known.add(cslot)
    pred={}; predfam={}
    for s in SLOTS:
        if s in show: shown_cnt[s]+=1
        if s in known: known_cnt[s]+=1
        pred[s]=r["truth"][s] if s in known else fmode(r["family"],s)   # oracle: known perfect, hidden=fammode
        predfam[s]=fmode(r["family"],s)
        if s not in known:
            fam_hidden_n[s]+=1
            if predfam[s]==r["truth"][s]: fam_hidden_acc[s]+=1
        if pred[s]==r["truth"][s]: oracle_acc[s]+=1
    woracle+=sum(W[s]*(pred[s]==r["truth"][s]) for s in SLOTS)/WSUM
    wfamall+=sum(W[s]*(predfam[s]==r["truth"][s]) for s in SLOTS)/WSUM

print(f"ORACLE (shown+corr perfect, hidden=fammode) CV={woracle/n:.4f}   (seq2seq 0.726)")
print(f"family-mode-all CV={wfamall/n:.4f}\n")
print(f"{'slot':10s} {'P(shown)':>8s} {'P(known)':>8s} {'oracleAcc':>9s} {'hidFamAcc':>9s}  seq2seq")
seq={'prep':.747,'activation':.820,'order':.867,'control':.750,'quench':.675,'workup':.403}
for s in SLOTS:
    hfa=fam_hidden_acc[s]/max(1,fam_hidden_n[s])
    print(f"{s:10s} {shown_cnt[s]/n:8.3f} {known_cnt[s]/n:8.3f} {oracle_acc[s]/n:9.3f} {hfa:9.3f}  {seq[s]:.3f}")
