"""Reference: realistic deterministic decoder (prefix-map + correction + conditional hidden).
NOT the submitted predictor (rules require a fine-tuned model) -- this just measures the
score a near-perfect decoder achieves, as a target for the T5."""
import sys, collections, re
import numpy as np, pandas as pd
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from aug import (SLOTS, parse_train_row, make_example, parse_seq, PREFIX, CORR_DESC)
W={"prep":2.20,"activation":0.85,"order":0.60,"control":3.00,"quench":4.00,"workup":0.25};WSUM=sum(W.values())
DS=Path(__file__).resolve().parent.parent/"dataset"
train=pd.read_csv(DS/"train.csv")
recs=[parse_train_row(r) for _,r in train.iterrows()]
def row_score(p,t): return sum(W[s]*(p.get(s)==t.get(s)) for s in SLOTS)/WSUM

def corr_slot_prefix(cn):
    cn=str(cn); slot=None
    for d,s in CORR_DESC.items():
        if d in cn: slot=s; break
    pf=PREFIX.findall(cn); return slot,(pf[-1] if pf else None)

def build_maps(tr):
    pfx2sv=collections.defaultdict(collections.Counter)  # prefix-> Counter((slot,val))
    fam=collections.defaultdict(lambda:collections.defaultdict(collections.Counter))
    # conditional: (fam, src_slot, src_val) -> Counter over tgt slot vals
    cond=collections.defaultdict(lambda:collections.defaultdict(collections.Counter))
    for r in tr:
        for s in SLOTS: fam[r["family"]][s][r["truth"][s]]+=1
        # prefix mapping from note (non-corrected) + correction
        for s,pf in r["note_pfx"].items():
            if s==r["cslot"]: continue
            pfx2sv[pf][(s,r["truth"][s])]+=1
        if r["cslot"] and r["cpfx"]: pfx2sv[r["cpfx"]][(r["cslot"],r["truth"][r["cslot"]])]+=1
        for a in SLOTS:
            for b in SLOTS:
                if a!=b: cond[(r["family"],a,r["truth"][a])][b][r["truth"][b]]+=1
    glob={s:collections.Counter([r["truth"][s] for r in tr]).most_common(1)[0][0] for s in SLOTS}
    return pfx2sv,fam,cond,glob

def decode_row(inp, pfx2sv, fam_tbl, cond, glob, family):
    # collect note prefixes (position-free) with candidate (slot,val)
    note=inp.split("\n")[1] if "\n" in inp else inp
    note_pfx=PREFIX.findall(note)
    # correction
    corr_line=[l for l in inp.split("\n") if any(d in l for d in CORR_DESC)]
    cs,cp=corr_slot_prefix(corr_line[0]) if corr_line else (None,None)
    # assign note prefixes to slots: greedy by descending top-count, distinct slots
    cands=[]
    for pf in note_pfx:
        for (slot,val),c in pfx2sv.get(pf,{}).items():
            cands.append((c,pf,slot,val))
    cands.sort(reverse=True)
    pred={}; used_pfx=set()
    for c,pf,slot,val in cands:
        if pf in used_pfx or slot in pred: continue
        pred[slot]=val; used_pfx.add(pf)
    # correction overrides
    if cs and cp:
        sv=pfx2sv.get(cp);
        if sv:
            # value for this corrected slot
            best=None;bc=-1
            for (slot,val),c in sv.items():
                if slot==cs and c>bc: best=val;bc=c
            if best is None: best=sv.most_common(1)[0][0][1]
            pred[cs]=best
    visible=dict(pred)
    # hidden slots via conditional inference (family + best visible link), fallback family-mode
    for s in SLOTS:
        if s in pred: continue
        scores=collections.Counter()
        for a,av in visible.items():
            ct=cond.get((family,a,av),{}).get(s)
            if ct:
                for val,n in ct.items(): scores[val]+=n*W[a]  # weight by source importance
        if scores: pred[s]=scores.most_common(1)[0][0]
        else:
            c=fam_tbl.get(family,{}).get(s); pred[s]=c.most_common(1)[0][0] if c else glob[s]
    return pred

from sklearn.model_selection import StratifiedKFold
fams=np.array([r["family"] for r in recs])
skf=StratifiedKFold(5,shuffle=True,random_state=1)
valrng=np.random.default_rng(999)
scores=[]; perslot=collections.defaultdict(list)
for tri,vai in skf.split(recs,fams):
    tr=[recs[i] for i in tri]; va=[recs[i] for i in vai]
    pfx2sv,fam_tbl,cond,glob=build_maps(tr)
    for r in va:
        for _ in range(3):
            inp,_=make_example(r,valrng,n_show=3)
            pred=decode_row(inp,pfx2sv,fam_tbl,cond,glob,r["family"])
            scores.append(row_score(pred,r["truth"]))
            for s in SLOTS: perslot[s].append(pred.get(s)==r["truth"][s])
print(f"DETERMINISTIC decoder CV weighted score: {np.mean(scores):.4f} (n={len(scores)})")
for s in SLOTS: print(f"   {s:10s} w={W[s]:<4} acc={np.mean(perslot[s]):.3f}")
