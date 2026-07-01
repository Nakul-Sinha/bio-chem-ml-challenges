"""Where is the headroom? Compare structured baselines to seq2seq (0.726) on the SAME val.
Values recur across rows -> test how much (family,slot), tags, and correction each contribute.
"""
import collections, numpy as np, pandas as pd
from pathlib import Path
from aug import (SLOTS, parse_train_row, make_example, PREFIX, CORR_DESC, get_family)

W={"prep":2.20,"activation":0.85,"order":0.60,"control":3.00,"quench":4.00,"workup":0.25}; WSUM=sum(W.values())
DS=Path(__file__).resolve().parent.parent/"dataset"; VAL_FRAC=0.15; VAL_REPS=3
train=pd.read_csv(DS/"train.csv"); recs=[parse_train_row(r) for _,r in train.iterrows()]
fams=np.array([r["family"] for r in recs]); rng=np.random.default_rng(42); val_idx=[]
for f in sorted(set(fams)):
    idx=np.where(fams==f)[0]; rng.shuffle(idx); val_idx+=list(idx[:int(len(idx)*VAL_FRAC)])
val_idx=set(int(x) for x in val_idx); tr=[r for i,r in enumerate(recs) if i not in val_idx]; va=[r for i,r in enumerate(recs) if i in val_idx]

# label space per slot
for s in SLOTS:
    vals=set(r["truth"][s] for r in recs)
    print(f"{s:10s} #values={len(vals)}")
# how deterministic is (family,slot)->value?
print("\n(family,slot)->value concentration (top-1 share of train):")
fs_mode={};
for s in SLOTS:
    shares=[]
    for f in sorted(set(fams)):
        c=collections.Counter(r["truth"][s] for r in tr if r["family"]==f)
        if c: fs_mode[(f,s)]=c.most_common(1)[0][0]; shares.append(c.most_common(1)[0][1]/sum(c.values()))
    print(f"  {s:10s} mean_top1={np.mean(shares):.3f}")

# maps
famtbl=collections.defaultdict(lambda:collections.defaultdict(collections.Counter))
fsp=collections.defaultdict(collections.Counter)   # (family,slot,prefix)->value
sp=collections.defaultdict(collections.Counter)    # (slot,prefix)->value
glob={}
for r in tr:
    for s in SLOTS:
        famtbl[r["family"]][s][r["truth"][s]]+=1
        if s in r["note_pfx"]:
            fsp[(r["family"],s,r["note_pfx"][s])][r["truth"][s]]+=1
            sp[(s,r["note_pfx"][s])][r["truth"][s]]+=1
    if r["cslot"] and r["cpfx"]:
        fsp[(r["family"],r["cslot"],r["cpfx"])][r["truth"][r["cslot"]]]+=1
        sp[(r["cslot"],r["cpfx"])][r["truth"][r["cslot"]]]+=1
glob={s:collections.Counter([r["truth"][s] for r in tr]).most_common(1)[0][0] for s in SLOTS}
def fmode(fam,s):
    c=famtbl.get(fam,{}).get(s); return c.most_common(1)[0][0] if c else glob[s]

# clean-ness of (family,slot,prefix)->value
clean=sum(1 for k,c in fsp.items() if len(c)==1); print(f"\n(family,slot,prefix)->value clean {clean}/{len(fsp)} = {clean/len(fsp):.3f}")

# rebuild fixed val degradation
valrng=np.random.default_rng(12345); examples=[]
for rep in range(VAL_REPS):
    for r in va:
        inp,_=make_example(r,valrng,n_show=3); examples.append((inp,r))

def wcv(predfn,label):
    acc={s:0 for s in SLOTS}; n=len(examples); wsc=0
    for inp,r in examples:
        pred=predfn(inp,r); truth=r["truth"]
        for s in SLOTS:
            if pred[s]==truth[s]: acc[s]+=1
        wsc+=sum(W[s]*(pred[s]==truth[s]) for s in SLOTS)/WSUM
    print(f"{label:34s} CV={wsc/n:.4f} | "+" ".join(f"{s[:2]}={acc[s]/n:.2f}" for s in SLOTS))

# baseline A: pure family mode (ignores all tags & correction)
def predA(inp,r): return {s:fmode(r["family"],s) for s in SLOTS}
# baseline B: family mode + correction override via (family,slot,prefix)
def parse_corr(inp,r):
    cs=None;cp=None
    for ln in inp.split("\n"):
        for desc,sl in CORR_DESC.items():
            if desc in ln: cs=sl; pf=PREFIX.findall(ln); cp=pf[-1] if pf else None
        if cs: break
    return cs,cp
def predB(inp,r):
    pred={s:fmode(r["family"],s) for s in SLOTS}
    cs,cp=parse_corr(inp,r)
    if cs and cp and (r["family"],cs,cp) in fsp: pred[cs]=fsp[(r["family"],cs,cp)].most_common(1)[0][0]
    elif cs and cp and (cs,cp) in sp: pred[cs]=sp[(cs,cp)].most_common(1)[0][0]
    return pred
# baseline C: B + assign note prefixes via (family,slot,prefix) best matching slot
def predC(inp,r):
    pred=predB(inp,r); cs,cp=parse_corr(inp,r)
    body=" ".join(l for l in inp.split("\n") if not any(d in l for d in CORR_DESC))
    for p in PREFIX.findall(body):
        # best (slot,value) for this prefix in this family
        best=None;bestn=0
        for s in SLOTS:
            c=fsp.get((r["family"],s,p))
            if c:
                v,nn=c.most_common(1)[0]
                if nn>bestn: best=(s,v);bestn=nn
        if best and best[0]!=cs: pred[best[0]]=best[1]
    return pred

wcv(predA,"A: family-mode only")
wcv(predB,"B: A + correction override")
wcv(predC,"C: B + note-prefix assign")
