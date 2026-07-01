"""Diagnostic: is model capacity the bottleneck, or information/decoding?
Build a DETERMINISTIC prefix->(slot,value) decoder from train, evaluate on the SAME
family-grouped val split + fixed val degradation as the CV (seed 42 / 12345, VAL_REPS=3).
Compare per-slot oracle-ish accuracy to the seq2seq (quench .675, control .750, workup .403).
Also report: how often each slot is SHOWN vs HIDDEN, and prefix->slot/value ambiguity.
"""
import re, collections, numpy as np, pandas as pd
from pathlib import Path
from aug import (SLOTS, parse_train_row, make_example, parse_seq, PREFIX,
                 CORR_DESC, note_prefix_by_slot, corr_slot_prefix, get_family)

W={"prep":2.20,"activation":0.85,"order":0.60,"control":3.00,"quench":4.00,"workup":0.25}; WSUM=sum(W.values())
DS=Path(__file__).resolve().parent.parent/"dataset"; VAL_FRAC=0.15; VAL_REPS=3
train=pd.read_csv(DS/"train.csv"); recs=[parse_train_row(r) for _,r in train.iterrows()]
fams=np.array([r["family"] for r in recs]); rng=np.random.default_rng(42); val_idx=[]
for f in sorted(set(fams)):
    idx=np.where(fams==f)[0]; rng.shuffle(idx); val_idx+=list(idx[:int(len(idx)*VAL_FRAC)])
val_idx=set(int(x) for x in val_idx); tr=[r for i,r in enumerate(recs) if i not in val_idx]; va=[r for i,r in enumerate(recs) if i in val_idx]

# ---- build maps from TRAIN ----
famtbl=collections.defaultdict(lambda:collections.defaultdict(collections.Counter))
pfx_sv=collections.defaultdict(collections.Counter)      # prefix_word -> Counter[(slot,value)]
pfx_val_byslot=collections.defaultdict(collections.Counter)  # (slot,prefix) -> Counter[value]
for r in tr:
    for s in SLOTS:
        famtbl[r["family"]][s][r["truth"][s]]+=1
        if s in r["note_pfx"]:
            p=r["note_pfx"][s]; pfx_sv[p][(s,r["truth"][s])]+=1; pfx_val_byslot[(s,p)][r["truth"][s]]+=1
    if r["cslot"] and r["cpfx"]:
        pfx_sv[r["cpfx"]][(r["cslot"],r["truth"][r["cslot"]])]+=1; pfx_val_byslot[(r["cslot"],r["cpfx"])][r["truth"][r["cslot"]]]+=1
glob={s:collections.Counter([r["truth"][s] for r in tr]).most_common(1)[0][0] for s in SLOTS}
def fmode(fam,s):
    c=famtbl.get(fam,{}).get(s); return c.most_common(1)[0][0] if c else glob[s]

# prefix ambiguity report
amb_slot=sum(1 for p,c in pfx_sv.items() if len({sl for (sl,_) in c})>1)
amb_val=sum(1 for p,c in pfx_sv.items() if len({v for (_,v) in c})>1)
print(f"prefixes: {len(pfx_sv)} | multi-SLOT {amb_slot} | multi-VALUE {amb_val}")
# value-uniqueness given (slot,prefix)
sv_clean=sum(1 for k,c in pfx_val_byslot.items() if len(c)==1); sv_tot=len(pfx_val_byslot)
print(f"(slot,prefix)->value clean {sv_clean}/{sv_tot} = {sv_clean/sv_tot:.3f}")

# ---- rebuild fixed val degradation, keep which slots were shown ----
valrng=np.random.default_rng(12345)
examples=[]  # (note_text, truth, fam, shown_slots, cslot)
for rep in range(VAL_REPS):
    for r in va:
        # replicate make_example's show selection deterministically by re-running with same rng
        inp,_=make_example(r,valrng,n_show=3)
        examples.append((inp,r["truth"],r["family"],r))

# stats: per slot, how often shown (note_pfx present & selected) vs hidden; correction target rate
shown_cnt={s:0 for s in SLOTS}; corr_cnt={s:0 for s in SLOTS}; tot=len(examples)
for inp,truth,fam,r in examples:
    # slots present as prefixes in the degraded note
    note_line=inp.split("\n")
    body=" ".join(note_line[1:])  # after header
    # correction slot
    if r["cslot"]: corr_cnt[r["cslot"]]+=1

# ---- deterministic decode of each degraded example ----
def decode(inp, r):
    lines=inp.split("\n"); header=lines[0]
    # find correction line (contains a CORR_DESC phrase)
    corr_slot=None; corr_pfx=None
    for ln in lines:
        for desc,sl in CORR_DESC.items():
            if desc in ln:
                corr_slot=sl; pf=PREFIX.findall(ln); corr_pfx=pf[-1] if pf else None
        if corr_slot: break
    # all prefixes in the note body (exclude the correction line to separate)
    note_body=" ".join(l for l in lines if not any(d in l for d in CORR_DESC))
    note_prefixes=PREFIX.findall(note_body)
    fam=get_family(header) if header else r["family"]
    pred={s:fmode(fam,s) for s in SLOTS}                    # base: family mode
    # assign each note prefix to its most likely (slot,value)
    used=set()
    for p in note_prefixes:
        if p in pfx_sv:
            for (sl,v),_ in pfx_sv[p].most_common():
                if sl not in used:
                    pred[sl]=v; used.add(sl); break
    # correction overrides
    if corr_slot and corr_pfx and (corr_slot,corr_pfx) in pfx_val_byslot:
        pred[corr_slot]=pfx_val_byslot[(corr_slot,corr_pfx)].most_common(1)[0][0]
    elif corr_slot and corr_pfx and corr_pfx in pfx_sv:
        for (sl,v),_ in pfx_sv[corr_pfx].most_common():
            if sl==corr_slot: pred[corr_slot]=v; break
    return pred, corr_slot

acc={s:0 for s in SLOTS}; n=len(examples); wsc=0
for inp,truth,fam,r in examples:
    pred,cs=decode(inp,r)
    for s in SLOTS:
        if pred[s]==truth[s]: acc[s]+=1
    wsc+=sum(W[s]*(pred[s]==truth[s]) for s in SLOTS)/WSUM
print(f"\nDETERMINISTIC decoder weighted CV={wsc/n:.4f}  (seq2seq ~0.726)")
for s in SLOTS:
    print(f"  {s:10s} det_acc={acc[s]/n:.3f}   corr_target_rate={corr_cnt[s]/tot:.3f}")
