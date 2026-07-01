"""Is the CV trustworthy? Check recipe (value-tuple) cardinality & repetition, and whether
test tag-prefixes/families match train. If recipes repeat heavily and test uses same prefixes,
CV should transfer; if val shares recipes with train it may be optimistic."""
import re, collections, numpy as np, pandas as pd
from pathlib import Path
from aug import SLOTS, parse_train_row, get_family, PREFIX
DS=Path(__file__).resolve().parent.parent/"dataset"
train=pd.read_csv(DS/"train.csv"); test=pd.read_csv(DS/"test.csv")
recs=[parse_train_row(r) for _,r in train.iterrows()]

# recipe = full value tuple
tuples=[tuple(r["truth"][s] for s in SLOTS) for r in recs]
tc=collections.Counter(tuples)
print(f"train rows={len(recs)}  distinct recipes={len(tc)}  singletons={sum(1 for v in tc.values() if v==1)}")
print(f"top recipe counts: {[n for _,n in tc.most_common(5)]}")
# per-family recipes
for f in sorted(set(r["family"] for r in recs)):
    ft=[t for t,r in zip(tuples,recs) if r["family"]==f]
    print(f"  {f:20s} rows={len(ft):4d} recipes={len(set(ft)):3d}")

# per-slot-pair recipes: are slots independent or correlated? (mutual predictability)
# quick: entropy of each slot vs conditional on family
print("\nrecipe repetition: fraction of rows whose recipe appears >=2x:",
      f"{sum(n for n in tc.values() if n>=2)/len(recs):.3f}")

# prefix vocab overlap train vs test
def prefixes(df,cols):
    s=set()
    for _,r in df.iterrows():
        for c in cols: s|=set(PREFIX.findall(str(r[c])))
    return s
tr_pfx=prefixes(train,["protocol_note","correction_notice"])
te_pfx=prefixes(test,["prompt"])
print(f"\ntrain prefixes={len(tr_pfx)} test prefixes={len(te_pfx)} test-not-in-train={len(te_pfx-tr_pfx)}")
# family distribution
trf=collections.Counter(get_family(r["prompt"]) for _,r in train.iterrows())
tef=collections.Counter(get_family(r["prompt"]) for _,r in test.iterrows())
print("train fam dist:",dict(trf)); print("test  fam dist:",dict(tef))

# correction-target slot distribution train vs test
from aug import CORR_DESC
def corr_slot(text):
    for d,sl in CORR_DESC.items():
        if d in str(text): return sl
    return None
trc=collections.Counter(corr_slot(r["correction_notice"]) for _,r in train.iterrows())
tec=collections.Counter(corr_slot(r["prompt"]) for _,r in test.iterrows())
print("train corr-slot:",dict(trc)); print("test  corr-slot:",dict(tec))
