"""Strict independent validation + sanity check of the C1 submission."""
import pandas as pd, numpy as np, collections, sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from aug import SLOTS, parse_seq, get_family
DS=Path(__file__).resolve().parent.parent/"dataset"
sub=pd.read_csv(Path(__file__).resolve().parent.parent/"submission.csv")
test=pd.read_csv(DS/"test.csv"); train=pd.read_csv(DS/"train.csv")
samp=pd.read_csv(DS/"sample_submission.csv")

# format checks
assert list(sub.columns)==["id","repaired_sequence"], sub.columns.tolist()
assert len(sub)==524, len(sub)
assert sub["id"].is_unique, "dup ids"
assert set(sub["id"])==set(test["id"]), "id set mismatch"
assert not sub.isna().any().any(), "NaNs"
# valid vocab per slot
vocab={s:set() for s in SLOTS}
for seq in train["repaired_sequence"]:
    d=parse_seq(seq)
    for s in SLOTS: vocab[s].add(d[s])
bad=0
for seq in sub["repaired_sequence"]:
    d=parse_seq(seq)
    if seq.count(";")!=5: bad+=1; continue
    order=[p.split("=")[0] for p in seq.split(";")]
    if order!=SLOTS: bad+=1; continue
    for s in SLOTS:
        if d.get(s) not in vocab[s]: bad+=1; break
print("malformed rows:",bad)
assert bad==0
print("FORMAT OK: 524 rows, columns/order/vocab all valid")

# distribution sanity: compare submission slot-value dist to train marginal
print("\nslot-value distribution (submission % vs train %):")
for s in SLOTS:
    subc=collections.Counter(parse_seq(x)[s] for x in sub["repaired_sequence"])
    trc=collections.Counter(parse_seq(x)[s] for x in train["repaired_sequence"])
    subt=sum(subc.values()); trt=sum(trc.values())
    print(f" [{s}]")
    for v,_ in trc.most_common():
        print(f"    {v:22s} sub={subc.get(v,0)/subt*100:5.1f}%  train={trc[v]/trt*100:5.1f}%")
    # degeneracy check: no single value should dominate >70% unless train does
    topfrac=max(subc.values())/subt
    if topfrac>0.7 and max(trc.values())/trt<0.5:
        print(f"    !! WARNING degenerate: top value {topfrac:.0%}")
# unique sequences in submission
print("\nunique sequences in submission:", sub["repaired_sequence"].nunique(),"/",len(sub))
print("sample predictions:")
for i in range(3): print("   ", sub.iloc[i]["id"], sub.iloc[i]["repaired_sequence"])
