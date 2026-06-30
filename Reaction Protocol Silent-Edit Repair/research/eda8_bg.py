"""Test 'background-operation hiding' hypothesis: are hidden slots biased to background values?
Indirect: if value V of slot S is preferentially hidden in test, visible test S will be depleted of V
relative to train marginal."""
import pandas as pd, numpy as np, re, collections, json
from pathlib import Path
DS = Path(__file__).resolve().parent.parent / "dataset"
RES=Path(__file__).resolve().parent
train = pd.read_csv(DS / "train.csv"); test=pd.read_csv(DS/"test.csv")
SLOTS = ["prep","activation","order","control","quench","workup"]
PREFIX = re.compile(r"\b([a-z]{3,6}-[a-z]{3,6})-\d+[A-Z]\b")
CORR_DESC = {"opening handling line":"prep","line before reactive contact":"activation",
    "line describing which material waits":"order","condition maintained during the hold":"control",
    "operation that ends reactivity":"quench","cleanup operation":"workup"}
pfx_map={k:tuple(v) for k,v in json.load(open(RES/"pfx_map.json")).items()}  # prefix-> [slot,value] majority
def parse_seq(s):
    d={}
    for p in str(s).split(";"):
        if "=" in p: k,v=p.split("=",1); d[k.strip()]=v.strip()
    return d
def corr(cn):
    cn=str(cn); slot=None
    for desc,sl in CORR_DESC.items():
        if desc in cn: slot=sl;break
    pf=PREFIX.findall(cn); return slot,(pf[-1] if pf else None)

# train marginal per slot
parsed=train["repaired_sequence"].apply(parse_seq)
train_marg={sl:collections.Counter() for sl in SLOTS}
for d in parsed:
    for sl in SLOTS: train_marg[sl][d[sl]]+=1

# test visible decode
test_vis={sl:collections.Counter() for sl in SLOTS}
for _,row in test.iterrows():
    for pf in PREFIX.findall(str(row["protocol_note"])):
        if pf in pfx_map:
            sl,val=pfx_map[pf]; test_vis[sl][val]+=1
    cs,cp=corr(row["correction_notice"])
    if cp and cp in pfx_map:
        sl,val=pfx_map[cp]; test_vis[sl][val]+=1

print("Per-slot value: train_marginal%  vs  test_visible%  (depleted in visible => preferentially hidden)")
for sl in SLOTS:
    tm=train_marg[sl]; tv=test_vis[sl]
    tmt=sum(tm.values()); tvt=sum(tv.values())
    print(f"\n[{sl}]  (test visible n={tvt})")
    for v,_ in tm.most_common():
        tmp=tm[v]/tmt*100; tvp=tv.get(v,0)/tvt*100 if tvt else 0
        flag="  <== depleted(hidden?)" if tvp<tmp-4 else ("  <== enriched" if tvp>tmp+4 else "")
        print(f"   {v:22s} train={tmp:5.1f}%  testvis={tvp:5.1f}%{flag}")
