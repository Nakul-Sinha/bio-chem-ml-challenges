"""EDA7: disambiguate noisy prefixes; sequence-template structure; missing-slot ceiling."""
import pandas as pd, numpy as np, re, collections, itertools, json
from pathlib import Path

DS = Path(__file__).resolve().parent.parent / "dataset"
train = pd.read_csv(DS / "train.csv"); test = pd.read_csv(DS / "test.csv")
SLOTS = ["prep","activation","order","control","quench","workup"]
W = {"prep":2.20,"activation":0.85,"order":0.60,"control":3.00,"quench":4.00,"workup":0.25}
WSUM=sum(W.values())
NOTE_KEY = {"setup":"prep","activation":"activation","order":"order","control":"control","stop":"quench","cleanup":"workup"}
PREFIX = re.compile(r"\b([a-z]{3,6}-[a-z]{3,6})-\d+[A-Z]\b")
PFXLET = re.compile(r"\b([a-z]{3,6}-[a-z]{3,6})-\d+([A-Z])\b")
CORR_DESC = {"opening handling line":"prep","line before reactive contact":"activation",
    "line describing which material waits":"order","condition maintained during the hold":"control",
    "operation that ends reactivity":"quench","cleanup operation":"workup"}
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
def corr(cn):
    cn=str(cn); slot=None
    for desc,sl in CORR_DESC.items():
        if desc in cn: slot=sl; break
    m=PFXLET.findall(cn); return slot,(m[-1] if m else None)
def note_pl_by_slot(note):
    out={}; note=str(note)
    for kw,slot in NOTE_KEY.items():
        m=re.search(rf"(?:^|[.\s]){kw}\b(.*?)(?:\.|$)", note, re.I)
        if m:
            pl=PFXLET.findall(m.group(1))
            if pl: out[slot]=pl[-1]  # (prefix,letter)
    return out
train["family"]=train["prompt"].apply(get_family); test["family"]=test["prompt"].apply(get_family)
train["cslot_pl"]=train["correction_notice"].apply(corr)

# (prefix,letter)->(slot,value)
pl2sv=collections.defaultdict(collections.Counter)
for _,row in train.iterrows():
    out=parse_seq(row["repaired_sequence"]); nt=note_pl_by_slot(row["protocol_note"])
    cs,cpl=row["cslot_pl"]
    for slot in SLOTS:
        if slot==cs:
            if cpl: pl2sv[cpl][(slot,out[slot])]+=1
        elif slot in nt: pl2sv[nt[slot]][(slot,out[slot])]+=1
cons=sum(1 for p,c in pl2sv.items() if len(c)==1)
print(f"(prefix,letter)->(slot,value): {len(pl2sv)} keys, consistent={cons} ({cons/len(pl2sv)*100:.1f}%)")
for k,c in pl2sv.items():
    if len(c)>1: print("  STILL NOISY",k,dict(c))
# test coverage of (prefix,letter)
test_pl=set()
for n in test["protocol_note"]: test_pl.update(PFXLET.findall(str(n)))
for c in test["correction_notice"]: test_pl.update(PFXLET.findall(str(c)))
print("test (prefix,letter) keys not in train:", len(test_pl-set(pl2sv)),"/",len(test_pl))

# Unique sequences overall & per family
train["seq"]=train["repaired_sequence"]
print("\nunique full sequences:", train["seq"].nunique(),"of",len(train))
for fam in sorted(train["family"].unique()):
    sub=train[train["family"]==fam]
    print(f"  {fam:18s}: {sub['seq'].nunique()} unique seqs / {len(sub)} rows")

# Missing-slot ceiling: P(slot | family) mode accuracy  vs  P(slot | family, other 5 slots known)
parsed=train["repaired_sequence"].apply(parse_seq)
df=pd.DataFrame(list(parsed)); df["family"]=train["family"].values
print("\nmode accuracy P(slot|family) [what we get if slot hidden, only family known]:")
wsum_vis=0
for sl in SLOTS:
    acc=df.groupby("family")[sl].apply(lambda s: s.value_counts().iloc[0]/len(s)).mean()
    print(f"  {sl:10s} w={W[sl]}: {acc:.3f}")

# Pairwise: does knowing other slots help predict a hidden slot? Use family+each other slot.
print("\nbest single-other-slot conditional mode accuracy for each hidden slot (given family+that slot):")
for target in SLOTS:
    best=0; bestsrc=None
    for src in SLOTS:
        if src==target: continue
        g=df.groupby(["family",src])[target].apply(lambda s: s.value_counts().iloc[0])
        n=df.groupby(["family",src])[target].size()
        acc=(g.sum()/n.sum())
        if acc>best: best=acc; bestsrc=src
    base=df.groupby("family")[target].apply(lambda s: s.value_counts().iloc[0]/len(s)).mean()
    print(f"  {target:10s}: family-only={base:.3f}  +best({bestsrc})={best:.3f}")
