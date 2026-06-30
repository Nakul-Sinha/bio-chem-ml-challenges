"""EDA6: PREFIX-based tag mapping (word1-word2 -> slot,value). Re-test decodability."""
import pandas as pd, numpy as np, re, collections, json
from pathlib import Path

DS = Path(__file__).resolve().parent.parent / "dataset"
train = pd.read_csv(DS / "train.csv"); test = pd.read_csv(DS / "test.csv")
SLOTS = ["prep","activation","order","control","quench","workup"]
NOTE_KEY = {"setup":"prep","activation":"activation","order":"order","control":"control","stop":"quench","cleanup":"workup"}
TAGFULL = re.compile(r"\b([a-z]{3,6}-[a-z]{3,6}-\d+[A-Z])\b")
PREFIX = re.compile(r"\b([a-z]{3,6}-[a-z]{3,6})-\d+[A-Z]\b")
CORR_DESC = {"opening handling line":"prep","line before reactive contact":"activation",
    "line describing which material waits":"order","condition maintained during the hold":"control",
    "operation that ends reactivity":"quench","cleanup operation":"workup"}
def parse_seq(s):
    d={}
    for p in str(s).split(";"):
        if "=" in p: k,v=p.split("=",1); d[k.strip()]=v.strip()
    return d
def corr_slot_and_pfx(cn):
    cn=str(cn); slot=None
    for desc,sl in CORR_DESC.items():
        if desc in cn: slot=sl; break
    pf=PREFIX.findall(cn); return slot,(pf[-1] if pf else None)
def note_pfx_by_slot(note):
    out={}; note=str(note)
    for kw,slot in NOTE_KEY.items():
        m=re.search(rf"(?:^|[.\s]){kw}\b(.*?)(?:\.|$)", note, re.I)
        if m:
            pf=PREFIX.findall(m.group(1))
            if pf: out[slot]=pf[-1]
    return out
def get_family(p):
    f=str(p).split("\n")[0].lower()
    for fam in ["imine reduction","resin exchange","cross coupling","carbonate closure",
                "salt metathesis","benzylic oxidation","acyl transfer","photoredox capture"]:
        if fam in f: return fam
    return "?"
train["family"]=train["prompt"].apply(get_family); test["family"]=test["prompt"].apply(get_family)
train["cslot"],train["cpfx"]=zip(*train["correction_notice"].apply(corr_slot_and_pfx))

# Build PREFIX -> (slot,value)
pfx2sv=collections.defaultdict(collections.Counter)
for _,row in train.iterrows():
    out=parse_seq(row["repaired_sequence"]); nt=note_pfx_by_slot(row["protocol_note"])
    cs,cp=row["cslot"],row["cpfx"]
    for slot in SLOTS:
        if slot==cs:
            if cp: pfx2sv[cp][(slot,out[slot])]+=1
        elif slot in nt:
            pfx2sv[nt[slot]][(slot,out[slot])]+=1
cons=sum(1 for p,c in pfx2sv.items() if len(c)==1)
print(f"PREFIX->(slot,value): {len(pfx2sv)} prefixes, consistent={cons} ({cons/len(pfx2sv)*100:.1f}%)")
for p,c in pfx2sv.items():
    if len(c)>1: print("  NOISY",p,dict(c))

# test prefix coverage
test_pfx=set()
for n in test["protocol_note"]: test_pfx.update(PREFIX.findall(str(n)))
for c in test["correction_notice"]: test_pfx.update(PREFIX.findall(str(c)))
print(f"\ntest unique prefixes: {len(test_pfx)}, NOT in train: {len(test_pfx-set(pfx2sv))} -> {sorted(test_pfx-set(pfx2sv))}")

# decode test (position-free prefix mapping + correction override)
pfx_map={p:c.most_common(1)[0][0] for p,c in pfx2sv.items()}
fill=collections.Counter(); miss=collections.Counter(); conflict=0
for _,row in test.iterrows():
    sv={}
    for pf in PREFIX.findall(str(row["protocol_note"])):
        if pf in pfx_map:
            sl,val=pfx_map[pf]
            if sl in sv and sv[sl]!=val: conflict+=1
            sv[sl]=val
    cs,cp=corr_slot_and_pfx(row["correction_notice"])
    if cp and cp in pfx_map: sl,val=pfx_map[cp]; sv[sl]=val
    fill[len(sv)]+=1
    for sl in SLOTS:
        if sl not in sv: miss[sl]+=1
print("\nslots filled per test row (prefix map):")
for k in sorted(fill): print(f"  {k}: {fill[k]}")
print("conflicts:",conflict)
print("missing slot counts:", {s:miss[s] for s in SLOTS})

# how many full tags per test note (raw)
ntags=[len(TAGFULL.findall(str(n))) for n in test["protocol_note"]]
print("\nnote tag-count distribution (test):", dict(collections.Counter(ntags)))
ntags_tr=[len(TAGFULL.findall(str(n))) for n in train["protocol_note"]]
print("note tag-count distribution (train):", dict(collections.Counter(ntags_tr)))
json.dump({f"{k}":list(v) for k,v in pfx_map.items()}, open(Path(__file__).parent/"pfx_map.json","w"))
