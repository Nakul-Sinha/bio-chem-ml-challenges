"""EDA5: test decodability with deterministic tag map; tag coverage; missing slots."""
import pandas as pd, numpy as np, re, collections, json
from pathlib import Path

DS = Path(__file__).resolve().parent.parent / "dataset"
RES = Path(__file__).resolve().parent
train = pd.read_csv(DS / "train.csv")
test = pd.read_csv(DS / "test.csv")
SLOTS = ["prep","activation","order","control","quench","workup"]
NOTE_KEY = {"setup":"prep","activation":"activation","order":"order","control":"control","stop":"quench","cleanup":"workup"}
TAG = re.compile(r"\b([a-z]{3,6}-[a-z]{3,6}-\d+[A-Z])\b")
CORR_DESC = {"opening handling line":"prep","line before reactive contact":"activation",
    "line describing which material waits":"order","condition maintained during the hold":"control",
    "operation that ends reactivity":"quench","cleanup operation":"workup"}
tag_map = {k:tuple(v) for k,v in json.loads((RES/"tag_map.json").read_text()).items()}

def corr_slot_and_tag(cn):
    cn=str(cn); slot=None
    for desc,sl in CORR_DESC.items():
        if desc in cn: slot=sl; break
    tags=TAG.findall(cn); return slot,(tags[-1] if tags else None)

def get_family(p):
    f=str(p).split("\n")[0].lower()
    for fam in ["imine reduction","resin exchange","cross coupling","carbonate closure",
                "salt metathesis","benzylic oxidation","acyl transfer","photoredox capture"]:
        if fam in f: return fam
    return "?"

# 1. test tag coverage
test_tags=set()
for n in test["protocol_note"]: test_tags.update(TAG.findall(str(n)))
for c in test["correction_notice"]: test_tags.update(TAG.findall(str(c)))
train_tags=set(tag_map.keys())
print("test unique tags:",len(test_tags))
print("test tags NOT in train map:",len(test_tags-train_tags), sorted(test_tags-train_tags)[:20])

# 2. decode each test row via tag map (position-free: each tag -> its slot,value)
test["family"]=test["prompt"].apply(get_family)
fill_counts=collections.Counter()
conflict=0; corr_unparsed=0
missing_slot_counter=collections.Counter()
decoded_rows=[]
for _,row in test.iterrows():
    note_tags=TAG.findall(str(row["protocol_note"]))
    slotval={}
    bad=False
    for t in note_tags:
        if t in tag_map:
            sl,val=tag_map[t]
            if sl in slotval and slotval[sl]!=val: conflict+=1
            slotval[sl]=val
    cs,ct=corr_slot_and_tag(row["correction_notice"])
    if cs is None or ct is None: corr_unparsed+=1
    if ct and ct in tag_map:
        sl,val=tag_map[ct]
        slotval[sl]=val  # correction overrides
    elif cs and ct: # tag unknown but slot known
        pass
    filled=len(slotval)
    fill_counts[filled]+=1
    for sl in SLOTS:
        if sl not in slotval: missing_slot_counter[sl]+=1
    decoded_rows.append(slotval)

print("\nslots filled per test row (note tags + correction):")
for k in sorted(fill_counts): print(f"  {k} slots: {fill_counts[k]} rows")
print("tag->slot conflicts within a row:",conflict)
print("corrections unparsed:",corr_unparsed)
print("\nwhich slots end up missing (need default):")
for sl in SLOTS: print(f"  {sl}: {missing_slot_counter[sl]}")

# 3. For rows fully filled (6), that's deterministic. For missing, need family default.
# Learn family default per slot from TRAIN (most common value per (family,slot)).
def parse_seq(s):
    d={}
    for p in str(s).split(";"):
        if "=" in p: k,v=p.split("=",1); d[k.strip()]=v.strip()
    return d
train["family"]=train["prompt"].apply(get_family)
fam_default=collections.defaultdict(dict)
fam_slot_dist=collections.defaultdict(collections.Counter)
for _,row in train.iterrows():
    out=parse_seq(row["repaired_sequence"])
    for sl in SLOTS: fam_slot_dist[(row["family"],sl)][out[sl]]+=1
print("\nper-(family,slot) value entropy (is family alone predictive?):")
for fam in sorted(train["family"].unique()):
    line=[fam[:16].ljust(16)]
    for sl in SLOTS:
        c=fam_slot_dist[(fam,sl)]; tot=sum(c.values()); top=c.most_common(1)[0]
        line.append(f"{sl[:4]}:{top[1]/tot*100:.0f}%")
    print("  "+" ".join(line))
