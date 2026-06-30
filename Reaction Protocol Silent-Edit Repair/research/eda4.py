"""EDA4: clean tag->(slot,value) mapping; missing-slot defaults; test parseability."""
import pandas as pd, numpy as np, re, collections, json
from pathlib import Path

DS = Path(__file__).resolve().parent.parent / "dataset"
train = pd.read_csv(DS / "train.csv")
test = pd.read_csv(DS / "test.csv")
SLOTS = ["prep","activation","order","control","quench","workup"]
NOTE_KEY = {"setup":"prep","activation":"activation","order":"order","control":"control","stop":"quench","cleanup":"workup"}
TAG = re.compile(r"\b([a-z]{3,6}-[a-z]{3,6}-\d+[A-Z])\b")

# correction slot-description -> slot
CORR_DESC = {
    "opening handling line":"prep",
    "line before reactive contact":"activation",
    "line describing which material waits":"order",
    "condition maintained during the hold":"control",
    "operation that ends reactivity":"quench",
    "cleanup operation":"workup",
}
def parse_seq(s):
    d={}
    for p in str(s).split(";"):
        if "=" in p:
            k,v=p.split("=",1); d[k.strip()]=v.strip()
    return d

def corr_slot_and_tag(cn):
    cn=str(cn); slot=None
    for desc,sl in CORR_DESC.items():
        if desc in cn:
            slot=sl; break
    tags=TAG.findall(cn)
    return slot, (tags[-1] if tags else None)

def note_tags(note):
    """slot->tag for labeled lines."""
    out={}; note=str(note)
    for kw,slot in NOTE_KEY.items():
        m=re.search(rf"(?:^|[.\s]){kw}\b(.*?)(?:\.|$)", note, re.I)
        if m:
            tags=TAG.findall(m.group(1))
            if tags: out[slot]=tags[-1]
    return out

# Verify corrections parse
train["cslot"],train["ctag"]=zip(*train["correction_notice"].apply(corr_slot_and_tag))
print("train corr slot parsed:", train["cslot"].notna().sum(),"/",len(train))
print("train corr tag parsed:", train["ctag"].notna().sum(),"/",len(train))
print("corr slot distribution:\n", train["cslot"].value_counts())

# Build clean tag->(slot,value)
tag2sv=collections.defaultdict(collections.Counter)
missing_by_family=collections.defaultdict(collections.Counter)  # (family,slot)->value when missing
def get_family(p):
    f=str(p).split("\n")[0].lower()
    for fam in ["imine reduction","resin exchange","cross coupling","carbonate closure",
                "salt metathesis","benzylic oxidation","acyl transfer","photoredox capture"]:
        if fam in f: return fam
    return "?"
train["family"]=train["prompt"].apply(get_family)

for _,row in train.iterrows():
    out=parse_seq(row["repaired_sequence"])
    nt=note_tags(row["protocol_note"])
    cs,ct=row["cslot"],row["ctag"]
    for slot in SLOTS:
        if slot==cs:
            if ct: tag2sv[ct][(slot,out[slot])]+=1
        else:
            if slot in nt:
                tag2sv[nt[slot]][(slot,out[slot])]+=1
            else:
                # missing line for non-corrected slot -> default
                missing_by_family[(row["family"],slot)][out[slot]]+=1

cons=sum(1 for t,c in tag2sv.items() if len(c)==1)
print(f"\nclean tag->(slot,value): {len(tag2sv)} tags, consistent={cons} ({cons/len(tag2sv)*100:.1f}%)")
for t,c in list(tag2sv.items()):
    if len(c)>1:
        tot=sum(c.values()); top=c.most_common(1)[0]
        print(f"  NOISY {t}: top={top[0]}{top[1]}/{tot}", dict(c))

# missing slot defaults
print(f"\nmissing-slot cases (non-corrected): {sum(sum(c.values()) for c in missing_by_family.values())}")
print("missing default consistency per (family,slot):")
mc=0; mt=0
for (fam,slot),c in sorted(missing_by_family.items()):
    mt+=1
    if len(c)==1: mc+=1
    else: print(f"  {fam:20s}/{slot:10s} -> {dict(c)}")
print(f"(family,slot) missing combos deterministic: {mc}/{mt}")

# Save the mapping for later
mapping={t:list(c.most_common(1)[0][0]) for t,c in tag2sv.items()}
Path(__file__).parent.joinpath("tag_map.json").write_text(json.dumps(mapping,indent=0))
print("saved tag_map.json with",len(mapping),"tags")
