"""Dissect the tag->value relationship. Is value encoded in the FULL tag, the suffix
(digits/letter), or a per-row cipher? Determines whether augmentation's suffix-randomization
is destroying real signal.
"""
import re, collections, numpy as np, pandas as pd
from pathlib import Path
from aug import SLOTS, get_family
DS=Path(__file__).resolve().parent.parent/"dataset"
train=pd.read_csv(DS/"train.csv")
FULLTAG=re.compile(r"\b([a-z]{3,6}-[a-z]{3,6})-(\d+)([A-Z])\b")
NOTE_KEY={"setup":"prep","activation":"activation","order":"order","control":"control","stop":"quench","cleanup":"workup"}

def slot_tags(note):
    """map slot->(prefix,digits,letter) by finding each keyword then the next tag."""
    out={}
    for kw,slot in NOTE_KEY.items():
        m=re.search(rf"(?:^|[.\s]){kw}\b(.*?)(?:\.|$)", str(note), re.I)
        if m:
            t=FULLTAG.search(m.group(1))
            if t: out[slot]=(t.group(1),t.group(2),t.group(3))
    return out

def parse_seq(s):
    d={}
    for p in str(s).split(";"):
        if "=" in p: k,v=p.split("=",1); d[k.strip()]=v.strip()
    return d

rows=[]
for _,r in train.iterrows():
    fam=get_family(r["prompt"]); tags=slot_tags(r["protocol_note"]); truth=parse_seq(r["repaired_sequence"])
    rows.append((fam,tags,truth))

# H: full-tag -> value  (does the same full tag always give the same value, anywhere?)
fulltag_val=collections.defaultdict(collections.Counter)
prefix_val=collections.defaultdict(collections.Counter)
suffixnum_val=collections.defaultdict(collections.Counter)   # digits -> value
suffixlet_val=collections.defaultdict(collections.Counter)   # letter -> value
letter_by_slot=collections.defaultdict(collections.Counter)
for fam,tags,truth in rows:
    for slot,(pfx,dig,let) in tags.items():
        v=truth[slot]
        fulltag_val[(pfx,dig,let)][v]+=1
        prefix_val[pfx][v]+=1
        suffixnum_val[dig][v]+=1
        suffixlet_val[let][v]+=1
        letter_by_slot[slot][let]+=1
def cleanpct(d):
    return sum(1 for k,c in d.items() if len(c)==1)/max(1,len(d))
print(f"full-tag  ->value clean {cleanpct(fulltag_val):.3f}  (n={len(fulltag_val)})")
print(f"prefix    ->value clean {cleanpct(prefix_val):.3f}  (n={len(prefix_val)})")
print(f"suffixDig ->value clean {cleanpct(suffixnum_val):.3f} (n={len(suffixnum_val)})")
print(f"suffixLet ->value clean {cleanpct(suffixlet_val):.3f} (n={len(suffixlet_val)})")

# per-row cipher: within a row, is prefix->value a bijection consistent across the row?
# test: does (prefix) determine value WITHIN a family? already knew ~no. Try (letter)->value within slot.
print("\nletter->value mutual info per slot (0=independent,1=deterministic):")
for s in SLOTS:
    # build letter->value counter for this slot
    lv=collections.defaultdict(collections.Counter)
    for fam,tags,truth in rows:
        if s in tags:
            _,_,let=tags[s]; lv[let][truth[s]]+=1
    # fraction of mass on the modal value averaged over letters
    tot=0;correct=0
    for let,c in lv.items():
        tot+=sum(c.values()); correct+=c.most_common(1)[0][1]
    print(f"  {s:10s} letter->modal acc={correct/max(1,tot):.3f}  (#letters={len(lv)})")

# does the DIGITS value (as int) map to value index within slot?
print("\ndigits(int) vs value: top pairs for quench:")
c=collections.Counter()
for fam,tags,truth in rows:
    if "quench" in tags:
        _,dig,_=tags["quench"]; c[(int(dig),truth["quench"])]+=1
for k,n in c.most_common(12): print("   ",k,n)
