"""EDA part 2: bench tag -> (slot,value) mapping, correction parsing, split structure."""
import pandas as pd, numpy as np, json, re, collections
from pathlib import Path

DS = Path(__file__).resolve().parent.parent / "dataset"
train = pd.read_csv(DS / "train.csv")
test = pd.read_csv(DS / "test.csv")

SLOTS = ["prep", "activation", "order", "control", "quench", "workup"]
NOTE_KEY = {"setup":"prep","activation":"activation","order":"order",
            "control":"control","stop":"quench","cleanup":"workup"}

def parse_seq(s):
    d = {}
    for part in str(s).split(";"):
        if "=" in part:
            k, v = part.split("=", 1)
            d[k.strip()] = v.strip()
    return d

TAG = re.compile(r"\b([a-z]{3,6}-[a-z]{3,6}-\d+[A-Z])\b")

# Build mapping note_slot_keyword -> tag, from protocol_note (labeled cases)
def parse_note_tags(note):
    """Return dict slot->tag for lines that start with a known keyword."""
    out = {}
    note = str(note)
    # split into sentences by '. '
    # Each labeled line: "<keyword> <text> <tag>."
    for kw, slot in NOTE_KEY.items():
        # find "<kw> ... <tag>" up to next period
        m = re.search(rf"\b{kw}\b(.*?)(?:\.|$)", note, re.I)
        if m:
            seg = m.group(1)
            tags = TAG.findall(seg)
            if tags:
                out[slot] = tags[-1]  # tag at end of the line
    return out

# Test how many train notes are fully labeled (all 6 slots found)
n_full = 0
note_tag_records = []  # (slot, tag, output_value)
for _, row in train.iterrows():
    nt = parse_note_tags(row["protocol_note"])
    out = parse_seq(row["repaired_sequence"])
    if len(nt) == 6:
        n_full += 1
    for slot, tag in nt.items():
        note_tag_records.append((slot, tag, out.get(slot)))
print(f"train notes fully labeled (6 slots): {n_full}/{len(train)}")

# Build tag -> (slot,value) mapping from note records, check consistency
# BUT corrected slot's note tag != output. So we must exclude corrected slot.
# First, identify corrected slot via correction_notice.
def correction_tag(cn):
    tags = TAG.findall(str(cn))
    return tags[-1] if tags else None

train["corr_tag"] = train["correction_notice"].apply(correction_tag)
print("train rows with a tag in correction_notice:", train["corr_tag"].notna().sum(), "/", len(train))

# Map: for non-corrected slots, note_tag -> value
tag2sv = collections.defaultdict(collections.Counter)  # tag -> Counter((slot,value))
for _, row in train.iterrows():
    nt = parse_note_tags(row["protocol_note"])
    out = parse_seq(row["repaired_sequence"])
    ctag = row["corr_tag"]
    for slot, tag in nt.items():
        if tag == ctag:
            continue  # this slot is overridden; note tag not reliable
        tag2sv[tag][(slot, out.get(slot))] += 1

# consistency: how many tags map to exactly one (slot,value)?
consistent = sum(1 for t,c in tag2sv.items() if len(c)==1)
print(f"\nunique note-tags (non-corrected): {len(tag2sv)}, consistent (1 mapping): {consistent}")
# show a few inconsistent
inc = [(t,c) for t,c in tag2sv.items() if len(c)>1]
print("inconsistent tag count:", len(inc))
for t,c in inc[:10]:
    print("   ", t, dict(c))

# Now check correction tag -> output value of corrected slot
# corrected slot value = the slot in output that differs from note-implied value
corr_tag2val = collections.defaultdict(collections.Counter)
for _, row in train.iterrows():
    nt = parse_note_tags(row["protocol_note"])
    out = parse_seq(row["repaired_sequence"])
    ctag = row["corr_tag"]
    if ctag is None:
        continue
    # which slot does corr apply to? the slot whose note tag == ctag won't help (it's new).
    # Instead: find slot where note tag maps (via tag2sv) to a value != output value
    for slot in SLOTS:
        ov = out.get(slot)
        ntag = nt.get(slot)
        # if note tag maps to a different value than output -> corrected slot
        implied = None
        if ntag and ntag in tag2sv and len(tag2sv[ntag])==1:
            implied = list(tag2sv[ntag])[0][1]
        if implied is not None and implied != ov:
            corr_tag2val[ctag][(slot, ov)] += 1
print(f"\ncorrection tags mapped: {len(corr_tag2val)}")
cc = sum(1 for t,c in corr_tag2val.items() if len(c)==1)
print(f"consistent correction tags: {cc}/{len(corr_tag2val)}")
for t,c in list(corr_tag2val.items())[:10]:
    print("   ", t, dict(c))
