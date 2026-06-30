"""Shared parsing + test-matched augmentation for Reaction Protocol Silent-Edit Repair.

Train notes are clean (6 labeled slots). Test notes are UNLABELED, show only 3 tags,
omit operations, and use unseen numeric tag suffixes (only the word-prefix is shared).
We degrade train rows into test-style examples so the fine-tuned seq2seq model learns to
(a) decode prefix->value position-free, (b) apply the correction, (c) infer hidden slots.
"""
import re, numpy as np

SLOTS = ["prep","activation","order","control","quench","workup"]
NOTE_KEY = {"setup":"prep","activation":"activation","order":"order",
            "control":"control","stop":"quench","cleanup":"workup"}
PREFIX = re.compile(r"\b([a-z]{3,6}-[a-z]{3,6})-\d+[A-Z]\b")
CORR_DESC = {  # slot-description phrase -> slot
    "opening handling line":"prep",
    "line before reactive contact":"activation",
    "line describing which material waits":"order",
    "condition maintained during the hold":"control",
    "operation that ends reactivity":"quench",
    "cleanup operation":"workup",
}
SLOT_DESC = {v:k for k,v in CORR_DESC.items()}
FAMILIES = ["imine reduction","resin exchange","cross coupling","carbonate closure",
            "salt metathesis","benzylic oxidation","acyl transfer","photoredox capture"]
HEADERS = ["The reaction family is logged as {f}.","Header family: {f}.",
           "The planner groups this run under {f}."]
NOTE_TPL = ["margin mark {t} is written beside that operation",
            "the operation is abbreviated only as {t}",
            "the copy keeps shorthand mark {t}",
            "the copied operation carries bench tag {t}",
            "the retyped line preserves local tag {t}"]
MISS_TPL = ["A secondary operation line is missing from the retyped page.",
            "The copy omits one background operation that must be inferred from context.",
            "One non-edited handling line is smudged in the copy."]
CORR_TPL = [
    "Audit repair: the {d} should carry local tag {t}; leave unrelated operations unchanged.",
    "Silent edit: replace the {d} with the operation marked {t} in the corrected record.",
    "Post-run note: repair the {d} to bench tag {t}; keep other operations from the note.",
    "QC note: corrected entry for the {d} is tagged {t}, not the copied line.",
]
REQUEST = "Generate the repaired canonical protocol sequence."

def parse_seq(s):
    d={}
    for p in str(s).split(";"):
        if "=" in p:
            k,v=p.split("=",1); d[k.strip()]=v.strip()
    return d

def get_family(prompt):
    f=str(prompt).split("\n")[0].lower()
    for fam in FAMILIES:
        if fam in f: return fam
    return "?"

def corr_slot_prefix(cn):
    cn=str(cn); slot=None
    for desc,sl in CORR_DESC.items():
        if desc in cn: slot=sl; break
    pf=PREFIX.findall(cn)
    return slot,(pf[-1] if pf else None)

def note_prefix_by_slot(note):
    out={}; note=str(note)
    for kw,slot in NOTE_KEY.items():
        m=re.search(rf"(?:^|[.\s]){kw}\b(.*?)(?:\.|$)", note, re.I)
        if m:
            pf=PREFIX.findall(m.group(1))
            if pf: out[slot]=pf[-1]
    return out

def parse_train_row(row):
    """Extract structured fields from a train row."""
    fam=get_family(row["prompt"])
    note_pfx=note_prefix_by_slot(row["protocol_note"])
    cslot,cpfx=corr_slot_prefix(row["correction_notice"])
    truth=parse_seq(row["repaired_sequence"])
    return dict(family=fam, note_pfx=note_pfx, cslot=cslot, cpfx=cpfx, truth=truth)

def seq_str(truth):
    return ";".join(f"{s}={truth[s]}" for s in SLOTS)

def rand_tag(prefix, rng):
    return f"{prefix}-{rng.integers(10,99)}{chr(ord('A')+int(rng.integers(0,9)))}"

def rand_code(rng):
    return chr(ord('A')+int(rng.integers(0,8)))+str(int(rng.integers(1000,3999)))

def make_example(rec, rng, n_show=3, randomize_suffix=True):
    """Build one test-style (input, target) pair from a parsed train record."""
    fam=rec["family"]; note_pfx=rec["note_pfx"]; cslot=rec["cslot"]; cpfx=rec["cpfx"]; truth=rec["truth"]
    header=rng.choice(HEADERS).format(f=fam)
    # choose which slots to show in the note (from the 6 note prefixes available)
    avail=[s for s in SLOTS if s in note_pfx]
    k=min(n_show,len(avail))
    show=list(rng.choice(avail,size=k,replace=False))
    parts=[]
    for s in show:
        pfx=note_pfx[s]
        tag=rand_tag(pfx,rng) if randomize_suffix else pfx
        parts.append(rng.choice(NOTE_TPL).format(t=tag))
    # join with mixed separators like test
    note="Audit copy "+rand_code(rng)+". "
    joined=""
    for i,p in enumerate(parts):
        sep="" if i==0 else ("; " if rng.random()<0.5 else ". ")
        joined+=sep+p
    note+=joined+". "+rng.choice(MISS_TPL)
    # correction
    if cslot is not None and cpfx is not None:
        ctag=rand_tag(cpfx,rng) if randomize_suffix else cpfx
        corr=rng.choice(CORR_TPL).format(d=SLOT_DESC[cslot],t=ctag)
    else:
        corr=""
    inp="\n".join(x for x in [header,note,corr,REQUEST] if x)
    return inp, seq_str(truth)

def make_clean_example(rec, rng, randomize_suffix=True):
    """Optional: an unlabeled example that shows ALL available note slots (denser signal)."""
    return make_example(rec, rng, n_show=6, randomize_suffix=randomize_suffix)
