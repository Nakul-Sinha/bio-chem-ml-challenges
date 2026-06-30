"""Explicit check of EVERY stated submission requirement for Challenge 1."""
import pandas as pd, csv
from pathlib import Path
ROOT=Path(__file__).resolve().parent.parent
DS=ROOT/"dataset"
SLOTS=["prep","activation","order","control","quench","workup"]

train=pd.read_csv(DS/"train.csv"); test=pd.read_csv(DS/"test.csv")
# valid value set per slot from PUBLIC TRAIN
def parse(s):
    d={}
    for p in str(s).split(";"):
        if "=" in p: k,v=p.split("=",1); d[k.strip()]=v.strip()
    return d
vocab={s:set() for s in SLOTS}
for seq in train["repaired_sequence"]:
    d=parse(seq)
    for s in SLOTS: vocab[s].add(d[s])

sub_path=ROOT/"submission.csv"
results=[]
def chk(name,ok,detail=""):
    results.append((name,ok,detail)); print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f"  -- {detail}" if detail else ""))

# --- File format: .csv with exact columns id,repaired_sequence ---
with open(sub_path,newline="") as f:
    rdr=csv.reader(f); header=next(rdr); body=list(rdr)
chk("File is .csv with exact header [id,repaired_sequence]", header==["id","repaired_sequence"], f"header={header}")
chk("Exactly 524 data rows (plus header)", len(body)==524, f"rows={len(body)}")
chk("No extra columns (every row has 2 fields)", all(len(r)==2 for r in body),
    f"bad={[i for i,r in enumerate(body) if len(r)!=2][:5]}")

sub=pd.read_csv(sub_path,dtype=str)
ids=sub["id"].tolist()
# --- ids ---
chk("Every test id present exactly once (set match)", set(ids)==set(test["id"].astype(str)),
    f"missing={len(set(test['id'].astype(str))-set(ids))} extra={len(set(ids)-set(test['id'].astype(str)))}")
chk("No duplicate ids", len(ids)==len(set(ids)), f"dups={len(ids)-len(set(ids))}")
chk("No missing/blank predictions", sub["repaired_sequence"].notna().all() and (sub["repaired_sequence"].str.len()>0).all())

# --- per-row structural checks ---
six=ok_order=ok_fmt=ok_val=0
bad_examples=[]
for seq in sub["repaired_sequence"]:
    parts=str(seq).split(";")
    if len(parts)==6: six+=1
    else:
        if len(bad_examples)<3: bad_examples.append(("not 6 parts",seq));
        continue
    names=[p.split("=",1)[0] for p in parts if "=" in p]
    if names==SLOTS: ok_order+=1
    else:
        if len(bad_examples)<3: bad_examples.append(("bad order",seq));
        continue
    if all(("=" in p and len(p.split("=",1))==2 and p.split("=",1)[1]!="") for p in parts): ok_fmt+=1
    else:
        if len(bad_examples)<3: bad_examples.append(("bad slot=value",seq));
        continue
    d=parse(seq)
    if all(d.get(s) in vocab[s] for s in SLOTS): ok_val+=1
    elif len(bad_examples)<3: bad_examples.append(("invalid value",seq))
chk("Exactly six ; -separated assignments in every row", six==len(sub), f"ok={six}/{len(sub)}")
chk("Slot names appear in exact order prep,activation,order,control,quench,workup", ok_order==len(sub), f"ok={ok_order}/{len(sub)}")
chk("Every assignment is slot=value (non-empty)", ok_fmt==len(sub), f"ok={ok_fmt}/{len(sub)}")
chk("Every value is one learned from public training examples", ok_val==len(sub), f"ok={ok_val}/{len(sub)}")
if bad_examples: print("  bad examples:",bad_examples)

# --- valid value sets for reference ---
print("\nValid value vocab per slot (from public train):")
for s in SLOTS: print(f"  {s}: {sorted(vocab[s])}")

print("\n=== SUMMARY:", "ALL PASS" if all(o for _,o,_ in results) else "SOME FAILED","===")
