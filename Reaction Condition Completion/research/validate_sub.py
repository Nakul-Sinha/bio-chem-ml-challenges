"""Validate C2 submission format + sanity distributions."""
import pandas as pd, numpy as np, collections
from pathlib import Path
ROOT=Path(__file__).resolve().parent.parent; DS=ROOT/"dataset"
sub=pd.read_csv(ROOT/"submission.csv",dtype=str)
test=pd.read_csv(DS/"test.csv"); samp=pd.read_csv(DS/"sample_submission.csv")
vocab=set(pd.read_csv(DS/"solvent_vocabulary.csv")["solvent_label"])
TEMPS={"cryogenic","cold","room","warm","hot"}; TIMES={"very_short","short","medium","long","very_long","overnight"}
def chk(n,ok,d=""): print(f"[{'PASS' if ok else 'FAIL'}] {n}"+(f" -- {d}" if d else ""))

chk("columns exact", list(sub.columns)==["reaction_id","pred_solvents","pred_temp_bin","pred_time_bin","pred_catalyst_present"], str(list(sub.columns)))
chk("row count == test", len(sub)==len(test), f"{len(sub)} vs {len(test)}")
chk("all test ids present once", set(sub["reaction_id"])==set(test["reaction_id"]) and sub["reaction_id"].is_unique)
chk("no NaN", not sub.isna().any().any())
chk("temp valid", set(sub["pred_temp_bin"])<=TEMPS, str(set(sub["pred_temp_bin"])-TEMPS))
chk("time valid", set(sub["pred_time_bin"])<=TIMES, str(set(sub["pred_time_bin"])-TIMES))
chk("catalyst in {0,1}", set(sub["pred_catalyst_present"])<={"0","1"}, str(set(sub["pred_catalyst_present"])))
# solvents: each label in vocab, non-empty, dup-free within cell
bad=0; allbad=set()
for s in sub["pred_solvents"]:
    parts=str(s).split("|")
    if len(parts)==0 or s=="" or s=="nan": bad+=1; continue
    if len(parts)!=len(set(parts)): bad+=1; continue
    for p in parts:
        if p not in vocab: allbad.add(p); bad+=1; break
chk("solvent labels valid/nonempty/dupfree", bad==0, f"bad={bad} unknown={list(allbad)[:5]}")

print("\n=== distributions ===")
for c in ["pred_temp_bin","pred_time_bin","pred_catalyst_present"]:
    print(c, sub[c].value_counts(normalize=True).round(3).to_dict())
card=sub["pred_solvents"].apply(lambda s:0 if s=="NONE" else len(str(s).split("|")))
print("solvent set sizes:", card.value_counts().sort_index().to_dict())
print("frac NONE:", (sub["pred_solvents"]=="NONE").mean().round(3))
print("top predicted solvents:", collections.Counter(x for s in sub["pred_solvents"] if s!="NONE" for x in s.split("|")).most_common(8))
print("\nSUMMARY: submission shape",sub.shape)
