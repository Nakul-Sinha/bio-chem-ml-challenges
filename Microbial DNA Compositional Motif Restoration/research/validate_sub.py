import pandas as pd, numpy as np
from pathlib import Path
ROOT=Path(__file__).resolve().parent.parent; DS=ROOT/"dataset"
sub=pd.read_csv(ROOT/"submission.csv",dtype=str); test=pd.read_csv(DS/"test.csv")
def chk(n,ok,d=""): print(f"[{'PASS' if ok else 'FAIL'}] {n}"+(f" -- {d}" if d else ""))
chk("columns exact", list(sub.columns)==["id","predicted_kmer_ids"], str(list(sub.columns)))
chk("row count == test", len(sub)==len(test), f"{len(sub)} vs {len(test)}")
chk("all test ids present once", set(sub["id"])==set(test["id"]) and sub["id"].is_unique)
chk("no NaN", not sub.isna().any().any())
bad=0; lens=[]
for s in sub["predicted_kmer_ids"]:
    ids=str(s).split(); lens.append(len(ids))
    if not (1<=len(ids)<=10): bad+=1; continue
    if len(ids)!=len(set(ids)): bad+=1; continue
    if not all(x.isdigit() and 1<=int(x)<=1024 for x in ids): bad+=1
chk("each cell 1-10 unique ints in 1..1024", bad==0, f"bad={bad}")
import collections
print("pred length dist:", dict(collections.Counter(lens)))
print("distinct first-preds:", sub["predicted_kmer_ids"].apply(lambda s: s.split()[0]).nunique())
print("rows fully identical to first row:", (sub["predicted_kmer_ids"]==sub["predicted_kmer_ids"].iloc[0]).sum())
print("sample:"); print(sub.head(3).to_string())
