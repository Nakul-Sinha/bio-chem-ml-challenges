"""EDA3: understand correction-notice phrasing -> slot, and clean tag->value mapping."""
import pandas as pd, numpy as np, re, collections
from pathlib import Path

DS = Path(__file__).resolve().parent.parent / "dataset"
train = pd.read_csv(DS / "train.csv")
test = pd.read_csv(DS / "test.csv")
SLOTS = ["prep","activation","order","control","quench","workup"]
NOTE_KEY = {"setup":"prep","activation":"activation","order":"order","control":"control","stop":"quench","cleanup":"workup"}
TAG = re.compile(r"\b([a-z]{3,6}-[a-z]{3,6}-\d+[A-Z])\b")

def parse_seq(s):
    d={}
    for p in str(s).split(";"):
        if "=" in p:
            k,v=p.split("=",1); d[k.strip()]=v.strip()
    return d

# --- Correction notice templates (mask tags) ---
def mask(cn):
    return TAG.sub("<TAG>", str(cn))
train["corr_tmpl"] = train["correction_notice"].apply(mask)
print("=== TOP CORRECTION TEMPLATES (train) ===")
for t,c in train["corr_tmpl"].value_counts().head(40).items():
    print(f"{c:4d}  {t}")

print("\n=== TOP CORRECTION TEMPLATES (test) ===")
test["corr_tmpl"] = test["correction_notice"].apply(mask)
for t,c in test["corr_tmpl"].value_counts().head(40).items():
    print(f"{c:4d}  {t}")
