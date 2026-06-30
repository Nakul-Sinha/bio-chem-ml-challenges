"""EDA for Reaction Protocol Silent-Edit Repair."""
import pandas as pd, numpy as np, json, re, collections
from pathlib import Path

DS = Path(__file__).resolve().parent.parent / "dataset"
train = pd.read_csv(DS / "train.csv")
test = pd.read_csv(DS / "test.csv")
samp = pd.read_csv(DS / "sample_submission.csv")
print("shapes:", train.shape, test.shape, samp.shape)
print("train cols:", train.columns.tolist())
print("test cols:", test.columns.tolist())

SLOTS = ["prep", "activation", "order", "control", "quench", "workup"]

def parse_seq(s):
    d = {}
    for part in str(s).split(";"):
        if "=" in part:
            k, v = part.split("=", 1)
            d[k.strip()] = v.strip()
    return d

# 1. value vocab per slot
parsed = train["repaired_sequence"].apply(parse_seq)
vocab = {sl: collections.Counter() for sl in SLOTS}
for d in parsed:
    for sl in SLOTS:
        vocab[sl][d.get(sl, "<MISSING>")] += 1
print("\n=== VALUE VOCAB PER SLOT (train outputs) ===")
for sl in SLOTS:
    print(f"\n[{sl}] n_unique={len(vocab[sl])}")
    for v, c in vocab[sl].most_common():
        print(f"    {v:30s} {c:5d}  ({c/len(train)*100:.1f}%)")

# 2. all outputs structurally valid?
bad = 0
for d in parsed:
    if any(sl not in d for sl in SLOTS):
        bad += 1
print(f"\noutputs missing a slot: {bad}")

# 3. family header extraction
def get_family(prompt):
    first = str(prompt).split("\n")[0]
    # patterns: "groups this run under X.", "Header family: X.", "reaction family is logged as X.", "Family tag: X"
    for pat in [r"under ([a-z ]+)\.", r"family[: ]+([a-z ]+)\.", r"family is logged as ([a-z ]+)\.",
                r"logged as ([a-z ]+)\.", r"family tag[: ]+([a-z ]+)\.", r"grouped as ([a-z ]+)\."]:
        m = re.search(pat, first, re.I)
        if m:
            return m.group(1).strip().lower()
    return "??:" + first[:60]

train["family"] = train["prompt"].apply(get_family)
test["family"] = test["prompt"].apply(get_family)
print("\n=== FAMILIES (train) ===")
print("n train families:", train["family"].nunique())
for f, c in train["family"].value_counts().items():
    print(f"    {f:30s} {c}")
print("\n=== FAMILIES (test) ===")
print("n test families:", test["family"].nunique())
unseen = set(test["family"]) - set(train["family"])
print("test families NOT in train:", unseen)
print("first-line examples that failed to parse family (train):")
for f in train["family"].unique():
    if f.startswith("??"):
        print("   ", f)
