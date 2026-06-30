import pandas as pd, numpy as np, collections, re
from pathlib import Path
DS=Path(__file__).resolve().parent.parent/"dataset"
train=pd.read_csv(DS/"train.csv"); test=pd.read_csv(DS/"test.csv")
vocab=pd.read_csv(DS/"solvent_vocabulary.csv")
print("train",train.shape,"test",test.shape,"vocab",vocab.shape)
print("vocab sources:",vocab["source"].value_counts().to_dict())
vocab_set=set(vocab["solvent_label"])

# label distributions
for col in ["temp_bin","time_bin","catalyst_present"]:
    print(f"\n=== {col} ===")
    print(train[col].value_counts(normalize=True).round(3).to_dict())

# solvent analysis
def sset(s):
    s=str(s)
    return [] if s in ("nan","NONE","") else s.split("|")
train["sl"]=train["solvent_labels"].apply(sset)
card=train["sl"].apply(len)
print("\n=== solvent set cardinality ===")
print(card.value_counts().sort_index().to_dict())
print("fraction NONE (empty):", (card==0).mean().round(3))
allsolv=collections.Counter(x for l in train["sl"] for x in l)
print("distinct solvents used in train:",len(allsolv))
print("top 15 solvents:",allsolv.most_common(15))
# any solvent labels outside vocab?
outside=set(allsolv)-vocab_set
print("solvent labels in train NOT in vocab:",len(outside), list(outside)[:10])
print("is OTHER used in train labels?", "OTHER" in allsolv)

# numeric feature distributions train vs test (detect shift)
print("\n=== numeric features train vs test (mean / p50 / p95) ===")
for col in ["reactant_count","product_count","smiles_length"]:
    tr=train[col]; te=test[col]
    print(f"  {col}: train {tr.mean():.1f}/{tr.median():.0f}/{tr.quantile(.95):.0f}  test {te.mean():.1f}/{te.median():.0f}/{te.quantile(.95):.0f}")

# Does catalyst_present correlate with metals in SMILES?
METALS=["Pd","Pt","Ni","Cu","Fe","Ru","Rh","Ir","Co","Zn","Mg","Ti","Pd+","[Pd]","Ag","Au","Mn","Mo","Sn","Li","Al","B"]
def has_metal(smi):
    left=str(smi).split(">>")[0]
    return int(any(re.search(r'\['+re.escape(m), left) or ("["+m+"]" in left) for m in ["Pd","Pt","Ni","Cu","Fe","Ru","Rh","Ir","Co","Ag","Au","Mn","Mo"]))
train["metal"]=train["reaction_smiles"].apply(has_metal)
print("\n=== catalyst vs transition-metal-in-SMILES ===")
print(pd.crosstab(train["catalyst_present"],train["metal"]))
print("catalyst_present rate:",train["catalyst_present"].mean().round(3))

# solvent appears literally as a reactant component?
def solvent_in_smiles(row):
    left=set(str(row["reaction_smiles"]).split(">>")[0].split("."))
    return any(s in left for s in row["sl"]) if row["sl"] else False
train["solv_in"]=train.apply(solvent_in_smiles,axis=1)
print("\nreactions where a true solvent appears as a reactant component:",train["solv_in"].mean().round(3))
