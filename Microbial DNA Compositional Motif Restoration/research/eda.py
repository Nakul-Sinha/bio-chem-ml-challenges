import pandas as pd, numpy as np, collections
from pathlib import Path
DS=Path(__file__).resolve().parent.parent/"dataset"
train=pd.read_csv(DS/"train.csv"); test=pd.read_csv(DS/"test.csv")
print("train",train.shape,"test",test.shape)
def toks(s): return [int(x) for x in str(s).split()]
train["kseq"]=train["kmer_seq"].apply(toks); train["gc"]=train["genome_context"].apply(toks)
test["gc"]=test["genome_context"].apply(toks)
test["mseq"]=test["masked_kmer_seq"].apply(lambda s:[x for x in str(s).split()])

# vocab
allk=collections.Counter(x for l in train["kseq"] for x in l)
allgc=collections.Counter(x for l in train["gc"] for x in l)
print("distinct tokens in train kmer_seq:",len(allk),"min/max:",min(allk),max(allk))
print("distinct tokens in genome_context:",len(allgc))
print("kmer_seq length dist:",collections.Counter(len(l) for l in train["kseq"]))
print("gc length dist:",collections.Counter(len(l) for l in train["gc"]))

# token id range -> vocab 1024?
allids=set(allk)|set(allgc)
print("union token ids:",len(allids),"range",min(allids),max(allids))

# global token frequency by RANK position in kmer_seq (position encodes rank)
print("\n=== most-frequent token at each rank position (0..15) ===")
for pos in range(16):
    c=collections.Counter(l[pos] for l in train["kseq"] if len(l)>pos)
    top=c.most_common(3)
    print(f" pos{pos}: {top}")

# genome grouping: cluster by genome_context. Use sorted-set as key.
def gckey(l): return tuple(sorted(set(l)))
train["gck"]=train["gc"].apply(gckey)
print("\nunique exact gc-set keys (train):",train["gck"].nunique(),"of",len(train))
# overlap-based: how similar are sibling gcs? sample
sizes=train.groupby("gck").size()
print("contigs per gc-set-key: mean",sizes.mean().round(2),"max",sizes.max(),"singletons",(sizes==1).sum())

# test mask_index distribution
print("\ntest mask_index dist:",collections.Counter(test["mask_index"]).most_common())

# how often is gold token in genome_context? (train: simulate masking each position)
in_gc=0; tot=0
for _,r in train.iterrows():
    gcset=set(r["gc"])
    for pos,t in enumerate(r["kseq"]):
        tot+=1; in_gc+= (t in gcset)
print(f"\nfrac of kmer_seq tokens present in own genome_context: {in_gc/tot:.3f}")
# restricted to mid positions (where masking happens) -- test mask_index range
mid=range(2,14)
in_gc=0; tot=0
for _,r in train.iterrows():
    gcset=set(r["gc"])
    for pos in mid:
        if pos<len(r["kseq"]): tot+=1; in_gc+=(r["kseq"][pos] in gcset)
print(f"frac present in gc (mid positions 2-13): {in_gc/tot:.3f}")
# most-frequent floor: predict global top-10
top10=[t for t,_ in allk.most_common(10)]
print("\nglobal top-10 tokens:",top10)
