"""rdkit-FREE featurizer: hashed character n-grams of reactant/product SMILES + string descriptors.
Same output shape contract as feat.py so the MLP pipeline is unchanged."""
import numpy as np, pandas as pd, zlib
from pathlib import Path
DS=Path(__file__).resolve().parent.parent/"dataset"
CACHE=Path(__file__).resolve().parent/"cache"; CACHE.mkdir(exist_ok=True)
NBITS=2048; NGRAMS=(2,3,4,5)

REAGENT_PATS={"boron":"B","Pd":"[Pd","Cu":"[Cu","Ni":"[Ni","Pt":"[Pt","Fe":"[Fe","Zn":"[Zn",
 "Li":"[Li","Mg":"[Mg","Na":"[Na","K_ion":"[K+","Cs":"[Cs","Al":"[Al","Cl_anion":"[Cl-]",
 "carbonate":"C([O-])([O-])=O","hydroxide":"[OH-]","hydride":"[H-]","amine_base":"CCN(CC)CC",
 "DIPEA":"CCN(C(C)C)C(C)C","pyridine":"c1ccncc1","TFA":"OC(=O)C(F)(F)F","acid_Cl":"C(=O)Cl",
 "azide":"[N-]=[N+]=[N-]","BOC":"OC(=O)","phosphine":"P(c1ccccc1)","fluoride":"[F-]",
 "tosyl":"S(=O)(=O)","nitro":"[N+](=O)[O-]"}

def ngram_vec(s):
    v=np.zeros(NBITS,dtype=np.float32); s=str(s)
    for n in NGRAMS:
        for i in range(len(s)-n+1):
            v[zlib.crc32(s[i:i+n].encode())%NBITS]=1.0
    return v

def desc(smi):
    left,_,right=str(smi).partition(">>"); f=[left.count(".")+1,right.count(".")+1,len(left),len(right),len(str(smi))]
    for ch in ["Cl","Br","F","N","O","S","P","B","c","=","#","+","-","[","@","/"]: f.append(str(smi).count(ch))
    for k,pat in REAGENT_PATS.items(): f.append(left.count(pat))
    return np.array(f,dtype=np.float32)

def featurize(df, tag):
    fcache=CACHE/f"featstr_{tag}.npz"
    if fcache.exists(): return np.load(fcache)["X"]
    R=np.zeros((len(df),NBITS),dtype=np.float32); P=np.zeros((len(df),NBITS),dtype=np.float32); D=[]
    for i,smi in enumerate(df["reaction_smiles"].astype(str)):
        left,_,right=smi.partition(">>"); R[i]=ngram_vec(left); P[i]=ngram_vec(right); D.append(desc(smi))
        if i%4000==0: print(f"  {tag} {i}/{len(df)}")
    X=np.hstack([R,P,P-R,np.vstack(D)]).astype(np.float32)
    np.savez_compressed(fcache,X=X); print(f"{tag} feat shape {X.shape}"); return X

if __name__=="__main__":
    tr=pd.read_csv(DS/"train.csv"); te=pd.read_csv(DS/"test.csv")
    print(featurize(tr,"train").shape, featurize(te,"test").shape)
