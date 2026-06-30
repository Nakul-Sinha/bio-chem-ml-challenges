"""rdkit-FREE token-level featurizer: hashed n-grams over CHEMICAL tokens (atoms/bonds/brackets)
of reactant/product SMILES + string descriptors. Diverse from char n-grams for ensembling."""
import numpy as np, pandas as pd, zlib, re
from pathlib import Path
DS=Path(__file__).resolve().parent.parent/"dataset"
CACHE=Path(__file__).resolve().parent/"cache"; CACHE.mkdir(exist_ok=True)
NBITS=2048; TOKGRAMS=(1,2,3)
TOKRE=re.compile(r"(\[[^\]]+\]|Br|Cl|Si|Se|Na|Li|Mg|Al|Ca|[BCNOFPSIbcnofps]|=|#|\(|\)|\.|-|\+|\\|/|@+|%[0-9]{2}|[0-9])")
REAGENT_PATS={"boron":"B","Pd":"[Pd","Cu":"[Cu","Ni":"[Ni","Pt":"[Pt","Fe":"[Fe","Zn":"[Zn",
 "Li":"[Li","Mg":"[Mg","Na":"[Na","K_ion":"[K+","Cs":"[Cs","Al":"[Al","Cl_anion":"[Cl-]",
 "carbonate":"C([O-])([O-])=O","hydroxide":"[OH-]","hydride":"[H-]","amine_base":"CCN(CC)CC",
 "DIPEA":"CCN(C(C)C)C(C)C","pyridine":"c1ccncc1","TFA":"OC(=O)C(F)(F)F","acid_Cl":"C(=O)Cl",
 "azide":"[N-]=[N+]=[N-]","BOC":"OC(=O)","phosphine":"P(c1ccccc1)","fluoride":"[F-]",
 "tosyl":"S(=O)(=O)","nitro":"[N+](=O)[O-]"}
def tok_vec(s):
    v=np.zeros(NBITS,dtype=np.float32); toks=TOKRE.findall(str(s))
    for n in TOKGRAMS:
        for i in range(len(toks)-n+1):
            v[zlib.crc32(("\x1f".join(toks[i:i+n])).encode())%NBITS]=1.0
    return v
def desc(smi):
    left,_,right=str(smi).partition(">>"); f=[left.count(".")+1,right.count(".")+1,len(left),len(right),len(str(smi))]
    for ch in ["Cl","Br","F","N","O","S","P","B","c","=","#","+","-","[","@","/"]: f.append(str(smi).count(ch))
    for k,pat in REAGENT_PATS.items(): f.append(left.count(pat))
    return np.array(f,dtype=np.float32)
def featurize(df, tag):
    fc=CACHE/f"feattok_{tag}.npz"
    if fc.exists(): return np.load(fc)["X"]
    R=np.zeros((len(df),NBITS),dtype=np.float32); P=np.zeros((len(df),NBITS),dtype=np.float32); D=[]
    for i,smi in enumerate(df["reaction_smiles"].astype(str)):
        left,_,right=smi.partition(">>"); R[i]=tok_vec(left); P[i]=tok_vec(right); D.append(desc(smi))
    X=np.hstack([R,P,P-R,np.vstack(D)]).astype(np.float32); np.savez_compressed(fc,X=X); return X
