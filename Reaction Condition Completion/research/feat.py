"""Featurize reactions: Morgan FPs of reactants/products + difference + descriptors. Cached."""
import numpy as np, pandas as pd, re
from pathlib import Path
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

DS=Path(__file__).resolve().parent.parent/"dataset"
CACHE=Path(__file__).resolve().parent/"cache"; CACHE.mkdir(exist_ok=True)
NBITS=2048; RAD=2

def morgan(smi):
    m=Chem.MolFromSmiles(smi) if smi else None
    if m is None: return np.zeros(NBITS,dtype=np.float32)
    fp=AllChem.GetMorganFingerprintAsBitVect(m,RAD,nBits=NBITS)
    a=np.zeros(NBITS,dtype=np.float32); DataStructs.ConvertToNumpyArray(fp,a); return a

# common reagent SMARTS-ish substrings (presence features informative for conditions)
REAGENT_PATS={
 "boron":"B","Pd":"[Pd","Cu":"[Cu","Ni":"[Ni","Pt":"[Pt","Fe":"[Fe","Zn":"[Zn",
 "Li":"[Li","Mg":"[Mg","Na":"[Na","K_ion":"[K+","Cs":"[Cs","Al":"[Al",
 "Cl_anion":"[Cl-]","carbonate":"C([O-])([O-])=O","hydroxide":"[OH-]","hydride":"[H-]",
 "amine_base_Et3N":"CCN(CC)CC","DIPEA":"CCN(C(C)C)C(C)C","pyridine":"c1ccncc1",
 "TFA":"OC(=O)C(F)(F)F","acid_Cl":"C(=O)Cl","azide":"[N-]=[N+]=[N-]","BOC":"OC(=O)",
 "phosphine":"P(c1ccccc1)","fluoride":"[F-]","tosyl":"S(=O)(=O)","nitro":"[N+](=O)[O-]",
}
def desc(smi):
    left,_,right=str(smi).partition(">>")
    feats=[]
    lc=left.count(".")+1; rc=right.count(".")+1
    feats+= [lc, rc, len(left), len(right), len(str(smi))]
    # atom-ish counts on full string (cheap, robust)
    for ch in ["Cl","Br","F","N","O","S","P","B","c","=","#","+","-","[","@","/"]:
        feats.append(str(smi).count(ch))
    for k,pat in REAGENT_PATS.items():
        feats.append(left.count(pat))
    return np.array(feats,dtype=np.float32)

def featurize(df, tag):
    fcache=CACHE/f"feat_{tag}.npz"
    if fcache.exists():
        d=np.load(fcache); return d["X"]
    R=np.zeros((len(df),NBITS),dtype=np.float32)
    P=np.zeros((len(df),NBITS),dtype=np.float32)
    D=[]
    for i,smi in enumerate(df["reaction_smiles"].astype(str)):
        left,_,right=smi.partition(">>")
        R[i]=morgan(left); P[i]=morgan(right); D.append(desc(smi))
        if i%2000==0: print(f"  {tag} {i}/{len(df)}")
    D=np.vstack(D)
    diff=P-R
    X=np.hstack([R,P,diff,D]).astype(np.float32)
    np.savez_compressed(fcache,X=X)
    print(f"{tag} feat shape {X.shape}")
    return X

if __name__=="__main__":
    tr=pd.read_csv(DS/"train.csv"); te=pd.read_csv(DS/"test.csv")
    Xtr=featurize(tr,"train"); Xte=featurize(te,"test")
    print("done",Xtr.shape,Xte.shape)
