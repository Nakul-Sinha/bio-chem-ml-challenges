"""Verify damage mechanism, group structure, near-duplicate leakage, signature rarity."""
import pandas as pd, numpy as np, re, collections
D="../dataset/"
train=pd.read_csv(D+"train.csv"); test=pd.read_csv(D+"test.csv")
obs=pd.read_csv(D+"observed_panel.csv")
dmg_group = obs.set_index('observed_index').damage_group.to_dict()  # O-index -> group 0..3

def parse(s):
    meta={}; ov=np.zeros(80,dtype=np.int16); tot=nz=None; panel=None
    for t in s.split():
        if t.startswith('O') and '_Q' in t:
            i=int(t[1:4]); ov[i]=int(t.split('_Q')[1])
        elif t.startswith('PANEL'): panel=t
        elif t.startswith('TOTAL_Q'): tot=int(t.split('_Q')[1])
        elif t.startswith('NZ_Q'): nz=int(t.split('_Q')[1])
        else:
            p=t.split('_')[0]; meta[p]=t
    return meta,ov,tot,nz,panel

rows=[parse(s) for s in train.source_sequence]
OV=np.stack([r[1] for r in rows])
panels=[r[4] for r in rows]

print("="*70);print("DAMAGE MECHANISM VERIFICATION")
# For each damage panel, check which O-indices are always zero
for dk in ['PANEL_DAMAGE_00','PANEL_DAMAGE_01','PANEL_DAMAGE_02','PANEL_DAMAGE_03']:
    idx=[i for i,p in enumerate(panels) if p==dk]
    sub=OV[idx]
    always_zero=set(np.where((sub==0).all(axis=0))[0])
    grp=int(dk[-1])
    expect=set(i for i,g in dmg_group.items() if g==grp)
    print(f"{dk}: n={len(idx)}  always-zero O count={len(always_zero)}  matches damage_group{grp}? {always_zero==expect}  extra_alwayszero={len(always_zero-expect)}")
# Normal rows: are damage-group features nonzero sometimes?
idxn=[i for i,p in enumerate(panels) if p=='PANEL_NORMAL']
subn=OV[idxn]
print("NORMAL rows: mean #nonzero O per row =", round((subn>0).sum(1).mean(),1))

print("="*70);print("METADATA GROUP STRUCTURE (COND x SEX x STAGE x PANEL)")
def meta_of(s,p):
    for t in s.split():
        if t.startswith(p+'_'): return t
    return p+'_NA'
for keys in [['COND'],['STAGE'],['COND','SEX','STAGE'],['COND','SEX','STAGE','PANEL']]:
    g=train.source_sequence.map(lambda s: tuple(meta_of(s,k) for k in keys))
    vc=g.value_counts()
    print(f"  group by {keys}: {vc.nunique() if False else len(vc)} groups; sizes min/med/max = {vc.min()}/{int(vc.median())}/{vc.max()}")

print("="*70);print("NEAR-DUPLICATE / LEAKAGE STRUCTURE")
# cosine-normalized presence pattern; find near-duplicate source rows
Xb=(OV>0).astype(np.float32)  # binary presence
norm=np.linalg.norm(Xb,axis=1,keepdims=True); Xn=Xb/np.clip(norm,1e-9,None)
# sample 500 rows, compute max off-diagonal cosine
import numpy.random as npr
rng=npr.default_rng(0); samp=rng.choice(len(train),400,replace=False)
S=Xn[samp]@Xn.T  # 400 x N
for k in range(len(samp)): S[k,samp[k]]=-1
mx=S.max(1)
print("presence-cosine to nearest other row: median=%.3f  p90=%.3f  frac>0.95=%.3f  frac>0.99=%.3f"%(
    np.median(mx), np.quantile(mx,0.9), (mx>0.95).mean(), (mx>0.99).mean()))
# Also on quantized values
Xv=OV.astype(np.float32); nv=np.linalg.norm(Xv,axis=1,keepdims=True); Xvn=Xv/np.clip(nv,1e-9,None)
Sv=Xvn[samp]@Xvn.T
for k in range(len(samp)): Sv[k,samp[k]]=-1
mxv=Sv.max(1)
print("value-cosine   to nearest other row: median=%.3f  p90=%.3f  frac>0.95=%.3f  frac>0.99=%.3f"%(
    np.median(mxv), np.quantile(mxv,0.9),(mxv>0.95).mean(),(mxv>0.99).mean()))

print("="*70);print("TARGET vs METADATA dependence (does target depend on COND/SEX/STAGE?)")
def tset(s):
    return frozenset() if s.strip()=='NONE' else frozenset(t for t in s.split())
TS=train.target_sequence.map(tset)
def active_idx(s):
    return set() if s.strip()=='NONE' else set(int(t[1:3]) for t in s.split())
AI=train.target_sequence.map(active_idx)
for key in ['COND','SEX','STAGE','PANEL']:
    g=train.source_sequence.map(lambda s: meta_of(s,key))
    print(f"  -- mean active-set size by {key}:")
    tmp=pd.DataFrame({'g':g,'n':AI.map(len)})
    print("    ", tmp.groupby('g').n.mean().round(2).to_dict())

print("="*70);print("SIGNATURE RARITY")
sig=train.target_sequence.map(lambda s:s.strip())
vc=sig.value_counts()
print("unique signatures:",len(vc),"| singletons:",(vc==1).sum(),f"({(vc==1).sum()/len(train)*100:.0f}% of rows are unique-signature)")
# token rarity: B2 tokens and rare (target,bin)
alltok=collections.Counter()
for s in train.target_sequence:
    if s.strip()!='NONE':
        for t in s.split(): alltok[t]+=1
rare_tok=[t for t,c in alltok.items() if c<40]
print("rarest tokens:",sorted(alltok.items(),key=lambda x:x[1])[:12])
print("B2 tokens present:",{t:c for t,c in alltok.items() if '_B2' in t})

print("="*70);print("PANEL damage vs target (does damage change the TARGET? it shouldn't - target is hidden truth)")
tmp=pd.DataFrame({'panel':[p for p in panels],'n':AI.map(len)})
print(tmp.groupby('panel').n.agg(['mean','count']).round(2).to_dict())
print("DONE")
