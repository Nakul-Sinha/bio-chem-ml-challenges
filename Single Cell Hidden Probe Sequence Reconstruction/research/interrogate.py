"""Thorough data interrogation for Single Cell Hidden Probe Sequence Reconstruction."""
import pandas as pd, numpy as np, re, json, collections
pd.set_option('display.width', 200); pd.set_option('display.max_columns', 50)

D = "../dataset/"
train = pd.read_csv(D+"train.csv")
test  = pd.read_csv(D+"test.csv")
samp  = pd.read_csv(D+"sample_submission.csv")
vocab = pd.read_csv(D+"sequence_vocabulary.csv")
obs   = pd.read_csv(D+"observed_panel.csv")
tgt   = pd.read_csv(D+"target_panel.csv")

print("="*70); print("SHAPES")
print("train", train.shape, "test", test.shape, "samp", samp.shape)
print("vocab", vocab.shape, "obs_panel", obs.shape, "tgt_panel", tgt.shape)

print("="*70); print("TRAIN HEAD")
for i in range(3):
    print("--- row", i, "id=", train.id.iloc[i])
    print("  SRC:", train.source_sequence.iloc[i])
    print("  TGT:", train.target_sequence.iloc[i])
print("TEST HEAD")
for i in range(2):
    print("  SRC:", test.source_sequence.iloc[i])
print("SAMPLE SUB HEAD"); print(samp.head(3).to_string())

print("="*70); print("VOCAB token_type counts"); print(vocab.token_type.value_counts())
print("--- vocab special/summary/metadata tokens ---")
for tt in ['special','summary']:
    print(tt, ":", vocab[vocab.token_type==tt].token.tolist())
print("--- metadata tokens (grouped by prefix) ---")
meta = vocab[vocab.token_type=='metadata'].token.tolist()
pref = collections.defaultdict(list)
for t in meta:
    p = t.split('_')[0]
    pref[p].append(t)
for p in sorted(pref): print(f"  {p}: n={len(pref[p])}  e.g. {pref[p][:6]}")
print("--- target_probe tokens ---")
print(sorted(vocab[vocab.token_type=='target_probe'].token.tolist()))

print("="*70); print("OBSERVED PANEL head + damage_group dist")
print(obs.head(4).to_string())
print("damage_group value_counts:"); print(obs.damage_group.value_counts())

print("="*70); print("TARGET PANEL (full) with derived active fraction")
tgt['active_frac'] = tgt.train_active_count/len(train)
tgt['b2_used'] = tgt.train_b2_count>0
print(tgt.to_string())
print("targets with B2 used:", tgt[tgt.b2_used].target_token_prefix.tolist())

# ---- Parse source sequences ----
print("="*70); print("SOURCE TOKEN STRUCTURE")
def parse_src(s): return s.split()
src_lens = train.source_sequence.map(lambda s: len(s.split()))
print("source length: min/median/max", src_lens.min(), int(src_lens.median()), src_lens.max())
# collect prefixes present per row and the metadata token values
first = train.source_sequence.iloc[0].split()
print("first row tokens (%d):"%len(first), first)

# Distribution of metadata token values across train
metacols = collections.defaultdict(collections.Counter)
Otokens_present = collections.Counter()
for s in train.source_sequence:
    toks = s.split()
    for t in toks:
        pfx = t.split('_')[0]
        if pfx in ('DOM','ASSAY','CTX','GENCTX','COND','SEX','STAGE','SAMPLE','PANEL','TOTAL','NZ'):
            metacols[pfx][t]+=1
        elif re.match(r'^O\d{3}$', pfx):
            Otokens_present[pfx]+=1
for k in ['DOM','ASSAY','CTX','GENCTX','COND','SEX','STAGE','SAMPLE','PANEL']:
    c = metacols[k]
    print(f"  {k}: {len(c)} distinct; top: {c.most_common(6)}")
print("  #distinct O-prefixes present:", len(Otokens_present), "(expect 80)")
print("  O-prefix presence min/max count:", min(Otokens_present.values()), max(Otokens_present.values()))

# PANEL_DAMAGE analysis
print("="*70); print("PANEL condition distribution (train vs test)")
def panel_of(s):
    for t in s.split():
        if t.startswith('PANEL'): return t
    return 'MISSING'
train['panel'] = train.source_sequence.map(panel_of)
test['panel']  = test.source_sequence.map(panel_of)
print("TRAIN:", train.panel.value_counts().to_dict())
print("TEST :", test.panel.value_counts().to_dict())

# DOM distribution train vs test (domain shift detection)
def tok_of(s, pfx):
    for t in s.split():
        if t.startswith(pfx+'_'): return t
    return pfx+'_NA'
for pfx in ['DOM','ASSAY','CTX','GENCTX','COND','SEX','STAGE','SAMPLE']:
    tr = train.source_sequence.map(lambda s: tok_of(s,pfx))
    te = test.source_sequence.map(lambda s: tok_of(s,pfx))
    trs, tes = set(tr.unique()), set(te.unique())
    print(f"  {pfx}: train_distinct={len(trs)} test_distinct={len(tes)} test_only={sorted(tes-trs)[:5]} train_only_n={len(trs-tes)}")

# ---- Parse target sequences ----
print("="*70); print("TARGET SEQUENCE STRUCTURE")
none_ct = (train.target_sequence.str.strip()=='NONE').sum()
print("NONE targets in train:", none_ct, f"({none_ct/len(train)*100:.1f}%)")
tgt_lens = train.target_sequence.map(lambda s: 0 if s.strip()=='NONE' else len(s.split()))
print("active target-set size: min/median/mean/max", tgt_lens.min(), int(tgt_lens.median()), round(tgt_lens.mean(),2), tgt_lens.max())
print("target-set size histogram:", dict(sorted(collections.Counter(tgt_lens).items())))

# Validate token format and canonical order
BIN_ORDER = {'B3':0,'B2':1,'B1':2}
def canon_key(t):
    m = re.match(r'^T(\d\d)_B([123])$', t)
    assert m, "bad token "+t
    ti = int(m.group(1)); b = int(m.group(2))
    return (-b, ti)   # descending bin, then ascending index
bad=0; noncanon=0; dup=0
allbins=collections.Counter()
for s in train.target_sequence:
    if s.strip()=='NONE': continue
    toks = s.split()
    for t in toks:
        if not re.match(r'^T(0\d|1[0-5])_B[123]$', t): bad+=1
        allbins[t.split('_')[1]]+=1
    if len(set(toks))!=len(toks): dup+=1
    keys = [canon_key(t) for t in toks]
    if keys != sorted(keys): noncanon+=1
print("malformed tokens:", bad, "| rows w/ duplicate tokens:", dup, "| rows violating canonical order:", noncanon)
print("bin frequency in targets:", dict(allbins))

# co-occurrence: how many distinct target signatures
sigs = train.target_sequence.map(lambda s: s.strip())
print("distinct target signatures:", sigs.nunique(), "of", len(train))
print("top signatures:", sigs.value_counts().head(8).to_dict())

# ---- Signal check: does source predict target? mutual info-ish quick check ----
print("="*70); print("QUICK SIGNAL CHECK: O-token quant value vs a target activation")
# Build a numeric matrix of O-values, correlate with T00 active
def ovec(s):
    v = np.full(80, -1, dtype=float)
    for t in s.split():
        m = re.match(r'^O(\d{3})_Q(\d+)$', t)
        if m: v[int(m.group(1))] = int(m.group(2))
    return v
OV = np.stack(train.source_sequence.map(ovec).values)
print("O-value matrix", OV.shape, "min/max", OV.min(), OV.max(), "| any -1 (missing)?", (OV==-1).any(), "count:", int((OV==-1).sum()))
# target active indicator
def tset(s):
    if s.strip()=='NONE': return set()
    return set(int(t[1:3]) for t in s.split())
TS = train.target_sequence.map(tset)
for ti in [0,5,13]:
    y = TS.map(lambda s: ti in s).astype(int).values
    # correlation of each O with y, report top abs
    cors=[]
    for j in range(80):
        col=OV[:,j]
        if col.std()>0:
            cors.append((abs(np.corrcoef(col,y)[0,1]), j))
    cors.sort(reverse=True)
    print(f"  T{ti:02d} active_rate={y.mean():.2f} top-|corr| O-features:", [(round(c,3),f'O{j:03d}') for c,j in cors[:5]])
print("DONE")
