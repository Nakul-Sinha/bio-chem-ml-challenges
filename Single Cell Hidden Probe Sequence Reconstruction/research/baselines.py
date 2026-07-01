"""Prior floor + logistic-regression tabular baseline on honest grouped CV.
Reports subset-weighted score, and grouped-vs-random gap (leakage check)."""
import numpy as np, pandas as pd, sys
from sklearn.model_selection import GroupKFold, KFold
from sklearn.linear_model import LogisticRegression
import srlib as L

D = "../dataset/"
train = pd.read_csv(D+"train.csv"); test = pd.read_csv(D+"test.csv")

# ---- features ----
def build_X(df):
    OV = np.zeros((len(df), 80), np.float32)
    TOT = np.zeros(len(df), np.float32); NZ = np.zeros(len(df), np.float32)
    DMG = np.full(len(df), -1, np.int64)       # damage group -1..3
    COND = []; SEX = []; STAGE = []
    for i, s in enumerate(df['source_sequence'].values):
        ov, tot, nz, panel, meta = L.parse_source(s)
        OV[i] = ov; TOT[i] = tot; NZ[i] = nz; DMG[i] = L.panel_damage_group(panel)
        COND.append(meta.get('COND','?')); SEX.append(meta.get('SEX','?')); STAGE.append(meta.get('STAGE','?'))
    return OV, TOT, NZ, DMG, np.array(COND), np.array(SEX), np.array(STAGE)

OV, TOT, NZ, DMG, COND, SEX, STAGE = build_X(train)
# targets: 16-dim bins
Y = np.stack([L.parse_target(s) for s in train['target_sequence']])   # (N,16) in {0,1,2,3}
N = len(train)
print("X", OV.shape, "Y", Y.shape, "active per row mean %.2f"%(Y>0).sum(1).mean())

# ---- groups + subset flags ----
gid, gkeys = L.make_groups(train)
gsize = pd.Series(gkeys).value_counts()
small_groups = set(gsize[gsize <= gsize.quantile(0.35)].index)   # rare contexts -> 'shifted' proxy
rare_tokens, tokcount = L.rare_token_set(train, thresh=160)
print("n groups=%d  small(shifted) groups=%d covering %d rows  | rare tokens=%d"%(
    len(gsize), len(small_groups), sum(gsize[g] for g in small_groups), len(rare_tokens)))

true_seqs = [L.target_tokens(s) for s in train['target_sequence']]
allflags = L.subset_flags(train, np.arange(N), rare_tokens, small_groups, gkeys)
print("flag coverage: damaged=%d rare=%d shifted=%d"%(allflags['damaged'].sum(), allflags['rare'].sum(), allflags['shifted'].sum()))

def eval_oof(pred_bins, tag):
    pred_seqs = [L.bins_to_seq(pred_bins[i]) for i in range(N)]
    r = L.weighted_score(pred_seqs, true_seqs, allflags)
    print(f"  [{tag}] FINAL={r['final']:.4f} | all={r['all']:.4f} shifted={r['shifted']:.4f} "
          f"damaged={r['damaged']:.4f} rare={r['rare']:.4f}")
    return r

# ================= 1) PRIOR FLOOR (source-independent) =================
print("="*70); print("PRIOR FLOOR: predict top-k most-active targets, each at its modal bin")
active_count = (Y > 0).sum(0)
order = np.argsort(-active_count)                # targets by activation freq
# modal active bin per target
modal_bin = np.zeros(16, int)
for t in range(16):
    vals = Y[:, t][Y[:, t] > 0]
    modal_bin[t] = np.bincount(vals).argmax() if len(vals) else 1
best = (None, -1)
for k in range(0, 17):
    pb = np.zeros((N, 16), int)
    for t in order[:k]:
        pb[:, t] = modal_bin[t]
    r = eval_oof(pb, f"prior k={k}")
    if r['final'] > best[1]:
        best = (k, r['final'])
print(f"  >> best prior: k={best[0]} FINAL={best[1]:.4f}")

# ================= 2) LOGISTIC REGRESSION per target, grouped CV =================
print("="*70); print("LOGISTIC REGRESSION (per-target 4-way), GROUPED 5-fold CV")
# feature matrix: O-values scaled + metadata one-hots + tot/nz + damage onehot
def feat(OV, TOT, NZ, DMG, COND, SEX, STAGE):
    conds = np.array([['COND_004','COND_025'].index(c) if c in ['COND_004','COND_025'] else 0 for c in COND])
    sexs  = np.array([1 if s=='SEX_006' else 0 for s in SEX])
    st_u = sorted(set(STAGE)); st_map={v:i for i,v in enumerate(st_u)}
    stg = np.array([st_map[s] for s in STAGE])
    stoh = np.eye(len(st_u))[stg]
    dmoh = np.eye(5)[DMG+1]   # -1..3 -> 0..4
    X = np.concatenate([OV/31.0, (TOT/63.0)[:,None], (NZ/60.0)[:,None],
                        conds[:,None], sexs[:,None], stoh, dmoh], axis=1).astype(np.float32)
    return X
X = feat(OV, TOT, NZ, DMG, COND, SEX, STAGE)
print("feature dim", X.shape[1])

def run_cv(splitter, groups, tag):
    oof = np.zeros((N, 16, 4), np.float32)   # predicted probs
    for tr, va in splitter.split(X, groups=groups):
        for t in range(16):
            classes = [0,1,2,3] if t in L.B2_TARGETS else [0,1,3]
            yt = Y[tr, t].copy()
            # map to available classes; unseen bins shouldn't occur
            clf = LogisticRegression(max_iter=300, C=1.0, class_weight=None)
            # skip degenerate
            if len(np.unique(yt)) < 2:
                oof[va, t, yt[0] if len(yt) else 0] = 1.0; continue
            clf.fit(X[tr], yt)
            proba = clf.predict_proba(X[va])
            for ci, c in enumerate(clf.classes_):
                oof[va, t, c] = proba[:, ci]
    return oof

gkf = GroupKFold(n_splits=5)
oof = run_cv(gkf, gid, "grouped")
# decode: argmax over 4 bins
pred_bins = oof.argmax(2)
eval_oof(pred_bins, "LR grouped argmax")

# random CV for leakage-gap comparison
kf = KFold(n_splits=5, shuffle=True, random_state=0)
oofr = np.zeros((N,16,4),np.float32)
for tr,va in kf.split(X):
    for t in range(16):
        yt=Y[tr,t]
        if len(np.unique(yt))<2:
            oofr[va,t,yt[0] if len(yt) else 0]=1.0; continue
        clf=LogisticRegression(max_iter=300,C=1.0); clf.fit(X[tr],yt)
        pr=clf.predict_proba(X[va])
        for ci,c in enumerate(clf.classes_): oofr[va,t,c]=pr[:,ci]
eval_oof(oofr.argmax(2), "LR RANDOM argmax (optimistic)")

# save grouped oof probs for threshold tuning later
np.save("oof_lr_grouped.npy", oof)
np.save("Y.npy", Y)
print("saved oof_lr_grouped.npy")
print("DONE")
