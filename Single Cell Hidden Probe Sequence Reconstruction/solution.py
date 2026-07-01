"""
Single Cell Hidden Probe Sequence Reconstruction — self-contained, self-timing solution.

Problem reduces to per-target ordinal prediction (16 hidden probes T00..T15, each
absent/B1/B2/B3) + deterministic canonical decode. Target *selection* dominates the score;
per-target active-AUC ~0.64 is a data ceiling (proven: LR==GBDT==kNN==MLP, joint modelling
adds nothing). Levers are ensemble calibration, a high-recall per-target-threshold decode,
modal-bin defaulting (argmax only for the 5 signal-bearing bin targets), and damage
augmentation for the damaged-panel subset.

Env: torch + numpy + pandas only (no sklearn/lightgbm, no pip, no external weights). Trains
from scratch at inference (data is tiny). Writes ./working/submission.csv and validates it.
Self-times to stay well under the A10G / 30-min budget.
"""
import os, sys, time, glob, re, math, warnings
import numpy as np, pandas as pd
import torch, torch.nn as nn, torch.nn.functional as F
warnings.filterwarnings('ignore'); np.seterr(all='ignore')

T_START = time.time()
BUDGET_S = float(os.environ.get("SOLUTION_BUDGET_S", "1500"))   # 25 min default, < 30 min wall
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
if DEV == 'cuda':   # verify CUDA actually runs kernels (some envs report available but mismatch arch)
    try:
        _t = torch.zeros(4, device='cuda'); float((_t + 1).sum().item())
    except Exception as e:
        print("[warn] CUDA present but unusable -> CPU fallback:", repr(e)[:120], flush=True); DEV = 'cpu'
RNG = np.random.default_rng(0)
torch.manual_seed(0)

# ----------------------------- data discovery -----------------------------
def find_data():
    try: here = os.path.dirname(os.path.abspath(__file__))
    except Exception: here = os.getcwd()
    cands = [os.getcwd(), here, os.path.join(here, "dataset"),
             os.path.join(os.getcwd(), "dataset"), "/kaggle/input"]
    cands += glob.glob("/kaggle/input/*") + glob.glob(os.path.join(here, "*"))
    for d in cands:
        try:
            if os.path.isfile(os.path.join(d, "train.csv")) and os.path.isfile(os.path.join(d, "test.csv")):
                return d
        except Exception:
            pass
    # recursive fallback: walk common roots for a dir containing both csvs
    for root in ["/kaggle/input", os.getcwd(), here]:
        if not os.path.isdir(root): continue
        for dp, _, fns in os.walk(root):
            if "train.csv" in fns and "test.csv" in fns:
                return dp
    raise FileNotFoundError("could not locate train.csv/test.csv")

DATA = find_data()
print("[data]", DATA, flush=True)
train = pd.read_csv(os.path.join(DATA, "train.csv"))
test  = pd.read_csv(os.path.join(DATA, "test.csv"))
try:
    obs = pd.read_csv(os.path.join(DATA, "observed_panel.csv"))
    DG = obs.set_index('observed_index').damage_group.reindex(range(80)).fillna(-1).astype(int).values
except Exception:
    DG = np.array([i % 4 for i in range(80)])   # fallback: 4 contiguous-ish groups

# ----------------------------- parsing / features -----------------------------
TOKEN_RE = re.compile(r'^T(\d\d)_B([123])$')
def parse_source(s):
    ov = np.zeros(80, np.float32); tot = 0.0; nz = 0.0; panel = 'PANEL_NORMAL'; meta = {}
    for t in s.split():
        if t[0] == 'O' and '_Q' in t: ov[int(t[1:4])] = float(t.split('_Q')[1])
        elif t.startswith('PANEL'): panel = t
        elif t.startswith('TOTAL_Q'): tot = float(t.split('_Q')[1])
        elif t.startswith('NZ_Q'): nz = float(t.split('_Q')[1])
        else: meta[t.split('_')[0]] = t
    return ov, tot, nz, panel, meta
def dmg_group(panel):
    return int(panel.split('_')[-1]) if panel.startswith('PANEL_DAMAGE_') else -1
def parse_target(s):
    b = np.zeros(16, np.int64); s = s.strip()
    if s and s != 'NONE':
        for t in s.split():
            m = TOKEN_RE.match(t)
            if m: b[int(m.group(1))] = int(m.group(2))
    return b

def parse_df(df):
    n = len(df); OV = np.zeros((n,80),np.float32); TOT=np.zeros(n,np.float32); NZ=np.zeros(n,np.float32)
    DMG=np.full(n,-1,np.int64); COND=[]; SEX=[]; STG=[]
    for i,s in enumerate(df['source_sequence'].values):
        ov,tot,nz,panel,meta = parse_source(s)
        OV[i]=ov; TOT[i]=tot; NZ[i]=nz; DMG[i]=dmg_group(panel)
        COND.append(0 if meta.get('COND')=='COND_004' else 1)
        SEX.append(1 if meta.get('SEX')=='SEX_006' else 0)
        STG.append(meta.get('STAGE','?'))
    return OV,TOT,NZ,DMG,np.array(COND,np.float32),np.array(SEX,np.float32),STG

OVtr,TOTtr,NZtr,DMGtr,CONDtr,SEXtr,STGtr = parse_df(train)
OVte,TOTte,NZte,DMGte,CONDte,SEXte,STGte = parse_df(test)
Y = np.stack([parse_target(s) for s in train['target_sequence']]).astype(np.int64)
N = len(train)

st_u = sorted(set(STGtr) | set(STGte)); st_map = {v:i for i,v in enumerate(st_u)}; nst=len(st_u)
STGtr_i = np.array([st_map[s] for s in STGtr]); STGte_i = np.array([st_map[s] for s in STGte])
lg = np.log1p(OVtr); MU = lg.mean(0); SD = lg.std(0)+1e-6

def build_feat(ov,tot,nz,dmg,cond,sex,stg_i):
    ovn=(np.log1p(ov)-MU)/SD; pres=(ov>0).astype(np.float32)
    dmoh=np.eye(5,dtype=np.float32)[dmg+1]; stoh=np.eye(nst,dtype=np.float32)[stg_i]
    return np.concatenate([ovn,pres,(tot/63.)[:,None],(nz/60.)[:,None],
                           cond[:,None],sex[:,None],dmoh,stoh],1).astype(np.float32)
DIN = build_feat(OVtr[:1],TOTtr[:1],NZtr[:1],DMGtr[:1],CONDtr[:1],SEXtr[:1],STGtr_i[:1]).shape[1]

allowed = np.zeros((16,4),bool)
for t in range(16):
    allowed[t,0]=True
    for b in [1,2,3]:
        if (Y[:,t]==b).any(): allowed[t,b]=True
AMASK = torch.tensor(allowed, device=DEV)
STRONG = [t for t in range(16) if allowed[t,2]]    # targets that use B2 = signal-bearing bins
print("[targets] B2/strong-bin targets:", STRONG, flush=True)

# ----------------------------- models -----------------------------
class MLP(nn.Module):
    def __init__(s,din,h=256,do=0.4,nl=2):
        super().__init__(); L=[]; d=din
        for _ in range(nl): L+=[nn.Linear(d,h),nn.BatchNorm1d(h),nn.ReLU(),nn.Dropout(do)]; d=h
        s.trunk=nn.Sequential(*L); s.head=nn.Linear(d,64)
    def forward(s,x): return s.head(s.trunk(x)).view(-1,16,4)
class Lin(nn.Module):
    def __init__(s,din): super().__init__(); s.f=nn.Linear(din,64)
    def forward(s,x): return s.f(x).view(-1,16,4)

def dmg_augment(ov,dmg,p=0.5):
    ov=ov.copy(); dmg=dmg.copy()
    for i in range(len(ov)):
        if dmg[i]==-1 and RNG.random()<p:
            g=int(RNG.integers(0,4)); ov[i,DG==g]=0; dmg[i]=g
    return ov,dmg

def train_one(kind, idx_tr, seed, epochs,
              OVs=None,TOTs=None,NZs=None,DMGs=None,CONDs=None,SEXs=None,STGs=None):
    """Train one model on rows idx_tr of the training set. Returns the fitted net."""
    torch.manual_seed(seed); np.random.seed(seed)
    yb = torch.tensor(Y[idx_tr], device=DEV)
    net = (MLP(DIN) if kind=='mlp' else Lin(DIN)).to(DEV)
    wd = 1e-4 if kind=='mlp' else 1e-3
    opt = torch.optim.AdamW(net.parameters(), lr=2e-3, weight_decay=wd)
    nb = max(1,(len(idx_tr)+127)//128)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=2e-3, total_steps=epochs*nb)
    for ep in range(epochs):
        net.train(); perm=np.random.permutation(len(idx_tr))
        ova,dmga = dmg_augment(OVtr[idx_tr], DMGtr[idx_tr])
        Xtr = torch.tensor(build_feat(ova,TOTtr[idx_tr],NZtr[idx_tr],dmga,
                                      CONDtr[idx_tr],SEXtr[idx_tr],STGtr_i[idx_tr]), device=DEV)
        for k in range(0,len(idx_tr),128):
            b=perm[k:k+128]
            if len(b)<2: continue
            lo=net(Xtr[b]).masked_fill(~AMASK.unsqueeze(0),-1e9)
            loss=F.cross_entropy(lo.reshape(-1,4), yb[b].reshape(-1))
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
    net.eval(); return net

def predict(net, OVx,TOTx,NZx,DMGx,CONDx,SEXx,STGx_i):
    X=torch.tensor(build_feat(OVx,TOTx,NZx,DMGx,CONDx,SEXx,STGx_i), device=DEV)
    with torch.no_grad():
        return F.softmax(net(X).masked_fill(~AMASK.unsqueeze(0),-1e9),2).cpu().numpy()

# ----------------------------- exact metric (for threshold tuning) -----------------------------
NONE_ID=48
def pack(bm,Lmax=16):
    b=np.asarray(bm); n=b.shape[0]; tgt=np.arange(16)[None,:]
    tokid=tgt*3+(b-1); key=np.where(b>0,-b*100+tgt,10**6)
    order=np.argsort(key,1); sk=np.take_along_axis(key,order,1); si=np.take_along_axis(tokid,order,1)
    val=sk<10**6; lens=val.sum(1)
    ids=np.full((n,Lmax),-1,np.int16); ids[:,:16]=np.where(val,si,-1).astype(np.int16)
    z=lens==0
    if z.any(): ids[z,0]=NONE_ID; lens=lens.copy(); lens[z]=1
    return ids,lens.astype(np.int64)
def ids_to_set(ids):
    n=ids.shape[0]; s=np.zeros((n,49),bool); rows=np.arange(n)[:,None].repeat(ids.shape[1],1)
    v=ids>=0; s[rows[v],ids[v]]=True; return s
def batch_edit(P,pl,Tt,tl,Lmax=16):
    n=P.shape[0]; D=np.zeros((n,Lmax+1,Lmax+1),np.int32); ar=np.arange(Lmax+1)
    D[:,:,0]=ar[None,:]; D[:,0,:]=ar[None,:]
    for i in range(1,Lmax+1):
        Pi=P[:,i-1]
        for j in range(1,Lmax+1):
            c=(Pi!=Tt[:,j-1]).astype(np.int32)
            D[:,i,j]=np.minimum(np.minimum(D[:,i-1,j]+1,D[:,i,j-1]+1),D[:,i-1,j-1]+c)
    return D[np.arange(n),pl,tl]
class Scorer:
    def __init__(s,Yt): s.T,s.tl=pack(Yt); s.ts=ids_to_set(s.T)
    def rows(s,pb,norm):
        P,pl=pack(pb); ps=ids_to_set(P); inter=(ps&s.ts).sum(1).astype(np.float64)
        la=pl.astype(np.float64); lb=s.tl.astype(np.float64); m=np.maximum(la,lb)
        pr=np.where(la>0,inter/la,0.); rc=np.where(lb>0,inter/lb,0.)
        f1=np.where((pr+rc)>0,2*pr*rc/(pr+rc),0.)
        if norm=='ratio': es=2*inter/(la+lb); ls=es
        else:
            lev=batch_edit(P,pl,s.T,s.tl).astype(np.float64)
            es=1-lev/(la+lb) if norm=='sum' else 1-lev/m; ls=inter/m
        return 0.5*es+0.3*f1+0.2*ls

# ----------------------------- decode + threshold tuning -----------------------------
def decode(oof, tau):
    pact=1-oof[:,:,0]; bins=np.ones((oof.shape[0],16),int); ab=oof[:,:,1:].argmax(2)+1
    for t in STRONG: bins[:,t]=ab[:,t]
    return np.where(pact>=tau[None,:],bins,0)

def make_groups():
    key=[(CONDtr[i],SEXtr[i],STGtr[i]) for i in range(N)]
    u={k:i for i,k in enumerate(sorted(set(key)))}
    return np.array([u[k] for k in key]), key
def subset_flags(gkeys):
    from collections import Counter
    c=Counter()
    for s in train['target_sequence']:
        s=s.strip()
        if s and s!='NONE':
            for t in s.split(): c[t]+=1
    rare={t for t,n in c.items() if n<160}
    gsz=Counter(gkeys); thr=sorted(gsz.values())[max(0,int(len(gsz)*0.35)-1)]
    small={g for g,z in gsz.items() if z<=thr}
    dmg=np.zeros(N,bool); rr=np.zeros(N,bool); sh=np.zeros(N,bool)
    for i in range(N):
        _,_,_,panel,_=parse_source(train['source_sequence'].iloc[i])
        dmg[i]=panel.startswith('PANEL_DAMAGE_')
        toks=set(train['target_sequence'].iloc[i].split()) if train['target_sequence'].iloc[i].strip()!='NONE' else set()
        rr[i]=len(toks & rare)>0; sh[i]=gkeys[i] in small
    return dict(damaged=dmg,rare=rr,shifted=sh)

def weighted(rows,flags):
    def mn(msk): msk=np.asarray(msk,bool); return float(rows[msk].mean()) if msk.any() else 0.0
    return 0.45*float(rows.mean())+0.25*mn(flags['shifted'])+0.20*mn(flags['damaged'])+0.10*mn(flags['rare'])

def tune_thresholds(oof, flags, scorer):
    """Tune per-target thresholds to maximise a norm-robust objective (mean of max & sum finals)."""
    def obj(tau):
        pb=decode(oof,tau)
        return 0.5*weighted(scorer.rows(pb,'max'),flags)+0.5*weighted(scorer.rows(pb,'sum'),flags)
    tau=np.full(16,0.10); cur=obj(tau)
    for _ in range(4):
        for t in range(16):
            bt,bv=tau[t],cur
            for c in np.arange(0.02,0.60,0.01):
                tau[t]=c; v=obj(tau)
                if v>bv+1e-9: bv,bt=v,c
            tau[t]=bt; cur=bv
    return tau,cur

# ----------------------------- submission validity -----------------------------
def canon_sort(tokens):
    best={}
    for tk in tokens:
        m=TOKEN_RE.match(tk)
        if m:
            ti=int(m.group(1)); b=int(m.group(2))
            if ti not in best or b>best[ti]: best[ti]=b
    return [f"T{ti:02d}_B{b}" for ti,b in sorted(best.items(),key=lambda kv:(-kv[1],kv[0]))]
def bins_to_str(bins):
    toks=canon_sort([f"T{t:02d}_B{int(bins[t])}" for t in range(16) if bins[t]>0])
    return "NONE" if not toks else " ".join(toks)
def check_submission(df, test_ids):
    if list(df.columns)!=['id','predicted_sequence']: return False,"columns"
    if df['id'].duplicated().any(): return False,"dup id"
    if set(df['id'])!=set(test_ids) or len(df)!=len(test_ids): return False,"id set/len"
    for i,seq in enumerate(df['predicted_sequence'].astype(str)):
        s=seq.strip()
        if s=='NONE': continue
        if not s or s=='nan': return False,f"empty@{i}"
        toks=s.split()
        if len(set(toks))!=len(toks): return False,f"dup tok@{i}"
        seen=set()
        for tk in toks:
            m=TOKEN_RE.match(tk)
            if not m or not(0<=int(m.group(1))<=15): return False,f"bad tok@{i}:{tk}"
            if int(tk[1:3]) in seen: return False,f"dup target@{i}"
            seen.add(int(tk[1:3]))
        if toks!=canon_sort(toks): return False,f"noncanon@{i}"
    return True,"OK"

# ----------------------------- run -----------------------------
def build_ensemble_oof(specs):
    """Grouped-CV OOF probs for threshold tuning. specs = list of (kind, seed, epochs)."""
    gid,gkeys=make_groups()
    oof=np.zeros((N,16,4),np.float32)
    # simple 5-fold GroupKFold
    ug=np.unique(gid); folds=[ug[i::5] for i in range(5)]
    for fg in folds:
        va=np.where(np.isin(gid,fg))[0]; tr=np.where(~np.isin(gid,fg))[0]
        acc=np.zeros((len(va),16,4),np.float32)
        for kind,seed,ep in specs:
            net=train_one(kind,tr,seed,ep)
            acc+=predict(net,OVtr[va],TOTtr[va],NZtr[va],DMGtr[va],CONDtr[va],SEXtr[va],STGtr_i[va])
        oof[va]=acc/len(specs)
    return oof,gkeys

def norm_probs(o):
    o=o.copy()
    for t in range(16): o[:,t,~allowed[t]]=0
    return o/(o.sum(2,keepdims=True)+1e-9)

# self-timing: warm up CUDA (init excluded), estimate a full 110-epoch MLP cost, pick the
# largest ensemble plan that fits the budget with margin.
_=train_one('mlp',np.arange(min(128,N)),999,1)          # CUDA/cuDNN warmup — not timed
if DEV=='cuda': torch.cuda.synchronize()
t0=time.time(); _=train_one('mlp',np.arange(N),998,6)
if DEV=='cuda': torch.cuda.synchronize()
per_model=max(0.3,(time.time()-t0)/6*110)               # ~sec for one full-data 110-epoch MLP
def cost(m,l,x): return per_model*(5*(0.8*m+0.4*l)+(m+0.5*l+x))   # CV(5-fold)+full-fit+extra
budget=max(60.0, BUDGET_S-per_model*3-60)
EXTRA=0; base_specs=[('mlp',0,90)]
for m,l,x in [(3,2,3),(3,1,2),(2,1,2),(2,1,0),(1,1,0),(1,0,0)]:
    if cost(m,l,x)<=budget*0.9:
        base_specs=[('mlp',s,110) for s in range(m)]+[('lin',100+s,80) for s in range(l)]; EXTRA=x; break
print(f"[timing] per_model~{per_model:.1f}s budget~{budget:.0f}s specs={base_specs} extra={EXTRA}", flush=True)

scorer=Scorer(Y)
oof,gkeys=build_ensemble_oof(base_specs)
oof=norm_probs(oof)
flags=subset_flags(gkeys)
tau,cvobj=tune_thresholds(oof,flags,scorer)
# report CV under all norms (with subset breakdown for transparency)
def subs(rows):
    def mn(m): m=np.asarray(m,bool); return float(rows[m].mean()) if m.any() else 0.0
    return (0.45*float(rows.mean())+0.25*mn(flags['shifted'])+0.20*mn(flags['damaged'])+0.10*mn(flags['rare']),
            float(rows.mean()),mn(flags['shifted']),mn(flags['damaged']),mn(flags['rare']))
for nm in ['max','sum','ratio']:
    fin,al,sh,dm,ra=subs(scorer.rows(decode(oof,tau),nm))
    print(f"[CV {nm:5s}] FINAL={fin:.4f} all={al:.4f} shifted={sh:.4f} damaged={dm:.4f} rare={ra:.4f}", flush=True)
print("[tau]",np.round(tau,2),flush=True)

# retrain on FULL train (add seeds if budget remains), predict test
full_specs=list(base_specs)
te_prob=np.zeros((len(test),16,4),np.float32); nmodels=0
for kind,seed,ep in full_specs:
    if time.time()-T_START > BUDGET_S-120: break
    net=train_one(kind,np.arange(N),seed,ep)
    te_prob+=predict(net,OVte,TOTte,NZte,DMGte,CONDte,SEXte,STGte_i); nmodels+=1
# spend leftover budget on extra MLP seeds
extra=3
for e in range(extra):
    if time.time()-T_START > BUDGET_S-120: break
    net=train_one('mlp',np.arange(N),1000+e,110)
    te_prob+=predict(net,OVte,TOTte,NZte,DMGte,CONDte,SEXte,STGte_i); nmodels+=1
te_prob=norm_probs(te_prob/max(1,nmodels))
print(f"[predict] ensembled {nmodels} full-data models", flush=True)

pred_bins=decode(te_prob,tau)
os.makedirs("./working", exist_ok=True)
sub=pd.DataFrame({'id':test['id'],
                  'predicted_sequence':[bins_to_str(pred_bins[i]) for i in range(len(test))]})
ok,msg=check_submission(sub,test['id'].tolist())
print("[check]",ok,msg,flush=True)
assert ok, f"submission invalid: {msg}"
sub.to_csv("./working/submission.csv", index=False)
print(f"[done] wrote ./working/submission.csv ({len(sub)} rows) in {time.time()-T_START:.0f}s; "
      f"avg pred set size {(pred_bins>0).sum(1).mean():.2f}", flush=True)
