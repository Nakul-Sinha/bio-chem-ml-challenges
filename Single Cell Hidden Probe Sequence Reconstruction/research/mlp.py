"""Torch MLP: shared trunk + 16 heads (4-way, B2 masked where unseen). Damage augmentation.
Grouped-CV OOF, threshold decode, exact weighted metric. Shippable in pure torch."""
import numpy as np, pandas as pd, time, argparse, srlib as L
import torch, torch.nn as nn, torch.nn.functional as F
from sklearn.model_selection import GroupKFold

ap=argparse.ArgumentParser()
ap.add_argument('--hidden',type=int,default=384); ap.add_argument('--layers',type=int,default=2)
ap.add_argument('--dropout',type=float,default=0.3); ap.add_argument('--dmg_aug',type=float,default=0.5)
ap.add_argument('--epochs',type=int,default=120); ap.add_argument('--lr',type=float,default=2e-3)
ap.add_argument('--wd',type=float,default=1e-4); ap.add_argument('--bs',type=int,default=128)
ap.add_argument('--seeds',type=int,default=3); ap.add_argument('--folds',type=int,default=5)
ap.add_argument('--tag',type=str,default='mlp'); ap.add_argument('--emb',type=int,default=0)
args=ap.parse_args()
dev='cuda' if torch.cuda.is_available() else 'cpu'
print("device",dev,"args",vars(args))

D="../dataset/"; train=pd.read_csv(D+"train.csv"); N=len(train)
# damage groups: O-index -> group
obs=pd.read_csv(D+"observed_panel.csv"); DG=obs.set_index('observed_index').damage_group.values  # len80

def parse_all(df):
    OV=np.zeros((len(df),80),np.float32);TOT=np.zeros(len(df),np.float32);NZ=np.zeros(len(df),np.float32)
    DMG=np.full(len(df),-1,np.int64);COND=[];SEX=[];STAGE=[]
    for i,s in enumerate(df['source_sequence'].values):
        ov,tot,nz,panel,meta=L.parse_source(s)
        OV[i]=ov;TOT[i]=tot;NZ[i]=nz;DMG[i]=L.panel_damage_group(panel)
        COND.append(0 if meta.get('COND')=='COND_004' else 1)
        SEX.append(1 if meta.get('SEX')=='SEX_006' else 0)
        STAGE.append(meta.get('STAGE','?'))
    st_u=sorted(set(STAGE));st_map={v:i for i,v in enumerate(st_u)}
    STG=np.array([st_map[s] for s in STAGE],np.int64)
    return OV,TOT,NZ,DMG,np.array(COND,np.float32),np.array(SEX,np.float32),STG,len(st_u)
OV,TOT,NZ,DMG,COND,SEX,STG,nstage=parse_all(train)
Y=np.stack([L.parse_target(s) for s in train['target_sequence']]).astype(np.int64)
mu=OV.mean(0); sd=OV.std(0)+1e-6

# valid bins mask per target: which classes are allowed (0 always; others if seen)
allowed=np.zeros((16,4),bool)
for t in range(16):
    allowed[t,0]=True
    for b in [1,2,3]:
        if (Y[:,t]==b).any(): allowed[t,b]=True
print("targets with B2 allowed:",[t for t in range(16) if allowed[t,2]])

gid,gkeys=L.make_groups(train)
gsize=pd.Series(gkeys).value_counts(); small=set(gsize[gsize<=gsize.quantile(0.35)].index)
rare_tokens,_=L.rare_token_set(train,160)
flags=L.subset_flags(train,np.arange(N),rare_tokens,small,gkeys)
true_seqs=[L.target_tokens(s) for s in train['target_sequence']]

def build_feat(ov,tot,nz,dmg,cond,sex,stg):
    ovn=(ov-mu)/sd
    pres=(ov>0).astype(np.float32)
    dmoh=np.eye(5,dtype=np.float32)[dmg+1]
    stoh=np.eye(nstage,dtype=np.float32)[stg]
    x=np.concatenate([ovn,pres,(tot/63.0)[:,None],(nz/60.0)[:,None],
                      cond[:,None],sex[:,None],dmoh,stoh],axis=1)
    return x.astype(np.float32)

class Net(nn.Module):
    def __init__(self,din,h,nl,do):
        super().__init__()
        layers=[]; d=din
        for _ in range(nl):
            layers+=[nn.Linear(d,h),nn.BatchNorm1d(h),nn.ReLU(),nn.Dropout(do)]; d=h
        self.trunk=nn.Sequential(*layers)
        self.head=nn.Linear(d,16*4)
    def forward(self,x):
        return self.head(self.trunk(x)).view(-1,16,4)

amask=torch.tensor(allowed,device=dev)  # (16,4)
def masked_logits(logits):
    return logits.masked_fill(~amask.unsqueeze(0), -1e9)

def dmg_augment(ov, dmg):
    """Randomly zero a damage group on NORMAL rows to mirror test damage."""
    ov=ov.copy(); dmg=dmg.copy()
    for i in range(len(ov)):
        if dmg[i]==-1 and np.random.rand()<args.dmg_aug:
            g=np.random.randint(0,4)
            ov[i, DG==g]=0; dmg[i]=g
    return ov,dmg

def train_fold(tr,va,seed):
    torch.manual_seed(seed); np.random.seed(seed)
    Xva=torch.tensor(build_feat(OV[va],TOT[va],NZ[va],DMG[va],COND[va],SEX[va],STG[va]),device=dev)
    yb=torch.tensor(Y[tr],device=dev)
    din=build_feat(OV[:1],TOT[:1],NZ[:1],DMG[:1],COND[:1],SEX[:1],STG[:1]).shape[1]
    net=Net(din,args.hidden,args.layers,args.dropout).to(dev)
    opt=torch.optim.AdamW(net.parameters(),lr=args.lr,weight_decay=args.wd)
    sched=torch.optim.lr_scheduler.OneCycleLR(opt,max_lr=args.lr,total_steps=args.epochs*((len(tr)+args.bs-1)//args.bs))
    ytr=Y[tr]
    for ep in range(args.epochs):
        net.train(); perm=np.random.permutation(len(tr))
        ova,dmga=dmg_augment(OV[tr],DMG[tr])
        Xtr=torch.tensor(build_feat(ova,TOT[tr],NZ[tr],dmga,COND[tr],SEX[tr],STG[tr]),device=dev)
        for k in range(0,len(tr),args.bs):
            idx=perm[k:k+args.bs]
            if len(idx)<2: continue   # BatchNorm needs >1 sample
            logits=masked_logits(net(Xtr[idx]))
            loss=F.cross_entropy(logits.reshape(-1,4), yb[idx].reshape(-1))
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
    net.eval()
    with torch.no_grad():
        pr=F.softmax(masked_logits(net(Xva)),dim=2).cpu().numpy()
    return pr

def run():
    oof=np.zeros((N,16,4),np.float32)
    gkf=GroupKFold(n_splits=args.folds)
    t0=time.time()
    for fi,(tr,va) in enumerate(gkf.split(OV,groups=gid)):
        acc=np.zeros((len(va),16,4),np.float32)
        for s in range(args.seeds):
            acc+=train_fold(tr,va,1000*s+fi)
        oof[va]=acc/args.seeds
        print(f"  fold{fi} done {time.time()-t0:.0f}s")
    return oof

oof=run()
np.save(f"oof_{args.tag}.npy",oof)
def score_bins(pb):
    ps=[L.bins_to_seq(pb[i]) for i in range(N)]; return L.weighted_score(ps,true_seqs,flags)
def decode_global(oof,tau):
    pact=1-oof[:,:,0]; binc=oof[:,:,1:].argmax(2)+1
    return np.where(pact>=tau,binc,0)
best=(-1,None,None)
for tau in np.arange(0.15,0.55,0.05):
    r=score_bins(decode_global(oof,tau))
    print(f"  tau={tau:.2f} FINAL={r['final']:.4f} all={r['all']:.4f} sh={r['shifted']:.4f} dm={r['damaged']:.4f} ra={r['rare']:.4f}")
    if r['final']>best[0]: best=(r['final'],tau,r)
print(f"[{args.tag}] >> BEST tau={best[1]:.2f} FINAL={best[0]:.4f}")
print("DONE")
