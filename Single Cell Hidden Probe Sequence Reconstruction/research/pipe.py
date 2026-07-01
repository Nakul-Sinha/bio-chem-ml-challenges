"""Development pipeline: pure-TORCH ensemble (linear + MLP seeds) w/ damage augmentation,
grouped-CV OOF, compared/ensembled with sklearn OOFs. Fast exact metric, all norms.
This is the basis for the shippable solution.py (torch-only)."""
import numpy as np, pandas as pd, time, srlib as L
import torch, torch.nn as nn, torch.nn.functional as F
from sklearn.model_selection import GroupKFold
from fastexact import FastScorer
import warnings; warnings.filterwarnings('ignore')
dev='cuda' if torch.cuda.is_available() else 'cpu'
D="../dataset/"; train=pd.read_csv(D+"train.csv"); N=len(train)
obs=pd.read_csv(D+"observed_panel.csv"); DG=obs.set_index('observed_index').damage_group.values

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
OV,TOT,NZ,DMG,COND,SEX,STG,nst=parse_all(train)
Y=np.stack([L.parse_target(s) for s in train['target_sequence']]).astype(np.int64)
lg=np.log1p(OV); mu=lg.mean(0); sd=lg.std(0)+1e-6
allowed=np.zeros((16,4),bool)
for t in range(16):
    allowed[t,0]=True
    for b in [1,2,3]:
        if (Y[:,t]==b).any(): allowed[t,b]=True
amask=torch.tensor(allowed,device=dev)
fs=FastScorer(Y)
gid,gkeys=L.make_groups(train)
gsize=pd.Series(gkeys).value_counts(); small=set(gsize[gsize<=gsize.quantile(0.35)].index)
rare_tokens,_=L.rare_token_set(train,160)
flags=L.subset_flags(train,np.arange(N),rare_tokens,small,gkeys)
STRONG=[5,6,12,13,14]

def build_feat(ov,tot,nz,dmg,cond,sex,stg):
    lgv=np.log1p(ov); ovn=(lgv-mu)/sd; pres=(ov>0).astype(np.float32)
    dmoh=np.eye(5,dtype=np.float32)[dmg+1]; stoh=np.eye(nst,dtype=np.float32)[stg]
    return np.concatenate([ovn,pres,(tot/63.)[:,None],(nz/60.)[:,None],
                           cond[:,None],sex[:,None],dmoh,stoh],1).astype(np.float32)
DIN=build_feat(OV[:1],TOT[:1],NZ[:1],DMG[:1],COND[:1],SEX[:1],STG[:1]).shape[1]

class MLP(nn.Module):
    def __init__(self,din,h=256,do=0.4,nl=2):
        super().__init__(); layers=[]; d=din
        for _ in range(nl): layers+=[nn.Linear(d,h),nn.BatchNorm1d(h),nn.ReLU(),nn.Dropout(do)]; d=h
        self.trunk=nn.Sequential(*layers); self.head=nn.Linear(d,64)
    def forward(self,x): return self.head(self.trunk(x)).view(-1,16,4)
class Lin(nn.Module):
    def __init__(self,din):
        super().__init__(); self.f=nn.Linear(din,64)
    def forward(self,x): return self.f(x).view(-1,16,4)

def dmg_aug(ov,dmg,p):
    ov=ov.copy(); dmg=dmg.copy()
    for i in range(len(ov)):
        if dmg[i]==-1 and np.random.rand()<p:
            g=np.random.randint(0,4); ov[i,DG==g]=0; dmg[i]=g
    return ov,dmg

def train_model(kind,tr,va,seed,epochs,dmgp=0.5):
    torch.manual_seed(seed); np.random.seed(seed)
    Xva=torch.tensor(build_feat(OV[va],TOT[va],NZ[va],DMG[va],COND[va],SEX[va],STG[va]),device=dev)
    yb=torch.tensor(Y[tr],device=dev)
    net=(MLP(DIN).to(dev) if kind=='mlp' else Lin(DIN).to(dev))
    opt=torch.optim.AdamW(net.parameters(),lr=2e-3,weight_decay=(1e-4 if kind=='mlp' else 1e-3))
    nb=(len(tr)+127)//128
    sched=torch.optim.lr_scheduler.OneCycleLR(opt,max_lr=2e-3,total_steps=epochs*nb)
    for ep in range(epochs):
        net.train(); perm=np.random.permutation(len(tr))
        ova,dmga=dmg_aug(OV[tr],DMG[tr],dmgp)
        Xtr=torch.tensor(build_feat(ova,TOT[tr],NZ[tr],dmga,COND[tr],SEX[tr],STG[tr]),device=dev)
        for k in range(0,len(tr),128):
            idx=perm[k:k+128]
            if len(idx)<2: continue
            lo=net(Xtr[idx]).masked_fill(~amask.unsqueeze(0),-1e9)
            loss=F.cross_entropy(lo.reshape(-1,4),yb[idx].reshape(-1))
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
    net.eval()
    with torch.no_grad():
        pr=F.softmax(net(Xva).masked_fill(~amask.unsqueeze(0),-1e9),2).cpu().numpy()
    return pr

def oof_for(kind,seeds,epochs):
    oof=np.zeros((N,16,4),np.float32)
    for fi,(tr,va) in enumerate(GroupKFold(5).split(OV,groups=gid)):
        acc=np.zeros((len(va),16,4),np.float32)
        for s in range(seeds): acc+=train_model(kind,tr,va,100*s+fi,epochs)
        oof[va]=acc/seeds
    return oof

t0=time.time()
oof_mlp=oof_for('mlp',3,110); np.save("oof_tmlp.npy",oof_mlp); print("tmlp %.0fs"%(time.time()-t0))
oof_lin=oof_for('lin',2,80);  np.save("oof_tlin.npy",oof_lin); print("tlin %.0fs"%(time.time()-t0))

def norm_probs(o):
    o=o.copy()
    for t in range(16): o[:,t,~allowed[t]]=0
    return o/(o.sum(2,keepdims=True)+1e-9)
def decode(oof,tau):
    pact=1-oof[:,:,0]; bins=np.ones((N,16),int); ab=oof[:,:,1:].argmax(2)+1
    for t in STRONG: bins[:,t]=ab[:,t]
    return np.where(pact>=tau[None,:],bins,0)
def tune(oof,norm):
    tau=np.full(16,0.10); cur=fs.weighted(decode(oof,tau),flags,norm)['final']
    for _ in range(3):
        for t in range(16):
            bt,bv=tau[t],cur
            for c in np.arange(0.02,0.60,0.01):
                tau[t]=c; v=fs.weighted(decode(oof,tau),flags,norm)['final']
                if v>bv+1e-9: bv,bt=v,c
            tau[t]=bt; cur=bv
    return tau,cur

sk={}
for nm,fn in [('gbdt','oof_gbdt_grouped.npy'),('lr','oof_lr_grouped.npy')]:
    try: sk[nm]=norm_probs(np.load(fn))
    except: pass
cand={'t-mlp':norm_probs(oof_mlp),'t-lin':norm_probs(oof_lin),
      'TORCH-ens':norm_probs((oof_mlp+oof_lin)/2)}
if sk: cand['TORCH+sk']=norm_probs((oof_mlp+oof_lin+sum(sk.values()))/(2+len(sk)))
print(f"\n{'model':11s} {'norm':5s} {'FINAL':>7s} {'all':>7s} {'shift':>7s} {'damg':>7s} {'rare':>7s}")
for nm,oof in cand.items():
    for norm in ['max','sum']:
        tau,f=tune(oof,norm); r=fs.weighted(decode(oof,tau),flags,norm)
        print(f"{nm:11s} {norm:5s} {r['final']:7.4f} {r['all']:7.4f} {r['shifted']:7.4f} {r['damaged']:7.4f} {r['rare']:7.4f}")
print("DONE pipe")
