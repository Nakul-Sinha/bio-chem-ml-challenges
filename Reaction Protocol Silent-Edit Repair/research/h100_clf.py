"""Classifier-head model: T5 encoder (mean-pooled) + 6 symmetric 6-way heads.
No autoregressive order bias (should fix the workup collapse). Same aug/val/metric as CV.
Usage: python h100_clf.py ENCODER EPOCHS K BS LR N_SEEDS   (e.g. t5-small 8 6 32 3e-4 1)
"""
import sys, os, time, collections, numpy as np, pandas as pd, torch, torch.nn as nn
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from transformers import T5TokenizerFast, T5EncoderModel, get_linear_schedule_with_warmup
sys.path.insert(0, str(Path(__file__).resolve().parent))
from aug import SLOTS, parse_train_row, make_example
WLOSS=os.environ.get("WLOSS","0")=="1"  # weight per-slot CE by competition slot weight

ENC     = sys.argv[1] if len(sys.argv)>1 else "t5-small"
EPOCHS  = int(sys.argv[2]) if len(sys.argv)>2 else 8
K       = int(sys.argv[3]) if len(sys.argv)>3 else 6
BS      = int(sys.argv[4]) if len(sys.argv)>4 else 32
LR      = float(sys.argv[5]) if len(sys.argv)>5 else 3e-4
N_SEEDS = int(sys.argv[6]) if len(sys.argv)>6 else 1
W={"prep":2.20,"activation":0.85,"order":0.60,"control":3.00,"quench":4.00,"workup":0.25}; WSUM=sum(W.values())
DS=Path("/mnt/work/c1"); MSRC=192; VAL_FRAC=0.15; VAL_REPS=3; dev="cuda"
torch.backends.cuda.matmul.allow_tf32=True; torch.backends.cudnn.allow_tf32=True
print(f"CLF ENC={ENC} EPOCHS={EPOCHS} K={K} BS={BS} LR={LR} N_SEEDS={N_SEEDS}",flush=True)

train=pd.read_csv(DS/"train.csv"); recs=[parse_train_row(r) for _,r in train.iterrows()]
fams=np.array([r["family"] for r in recs]); rng=np.random.default_rng(42); val_idx=[]
for f in sorted(set(fams)):
    idx=np.where(fams==f)[0]; rng.shuffle(idx); val_idx+=list(idx[:int(len(idx)*VAL_FRAC)])
val_idx=set(int(x) for x in val_idx); tr=[r for i,r in enumerate(recs) if i not in val_idx]; va=[r for i,r in enumerate(recs) if i in val_idx]
# value vocab per slot (from full train to be safe)
V2I={s:{} for s in SLOTS}; I2V={s:[] for s in SLOTS}
for r in recs:
    for s in SLOTS:
        v=r["truth"][s]
        if v not in V2I[s]: V2I[s][v]=len(I2V[s]); I2V[s].append(v)
NC={s:len(I2V[s]) for s in SLOTS}; print("classes/slot:",NC,flush=True)

tok=T5TokenizerFast.from_pretrained(ENC)
class DSc(Dataset):
    def __init__(s,items): s.it=items  # (src, truthdict)
    def __len__(s): return len(s.it)
    def __getitem__(s,i):
        src,truth=s.it[i]; x=tok(src,max_length=MSRC,truncation=True,padding="max_length",return_tensors="pt")
        y=torch.tensor([V2I[sl][truth[sl]] for sl in SLOTS])
        return x["input_ids"].squeeze(0),x["attention_mask"].squeeze(0),y

class Net(nn.Module):
    def __init__(s):
        super().__init__(); s.enc=T5EncoderModel.from_pretrained(ENC); h=s.enc.config.d_model
        s.drop=nn.Dropout(0.2); s.heads=nn.ModuleList([nn.Linear(h,NC[sl]) for sl in SLOTS])
    def forward(s,ids,mask):
        o=s.enc(input_ids=ids,attention_mask=mask).last_hidden_state
        m=mask.unsqueeze(-1).float(); z=(o*m).sum(1)/m.sum(1).clamp(min=1); z=s.drop(z)
        return [head(z) for head in s.heads]

valrng=np.random.default_rng(12345); val_items=[]; val_truth=[]
for rep in range(VAL_REPS):
    for r in va:
        inp,_=make_example(r,valrng,n_show=3); val_items.append((inp,r["truth"])); val_truth.append(r["truth"])

def row_score(p,t): return sum(W[s]*(p.get(s)==t.get(s)) for s in SLOTS)/WSUM
def train_one(seed):
    torch.manual_seed(seed); ar=np.random.default_rng(seed+1); items=[]
    for r in tr:
        for _ in range(K):
            inp,_=make_example(r,ar,n_show=3); items.append((inp,r["truth"]))
        inp,_=make_example(r,ar,n_show=6); items.append((inp,r["truth"]))
    net=Net().to(dev); dl=DataLoader(DSc(items),batch_size=BS,shuffle=True,num_workers=8,pin_memory=True)
    opt=torch.optim.AdamW(net.parameters(),lr=LR,weight_decay=1e-5); ce=nn.CrossEntropyLoss()
    _wm=np.array([W[s] for s in SLOTS]); _wm=_wm/_wm.mean(); LW=[float(x) for x in _wm] if WLOSS else [1.0]*len(SLOTS)
    tot=len(dl)*EPOCHS; sch=get_linear_schedule_with_warmup(opt,int(0.05*tot),tot); net.train()
    for ep in range(EPOCHS):
        te=time.time()
        for ids,mask,y in dl:
            ids,mask,y=ids.to(dev),mask.to(dev),y.to(dev)
            with torch.autocast("cuda",dtype=torch.bfloat16): logits=net(ids,mask)
            loss=sum(LW[j]*ce(logits[j],y[:,j]) for j in range(len(SLOTS)))
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(net.parameters(),1.0); opt.step(); sch.step()
        print(f"  seed{seed} ep{ep} {time.time()-te:.1f}s loss={loss.item():.2f}",flush=True)
    return net

def predict(net):
    net.eval(); probs=[[] for _ in SLOTS]
    with torch.no_grad():
        for i in range(0,len(val_items),256):
            batch=val_items[i:i+256]; x=tok([b[0] for b in batch],max_length=MSRC,truncation=True,padding=True,return_tensors="pt").to(dev)
            with torch.autocast("cuda",dtype=torch.bfloat16): logits=net(x["input_ids"],x["attention_mask"])
            for j in range(len(SLOTS)): probs[j].append(torch.softmax(logits[j].float(),1).cpu().numpy())
    return [np.concatenate(p) for p in probs]

t0=time.time(); seed_probs=[]
for sd in range(N_SEEDS):
    net=train_one(100+sd); pr=predict(net); seed_probs.append(pr)
    parsed=[{SLOTS[j]:I2V[SLOTS[j]][pr[j][i].argmax()] for j in range(len(SLOTS))} for i in range(len(val_truth))]
    sc=np.mean([row_score(parsed[i],val_truth[i]) for i in range(len(parsed))])
    print(f"seed {sd} CV={sc:.4f} [{time.time()-t0:.0f}s]",flush=True); del net; torch.cuda.empty_cache()

avg=[sum(seed_probs[m][j] for m in range(N_SEEDS))/N_SEEDS for j in range(len(SLOTS))]
parsed=[{SLOTS[j]:I2V[SLOTS[j]][avg[j][i].argmax()] for j in range(len(SLOTS))} for i in range(len(val_truth))]
print(f"ENSEMBLE({N_SEEDS}) CV={np.mean([row_score(parsed[i],val_truth[i]) for i in range(len(parsed))]):.4f}",flush=True)
for s in SLOTS:
    j=SLOTS.index(s); print(f"  {s:10s} acc={np.mean([parsed[i][s]==val_truth[i][s] for i in range(len(parsed))]):.3f}",flush=True)
