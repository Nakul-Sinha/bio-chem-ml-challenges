"""Classifier final-submission generator: trains N_SEEDS classifier-head models on ALL train,
averages per-slot probabilities over seeds, writes submission.csv. Usage:
  python h100_clf_final.py ENCODER EPOCHS K BS LR N_SEEDS
"""
import sys, time, collections, numpy as np, pandas as pd, torch, torch.nn as nn
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from transformers import T5TokenizerFast, T5EncoderModel, get_linear_schedule_with_warmup
sys.path.insert(0, str(Path(__file__).resolve().parent))
from aug import SLOTS, parse_train_row, make_example, get_family

ENC     = sys.argv[1] if len(sys.argv)>1 else "t5-small"
EPOCHS  = int(sys.argv[2]) if len(sys.argv)>2 else 10
K       = int(sys.argv[3]) if len(sys.argv)>3 else 6
BS      = int(sys.argv[4]) if len(sys.argv)>4 else 32
LR      = float(sys.argv[5]) if len(sys.argv)>5 else 3e-4
N_SEEDS = int(sys.argv[6]) if len(sys.argv)>6 else 4
DS=Path("/mnt/work/c1"); MSRC=192; dev="cuda"
torch.backends.cuda.matmul.allow_tf32=True; torch.backends.cudnn.allow_tf32=True
print(f"CLF-FINAL ENC={ENC} EPOCHS={EPOCHS} K={K} BS={BS} LR={LR} N_SEEDS={N_SEEDS}",flush=True)

train=pd.read_csv(DS/"train.csv"); test=pd.read_csv(DS/"test.csv"); recs=[parse_train_row(r) for _,r in train.iterrows()]
V2I={s:{} for s in SLOTS}; I2V={s:[] for s in SLOTS}
for r in recs:
    for s in SLOTS:
        v=r["truth"][s]
        if v not in V2I[s]: V2I[s][v]=len(I2V[s]); I2V[s].append(v)
NC={s:len(I2V[s]) for s in SLOTS}
tok=T5TokenizerFast.from_pretrained(ENC)
class DSc(Dataset):
    def __init__(s,items): s.it=items
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

srcs=test["prompt"].astype(str).tolist()
def train_one(seed):
    torch.manual_seed(seed); ar=np.random.default_rng(seed+1); items=[]
    for r in recs:
        for _ in range(K):
            inp,_=make_example(r,ar,n_show=3); items.append((inp,r["truth"]))
        inp,_=make_example(r,ar,n_show=6); items.append((inp,r["truth"]))
    net=Net().to(dev); dl=DataLoader(DSc(items),batch_size=BS,shuffle=True,num_workers=8,pin_memory=True)
    opt=torch.optim.AdamW(net.parameters(),lr=LR,weight_decay=1e-5); ce=nn.CrossEntropyLoss()
    tot=len(dl)*EPOCHS; sch=get_linear_schedule_with_warmup(opt,int(0.05*tot),tot); net.train()
    for ep in range(EPOCHS):
        for ids,mask,y in dl:
            ids,mask,y=ids.to(dev),mask.to(dev),y.to(dev)
            with torch.autocast("cuda",dtype=torch.bfloat16): logits=net(ids,mask)
            loss=sum(ce(logits[j],y[:,j]) for j in range(len(SLOTS)))
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(net.parameters(),1.0); opt.step(); sch.step()
    return net
def predict(net):
    net.eval(); probs=[[] for _ in SLOTS]
    with torch.no_grad():
        for i in range(0,len(srcs),256):
            x=tok(srcs[i:i+256],max_length=MSRC,truncation=True,padding=True,return_tensors="pt").to(dev)
            with torch.autocast("cuda",dtype=torch.bfloat16): logits=net(x["input_ids"],x["attention_mask"])
            for j in range(len(SLOTS)): probs[j].append(torch.softmax(logits[j].float(),1).cpu().numpy())
    return [np.concatenate(p) for p in probs]

t0=time.time(); acc=[0]*len(SLOTS)
for sd in range(N_SEEDS):
    net=train_one(42+sd); pr=predict(net)
    for j in range(len(SLOTS)): acc[j]=acc[j]+pr[j]/N_SEEDS
    print(f"seed {sd} done [{time.time()-t0:.0f}s]",flush=True); del net; torch.cuda.empty_cache()
rows=[]
for i,(_,trow) in enumerate(test.iterrows()):
    pred={SLOTS[j]:I2V[SLOTS[j]][acc[j][i].argmax()] for j in range(len(SLOTS))}
    rows.append({"id":trow["id"],"repaired_sequence":";".join(f"{s}={pred[s]}" for s in SLOTS)})
sub=pd.DataFrame(rows,columns=["id","repaired_sequence"])
assert len(sub)==len(test) and sub["id"].is_unique and set(sub["id"])==set(test["id"])
for s in sub["repaired_sequence"]: assert s.count(";")==5
sub.to_csv(DS/"submission.csv",index=False); print("wrote submission.csv",sub.shape,f"[{time.time()-t0:.0f}s]",flush=True)
