"""H100 final-submission generator. Trains N_SEEDS models on ALL train data, generates
test predictions (snap-to-vocab + family-mode fallback), per-slot majority vote, writes
submission.csv. Parametrized so it can be fired with the winning CV config.  Usage:
  python h100_final.py MODEL EPOCHS K BS LR N_SEEDS
"""
import sys, time, collections, numpy as np, pandas as pd, torch
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from transformers import T5TokenizerFast, T5ForConditionalGeneration, get_linear_schedule_with_warmup
sys.path.insert(0, str(Path(__file__).resolve().parent))
from aug import SLOTS, parse_train_row, make_example, parse_seq, seq_str, get_family

MODEL   = sys.argv[1] if len(sys.argv)>1 else "t5-base"
EPOCHS  = int(sys.argv[2]) if len(sys.argv)>2 else 6
K       = int(sys.argv[3]) if len(sys.argv)>3 else 6
BS      = int(sys.argv[4]) if len(sys.argv)>4 else 16
LR      = float(sys.argv[5]) if len(sys.argv)>5 else 2e-4
N_SEEDS = int(sys.argv[6]) if len(sys.argv)>6 else 1
DS=Path("/mnt/work/c1"); MSRC=192; MTGT=48; dev="cuda"
torch.backends.cuda.matmul.allow_tf32=True; torch.backends.cudnn.allow_tf32=True
print(f"FINAL MODEL={MODEL} EPOCHS={EPOCHS} K={K} BS={BS} LR={LR} N_SEEDS={N_SEEDS}",flush=True)

class DS_(Dataset):
    def __init__(s,pairs,tok): s.p=pairs; s.tok=tok
    def __len__(s): return len(s.p)
    def __getitem__(s,i):
        src,tgt=s.p[i]; x=s.tok(src,max_length=MSRC,truncation=True,padding="max_length",return_tensors="pt")
        y=s.tok(text_target=tgt,max_length=MTGT,truncation=True,padding="max_length",return_tensors="pt")
        lab=y["input_ids"].squeeze(0); lab[lab==s.tok.pad_token_id]=-100
        return {"input_ids":x["input_ids"].squeeze(0),"attention_mask":x["attention_mask"].squeeze(0),"labels":lab}

train=pd.read_csv(DS/"train.csv"); test=pd.read_csv(DS/"test.csv")
recs=[parse_train_row(r) for _,r in train.iterrows()]
vocab={s:set() for s in SLOTS}; famtbl=collections.defaultdict(lambda:collections.defaultdict(collections.Counter))
for r in recs:
    for s in SLOTS: vocab[s].add(r["truth"][s]); famtbl[r["family"]][s][r["truth"][s]]+=1
glob={s:collections.Counter([r["truth"][s] for r in recs]).most_common(1)[0][0] for s in SLOTS}
def fmode(fam,s):
    c=famtbl.get(fam,{}).get(s); return c.most_common(1)[0][0] if c else glob[s]
srcs=test["prompt"].astype(str).tolist()

def train_one(seed):
    torch.manual_seed(seed); ar=np.random.default_rng(seed+1); pairs=[]
    for r in recs:
        for _ in range(K): pairs.append(make_example(r,ar,n_show=3))
        pairs.append(make_example(r,ar,n_show=6))
    ar.shuffle(pairs)
    tok=T5TokenizerFast.from_pretrained(MODEL); model=T5ForConditionalGeneration.from_pretrained(MODEL).to(dev)
    dl=DataLoader(DS_(pairs,tok),batch_size=BS,shuffle=True,num_workers=8,pin_memory=True)
    opt=torch.optim.AdamW(model.parameters(),lr=LR)
    tot=len(dl)*EPOCHS; sch=get_linear_schedule_with_warmup(opt,int(0.05*tot),tot); model.train()
    for ep in range(EPOCHS):
        for b in dl:
            b={k:v.to(dev,non_blocking=True) for k,v in b.items()}
            with torch.autocast("cuda",dtype=torch.bfloat16): loss=model(**b).loss
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step(); sch.step()
    return tok,model

def predict(tok,model):
    model.eval(); out=[]
    for i in range(0,len(srcs),128):
        enc=tok(srcs[i:i+128],max_length=MSRC,truncation=True,padding=True,return_tensors="pt").to(dev)
        with torch.no_grad(), torch.autocast("cuda",dtype=torch.bfloat16):
            g=model.generate(**enc,max_new_tokens=MTGT,num_beams=1)
        out+=tok.batch_decode(g,skip_special_tokens=True)
    return out

t0=time.time(); per_seed=[]
for sd in range(N_SEEDS):
    tok,model=train_one(42+sd); outs=predict(tok,model)
    sp=[]
    for o,(_,trow) in zip(outs,test.iterrows()):
        fam=get_family(trow["prompt"]); d=parse_seq(o)
        sp.append({s:(d.get(s) if d.get(s) in vocab[s] else fmode(fam,s)) for s in SLOTS})
    per_seed.append(sp); print(f"seed {sd} done [{time.time()-t0:.0f}s]",flush=True)
    del model; torch.cuda.empty_cache()

rows=[]
for i,(_,trow) in enumerate(test.iterrows()):
    pred={}
    for s in SLOTS:
        votes=collections.Counter(per_seed[m][i][s] for m in range(N_SEEDS)); pred[s]=votes.most_common(1)[0][0]
    rows.append({"id":trow["id"],"repaired_sequence":seq_str(pred)})
sub=pd.DataFrame(rows,columns=["id","repaired_sequence"])
assert list(sub.columns)==["id","repaired_sequence"] and len(sub)==len(test) and sub["id"].is_unique and set(sub["id"])==set(test["id"])
for s in sub["repaired_sequence"]:
    d=parse_seq(s); assert all(k in d and d[k] in vocab[k] for k in SLOTS) and s.count(";")==5
sub.to_csv(DS/"submission.csv",index=False)
print("wrote submission.csv:",sub.shape,f"[{time.time()-t0:.0f}s]",flush=True)
