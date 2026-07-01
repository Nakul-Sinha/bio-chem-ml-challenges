"""H100 CV harness — mirrors cv_t5_ens.py exactly (same weighted metric, family-grouped
split seed 42, fixed val degradation seed 12345, VAL_REPS=3) so t5-base/large scores are
directly comparable to the t5-small baseline (single seeds 0.725-0.732, 3-seed ens 0.7268).
Reports per-epoch timing to project A10G runtime.  Usage:
  python h100_cv.py MODEL EPOCHS K BS LR N_SEEDS
  e.g. python h100_cv.py t5-base 6 6 16 2e-4 1
"""
import sys, os, time, collections, numpy as np, pandas as pd, torch
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from transformers import T5TokenizerFast, T5ForConditionalGeneration, get_linear_schedule_with_warmup
sys.path.insert(0, str(Path(__file__).resolve().parent))
from aug import SLOTS, parse_train_row, make_example, parse_seq
# target slot order (affects only the generated string, not the metric). env TGT_ORDER=weight or csv
_WORD={"prep":2.20,"activation":0.85,"order":0.60,"control":3.00,"quench":4.00,"workup":0.25}
_o=os.environ.get("TGT_ORDER","")
if _o=="weight": TGT=sorted(SLOTS,key=lambda s:-_WORD[s])
elif _o: TGT=_o.split(",")
else: TGT=list(SLOTS)
def seq_str_ord(t): return ";".join(f"{s}={t[s]}" for s in TGT)
print("TGT order:",TGT,flush=True)

MODEL   = sys.argv[1] if len(sys.argv)>1 else "t5-base"
EPOCHS  = int(sys.argv[2]) if len(sys.argv)>2 else 6
K       = int(sys.argv[3]) if len(sys.argv)>3 else 6
BS      = int(sys.argv[4]) if len(sys.argv)>4 else 16
LR      = float(sys.argv[5]) if len(sys.argv)>5 else 2e-4
N_SEEDS = int(sys.argv[6]) if len(sys.argv)>6 else 1

W={"prep":2.20,"activation":0.85,"order":0.60,"control":3.00,"quench":4.00,"workup":0.25}; WSUM=sum(W.values())
DS=Path("/mnt/work/c1"); MSRC=192; MTGT=48; VAL_FRAC=0.15; VAL_REPS=3
dev="cuda"
torch.backends.cuda.matmul.allow_tf32=True; torch.backends.cudnn.allow_tf32=True
print(f"MODEL={MODEL} EPOCHS={EPOCHS} K={K} BS={BS} LR={LR} N_SEEDS={N_SEEDS}",flush=True)

def row_score(p,t): return sum(W[s]*(p.get(s)==t.get(s)) for s in SLOTS)/WSUM
class DS_(Dataset):
    def __init__(s,pairs,tok): s.p=pairs; s.tok=tok
    def __len__(s): return len(s.p)
    def __getitem__(s,i):
        src,tgt=s.p[i]; x=s.tok(src,max_length=MSRC,truncation=True,padding="max_length",return_tensors="pt")
        y=s.tok(text_target=tgt,max_length=MTGT,truncation=True,padding="max_length",return_tensors="pt")
        lab=y["input_ids"].squeeze(0); lab[lab==s.tok.pad_token_id]=-100
        return {"input_ids":x["input_ids"].squeeze(0),"attention_mask":x["attention_mask"].squeeze(0),"labels":lab}

train=pd.read_csv(DS/"train.csv"); recs=[parse_train_row(r) for _,r in train.iterrows()]
fams=np.array([r["family"] for r in recs]); rng=np.random.default_rng(42); val_idx=[]
for f in sorted(set(fams)):
    idx=np.where(fams==f)[0]; rng.shuffle(idx); val_idx+=list(idx[:int(len(idx)*VAL_FRAC)])
val_idx=set(int(x) for x in val_idx); tr=[r for i,r in enumerate(recs) if i not in val_idx]; va=[r for i,r in enumerate(recs) if i in val_idx]
vocab={s:set() for s in SLOTS}; famtbl=collections.defaultdict(lambda:collections.defaultdict(collections.Counter))
for r in tr:
    for s in SLOTS: vocab[s].add(r["truth"][s]); famtbl[r["family"]][s][r["truth"][s]]+=1
glob={s:collections.Counter([r["truth"][s] for r in tr]).most_common(1)[0][0] for s in SLOTS}
def fmode(fam,s):
    c=famtbl.get(fam,{}).get(s); return c.most_common(1)[0][0] if c else glob[s]
print(f"train {len(tr)} val {len(va)}",flush=True)

valrng=np.random.default_rng(12345); val_srcs=[]; val_truth=[]; val_fam=[]
for rep in range(VAL_REPS):
    for r in va:
        inp,_=make_example(r,valrng,n_show=3); val_srcs.append(inp); val_truth.append(r["truth"]); val_fam.append(r["family"])

def train_one(seed):
    torch.manual_seed(seed); ar=np.random.default_rng(seed+1); pairs=[]
    for r in tr:
        for _ in range(K):
            inp,_=make_example(r,ar,n_show=3); pairs.append((inp,seq_str_ord(r["truth"])))
        inp,_=make_example(r,ar,n_show=6); pairs.append((inp,seq_str_ord(r["truth"])))
    ar.shuffle(pairs)
    tok=T5TokenizerFast.from_pretrained(MODEL); model=T5ForConditionalGeneration.from_pretrained(MODEL).to(dev)
    dl=DataLoader(DS_(pairs,tok),batch_size=BS,shuffle=True,num_workers=8,pin_memory=True)
    opt=torch.optim.AdamW(model.parameters(),lr=LR)
    tot=len(dl)*EPOCHS; sch=get_linear_schedule_with_warmup(opt,int(0.05*tot),tot); model.train()
    for ep in range(EPOCHS):
        te=time.time()
        for b in dl:
            b={k:v.to(dev,non_blocking=True) for k,v in b.items()}
            with torch.autocast("cuda",dtype=torch.bfloat16): loss=model(**b).loss
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step(); sch.step()
        print(f"  seed{seed} ep{ep} {time.time()-te:.1f}s loss={loss.item():.3f}",flush=True)
    return tok,model

def predict(tok,model,srcs):
    model.eval(); out=[]
    for i in range(0,len(srcs),128):
        enc=tok(srcs[i:i+128],max_length=MSRC,truncation=True,padding=True,return_tensors="pt").to(dev)
        with torch.no_grad(), torch.autocast("cuda",dtype=torch.bfloat16):
            g=model.generate(**enc,max_new_tokens=MTGT,num_beams=1)
        out+=tok.batch_decode(g,skip_special_tokens=True)
    return out

t0=time.time(); all_parsed=[]
for s in range(N_SEEDS):
    ttr=time.time(); tok,model=train_one(100+s); traint=time.time()-ttr
    outs=predict(tok,model,val_srcs)
    parsed=[]
    for o,fam in zip(outs,val_fam):
        d=parse_seq(o); pr={sl:(d.get(sl) if d.get(sl) in vocab[sl] else fmode(fam,sl)) for sl in SLOTS}; parsed.append(pr)
    all_parsed.append(parsed)
    sc=np.mean([row_score(parsed[i],val_truth[i]) for i in range(len(parsed))])
    print(f"seed {s} single CV={sc:.4f}  train={traint:.0f}s  [{time.time()-t0:.0f}s]",flush=True)
    del model; torch.cuda.empty_cache()

if N_SEEDS>1:
    ens=[]
    for i in range(len(val_truth)):
        pr={}
        for sl in SLOTS:
            votes=collections.Counter(all_parsed[m][i][sl] for m in range(N_SEEDS)); pr[sl]=votes.most_common(1)[0][0]
        ens.append(pr)
    print(f"ENSEMBLE ({N_SEEDS} seeds) CV={np.mean([row_score(ens[i],val_truth[i]) for i in range(len(ens))]):.4f}",flush=True)
    fin=ens
else:
    fin=all_parsed[0]
for sl in SLOTS:
    print(f"  {sl:10s} acc={np.mean([fin[i][sl]==val_truth[i][sl] for i in range(len(fin))]):.3f}",flush=True)
