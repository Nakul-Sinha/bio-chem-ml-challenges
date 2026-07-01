"""Blend reordered-seq2seq (weight order) + classifier heads on the SAME val split.
seq2seq -> per-slot vote distribution over M seeds; classifier -> avg softmax over N seeds.
Sweep blend weight w: pred = argmax( w*seq2seq_votes + (1-w)*clf_probs ). Reports each alone + blend.
Usage: python h100_blend.py M_SEQ N_CLF EPOCHS_SEQ EPOCHS_CLF
"""
import sys, time, collections, numpy as np, pandas as pd, torch, torch.nn as nn
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from transformers import (T5TokenizerFast, T5ForConditionalGeneration, T5EncoderModel,
                          get_linear_schedule_with_warmup)
sys.path.insert(0, str(Path(__file__).resolve().parent))
from aug import SLOTS, parse_train_row, make_example, parse_seq

M_SEQ   = int(sys.argv[1]) if len(sys.argv)>1 else 3
N_CLF   = int(sys.argv[2]) if len(sys.argv)>2 else 3
EP_SEQ  = int(sys.argv[3]) if len(sys.argv)>3 else 6
EP_CLF  = int(sys.argv[4]) if len(sys.argv)>4 else 10
W={"prep":2.20,"activation":0.85,"order":0.60,"control":3.00,"quench":4.00,"workup":0.25}; WSUM=sum(W.values())
WORD=W; TGT=sorted(SLOTS,key=lambda s:-WORD[s])   # weight order for seq2seq target
DS=Path("/mnt/work/c1"); MSRC=192; MTGT=48; VAL_FRAC=0.15; VAL_REPS=3; dev="cuda"
torch.backends.cuda.matmul.allow_tf32=True; torch.backends.cudnn.allow_tf32=True
print(f"BLEND M_SEQ={M_SEQ} N_CLF={N_CLF} EP_SEQ={EP_SEQ} EP_CLF={EP_CLF} TGT={TGT}",flush=True)

train=pd.read_csv(DS/"train.csv"); recs=[parse_train_row(r) for _,r in train.iterrows()]
fams=np.array([r["family"] for r in recs]); rng=np.random.default_rng(42); val_idx=[]
for f in sorted(set(fams)):
    idx=np.where(fams==f)[0]; rng.shuffle(idx); val_idx+=list(idx[:int(len(idx)*VAL_FRAC)])
val_idx=set(int(x) for x in val_idx); tr=[r for i,r in enumerate(recs) if i not in val_idx]; va=[r for i,r in enumerate(recs) if i in val_idx]
vocab={s:set() for s in SLOTS}; famtbl=collections.defaultdict(lambda:collections.defaultdict(collections.Counter))
V2I={s:{} for s in SLOTS}; I2V={s:[] for s in SLOTS}
for r in recs:
    for s in SLOTS:
        v=r["truth"][s]
        if v not in V2I[s]: V2I[s][v]=len(I2V[s]); I2V[s].append(v)
for r in tr:
    for s in SLOTS: vocab[s].add(r["truth"][s]); famtbl[r["family"]][s][r["truth"][s]]+=1
glob={s:collections.Counter([r["truth"][s] for r in tr]).most_common(1)[0][0] for s in SLOTS}
def fmode(fam,s):
    c=famtbl.get(fam,{}).get(s); return c.most_common(1)[0][0] if c else glob[s]
NC={s:len(I2V[s]) for s in SLOTS}
def seq_str_ord(t): return ";".join(f"{s}={t[s]}" for s in TGT)

valrng=np.random.default_rng(12345); val_srcs=[]; val_truth=[]; val_fam=[]
for rep in range(VAL_REPS):
    for r in va:
        inp,_=make_example(r,valrng,n_show=3); val_srcs.append(inp); val_truth.append(r["truth"]); val_fam.append(r["family"])
N=len(val_srcs)
def row_score(p,t): return sum(W[s]*(p.get(s)==t.get(s)) for s in SLOTS)/WSUM

# ---- seq2seq ----
tok=T5TokenizerFast.from_pretrained("t5-small")
class DSs(Dataset):
    def __init__(s,p): s.p=p
    def __len__(s): return len(s.p)
    def __getitem__(s,i):
        src,tgt=s.p[i]; x=tok(src,max_length=MSRC,truncation=True,padding="max_length",return_tensors="pt")
        y=tok(text_target=tgt,max_length=MTGT,truncation=True,padding="max_length",return_tensors="pt")
        lab=y["input_ids"].squeeze(0); lab[lab==tok.pad_token_id]=-100
        return x["input_ids"].squeeze(0),x["attention_mask"].squeeze(0),lab
def seq_votes():
    votes=[{s:collections.Counter() for s in SLOTS} for _ in range(N)]
    for sd in range(M_SEQ):
        torch.manual_seed(100+sd); ar=np.random.default_rng(101+sd); pairs=[]
        for r in tr:
            for _ in range(6): inp,_=make_example(r,ar,n_show=3); pairs.append((inp,seq_str_ord(r["truth"])))
            inp,_=make_example(r,ar,n_show=6); pairs.append((inp,seq_str_ord(r["truth"])))
        ar.shuffle(pairs)
        model=T5ForConditionalGeneration.from_pretrained("t5-small").to(dev)
        dl=DataLoader(DSs(pairs),batch_size=16,shuffle=True,num_workers=8,pin_memory=True)
        opt=torch.optim.AdamW(model.parameters(),lr=3e-4); tot=len(dl)*EP_SEQ
        sch=get_linear_schedule_with_warmup(opt,int(.05*tot),tot); model.train()
        for ep in range(EP_SEQ):
            for ids,mask,lab in dl:
                ids,mask,lab=ids.to(dev),mask.to(dev),lab.to(dev)
                with torch.autocast("cuda",dtype=torch.bfloat16): loss=model(input_ids=ids,attention_mask=mask,labels=lab).loss
                opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step(); sch.step()
        model.eval(); outs=[]
        for i in range(0,N,128):
            enc=tok(val_srcs[i:i+128],max_length=MSRC,truncation=True,padding=True,return_tensors="pt").to(dev)
            with torch.no_grad(), torch.autocast("cuda",dtype=torch.bfloat16):
                g=model.generate(**enc,max_new_tokens=MTGT,num_beams=1)
            outs+=tok.batch_decode(g,skip_special_tokens=True)
        for i,(o,fam) in enumerate(zip(outs,val_fam)):
            d=parse_seq(o)
            for s in SLOTS:
                v=d.get(s) if d.get(s) in vocab[s] else fmode(fam,s); votes[i][s][v]+=1
        print(f"  seq seed{sd} done",flush=True); del model; torch.cuda.empty_cache()
    # to prob matrices per slot
    P={s:np.zeros((N,NC[s]),dtype=np.float32) for s in SLOTS}
    for i in range(N):
        for s in SLOTS:
            for v,c in votes[i][s].items():
                if v in V2I[s]: P[s][i,V2I[s][v]]=c/M_SEQ
    return P

# ---- classifier ----
class Net(nn.Module):
    def __init__(s):
        super().__init__(); s.enc=T5EncoderModel.from_pretrained("t5-small"); h=s.enc.config.d_model
        s.drop=nn.Dropout(0.2); s.heads=nn.ModuleList([nn.Linear(h,NC[sl]) for sl in SLOTS])
    def forward(s,ids,mask):
        o=s.enc(input_ids=ids,attention_mask=mask).last_hidden_state
        m=mask.unsqueeze(-1).float(); z=(o*m).sum(1)/m.sum(1).clamp(min=1); z=s.drop(z)
        return [head(z) for head in s.heads]
class DSc(Dataset):
    def __init__(s,it): s.it=it
    def __len__(s): return len(s.it)
    def __getitem__(s,i):
        src,truth=s.it[i]; x=tok(src,max_length=MSRC,truncation=True,padding="max_length",return_tensors="pt")
        y=torch.tensor([V2I[sl][truth[sl]] for sl in SLOTS])
        return x["input_ids"].squeeze(0),x["attention_mask"].squeeze(0),y
def clf_probs():
    acc={s:np.zeros((N,NC[s]),dtype=np.float32) for s in SLOTS}
    for sd in range(N_CLF):
        torch.manual_seed(100+sd); ar=np.random.default_rng(101+sd); items=[]
        for r in tr:
            for _ in range(6): inp,_=make_example(r,ar,n_show=3); items.append((inp,r["truth"]))
            inp,_=make_example(r,ar,n_show=6); items.append((inp,r["truth"]))
        net=Net().to(dev); dl=DataLoader(DSc(items),batch_size=32,shuffle=True,num_workers=8,pin_memory=True)
        opt=torch.optim.AdamW(net.parameters(),lr=3e-4,weight_decay=1e-5); ce=nn.CrossEntropyLoss()
        tot=len(dl)*EP_CLF; sch=get_linear_schedule_with_warmup(opt,int(.05*tot),tot); net.train()
        for ep in range(EP_CLF):
            for ids,mask,y in dl:
                ids,mask,y=ids.to(dev),mask.to(dev),y.to(dev)
                with torch.autocast("cuda",dtype=torch.bfloat16): lg=net(ids,mask)
                loss=sum(ce(lg[j],y[:,j]) for j in range(len(SLOTS)))
                opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(net.parameters(),1.0); opt.step(); sch.step()
        net.eval()
        with torch.no_grad():
            for i in range(0,N,256):
                x=tok(val_srcs[i:i+256],max_length=MSRC,truncation=True,padding=True,return_tensors="pt").to(dev)
                with torch.autocast("cuda",dtype=torch.bfloat16): lg=net(x["input_ids"],x["attention_mask"])
                for j,s in enumerate(SLOTS): acc[s][i:i+256]+=torch.softmax(lg[j].float(),1).cpu().numpy()/N_CLF
        print(f"  clf seed{sd} done",flush=True); del net; torch.cuda.empty_cache()
    return acc

t0=time.time(); Pseq=seq_votes(); print(f"seq done [{time.time()-t0:.0f}s]",flush=True)
Pclf=clf_probs(); print(f"clf done [{time.time()-t0:.0f}s]",flush=True)
def score_from(P):
    sc=0
    for i in range(N):
        pred={s:I2V[s][P[s][i].argmax()] for s in SLOTS}; sc+=row_score(pred,val_truth[i])
    return sc/N
print(f"\nseq2seq(reorder) alone CV={score_from(Pseq):.4f}")
print(f"classifier alone    CV={score_from(Pclf):.4f}")
for w in [0.3,0.4,0.5,0.6,0.7]:
    P={s:w*Pseq[s]+(1-w)*Pclf[s] for s in SLOTS}
    print(f"blend w_seq={w:.1f}       CV={score_from(P):.4f}")
