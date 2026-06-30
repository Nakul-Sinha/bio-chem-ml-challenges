"""CV: fine-tune T5 on test-matched augmented data; score weighted metric on degraded val."""
import sys, os, time, argparse, collections, re
import numpy as np, pandas as pd, torch
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from transformers import T5TokenizerFast, T5ForConditionalGeneration, get_linear_schedule_with_warmup
sys.path.insert(0, str(Path(__file__).resolve().parent))
import aug
from aug import SLOTS, parse_train_row, make_example, seq_str, parse_seq

W = {"prep":2.20,"activation":0.85,"order":0.60,"control":3.00,"quench":4.00,"workup":0.25}; WSUM=sum(W.values())
DS = Path(__file__).resolve().parent.parent / "dataset"

def set_seed(s):
    np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)

def row_score(pred,true):
    return sum(W[s]*(pred.get(s)==true.get(s)) for s in SLOTS)/WSUM

class DS_(Dataset):
    def __init__(self, pairs, tok, msrc, mtgt):
        self.pairs=pairs; self.tok=tok; self.msrc=msrc; self.mtgt=mtgt
    def __len__(self): return len(self.pairs)
    def __getitem__(self,i):
        src,tgt=self.pairs[i]
        x=self.tok(src,max_length=self.msrc,truncation=True,padding="max_length",return_tensors="pt")
        y=self.tok(text_target=tgt,max_length=self.mtgt,truncation=True,padding="max_length",return_tensors="pt")
        labels=y["input_ids"].squeeze(0); labels[labels==self.tok.pad_token_id]=-100
        return {"input_ids":x["input_ids"].squeeze(0),"attention_mask":x["attention_mask"].squeeze(0),"labels":labels}

def build_vocab(parsed_records):
    voc={s:collections.Counter() for s in SLOTS}
    for r in parsed_records:
        for s in SLOTS: voc[s][r["truth"][s]]+=1
    return {s:set(voc[s]) for s in voc}, voc

def fam_mode_table(records):
    tbl=collections.defaultdict(lambda: collections.defaultdict(collections.Counter))
    for r in records:
        for s in SLOTS: tbl[r["family"]][s][r["truth"][s]]+=1
    return tbl

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--model",default="t5-small")
    ap.add_argument("--epochs",type=int,default=6)
    ap.add_argument("--K",type=int,default=8)          # augment copies per row
    ap.add_argument("--bs",type=int,default=16)
    ap.add_argument("--lr",type=float,default=3e-4)
    ap.add_argument("--msrc",type=int,default=192)
    ap.add_argument("--mtgt",type=int,default=48)
    ap.add_argument("--val_frac",type=float,default=0.15)
    ap.add_argument("--seed",type=int,default=42)
    ap.add_argument("--val_reps",type=int,default=3)    # avg metric over N degradations of val
    ap.add_argument("--limit",type=int,default=0)       # debug: cap train rows
    args=ap.parse_args(); print(vars(args))
    set_seed(args.seed)
    dev="cuda" if torch.cuda.is_available() else "cpu"

    train=pd.read_csv(DS/"train.csv")
    if args.limit: train=train.iloc[:args.limit].reset_index(drop=True)
    recs=[parse_train_row(r) for _,r in train.iterrows()]
    fams=np.array([r["family"] for r in recs])
    # stratified split by family
    rng=np.random.default_rng(args.seed)
    val_idx=[]
    for f in set(fams):
        idx=np.where(fams==f)[0]; rng.shuffle(idx)
        n=int(len(idx)*args.val_frac); val_idx+=list(idx[:n])
    val_idx=set(val_idx)
    tr=[r for i,r in enumerate(recs) if i not in val_idx]
    va=[r for i,r in enumerate(recs) if i in val_idx]
    print(f"train rows {len(tr)} val rows {len(va)}")
    vocab,_=build_vocab(tr); famtbl=fam_mode_table(tr)
    glob_mode={s:max(collections.Counter([r['truth'][s] for r in tr]).items(),key=lambda kv:kv[1])[0] for s in SLOTS}
    def fmode(fam,s):
        c=famtbl.get(fam,{}).get(s); return c.most_common(1)[0][0] if c else glob_mode[s]

    # build augmented training pairs
    augrng=np.random.default_rng(args.seed+1)
    pairs=[]
    for r in tr:
        for _ in range(args.K):
            pairs.append(make_example(r,augrng,n_show=3))
        pairs.append(make_example(r,augrng,n_show=6))  # one denser
    augrng.shuffle(pairs)
    print("augmented train pairs:",len(pairs))

    tok=T5TokenizerFast.from_pretrained(args.model)
    model=T5ForConditionalGeneration.from_pretrained(args.model).to(dev)
    ds=DS_(pairs,tok,args.msrc,args.mtgt)
    dl=DataLoader(ds,batch_size=args.bs,shuffle=True,num_workers=0,pin_memory=True)
    opt=torch.optim.AdamW(model.parameters(),lr=args.lr)
    total=len(dl)*args.epochs
    sch=get_linear_schedule_with_warmup(opt,int(0.05*total),total)
    use_bf16=torch.cuda.is_available()
    model.train(); t0=time.time()
    for ep in range(args.epochs):
        tot=0
        for bi,b in enumerate(dl):
            b={k:v.to(dev) for k,v in b.items()}
            with torch.autocast("cuda",dtype=torch.bfloat16,enabled=use_bf16):
                out=model(**b); loss=out.loss
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
            opt.step(); sch.step(); tot+=loss.item()
        print(f"  epoch {ep+1}/{args.epochs} loss {tot/len(dl):.4f}  [{time.time()-t0:.0f}s]")

    # eval on degraded val (avg over reps)
    model.eval()
    valrng=np.random.default_rng(12345)
    def predict(srcs):
        preds=[]
        for i in range(0,len(srcs),64):
            chunk=srcs[i:i+64]
            enc=tok(chunk,max_length=args.msrc,truncation=True,padding=True,return_tensors="pt").to(dev)
            with torch.no_grad(), torch.autocast("cuda",dtype=torch.bfloat16,enabled=use_bf16):
                g=model.generate(**enc,max_new_tokens=48,num_beams=1)
            preds+=tok.batch_decode(g,skip_special_tokens=True)
        return preds
    all_scores=[]; perslot=collections.defaultdict(list)
    for rep in range(args.val_reps):
        srcs=[]; truths=[]; famsv=[]
        for r in va:
            inp,_=make_example(r,valrng,n_show=3)
            srcs.append(inp); truths.append(r["truth"]); famsv.append(r["family"])
        outs=predict(srcs)
        for o,true,fam in zip(outs,truths,famsv):
            d=parse_seq(o)
            pred={}
            for s in SLOTS:
                v=d.get(s)
                if v not in vocab[s]:  # invalid/missing -> family mode (allowed post-proc)
                    v=fmode(fam,s)
                pred[s]=v
            all_scores.append(row_score(pred,true))
            for s in SLOTS: perslot[s].append(pred[s]==true[s])
    print(f"\n=== CV weighted score: {np.mean(all_scores):.4f} (n={len(all_scores)}) ===")
    for s in SLOTS: print(f"   {s:10s} w={W[s]:<4} acc={np.mean(perslot[s]):.3f}")

if __name__=="__main__":
    main()
