import os, re, sys, time, collections
from pathlib import Path
import numpy as np, pandas as pd, torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import T5TokenizerFast, T5EncoderModel, get_linear_schedule_with_warmup

SEED       = 42
ENC        = "t5-small"
EPOCHS     = 10
K_AUG      = 6
N_DENSE    = 1
BATCH      = 32
LR         = 3e-4
MAX_SRC    = 192
MAX_SEEDS  = 5
BUDGET_S   = 1380

SLOTS   = ["prep","activation","order","control","quench","workup"]
NOTE_KEY= {"setup":"prep","activation":"activation","order":"order","control":"control","stop":"quench","cleanup":"workup"}
PREFIX  = re.compile(r"\b([a-z]{3,6}-[a-z]{3,6})-\d+[A-Z]\b")
CORR_DESC={"opening handling line":"prep","line before reactive contact":"activation",
    "line describing which material waits":"order","condition maintained during the hold":"control",
    "operation that ends reactivity":"quench","cleanup operation":"workup"}
SLOT_DESC={v:k for k,v in CORR_DESC.items()}
FAMILIES=["imine reduction","resin exchange","cross coupling","carbonate closure",
          "salt metathesis","benzylic oxidation","acyl transfer","photoredox capture"]
HEADERS=["The reaction family is logged as {f}.","Header family: {f}.","The planner groups this run under {f}."]
NOTE_TPL=["margin mark {t} is written beside that operation","the operation is abbreviated only as {t}",
          "the copy keeps shorthand mark {t}","the copied operation carries bench tag {t}","the retyped line preserves local tag {t}"]
MISS_TPL=["A secondary operation line is missing from the retyped page.",
          "The copy omits one background operation that must be inferred from context.",
          "One non-edited handling line is smudged in the copy."]
CORR_TPL=["Audit repair: the {d} should carry local tag {t}; leave unrelated operations unchanged.",
    "Silent edit: replace the {d} with the operation marked {t} in the corrected record.",
    "Post-run note: repair the {d} to bench tag {t}; keep other operations from the note.",
    "QC note: corrected entry for the {d} is tagged {t}, not the copied line."]
REQUEST="Generate the repaired canonical protocol sequence."

def find_data_dir():
    here=Path(__file__).resolve().parent
    for c in [here/"dataset"/"public",here/"dataset",Path("dataset/public"),Path("dataset"),here,Path(".")]:
        if (c/"train.csv").exists() and (c/"test.csv").exists(): return c
    raise FileNotFoundError("train.csv/test.csv not found")
def parse_seq(s):
    d={}
    for p in str(s).split(";"):
        if "=" in p: k,v=p.split("=",1); d[k.strip()]=v.strip()
    return d
def get_family(prompt):
    f=str(prompt).split("\n")[0].lower()
    for fam in FAMILIES:
        if fam in f: return fam
    return "?"
def corr_slot_prefix(cn):
    cn=str(cn); slot=None
    for desc,sl in CORR_DESC.items():
        if desc in cn: slot=sl; break
    pf=PREFIX.findall(cn); return slot,(pf[-1] if pf else None)
def note_prefix_by_slot(note):
    out={}; note=str(note)
    for kw,slot in NOTE_KEY.items():
        m=re.search(rf"(?:^|[.\s]){kw}\b(.*?)(?:\.|$)", note, re.I)
        if m:
            pf=PREFIX.findall(m.group(1))
            if pf: out[slot]=pf[-1]
    return out
def parse_train_row(row):
    return dict(family=get_family(row["prompt"]),note_pfx=note_prefix_by_slot(row["protocol_note"]),
                **dict(zip(("cslot","cpfx"),corr_slot_prefix(row["correction_notice"]))),truth=parse_seq(row["repaired_sequence"]))
def rand_tag(pfx,rng): return f"{pfx}-{rng.integers(10,99)}{chr(ord('A')+int(rng.integers(0,9)))}"
def rand_code(rng): return chr(ord('A')+int(rng.integers(0,8)))+str(int(rng.integers(1000,3999)))
def make_example(rec,rng,n_show=3):
    fam,note_pfx,cslot,cpfx=rec["family"],rec["note_pfx"],rec["cslot"],rec["cpfx"]
    header=rng.choice(HEADERS).format(f=fam); avail=[s for s in SLOTS if s in note_pfx]
    k=min(n_show,len(avail)); show=list(rng.choice(avail,size=k,replace=False)) if avail else []
    note="Audit copy "+rand_code(rng)+". "; joined=""
    for i,s in enumerate(show):
        tag=rand_tag(note_pfx[s],rng); sep="" if i==0 else ("; " if rng.random()<0.5 else ". ")
        joined+=sep+rng.choice(NOTE_TPL).format(t=tag)
    note+=joined+". "+rng.choice(MISS_TPL); corr=""
    if cslot and cpfx: corr=rng.choice(CORR_TPL).format(d=SLOT_DESC[cslot],t=rand_tag(cpfx,rng))
    return "\n".join(x for x in [header,note,corr,REQUEST] if x)

def main():
    np.random.seed(SEED); torch.manual_seed(SEED)
    torch.backends.cuda.matmul.allow_tf32=True; torch.backends.cudnn.allow_tf32=True
    dev="cuda" if torch.cuda.is_available() else "cpu"
    bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    DATA=find_data_dir(); print("data dir:",DATA,"device:",dev,flush=True)
    train=pd.read_csv(DATA/"train.csv"); test=pd.read_csv(DATA/"test.csv")
    recs=[parse_train_row(r) for _,r in train.iterrows()]
    V2I={s:{} for s in SLOTS}; I2V={s:[] for s in SLOTS}
    famtbl=collections.defaultdict(lambda:collections.defaultdict(collections.Counter))
    for r in recs:
        for s in SLOTS:
            v=r["truth"][s]
            if v not in V2I[s]: V2I[s][v]=len(I2V[s]); I2V[s].append(v)
            famtbl[r["family"]][s][v]+=1
    glob={s:collections.Counter([r["truth"][s] for r in recs]).most_common(1)[0][0] for s in SLOTS}
    def fmode(fam,s):
        c=famtbl.get(fam,{}).get(s); return c.most_common(1)[0][0] if c else glob[s]
    NC={s:len(I2V[s]) for s in SLOTS}
    tok=T5TokenizerFast.from_pretrained(ENC)
    srcs=test["prompt"].astype(str).tolist()

    class DSc(Dataset):
        def __init__(s,items): s.it=items
        def __len__(s): return len(s.it)
        def __getitem__(s,i):
            src,truth=s.it[i]; x=tok(src,max_length=MAX_SRC,truncation=True,padding="max_length",return_tensors="pt")
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

    def train_one(seed):
        torch.manual_seed(seed); ar=np.random.default_rng(seed+1); items=[]
        for r in recs:
            for _ in range(K_AUG): items.append((make_example(r,ar,n_show=3),r["truth"]))
            for _ in range(N_DENSE): items.append((make_example(r,ar,n_show=6),r["truth"]))
        net=Net().to(dev); dl=DataLoader(DSc(items),batch_size=BATCH,shuffle=True,num_workers=min(8,os.cpu_count() or 2),pin_memory=True)
        opt=torch.optim.AdamW(net.parameters(),lr=LR,weight_decay=1e-5); ce=nn.CrossEntropyLoss()
        tot=len(dl)*EPOCHS; sch=get_linear_schedule_with_warmup(opt,int(0.05*tot),tot); net.train()
        for ep in range(EPOCHS):
            for ids,mask,y in dl:
                ids,mask,y=ids.to(dev),mask.to(dev),y.to(dev)
                with torch.autocast("cuda",dtype=torch.bfloat16,enabled=bf16): logits=net(ids,mask)
                loss=sum(ce(logits[j],y[:,j]) for j in range(len(SLOTS)))
                opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(net.parameters(),1.0); opt.step(); sch.step()
        return net
    def predict(net):
        net.eval(); probs=[[] for _ in SLOTS]
        with torch.no_grad():
            for i in range(0,len(srcs),256):
                x=tok(srcs[i:i+256],max_length=MAX_SRC,truncation=True,padding=True,return_tensors="pt").to(dev)
                with torch.autocast("cuda",dtype=torch.bfloat16,enabled=bf16): logits=net(x["input_ids"],x["attention_mask"])
                for j in range(len(SLOTS)): probs[j].append(torch.softmax(logits[j].float(),1).cpu().numpy())
        return [np.concatenate(p) for p in probs]

    t0=time.time(); accum=None; nseed=0
    for sd in range(MAX_SEEDS):
        if nseed>=1:
            per=(time.time()-t0)/nseed
            if (time.time()-t0)+per>BUDGET_S: break
        net=train_one(SEED+sd); pr=predict(net)
        accum=pr if accum is None else [a+b for a,b in zip(accum,pr)]
        nseed+=1; print(f"seed {sd} done nseed={nseed} [{time.time()-t0:.0f}s]",flush=True)
        del net
        if torch.cuda.is_available(): torch.cuda.empty_cache()

    rows=[]
    for i,(_,trow) in enumerate(test.iterrows()):
        fam=get_family(trow["prompt"])
        pred={}
        for j,s in enumerate(SLOTS):
            k=int(accum[j][i].argmax()); pred[s]=I2V[s][k] if I2V[s] else fmode(fam,s)
        rows.append({"id":trow["id"],"repaired_sequence":";".join(f"{s}={pred[s]}" for s in SLOTS)})
    sub=pd.DataFrame(rows,columns=["id","repaired_sequence"])
    assert list(sub.columns)==["id","repaired_sequence"] and len(sub)==len(test) and sub["id"].is_unique and set(sub["id"])==set(test["id"])
    for s in sub["repaired_sequence"]:
        d=parse_seq(s); assert all(k in d for k in SLOTS) and s.count(";")==5
    Path("working").mkdir(exist_ok=True); sub.to_csv("working/submission.csv",index=False); sub.to_csv("submission.csv",index=False)
    print(f"wrote submission.csv seeds={nseed}",sub.shape,f"[{time.time()-t0:.0f}s]",flush=True)

if __name__=="__main__":
    main()
