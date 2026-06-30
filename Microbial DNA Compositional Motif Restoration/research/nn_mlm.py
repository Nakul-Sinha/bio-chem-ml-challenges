"""Masked-LM transformer over contig signature (16) + genome_context (16). Group-CV, MRR@10."""
import sys, time, numpy as np, pandas as pd, torch, torch.nn as nn
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from common import load, genome_key, mrr10, group_folds, VOCAB
dev="cuda" if torch.cuda.is_available() else "cpu"
MASK=VOCAB+1  # 1025 ; tokens are 1..1024, 0=pad
MASKPOS=list(range(4,16))
D=192; LAYERS=4; HEADS=6; FF=512; DROP=0.1
EPOCHS=30; BS=256; LR=2e-3

tr,te=load()
keys=[genome_key(g) for g in tr["gc"]]
folds=group_folds(keys,n=5,seed=0)
K=np.array([r for r in tr["kseq"]],dtype=np.int64)        # (N,16) tokens 1..1024
G=np.array([r for r in tr["gc"]],dtype=np.int64)          # (N,16)

class MLM(nn.Module):
    def __init__(s):
        super().__init__()
        s.tok=nn.Embedding(VOCAB+2,D,padding_idx=0)        # 0..1025
        s.pos=nn.Embedding(32,D); s.seg=nn.Embedding(2,D)
        enc=nn.TransformerEncoderLayer(D,HEADS,FF,DROP,batch_first=True,activation="gelu")
        s.tr=nn.TransformerEncoder(enc,LAYERS)
        s.head=nn.Sequential(nn.LayerNorm(D),nn.Linear(D,D),nn.GELU(),nn.Linear(D,VOCAB))
    def forward(s,contig,gc):
        B=contig.shape[0]
        x=torch.cat([contig,gc],1)                          # (B,32)
        posid=torch.arange(32,device=x.device)[None,:].expand(B,32)
        segid=torch.cat([torch.zeros(16,dtype=torch.long),torch.ones(16,dtype=torch.long)]).to(x.device)[None,:].expand(B,32)
        h=s.tok(x)+s.pos(posid)+s.seg(segid)
        h=s.tr(h)
        return s.head(h[:,:16])                              # (B,16,VOCAB) logits for contig positions

def make_batch(Kb,Gb,rng,train=True,mask_pos=None):
    contig=torch.tensor(Kb.copy()); gc=torch.tensor(Gb)
    B=len(Kb); mp=np.empty(B,dtype=np.int64); gold=np.empty(B,dtype=np.int64)
    for i in range(B):
        p=rng.choice(MASKPOS) if train else mask_pos[i]
        gold[i]=Kb[i,p]; mp[i]=p; contig[i,p]=MASK
    return contig.to(dev),gc.to(dev),torch.tensor(mp).to(dev),torch.tensor(gold-1).to(dev)

def train_model(tri,seed=0):
    torch.manual_seed(seed); rng=np.random.default_rng(seed)
    net=MLM().to(dev); opt=torch.optim.AdamW(net.parameters(),lr=LR,weight_decay=1e-4)
    Kt,Gt=K[tri],G[tri]; n=len(tri)
    steps=EPOCHS*((n+BS-1)//BS); sch=torch.optim.lr_scheduler.OneCycleLR(opt,LR,total_steps=steps)
    ce=nn.CrossEntropyLoss()
    for ep in range(EPOCHS):
        net.train(); perm=rng.permutation(n)
        for i in range(0,n,BS):
            idx=perm[i:i+BS]; contig,gc,mp,gold=make_batch(Kt[idx],Gt[idx],rng,True)
            logits=net(contig,gc)                            # (B,16,V)
            lg=logits[torch.arange(len(idx)),mp]             # (B,V)
            loss=ce(lg,gold)
            opt.zero_grad(); loss.backward(); opt.step(); sch.step()
    return net

@torch.no_grad()
def predict(net,Kb,Gb,mask_pos):
    net.eval(); contig=torch.tensor(Kb.copy());
    for i in range(len(Kb)): contig[i,mask_pos[i]]=MASK
    out=[]
    for i in range(0,len(Kb),512):
        c=contig[i:i+512].to(dev); g=torch.tensor(Gb[i:i+512]).to(dev)
        logits=net(c,g); mp=torch.tensor(mask_pos[i:i+512]).to(dev)
        lg=logits[torch.arange(len(mp)),mp].cpu().numpy()    # (b,V) for tokens 1..1024
        out.append(lg)
    return np.concatenate(out)

def topk_excl(scores,vis,k=10):
    s=scores.copy()
    for v in vis:
        if v is not None and 1<=v<=VOCAB: s[v-1]=-1e9
    idx=np.argpartition(-s,k)[:k]; idx=idx[np.argsort(-s[idx])]
    return [int(i+1) for i in idx]

if __name__=="__main__":
    t0=time.time(); scores=[]
    for f in range(5):
        tri=np.where(folds!=f)[0]; vai=np.where(folds==f)[0]
        net=train_model(tri,seed=0)
        # eval: mask positions 4..15 on val
        gold=[]; ranked=[]
        Kv,Gv=K[vai],G[vai]
        for pos in MASKPOS:
            sc=predict(net,Kv,Gv,np.full(len(vai),pos))
            for j in range(len(vai)):
                g=Kv[j,pos]; vis=[Kv[j,p] for p in range(16) if p!=pos]
                gold.append(g); ranked.append(topk_excl(sc[j],vis))
        m=mrr10(gold,ranked); scores.append(m); print(f"fold{f} MRR@10={m:.4f} [{time.time()-t0:.0f}s]")
    print(f"\nCV MRR@10 (NN-MLM) = {np.mean(scores):.4f}")
