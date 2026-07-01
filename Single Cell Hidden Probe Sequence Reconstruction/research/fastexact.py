"""Fast EXACT vectorized weighted metric. Verified against srlib.row_score for all norms.
LCS=|intersection| (canonical), F1 from sets, edit via batched Levenshtein DP across all rows."""
import numpy as np, srlib as L

NONE_ID = 48
def bins_to_ids(bins):
    """bins:(16,) -> canonical token-id array (ids in 0..47), NONE->[48]."""
    toks=[(t,int(bins[t])) for t in range(16) if bins[t]>0]
    toks.sort(key=lambda kv:(-kv[1],kv[0]))
    if not toks: return np.array([NONE_ID],dtype=np.int16)
    return np.array([t*3+(b-1) for t,b in toks],dtype=np.int16)

def pack(bins_mat, Lmax=16):
    """Vectorized: bins_mat:(N,16)->(ids (N,Lmax) pad -1, lens). NONE row -> [48].
    Canonical order = descending bin, then ascending target index."""
    b=np.asarray(bins_mat); N=b.shape[0]
    tgt=np.arange(16)[None,:]
    tokid=tgt*3+(b-1)                       # valid where b>0
    key=np.where(b>0, -b*100+tgt, 10**6)    # absent -> large key (sorts last)
    order=np.argsort(key,axis=1)            # (N,16)
    sorted_key=np.take_along_axis(key,order,axis=1)
    sorted_id=np.take_along_axis(tokid,order,axis=1)
    valid=sorted_key<10**6                  # (N,16) which sorted slots are real tokens
    lens=valid.sum(1)
    ids=np.full((N,Lmax),-1,dtype=np.int16)
    ids[:,:16]=np.where(valid,sorted_id,-1).astype(np.int16)
    none_rows=(lens==0)
    if none_rows.any():
        ids[none_rows,0]=NONE_ID; lens=lens.copy(); lens[none_rows]=1
    return ids,lens.astype(np.int64)

def batch_edit(P,plen,T,tlen,Lmax=16):
    N=P.shape[0]
    D=np.zeros((N,Lmax+1,Lmax+1),dtype=np.int32)
    ar=np.arange(Lmax+1)
    D[:,:,0]=ar[None,:]; D[:,0,:]=ar[None,:]
    for i in range(1,Lmax+1):
        Pi=P[:,i-1]
        for j in range(1,Lmax+1):
            cost=(Pi!=T[:,j-1]).astype(np.int32)
            D[:,i,j]=np.minimum(np.minimum(D[:,i-1,j]+1,D[:,i,j-1]+1),D[:,i-1,j-1]+cost)
    return D[np.arange(N),plen,tlen]

def ids_to_set(ids):
    """ids:(N,Lmax) with pad -1 -> (N,49) bool membership. Vectorized."""
    N=ids.shape[0]; s=np.zeros((N,49),bool)
    rows=np.arange(N)[:,None].repeat(ids.shape[1],1)
    val=ids>=0
    s[rows[val],ids[val]]=True
    return s

class FastScorer:
    """Precompute truth packing once; score many predictions fast under any norm."""
    def __init__(self, Y):
        self.Y=Y; self.N=len(Y)
        self.T,self.tlen=pack(Y)
        self.tset=ids_to_set(self.T)
    def rows(self, pred_bins, norm='max'):
        N=self.N
        P,plen=pack(pred_bins)
        pset=ids_to_set(P)
        inter=(pset&self.tset).sum(1).astype(np.float64)
        la=plen.astype(np.float64); lb=self.tlen.astype(np.float64)
        m=np.maximum(la,lb)
        prec=np.where(la>0,inter/la,0.0); rec=np.where(lb>0,inter/lb,0.0)
        f1=np.where((prec+rec)>0,2*prec*rec/(prec+rec),0.0)
        if norm=='ratio':
            es=2*inter/(la+lb); ls=2*inter/(la+lb)
        else:
            lev=batch_edit(P,plen,self.T,self.tlen).astype(np.float64)
            if norm=='sum': es=1-lev/(la+lb)
            else: es=1-lev/m
            ls=inter/m
        return 0.5*es+0.3*f1+0.2*ls
    def weighted(self, pred_bins, flags, norm='max'):
        rs=self.rows(pred_bins,norm)
        def mn(msk):
            msk=np.asarray(msk,bool); return float(rs[msk].mean()) if msk.any() else float('nan')
        s_all=float(rs.mean())
        return dict(final=0.45*s_all+0.25*mn(flags['shifted'])+0.20*mn(flags['damaged'])+0.10*mn(flags['rare']),
                    all=s_all,shifted=mn(flags['shifted']),damaged=mn(flags['damaged']),rare=mn(flags['rare']),rows=rs)

if __name__=="__main__":
    import pandas as pd, random
    Y=np.load("Y.npy"); fs=FastScorer(Y)
    # verify vs srlib on random predictions
    rng=random.Random(1); N=len(Y)
    def randpred():
        pb=np.zeros((N,16),int)
        for i in range(N):
            for t in range(16):
                r=rng.random()
                if r<0.3: pb[i,t]=rng.choice([1,1,2,3])
        return pb
    pb=randpred()
    for norm in ['max','sum','ratio']:
        fast=fs.rows(pb,norm)
        slow=np.array([L.row_score(L.bins_to_seq(pb[i]), L.bins_to_seq(Y[i]), norm) for i in range(300)])
        d=np.abs(fast[:300]-slow).max()
        print(f"norm={norm}: max|fast-slow| over 300 rows = {d:.2e}")
    print("verify done")
