"""Composite metric + CV proxies for the hidden rare/shifted/catalyst-positive tracks."""
import numpy as np
from sklearn.metrics import balanced_accuracy_score, f1_score

TEMPS=["cryogenic","cold","room","warm","hot"]
TIMES=["very_short","short","medium","long","very_long","overnight"]

def set_f1_row(pred,true):
    ps,ts=set(pred),set(true)
    if not ps and not ts: return 1.0
    if not ps or not ts: return 0.0
    inter=len(ps&ts)
    if inter==0: return 0.0
    p=inter/len(ps); r=inter/len(ts); return 2*p*r/(p+r)

def row_score(ps,pt,ptime,pcat, ts,tt,ttime,tcat):
    return (0.40*set_f1_row(ps,ts)+0.20*(pt==tt)+0.20*(ptime==ttime)+0.20*(int(pcat)==int(tcat)))

def composite(pred, true, rare_mask, shift_mask, debug=False):
    """pred/true: dict with keys solv(list of lists), temp, time, cat (arrays)."""
    n=len(true["temp"])
    sf1=np.array([set_f1_row(pred["solv"][i],true["solv"][i]) for i in range(n)])
    tempok=np.array([pred["temp"][i]==true["temp"][i] for i in range(n)])
    timeok=np.array([pred["time"][i]==true["time"][i] for i in range(n)])
    catok=np.array([int(pred["cat"][i])==int(true["cat"][i]) for i in range(n)])
    rs=0.40*sf1+0.20*tempok+0.20*timeok+0.20*catok
    SolventSetF1=sf1.mean()
    TempBA=balanced_accuracy_score(true["temp"],pred["temp"])
    TimeBA=balanced_accuracy_score(true["time"],pred["time"])
    CatMF1=f1_score(true["cat"],[int(x) for x in pred["cat"]],average="macro",zero_division=0)
    Rare=rs[rare_mask].mean() if rare_mask.sum() else 0.0
    Shift=rs[shift_mask].mean() if shift_mask.sum() else 0.0
    catpos=np.array(true["cat"])==1
    CatPos=rs[catpos].mean() if catpos.sum() else 0.0
    Exact=np.mean([(set(pred["solv"][i])==set(true["solv"][i]) and tempok[i] and timeok[i] and catok[i]) for i in range(n)])
    score=(0.18*SolventSetF1+0.10*TempBA+0.10*TimeBA+0.08*CatMF1
           +0.20*Rare+0.18*Shift+0.08*CatPos+0.08*Exact)
    if debug:
        print(f"  SolvF1={SolventSetF1:.3f} TempBA={TempBA:.3f} TimeBA={TimeBA:.3f} CatMF1={CatMF1:.3f} "
              f"Rare={Rare:.3f} Shift={Shift:.3f} CatPos={CatPos:.3f} Exact={Exact:.3f}")
    return score

def make_masks(df):
    """Proxy rare & shifted masks from observable rarity/complexity (hidden tracks unknown)."""
    rare = (df["temp_bin"].isin(["cryogenic","hot"]) | df["time_bin"].isin(["very_short","very_long","overnight"])
            | (df["solvent_labels"].astype(str).str.contains("\\|"))
            | (df["smiles_length"]>=df["smiles_length"].quantile(0.8))).values
    shift = (df["smiles_length"]>=df["smiles_length"].quantile(0.8)).values  # test skews long/complex
    return rare, shift
