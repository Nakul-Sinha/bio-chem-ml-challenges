"""EXACT reproduction of the Spectral Route Robustness Score.
Final = 0.40*MacroF1 + 0.35*BalancedAccuracy + 0.10*AnchorGateF1 + 0.15*StressMacroF1
All over integer class ids 0..5; anchor = route-aphelion = id 0.
"""
import numpy as np

CLASSES = list(range(6))
ANCHOR = 0  # route-aphelion

def macro_f1(y_true, y_pred, labels=CLASSES):
    y_true = np.asarray(y_true); y_pred = np.asarray(y_pred)
    f1s = []
    for c in labels:
        tp = int(np.sum((y_true == c) & (y_pred == c)))
        fp = int(np.sum((y_true != c) & (y_pred == c)))
        fn = int(np.sum((y_true == c) & (y_pred != c)))
        den = 2*tp + fp + fn
        f1s.append(0.0 if den == 0 else 2*tp/den)
    return float(np.mean(f1s))

def balanced_accuracy(y_true, y_pred, labels=CLASSES):
    y_true = np.asarray(y_true); y_pred = np.asarray(y_pred)
    recs = []
    for c in labels:
        pos = int(np.sum(y_true == c))
        if pos == 0:
            recs.append(0.0)
        else:
            tp = int(np.sum((y_true == c) & (y_pred == c)))
            recs.append(tp/pos)
    return float(np.mean(recs))

def anchor_gate_f1(y_true, y_pred, anchor=ANCHOR):
    y_true = np.asarray(y_true); y_pred = np.asarray(y_pred)
    tp = int(np.sum((y_true == anchor) & (y_pred == anchor)))
    fp = int(np.sum((y_true != anchor) & (y_pred == anchor)))
    fn = int(np.sum((y_true == anchor) & (y_pred != anchor)))
    den = 2*tp + fp + fn
    return 0.0 if den == 0 else 2*tp/den

def stress_macro_f1(y_true, y_pred, stress_mask, labels=CLASSES):
    y_true = np.asarray(y_true); y_pred = np.asarray(y_pred)
    m = np.asarray(stress_mask).astype(bool)
    yt, yp = y_true[m], y_pred[m]
    if len(yt) < 2 or len(np.unique(yt)) < 2:
        return macro_f1(y_true, y_pred, labels)  # fallback to overall
    return macro_f1(yt, yp, labels)

def final_score(y_true, y_pred, stress_mask):
    mf = macro_f1(y_true, y_pred)
    ba = balanced_accuracy(y_true, y_pred)
    gf = anchor_gate_f1(y_true, y_pred)
    sf = stress_macro_f1(y_true, y_pred, stress_mask)
    final = 0.40*mf + 0.35*ba + 0.10*gf + 0.15*sf
    return dict(Final=final, MacroF1=mf, BalancedAcc=ba, AnchorGateF1=gf, StressMacroF1=sf)

def print_scores(d, tag=""):
    print(f"{tag:16s} Final {d['Final']:.4f} | MacroF1 {d['MacroF1']:.4f} "
          f"BalAcc {d['BalancedAcc']:.4f} Gate {d['AnchorGateF1']:.4f} Stress {d['StressMacroF1']:.4f}")

if __name__ == "__main__":
    # sanity: perfect prediction -> 1.0 ; single-class -> low
    y = np.array([0,1,2,3,4,5,0,1]); s = np.array([1,1,0,0,1,1,0,0])
    print_scores(final_score(y, y, s), "perfect")
    print_scores(final_score(y, np.zeros_like(y), s), "all-anchor")
