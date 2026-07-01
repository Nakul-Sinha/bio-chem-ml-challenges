"""Submission checker — asserts EVERY grading rule from the description (§9, §14).
Usage: python checker.py <submission.csv> <sample_submission.csv>
Exits non-zero and prints the first violation if invalid; prints PASS if valid.
Run this before every ship. A schema slip scores zero.
"""
import sys, numpy as np, pandas as pd

VALID_LABELS = {"route-aphelion","route-borealis","route-cygnus","route-driftwood","route-equinox","route-fjord"}
VALID_IDS = {str(i) for i in range(6)}
REQUIRED = ["id","target","stress_flag"]
FORBIDDEN = {"target_id","visibility","answer","answers","label","class","class_id","y",
             "fold","split","is_private","private","public","prediction"}
FORBIDDEN_PREFIX = ("unnamed:","hint_","evidence_","target_","true_","gt_","score_route_")

def fail(msg): print("FAIL:", msg); sys.exit(1)

def check(sub_path, ss_path):
    ss = pd.read_csv(ss_path)
    sub = pd.read_csv(sub_path)
    cols = list(sub.columns)
    # exact columns, exact order
    if cols != REQUIRED: fail(f"columns must be exactly {REQUIRED} in order, got {cols}")
    for c in cols:
        cl=c.lower()
        if cl in FORBIDDEN: fail(f"forbidden column present: {c}")
        if cl.startswith(FORBIDDEN_PREFIX): fail(f"forbidden column prefix: {c}")
    # row count
    if len(sub) != len(ss): fail(f"row count {len(sub)} != expected {len(ss)}")
    # ids: present, non-empty, unique, match set, and SAME ORDER as sample_submission
    if sub["id"].isna().any() or (sub["id"].astype(str).str.len()==0).any(): fail("empty/missing ids")
    if sub["id"].duplicated().any(): fail("duplicate ids")
    if set(sub["id"]) != set(ss["id"]): fail("id set does not match sample_submission")
    if not (sub["id"].astype(str).values == ss["id"].astype(str).values).all():
        fail("ids not in the same row order as sample_submission")
    # target validity
    tgt = sub["target"].astype(str)
    bad = tgt[~tgt.isin(VALID_LABELS | VALID_IDS)]
    if len(bad): fail(f"invalid target values e.g. {bad.iloc[0]!r} (n={len(bad)})")
    if sub["target"].isna().any(): fail("missing target values")
    # stress_flag pass-through unchanged
    if not (sub["stress_flag"].astype(str).values == ss["stress_flag"].astype(str).values).all():
        fail("stress_flag not passed through unchanged from sample_submission")
    # finite check (stress_flag numeric)
    sf = pd.to_numeric(sub["stress_flag"], errors="coerce")
    if sf.isna().any() or not np.isfinite(sf.values).all(): fail("non-finite stress_flag")
    print(f"PASS: {len(sub)} rows, columns {cols}, ids ordered & matched, targets valid, stress_flag pass-through.")
    print("target dist:", sub["target"].value_counts().to_dict())

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("usage: python checker.py <submission.csv> <sample_submission.csv>"); sys.exit(2)
    check(sys.argv[1], sys.argv[2])
