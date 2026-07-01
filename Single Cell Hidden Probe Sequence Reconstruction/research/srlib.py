"""Core library: exact metric, parsing, canonical decode, grouped CV, submission checker.
Everything gates on this. Kept dependency-light (numpy/pandas only) so it can be reused
inside the shipped solution.py too."""
import numpy as np, pandas as pd, re

# ----------------------------- token / canonical rules -----------------------------
TOKEN_RE = re.compile(r'^T(\d\d)_B([123])$')
# 11 targets never use B2 in train; 5 do. We still ALLOW B2 only where seen, but the
# metric itself accepts any T00-15_B1/2/3. Canonical order = descending bin, then index.
B2_TARGETS = {5, 6, 12, 13, 14}

def is_valid_token(t):
    m = TOKEN_RE.match(t)
    if not m: return False
    return 0 <= int(m.group(1)) <= 15

def canon_sort(tokens):
    """Return tokens sorted canonically: descending bin, then ascending target index.
    Duplicates (same target) collapsed keeping the highest bin."""
    best = {}
    for t in tokens:
        m = TOKEN_RE.match(t)
        if not m: continue
        ti = int(m.group(1)); b = int(m.group(2))
        if ti not in best or b > best[ti]:
            best[ti] = b
    order = sorted(best.items(), key=lambda kv: (-kv[1], kv[0]))
    return [f"T{ti:02d}_B{b}" for ti, b in order]

def bins_to_seq(bins):
    """bins: length-16 array of ints in {0,1,2,3}. 0=absent. Returns canonical token list."""
    toks = [(ti, int(b)) for ti, b in enumerate(bins) if b > 0]
    toks.sort(key=lambda kv: (-kv[1], kv[0]))
    return [f"T{ti:02d}_B{b}" for ti, b in toks]

def seq_to_str(tokens):
    return "NONE" if len(tokens) == 0 else " ".join(tokens)

def parse_target(s):
    """target_sequence string -> length-16 int array of bins (0 absent)."""
    b = np.zeros(16, dtype=np.int64)
    s = s.strip()
    if s == "NONE" or s == "":
        return b
    for t in s.split():
        m = TOKEN_RE.match(t)
        if m:
            b[int(m.group(1))] = int(m.group(2))
    return b

def target_tokens(s):
    s = s.strip()
    if s == "NONE" or s == "":
        return []
    return s.split()

# ----------------------------- exact row metric -----------------------------
def _lev(a, b):
    la, lb = len(a), len(b)
    if la == 0: return lb
    if lb == 0: return la
    prev = list(range(lb + 1))
    for i in range(1, la + 1):
        cur = [i] + [0] * lb
        ai = a[i - 1]
        for j in range(1, lb + 1):
            cost = 0 if ai == b[j - 1] else 1
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost)
        prev = cur
    return prev[lb]

def _lcs(a, b):
    la, lb = len(a), len(b)
    if la == 0 or lb == 0: return 0
    dp = [0] * (lb + 1)
    for i in range(1, la + 1):
        prev = 0; ai = a[i - 1]
        for j in range(1, lb + 1):
            tmp = dp[j]
            dp[j] = prev + 1 if ai == b[j - 1] else max(dp[j], dp[j - 1])
            prev = tmp
    return dp[lb]

def row_score(pred_tokens, true_tokens, norm='max'):
    """Exact row score: 0.5*edit_sim + 0.3*token_F1 + 0.2*order_LCS.
    NONE is represented as the single literal token 'NONE'.
    norm selects the (unknown) grader convention for normalized_edit_similarity/lcs:
      'max'   edit=1-lev/max(la,lb),      lcs=LCS/max         (standard NED; primary)
      'sum'   edit=1-lev/(la+lb),         lcs=LCS/max
      'ratio' edit=2*LCS/(la+lb),         lcs=2*LCS/(la+lb)   (difflib/Levenshtein.ratio)
    """
    p = pred_tokens if len(pred_tokens) else ["NONE"]
    t = true_tokens if len(true_tokens) else ["NONE"]
    la, lb = len(p), len(t); m = max(la, lb)
    ps, ts = set(p), set(t)
    inter = len(ps & ts)
    prec = inter / la if la else 0.0
    rec = inter / lb if lb else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    if norm == 'ratio':
        edit_sim = 2 * inter / (la + lb); lcs_sim = 2 * inter / (la + lb)
    elif norm == 'sum':
        edit_sim = 1.0 - _lev(p, t) / (la + lb); lcs_sim = _lcs(p, t) / m
    else:  # 'max'
        edit_sim = 1.0 - _lev(p, t) / m; lcs_sim = _lcs(p, t) / m
    return 0.5 * edit_sim + 0.3 * f1 + 0.2 * lcs_sim

# ----------------------------- subset-weighted CV score -----------------------------
def weighted_score(pred_seqs, true_seqs, flags, norm='max'):
    """pred_seqs, true_seqs: lists of token-lists (canonical). flags: dict of
    boolean np arrays 'shifted','damaged','rare' (len N). Returns dict of subset scores
    and the final weighted number."""
    N = len(true_seqs)
    rs = np.array([row_score(pred_seqs[i], true_seqs[i], norm) for i in range(N)])
    def mean(mask):
        mask = np.asarray(mask, bool)
        return float(rs[mask].mean()) if mask.any() else float('nan')
    s_all = float(rs.mean())
    s_sh = mean(flags['shifted']); s_dm = mean(flags['damaged']); s_ra = mean(flags['rare'])
    final = 0.45 * s_all + 0.25 * s_sh + 0.20 * s_dm + 0.10 * s_ra
    return dict(final=final, all=s_all, shifted=s_sh, damaged=s_dm, rare=s_ra,
                n=N, n_sh=int(np.asarray(flags['shifted']).sum()),
                n_dm=int(np.asarray(flags['damaged']).sum()),
                n_ra=int(np.asarray(flags['rare']).sum()), rows=rs)

# ----------------------------- feature parsing -----------------------------
META_KEYS = ['DOM','ASSAY','CTX','GENCTX','COND','SEX','STAGE','SAMPLE']
def parse_source(s):
    """-> (ov[80] int, tot, nz, panel_str, meta dict)."""
    ov = np.zeros(80, dtype=np.int64); tot = 0; nz = 0; panel = 'PANEL_NORMAL'; meta = {}
    for t in s.split():
        if t[0] == 'O' and '_Q' in t:
            ov[int(t[1:4])] = int(t.split('_Q')[1])
        elif t.startswith('PANEL'):
            panel = t
        elif t.startswith('TOTAL_Q'):
            tot = int(t.split('_Q')[1])
        elif t.startswith('NZ_Q'):
            nz = int(t.split('_Q')[1])
        else:
            meta[t.split('_')[0]] = t
    return ov, tot, nz, panel, meta

def panel_damage_group(panel):
    """PANEL_DAMAGE_0k -> k; PANEL_NORMAL -> -1."""
    if panel.startswith('PANEL_DAMAGE_'):
        return int(panel.split('_')[-1])
    return -1

# ----------------------------- submission checker -----------------------------
def check_submission(df, test_ids):
    """Validate submission df against contract. Returns (ok, msg)."""
    if list(df.columns) != ['id', 'predicted_sequence']:
        return False, f"columns must be exactly ['id','predicted_sequence'], got {list(df.columns)}"
    if df['id'].duplicated().any():
        return False, "duplicate ids"
    if set(df['id']) != set(test_ids):
        miss = set(test_ids) - set(df['id']); extra = set(df['id']) - set(test_ids)
        return False, f"id mismatch: missing {len(miss)} extra {len(extra)}"
    if len(df) != len(test_ids):
        return False, f"row count {len(df)} != {len(test_ids)}"
    for i, seq in enumerate(df['predicted_sequence'].astype(str)):
        s = seq.strip()
        if s == 'NONE':
            continue
        if s == '' or s == 'nan':
            return False, f"empty prediction at row {i}"
        toks = s.split()
        if len(set(toks)) != len(toks):
            return False, f"duplicate token at row {i}: {s}"
        seen_t = set()
        for t in toks:
            if not is_valid_token(t):
                return False, f"invalid token '{t}' at row {i}"
            ti = int(t[1:3])
            if ti in seen_t:
                return False, f"duplicate target id at row {i}: {s}"
            seen_t.add(ti)
        if toks != canon_sort(toks):
            return False, f"non-canonical order at row {i}: {s}"
    return True, "OK"

# ----------------------------- grouped CV + subset flags -----------------------------
def make_groups(train):
    """Biological-context group key = (COND,SEX,STAGE). Returns np array of group ids."""
    def key(s):
        _, _, _, _, meta = parse_source(s)
        return (meta.get('COND','?'), meta.get('SEX','?'), meta.get('STAGE','?'))
    keys = train['source_sequence'].map(key)
    uniq = {k: i for i, k in enumerate(sorted(set(keys)))}
    return keys.map(uniq).values, keys.values

def rare_token_set(train, thresh=160):
    """Tokens with train count < thresh are 'rare'. Default catches all B2 and low-freq B3."""
    from collections import Counter
    c = Counter()
    for s in train['target_sequence']:
        for t in target_tokens(s):
            c[t] += 1
    return {t for t, n in c.items() if n < thresh}, c

def subset_flags(train_df, idx, rare_tokens, small_groups, group_keys):
    """Build boolean flags for a set of row-indices idx (positions into train_df)."""
    dmg = np.zeros(len(idx), bool); rare = np.zeros(len(idx), bool); shifted = np.zeros(len(idx), bool)
    for k, i in enumerate(idx):
        s = train_df['source_sequence'].iloc[i]
        _, _, _, panel, _ = parse_source(s)
        dmg[k] = panel.startswith('PANEL_DAMAGE_')
        toks = set(target_tokens(train_df['target_sequence'].iloc[i]))
        rare[k] = len(toks & rare_tokens) > 0
        shifted[k] = group_keys[i] in small_groups
    return dict(damaged=dmg, rare=rare, shifted=shifted)

if __name__ == "__main__":
    # ---- self-test the metric on known cases ----
    def rs(a, b): return round(row_score(a, b), 4)
    assert rs([], []) == 1.0, "NONE vs NONE"
    assert rs(["T01_B1"], ["T01_B1"]) == 1.0
    assert rs([], ["T01_B1"]) == 0.0, "NONE vs active"
    # one correct of two
    print("perfect:", rs(["T06_B3","T01_B1"], ["T06_B3","T01_B1"]))
    print("half:", rs(["T06_B3"], ["T06_B3","T01_B1"]))
    print("wrong bin:", rs(["T06_B1"], ["T06_B3"]))
    print("extra fp:", rs(["T06_B3","T01_B1"], ["T06_B3"]))
    print("selftest passed")
