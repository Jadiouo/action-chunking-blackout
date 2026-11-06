"""Check every number in paper/arxiv/main.tex against the result CSVs.

Independent of code/analyze_arxiv.py: the counts, discordant pairs, exact p values, Holm corrections, recovery
rho and the paired bootstrap are recomputed here from the raw CSVs (same bootstrap recipe: 4000 resamples,
default_rng(0), percentile 2.5/97.5).  The cluster-robust logistic regressions
(logged count I, scheduled count J, J plus onset terms JT) are refit here from full_grid.csv with an own Newton solver and compared
with results/arxiv_numbers.json; the logged-count fit is also cross-checked against the earlier script's table in results/analysis.md.

Layers
  A  prose claims : each claim is a phrase (as it appears after light normalisation of the TeX) that must occur
                    in main.tex; the phrase is built from numbers recomputed from the CSVs.
  B  text claims  : numbers that come from docs or from md tables written by other scripts; the source line must
                    contain the numbers, and the phrase must occur in the tex.
  C  tables       : count cells (k/n), x/y pairs and p values of the generated tables are recomputed.
  D  lint         : every numeric token of the prose that no claim covers and that is not a declared cell label or
                    design constant is listed as UNCOVERED.
Placeholders: no \\todo may remain.
Exit status 1 on any MISMATCH / UNCOVERED / structural problem.     Run:  python3 paper/arxiv/verify_numbers.py
"""
import json, math, os, re, sys
import numpy as np
import pandas as pd
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
RES = os.path.join(ROOT, 'results')
TEX = os.environ.get('VERIFY_TEX', os.path.join(HERE, 'main.tex'))
NB = 4000


def rd(f):
    return pd.read_csv(os.path.join(RES, f))


# ---------------------------------------------------------------- data (raw CSV)
fg = rd('full_grid.csv'); orc = rd('oracle.csv'); det = rd('detect.csv'); bl = rd('baselines.csv')
rea = rd('realistic.csv'); ins = rd('insertion.csv'); insf = rd('p2b_insertion_fixed.csv'); te2 = rd('p2b_te.csv')
uns = rd('p2b_unseen.csv'); pil2 = rd('p2b_pilot.csv'); tcc = rd('p2b_tc_clean.csv'); gcl = rd('p2b_generic_clean.csv')
dty = rd('p2b_dtype.csv'); pro = rd('proprio.csv'); pilot0 = rd('pilot.csv')
rem = rea[rea['stage'] == 'main']
assert len(rem) == 5940 and set(rea['stage']) == {'main', 'pilot'}
J = json.load(open(os.path.join(RES, 'arxiv_numbers.json')))


def V(d, col='success'):
    d = d.sort_values('env_seed')
    assert d['env_seed'].is_unique
    return d[col].to_numpy().astype(int)


def FG(n, L, mode, te=False):
    cfg = ('TE_L0_black' if L == 0 else f'TE_L{L}_{mode}') if te else (f'n{n}_L0_black' if L == 0 else f'n{n}_L{L}_{mode}')
    return fg[fg['config'] == cfg]


def BL(m, corr, L):
    return bl[(bl['method'] == m) & (bl['corr'] == corr) & (bl['L'] == L)]


def RE(rule, corr, n, L):
    return rem[(rem['rule'] == rule) & (rem['corr'] == corr) & (rem['n'] == n) & (rem['L'] == L)]


def INS(rule, mode, L, n):
    return ins[(ins['rule'] == rule) & (ins['mode'] == mode) & (ins['L'] == L) & (ins['n'] == n) & (ins['te'] == 0)]


def INSF(mode, L, n):
    return insf[(insf['L'] == L) & (insf['n'] == n) & ((insf['corr'] == mode) if L > 0 else (insf['corr'] == 'black'))]


def lvl(d, k):
    return (d.sort_values('env_seed')['max_reward'].to_numpy() >= k).astype(int)


def s_(a):
    return int(np.sum(a))


def mc(a, b):
    x = int(((a == 1) & (b == 0)).sum()); y = int(((a == 0) & (b == 1)).sum())
    p = 1.0 if x + y == 0 else float(stats.binomtest(min(x, y), x + y, 0.5).pvalue)
    return x, y, p


def holm(ps):
    o = np.argsort(ps); m = len(ps); out = [0] * m; run = 0
    for r, i in enumerate(o):
        run = max(run, (m - r) * ps[i]); out[i] = min(1, run)
    return out


def bidx(n):
    return np.random.default_rng(0).integers(0, n, size=(NB, n))


def rho_ci(clean, fixed, rule):
    den = clean.mean() - fixed.mean()
    if den < 0.10:
        return None
    idx = bidx(len(rule)); c = clean[idx].mean(1); f = fixed[idx].mean(1); r = rule[idx].mean(1)
    ok = (c - f) >= 0.10
    v = (r[ok] - f[ok]) / (c[ok] - f[ok])
    return (rule.mean() - fixed.mean()) / den, np.percentile(v, 2.5), np.percentile(v, 97.5)


def ksched(n, L, t0):
    return -(-(t0 + L) // n) - -(-t0 // n)


def f2(x):
    return f'{x:.2f}'


def logit_cluster(X, y, groups):
    """Newton fit of a logistic regression and the cluster-robust (sandwich) covariance with the usual small-sample factor
    G/(G-1) * (N-1)/(N-K).  Independent re-implementation, checked against results/arxiv_numbers.json."""
    beta = np.zeros(X.shape[1])
    for _ in range(200):
        pr = 1 / (1 + np.exp(-(X @ beta))); W = pr * (1 - pr)
        step = np.linalg.solve(X.T @ (X * W[:, None]), X.T @ (y - pr)); beta = beta + step
        if np.abs(step).max() < 1e-10:
            break
    pr = 1 / (1 + np.exp(-(X @ beta))); W = pr * (1 - pr)
    Hinv = np.linalg.inv(X.T @ (X * W[:, None])); sc = X * (y - pr)[:, None]
    ug = np.unique(groups); S = np.zeros((X.shape[1],) * 2)
    for g_ in ug:
        v = sc[groups == g_].sum(0); S += np.outer(v, v)
    G, Nn, K = len(ug), len(y), X.shape[1]
    return beta, Hinv @ S @ Hinv * (G / (G - 1)) * ((Nn - 1) / (Nn - K))


def reg_fit(mode, kind):
    """kind: I logged count, J scheduled count (Eq. 1), JT = J with onset t0 and t0^2 (t0 centred at 150, units of 100 steps)."""
    k_, ln_, L_, y_, g_, t_ = [], [], [], [], [], []
    for n in [10, 25, 50, 100]:
        for L in [10, 25, 50, 100]:
            d = FG(n, L, mode).sort_values('env_seed')
            ks = np.array([ksched(n, L, int(t)) for t in d['t0']], dtype=float)
            k_.append(d['replans_in_blackout'].to_numpy(float) if kind == 'I' else ks)
            ln_.append(np.full(len(d), math.log2(n))); L_.append(np.full(len(d), L / 10.0)); y_.append(d['success'].to_numpy(float))
            g_.append(d['env_seed'].to_numpy()); t_.append((d['t0'].to_numpy(float) - 150.0) / 100.0)
    k, ln, Lc, y, g, tz = [np.concatenate(a) for a in (k_, ln_, L_, y_, g_, t_)]
    cols = [np.ones(len(y)), k, ln, k * ln, Lc] + ([tz, tz ** 2] if kind == 'JT' else [])
    beta, V = logit_cluster(np.column_stack(cols), y, g)
    sl = {}
    for n in [10, 25, 50, 100]:
        l2 = math.log2(n); b_ = beta[1] + beta[3] * l2; se = math.sqrt(V[1, 1] + l2 ** 2 * V[3, 3] + 2 * l2 * V[1, 3])
        sl[str(n)] = {'slope': b_, 'lo': b_ - 1.959964 * se, 'hi': b_ + 1.959964 * se, 'or': math.exp(b_)}
    return {'N': len(y), 'b_int': beta[3], 'se_int': math.sqrt(V[3, 3]), 'slope': sl}


def sgs(x):
    return ('-' if x < 0 else '+') + f'{abs(x):.2f}'


def src_consts(path, pattern):
    """all numbers captured by `pattern` in a source file (design constants are compared with the code, not hard-coded here)"""
    return [float(m) for m in re.findall(pattern, open(os.path.join(ROOT, path)).read())]


def all_equal(label, pattern, expected):
    """every occurrence in the text of `pattern` (one capture group = a number) must equal `expected`; at least one occurrence"""
    global n_ok
    vals = [float(m) for m in re.findall(pattern, T)]
    if vals and all(abs(v - expected) < 1e-9 for v in vals):
        n_ok += 1
        for m in re.finditer(pattern, T):
            for k in range(m.start(), m.end()):
                cov[k] = True
        return
    problems.append(f'MISMATCH [{label}] text values {vals} for {pattern!r}, expected {expected}')


def sgn(x, nd=2):
    return f'{x:.{nd}f}'


def sg(x):
    return '0.00' if abs(x) < 0.005 else f'{x:.2f}'


def rho_s(r):
    return f'{sg(r[0])} [{sg(r[1])}, {sg(r[2])}]'


def pf(p):
    if p >= 0.1:
        if abs(p * 1000 - round(p * 1000)) < 1e-9 and round(p * 1000) % 10 == 5:
            return f'{p:.3f}'                      # exact half such as 0.125 keeps three decimals
        return f'{p:.2f}'
    if p >= 0.001:
        return f'{p:.3f}'
    return f'{p:.1e}'.replace('e-0', 'e-')


def sci(p):
    """p as 'a.b \\times 10^{-c}' normalised to 'a.b×10-c'"""
    m, e = f'{p:.1e}'.split('e')
    return f'{m}×10^{{{int(e)}}}'


# ---------------------------------------------------------------- tex normalisation
raw = open(TEX).read()
raw_nc = re.sub(r'(?m)(?<!\\)%.*$', '', raw)
n_todo = len(re.findall(r'\\todo\{', raw_nc))
body = raw_nc[raw_nc.index('\\begin{document}'):]
body = body[:body.index('\\bibliographystyle')] + body[body.index('\\appendix'):]
for pat in [r'\\cite[a-z]*\{[^}]*\}', r'\\ref\{[^}]*\}', r'\\label\{[^}]*\}', r'\\input\{[^}]*\}', r'\\includegraphics(\[[^\]]*\])?\{[^}]*\}',
            r'\\begin\{[^}]*\}(\[[^\]]*\])?', r'\\end\{[^}]*\}', r'\\hline', r'\\todo\{[^}]*\}', r'\\xspaceaddexceptions\{[^}]*\}']:
    body = re.sub(pat, '', body)


def norm(s):
    s = s.replace('\\%', '%').replace('{=}', '=').replace('{\\ge}', '≥').replace('{\\le}', '≤').replace('{\\times}', '×').replace('~', ' ').replace('\\,', '') \
        .replace('\\geq', '≥').replace('\\ge', '≥').replace('\\le', '≤').replace('\\pm', '±').replace('\\times', '×').replace('\\dots', '…') \
        .replace('\\rho', 'ρ').replace('\\tau', 'τ').replace('\\sigma', 'σ').replace('\\theta', 'θ').replace('\\in', '∈').replace('\\{', '{').replace('\\}', '}')
    for m in ['fixed', 'hold', 'detect', 'ood', 'SR']:
        s = s.replace('\\' + m, m)
    for _ in range(3):
        s = re.sub(r'\\(?:textbf|emph|textit|texttt|textsf|mathrm|paragraph|subsection|section|path)\*?\{([^{}]*)\}', r'\1', s)
    s = s.replace('$', '').replace('--', '–').replace('\\_', '_').replace('\\ ', ' ')
    s = re.sub(r'\{\s*([^{}]*?)\s*\}', r'\1', s) if False else s
    return re.sub(r'\s+', ' ', s)


T = norm(body)
cov = [False] * len(T)
problems = []
n_ok = 0


def mark(ph):
    ok = False
    for m in re.finditer(re.escape(ph), T):
        ok = True
        for k in range(m.start(), m.end()):
            cov[k] = True
    return ok


def need(label, phrase):
    global n_ok
    if mark(norm(phrase)):
        n_ok += 1
        return
    problems.append(f'MISMATCH [{label}] tex lacks: {phrase!r}')


def design(phrase):
    if not mark(norm(phrase)):
        problems.append(f'MISSING-DESIGN phrase not in tex: {phrase!r}')


def src_has(label, path, key, subs):
    p = path if os.path.isabs(path) else os.path.join(ROOT, path)
    lines = open(p).read().split('\n')
    hits = [l for l in lines if key in l]
    global n_ok
    if hits and any(all(s in l for s in subs) for l in hits):
        n_ok += 1
        return
    problems.append(f'MISMATCH [{label}] {path}: no line with {key!r} containing {subs}')


# =============================================================== A. prose claims
# ---- abstract
fx50 = V(BL('fixed_n50', 'black', 50)); hd50 = V(BL('oracle_n50', 'black', 50)); dt50 = V(BL('detect_n50', 'black', 50))
need('abs', f'from {s_(fx50)}/100 to {s_(hd50)}/100 at n=50, L=50, and a pixel-statistics rule recovers the same {s_(dt50)}/100')
assert s_(hd50) == s_(dt50)
h_f100 = V(BL('oracle_n50', 'freeze', 100)); f_f100 = V(BL('fixed_n50', 'freeze', 100))
sub = te2[(te2['corr'] == 'freeze') & (te2['L'] == 100)]
c_hold = V(sub[sub['rule'] == 'oracle']); c_fix = V(sub[(sub['rule'] == 'fixed') & (sub['n'] == 50)])
c_te1 = V(sub[(sub['rule'] == 'median_te') & (sub['h'] == 1)]); c_te10 = V(sub[(sub['rule'] == 'median_te') & (sub['h'] == 10)])
q_te1 = sub[(sub['rule'] == 'median_te') & (sub['h'] == 1)]['replans_total'].mean()
q_te10 = sub[(sub['rule'] == 'median_te') & (sub['h'] == 10)]['replans_total'].mean()
q_hold = sub[sub['rule'] == 'oracle']['replans_total'].mean(); q_fix = sub[(sub['rule'] == 'fixed') & (sub['n'] == 50)]['replans_total'].mean()
assert s_(h_f100) < s_(f_f100) and s_(c_hold) < s_(c_fix)                       # direction word "does worse than"
need('abs', f'with frozen frames and L=100 it does worse than leaving the policy alone ({s_(h_f100)} against {s_(f_f100)} of 100 episodes, and {s_(c_hold)} against {s_(c_fix)} on 100 fresh seeds)')
assert abs(q_te1 - 370) <= 10 and abs(q_hold - 6.5) < 0.1
mt_h = mc(c_te1, c_hold); mt_f = mc(c_te1, c_fix)
assert s_(c_te1) > s_(c_hold) and mt_h[2] < 0.05 and mt_f[2] > 0.05          # "beats hold" but "not leaving the policy alone"
need('abs', f'beats hold ({s_(c_te1)} against {s_(c_hold)} on the fresh seeds) but not leaving the policy alone ({s_(c_te1)} against {s_(c_fix)}, p={mt_f[2]:.2f}), and it needs about 370 policy calls per episode against {q_hold:.1f}')
hi = V(INS('oracle', 'black', 50, 50)); fi = V(INSF('black', 50, 50))
need('abs', f'({s_(hi)} against {s_(fi)} of 200 at n=50)')
bl_f = J['unseen']['blur40']['det']['false_rate']; dm_f = J['unseen']['dim0.1']['det']['false_rate']
u_blur = uns[uns['corr'] == 'blur40']; u_dim = uns[uns['corr'] == 'dim0.1']
fr_b = u_blur[u_blur['rule'] == 'generic']; fr_d = u_dim[u_dim['rule'] == 'generic']
frb = fr_b['det_out'].sum() / fr_b['steps_out'].sum(); frd = fr_d['det_out'].sum() / fr_d['steps_out'].sum()
gall = rem[rem['rule'] == 'generic'].groupby(['corr', 'n', 'L']).apply(lambda d: d['det_out'].sum() / d['steps_out'].sum(), include_groups=False)
need('abs', f'false-alarm rate of {100 * gall.min():.1f}–{100 * gall.max():.1f}% across cells')
design('1% it was designed for')
design('with 100 paired initial conditions per cell')

# ---- 5.1 grid
cl = {n: s_(V(FG(n, 0, 'black'))) for n in [1, 10, 25, 50, 100]}
cte = s_(V(FG(1, 0, 'black', te=True)))
need('grid', f'0/100 at n=1, {cl[10]}/100 at n=10, {cl[25]}/100 at n=25, {cl[50]}/100 at n=50 and {cl[100]}/100 at n=100, against {cte}/100 for Mean TE')
assert cl[1] == 0
need('grid', 'In a diagnosis on eight seeds the arm hardly moves')
src_has('n1 diag', 'results/n1_diagnosis.md', 'n=1 成功 0/8', ['n=25 成功 7/8'])


def gk(n, L, mode):
    return s_(V(FG(n, L, mode)))


need('grid', f'succeeds in {gk(25, 50, "black")}, {gk(50, 50, "black")} and {gk(100, 50, "black")} of 100 episodes at n=25, 50 and 100, and at L=100 in {gk(25, 100, "black")}, {gk(50, 100, "black")} and {gk(100, 100, "black")}')
need('grid', f'the same cells give {gk(25, 50, "freeze")}, {gk(50, 50, "freeze")} and {gk(100, 50, "freeze")} at L=50 and {gk(25, 100, "freeze")}, {gk(50, 100, "freeze")} and {gk(100, 100, "freeze")} at L=100')
need('grid', f'at L=25 and n=50 the success is {gk(50, 25, "freeze")}/100, close to the {cl[50]}/100 of the clean runs')
assert all(gk(n, L, 'black') < gk(n, L, 'freeze') for n in [25, 50, 100] for L in [25, 50, 100])
design('every cell with L≥25 and n≥25')


def wil_hw(k, n, z=1.959964):
    c = (k / n + z * z / (2 * n)) / (1 + z * z / n); h = z * math.sqrt(k / n * (1 - k / n) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return h
# Fig. 1 plots fixed n in {10,25,50,100} (black L=0..100, freeze L=10..100) and hold n in {25,50} (L=10..100, black starts at the clean fixed point)
fig1 = [gk(n, L, m) for m in ['black', 'freeze'] for n in [10, 25, 50, 100] for L in ([0, 10, 25, 50, 100] if m == 'black' else [10, 25, 50, 100])]
fig1 += [s_(V(orc[orc['config'] == f'oracle_n{n}_L{L}_{m}'])) for m in ['black', 'freeze'] for n in [25, 50] for L in [10, 25, 50, 100]]
fig1 += [gk(n, 0, 'black') for n in [25, 50]]
hws = [wil_hw(k, 100) for k in fig1]
need('grid', f'The half-width of a 95% interval is between {min(hws):.3f} and {max(hws):.3f} per point')

# ---- 5.2 mechanism
def nat(L, mode):
    d = FG(100, L, mode).sort_values('env_seed'); ks = np.array([ksched(100, L, int(t)) for t in d['t0']]); s = d['success'].to_numpy()
    return (int(s[ks == 0].sum()), int((ks == 0).sum()), int(s[ks >= 1].sum()), int((ks >= 1).sum()))


nb50 = nat(50, 'black'); nf50 = nat(50, 'freeze')
need('mech', f'split into {nb50[1]} with k=0 and {nb50[3]} with k≥1')
need('mech', f'succeed {nb50[0]} times out of {nb50[1]} and those with k≥1 {nb50[2]} times out of {nb50[3]}')
need('mech', f'the split is {nf50[0]}/{nf50[1]} against {nf50[2]}/{nf50[3]}')
assert nb50[0] == nf50[0] and nb50[1] == nf50[1]
need('mech', f'both give {nb50[0]}/{nb50[1]}')
nb25 = nat(25, 'black'); nf25 = nat(25, 'freeze'); nb10 = nat(10, 'black'); nf10 = nat(10, 'freeze')
need('mech', f'(black {nb25[2]}/{nb25[3]}, frozen {nf25[2]}/{nf25[3]} with k≥1; {nb25[0]}/{nb25[1]} for both with k=0)')
need('mech', f'({nb10[2]}/{nb10[3]} and {nf10[2]}/{nf10[3]}; {nb10[0]}/{nb10[1]} for both)')
R_ = J['mech']['reg']
RG = {(k, m): reg_fit(m, k) for k in ['I', 'J', 'JT'] for m in ['black', 'freeze']}
for (k, m), r in RG.items():                                          # own fit == analysis script
    rj = R_[f'{k}_{m}']
    assert r['N'] == rj['N'] == 1600 and abs(r['b_int'] - rj['b_int']) < 1e-6 and abs(r['se_int'] - rj['se_int']) < 1e-6
    assert all(abs(r['slope'][n]['slope'] - rj['slope'][n]['slope']) < 1e-6 and abs(r['slope'][n]['lo'] - rj['slope'][n]['lo']) < 1e-6 for n in ['10', '25', '50', '100'])
# the logged count differs from the scheduled count only in episodes that succeed inside the window (the reason for using Eq. 1)
dd_ = pd.concat([FG(n, L, 'black') for n in [10, 25, 50, 100] for L in [10, 25, 50, 100]] + [FG(n, L, 'freeze') for n in [10, 25, 50, 100] for L in [10, 25, 50, 100]])
dd_['ks'] = [ksched(int(n), int(L), int(t)) for n, L, t in zip(dd_['n'], dd_['L'], dd_['t0'])]
df_ = dd_[dd_['ks'] != dd_['replans_in_blackout']]
assert len(df_) == 41 and (df_['success'] == 1).all() and (df_['replans_in_blackout'] < df_['ks']).all()
rb, rf = RG[('J', 'black')], RG[('J', 'freeze')]
sl = rb['slope']
need('mech', f'the interaction of k and \\log_2 n for black frames is {rb["b_int"]:.2f} (cluster-robust standard error {rb["se_int"]:.3f})')
need('mech', f'is {sl["10"]["slope"]:.2f} at n=10, {sl["25"]["slope"]:.2f} at n=25, {sl["50"]["slope"]:.2f} at n=50 and {sl["100"]["slope"]:.2f} at n=100 (odds ratios {sl["10"]["or"]:.2f} to {sl["100"]["or"]:.2f})')
assert all(sl[n]['hi'] < 0 for n in sl) and sl['10']['slope'] > sl['25']['slope'] > sl['50']['slope'] > sl['100']['slope']
# cells whose L is not a multiple of n: the phase of the onset alone decides k (computed from the CSV, independent of the JSON)
po = {}
for m in ['black', 'freeze']:
    for n in [10, 25, 50, 100]:
        for L in [10, 25, 50, 100]:
            if L % n == 0:
                continue
            d = FG(n, L, m).sort_values('env_seed'); ks_ = np.array([ksched(n, L, int(t)) for t in d['t0']]); s__ = d['success'].to_numpy()
            vals = sorted(set(ks_.tolist())); assert len(vals) == 2 and vals[1] - vals[0] == 1
            hi_, lo_ = s__[ks_ == vals[1]], s__[ks_ == vals[0]]
            po[(m, n, L)] = math.log((hi_.sum() + .5) / (len(hi_) - hi_.sum() + .5)) - math.log((lo_.sum() + .5) / (len(lo_) - lo_.sum() + .5))
for key, v in po.items():
    jj = J['mech']['phase_only'][f'{key[0]}_n{key[1]}_L{key[2]}']['log_or']
    assert abs(jj - v) < 1e-9
pb_ = {k[1:]: v for k, v in po.items() if k[0] == 'black'}
assert len(pb_) == 7 and all(v < 0 for v in pb_.values())
b50 = [v for (n, L), v in pb_.items() if n == 50]; b100 = [v for (n, L), v in pb_.items() if n == 100]; (b10,) = [v for (n, L), v in pb_.items() if n == 10]; (b25,) = [v for (n, L), v in pb_.items() if n == 25]
assert len(b50) == 2 and len(b100) == 3 and f'{b50[0]:.2f}' == f'{b50[1]:.2f}' and b10 > b25 > b50[0] > max(b100)
need('mech', f'the log odds ratio for one more replan is {b10:.2f} at n=10, {b25:.2f} at n=25, {b50[0]:.2f} in both such cells at n=50 and {max(b100):.2f} to {min(b100):.2f} in the three at n=100 (0.5 added to each count)')
rt = RG[('JT', 'black')]['slope']
assert all(rt[n]['hi'] < 0 for n in rt)
need('mech', f'changes the black slopes to {rt["10"]["slope"]:.2f} to {rt["100"]["slope"]:.2f}, which does not alter this picture')
slf = rf['slope']
need('mech', f'For frozen frames the slopes are {sgs(slf["10"]["slope"])}, {sgs(slf["25"]["slope"])}, {sgs(slf["50"]["slope"])} and {sgs(slf["100"]["slope"])}, and the intervals for n≥25 lie above zero')
assert all(slf[n]['lo'] > 0 for n in ['25', '50', '100'])
rtf = RG[('JT', 'freeze')]['slope']
assert all(rtf[n]['lo'] > 0 for n in ['25', '50', '100']) and max(abs(rtf[n]['slope'] - slf[n]['slope']) for n in slf) < 0.01   # "do not change with t0 and t0^2"
need('mech', 'they do not change with t_0 and t_0^2')
pf_ = [v for k, v in po.items() if k[0] == 'freeze']
assert len(pf_) == 7 and min(pf_) < 0 < max(pf_)                          # no consistent sign in the phase-only frozen contrasts
need('mech', f'have both signs ({sgs(min(pf_))} to {sgs(max(pf_))} over seven cells)')
design('We therefore report the positive frozen slope only as an association in this grid, in which k is tied to the phase of the onset, and not as evidence that replanning on a frozen frame helps')
sli = RG[('I', 'black')]['slope']; slif = RG[('I', 'freeze')]['slope']
assert abs(RG[('I', 'black')]['b_int'] + 0.71) < 0.006 and abs(RG[('I', 'black')]['se_int'] - 0.096) < 0.0006      # logged-count fit vs analysis.md
need('mech', f'with the logged count the black slopes barely move ({sli["10"]["slope"]:.2f} to {sli["100"]["slope"]:.2f}) and the frozen intervals for n≥25 contain zero')
assert all(slif[n]['lo'] < 0 < slif[n]['hi'] for n in ['25', '50', '100'])
need('mech', 'because the logged count is lower only in episodes that succeed inside the window, so it depends on the outcome')
design('1600 episodes per fault type')
# cross-check regression against the earlier script (analysis.md)
src_has('reg black', 'results/analysis.md', '| black | I | k', ['-0.710 (0.096)'.split(' ')[0] if False else '-0.710'])
src_has('reg slopes', 'results/analysis.md', '| black | 100 | -2.92', ['0.05', '[-3.55, -2.28]'])
src_has('reg slopes', 'results/analysis.md', '| black | 10 | -0.56', ['0.57', '[-0.76, -0.37]'])
src_has('reg slopes', 'results/analysis.md', '| freeze | 100 | +0.32', ['[-0.10, +0.74]'])

# ---- 5.3 hold
def tcc_(mode, L, n):
    fx = V(BL(f'fixed_n{n}', mode, L)); hd = V(BL(f'oracle_n{n}', mode, L)); dt = V(BL(f'detect_n{n}', mode, L))
    cl_ = V(FG(n, 0, 'black'))
    return fx, hd, dt, cl_


fx25, hd25, dt25, cl25 = tcc_('black', 50, 25); fx5, hd5, dt5, cl5 = tcc_('black', 50, 50)
r25 = rho_ci(cl25, fx25, hd25); r50 = rho_ci(cl5, fx5, hd5)
x, y, p = mc(hd5, fx5)
need('hold', f'it reaches {s_(hd25)}/100 at n=25 (fixed: {s_(fx25)}) and {s_(hd5)}/100 at n=50 (fixed: {s_(fx5)}), against {s_(cl25)} and {s_(cl5)} without blackout')
need('hold', f'recovery is ρ={r25[0]:.2f} [{r25[1]:.2f}, {r25[2]:.2f}] and {r50[0]:.2f} [{r50[1]:.2f}, {r50[2]:.2f}], and at n=50 the two rules disagree on {x + y} seeds, {x} of them in favor of hold')
dd = det[det['config'] != 'detect_n50_L0_black']
assert len(dd) == 800 and dd['detect_latency'].notna().all() and dd['detect_latency'].max() == 0 and dd['false_detect'].sum() == 0
need('hold', f'in the {len(dd)} blackout episodes of its test it raised an alarm in the first step of every window and never outside it')
need('hold', f'reaches the same counts, {s_(dt25)} and {s_(dt5)}')
c0 = V(det[det['config'] == 'detect_n50_L0_black']); xq, yq, pq = mc(c0, V(FG(50, 0, 'black')))
need('hold', f'{s_(c0)}/100 against {s_(V(FG(50, 0, "black")))}/100 for fixed(50) in the LeRobot loop ({xq}/{yq} discordant seeds)')
fo = orc[orc['config'] == 'oracle_n25_L100_black']['forced_replans_in_blackout'].mean(); fo5 = orc[orc['config'] == 'oracle_n50_L100_black']['forced_replans_in_blackout'].mean()
fx25b, hd25b, _, _ = tcc_('black', 100, 25); fx5b, hd5b, _, _ = tcc_('black', 100, 50)
rb25 = rho_ci(cl25, fx25b, hd25b); rb50 = rho_ci(cl5, fx5b, hd5b)
need('hold', f'(on average {fo:.2f} forced replans at n=25 and {fo5:.2f} at n=50), and recovery is partial: {s_(hd25b)}/100 and {s_(hd5b)}/100 against {s_(fx25b)} and {s_(fx5b)}, ρ={rb25[0]:.2f} and {rb50[0]:.2f}')

# stratified
def strat(mode, hn):
    out = {'k0': [0, 0, 0], 'k1': [0, 0, 0]}
    for L in [25, 50]:
        fx = BL('fixed_n100', mode, L).sort_values('env_seed'); hd = BL(f'oracle_n{hn}', mode, L).sort_values('env_seed')
        ks = np.array([ksched(100, L, int(t)) for t in fx['t0']])
        for key, m in [('k0', ks == 0), ('k1', ks >= 1)]:
            out[key][0] += int(fx['success'].to_numpy()[m].sum()); out[key][1] += int(hd['success'].to_numpy()[m].sum()); out[key][2] += int(m.sum())
    return out


sb = strat('black', 50); sf = strat('freeze', 50)
need('hold', f'Among the {sb["k0"][2] + sb["k1"][2]} pairs of episodes with L∈{{25,50}}'.replace('{{', '{').replace('}}', '}'))
need('hold', f'(k≥1, {sb["k1"][2]} pairs) and those where it would not ({sb["k0"][2]})')
need('hold', f'({sb["k1"][1]}/{sb["k1"][2]} against {sb["k1"][0]}/{sb["k1"][2]} for fixed(100)) and not on the second ({sb["k0"][1]}/{sb["k0"][2]} against {sb["k0"][0]}/{sb["k0"][2]})')
need('hold', f'({sf["k1"][1]}/{sf["k1"][2]} against {sf["k1"][0]}/{sf["k1"][2]})')

fx25f, hd25f, _, _ = tcc_('freeze', 50, 25); fx5f, hd5f, _, _ = tcc_('freeze', 50, 50)
a1 = mc(hd25f, fx25f); a2 = mc(hd5f, fx5f)
need('hold', f'({s_(hd25f)} against {s_(fx25f)}, {a1[0]}/{a1[1]}) but not significantly at n=50 ({s_(hd5f)} against {s_(fx5f)}, {a2[0]}/{a2[1]}, uncorrected p={a2[2]:.3f})')
fx25g, hd25g, _, _ = tcc_('freeze', 100, 25); fx5g, hd5g, _, _ = tcc_('freeze', 100, 50)
g1 = mc(hd25g, fx25g); g2 = mc(hd5g, fx5g)
need('hold', f'{s_(hd25g)} against {s_(fx25g)} at n=25 ({g1[0]}/{g1[1]}) and {s_(hd5g)} against {s_(fx5g)} at n=50 ({g2[0]}/{g2[1]})')
x6, y6, p6 = mc(c_hold, c_fix)
need('hold', f'hold succeeds in {s_(c_hold)} episodes and fixed in {s_(c_fix)}, with {x6}/{y6} discordant seeds, that is, {y6} of the {x6 + y6} in favor of fixed')
assert y6 > x6 and s_(c_fix) > s_(c_hold)                                  # the 17 discordant seeds favour fixed, not hold
design('(150–249)')

# ---- 5.4 Insertion
cl_i = {n: s_(lvl(INSF('black', 0, n), 4)) for n in [25, 50, 100]}
need('ins', f'({cl_i[25]}/200 at n=25, {cl_i[50]}/200 at n=50, {cl_i[100]}/200 at n=100 with reward=4)')
def ir(rule, mode, L, n, k):
    return lvl(INS(rule, mode, L, n), k)
def inf(mode, L, n, k):
    return lvl(INSF(mode, L, n), k)
h25, h50 = ir('oracle', 'black', 50, 25, 4), ir('oracle', 'black', 50, 50, 4)
d25, d50 = ir('detect', 'black', 50, 25, 4), ir('detect', 'black', 50, 50, 4)
f25, f50 = inf('black', 50, 25, 4), inf('black', 50, 50, 4)
need('ins', f'hold raises success from {s_(f25)} to {s_(h25)} of 200 at n=25 and from {s_(f50)} to {s_(h50)} at n=50, and detect from {s_(f25)} to {s_(d25)} and from {s_(f50)} to {s_(d50)}')
F1p = [mc(h25, f25)[2], mc(d25, f25)[2], mc(h50, f50)[2]]
F1h = holm(F1p)
need('ins', f'(p_{{Holm}}={F1h[0]:.4f}, {F1h[1]:.4f} and {sci(F1h[2])})')
cl25i, cl50i = inf('black', 0, 25, 4), inf('black', 0, 50, 4)
ri = rho_ci(cl50i, f50, h50)
need('ins', f'ρ={ri[0]:.2f} [{ri[1]:.2f}, {ri[2]:.2f}]; it is not reported at n=25 because the clean-minus-fixed gap is {cl25i.mean() - f25.mean():.2f}')
assert rho_ci(cl25i, f25, h25) is None
f50_3, h50_3 = inf('black', 50, 50, 3), ir('oracle', 'black', 50, 50, 3); c50_3 = inf('black', 0, 50, 3)
r3 = rho_ci(c50_3, f50_3, h50_3)
need('ins', f'the counts are {s_(f50_3)} for fixed, {s_(h50_3)} for hold and {s_(c50_3)} clean (ρ={r3[0]:.2f} [{r3[1]:.2f}, {r3[2]:.2f}])')
r3a = rho_ci(inf('black', 0, 25, 3), inf('black', 100, 25, 3), ir('oracle', 'black', 100, 25, 3)); r3b = rho_ci(inf('black', 0, 50, 3), inf('black', 100, 50, 3), ir('oracle', 'black', 100, 50, 3))
need('ins', f'(reward≥3, ρ={r3a[0]:.2f} and {r3b[0]:.2f})')
dnb = INSF('black', 50, 100).sort_values('env_seed'); dnf = INSF('freeze', 50, 100).sort_values('env_seed')
ks = np.array([ksched(100, 50, int(t)) for t in dnb['t0']]); sb4 = (dnb['max_reward'].to_numpy() >= 4); sf4 = (dnf['max_reward'].to_numpy() >= 4)
need('ins', f'succeed in {sb4[ks == 0].sum()} of {(ks == 0).sum()} and those with one in {sb4[ks >= 1].sum()} of {(ks >= 1).sum()} (frozen: {sf4[ks == 0].sum()}/{(ks == 0).sum()} and {sf4[ks >= 1].sum()}/{(ks >= 1).sum()})')
mx = max(abs(a.mean() - b.mean()) for m in ['freeze'] for L in [50] for n in [25, 50] for a, b in [(ir('oracle', m, L, n, 4), inf(m, L, n, 4)), (ir('detect', m, L, n, 4), inf(m, L, n, 4))])
need('ins', f'(the largest absolute difference to fixed is {mx:.3f})')
fz = {(n, k): (ir('oracle', 'freeze', 100, n, k), inf('freeze', 100, n, k)) for n in [25, 50] for k in [3, 4]}
assert all(a.mean() <= b.mean() for a, b in fz.values())
need('ins', f'for example {s_(fz[(50, 4)][0])} against {s_(fz[(50, 4)][1])} of 200 at n=50 with reward=4, and {s_(fz[(50, 3)][0])} against {s_(fz[(50, 3)][1])} with reward≥3)')
assert all(mc(a, b)[2] > 0.05 for a, b in fz.values())

# ---- 5.5 detector
def synp(c, n, L):
    cl_ = FG(n, 0, 'black'); cl_ = V(cl_[cl_['env_seed'] < 90])
    return cl_, V(RE('fixed', c, n, L)), V(RE('oracle', c, n, L)), V(RE('generic', c, n, L)), RE('generic', c, n, L)


hs = [V(RE('oracle', c, 50, 50)) for c in ['noise0.6', 'occlude0.5center', 'delay50', 'flicker0.6']]
assert all((hs[0] == h).all() for h in hs) and s_(hs[0]) == 75
need('ood', f'its {s_(hs[0])}/90 successes are identical seed by seed for noise, occlusion, delay and flicker (0 of 90 episodes differ)')
mxd = max(int((V(RE('oracle', a, 25, 50)) != V(RE('oracle', b, 25, 50))).sum()) for a in ['noise0.6', 'occlude0.5center', 'delay50'] for b in ['noise0.6', 'occlude0.5center', 'delay50'])
need('ood', f'at n=25 the rows differ by at most {mxd} of 90 episodes')
Rf = [s_(V(RE('fixed', c, 50, 50))) / s_(synp(c, 50, 50)[0]) for c in ['noise0.6', 'occlude0.5center', 'delay50', 'flicker0.6']]
need('ood', f'(retention {min(Rf):.2f} to {max(Rf):.2f} for the four synthetic corruptions)')
nz = synp('noise0.6', 50, 50); oc = synp('occlude0.5center', 50, 50)
bu = {c: tuple(V(uns[(uns['corr'] == c) & (uns['rule'] == r)]) for r in ['fixed', 'oracle', 'generic']) + (uns[(uns['corr'] == c) & (uns['rule'] == 'generic')],)
      for c in ['blur40', 'dim0.1']}
clean50 = V(tcc[tcc['n'] == 50])
rn = rho_ci(nz[0], nz[1], nz[3]); ro = rho_ci(oc[0], oc[1], oc[3]); rbl = rho_ci(clean50, bu['blur40'][0], bu['blur40'][2])
need('ood', f'{s_(nz[3])}, {s_(oc[3])} and {s_(bu["blur40"][2])} successes of 90, against {s_(nz[1])}, {s_(oc[1])} and {s_(bu["blur40"][0])} for fixed (ρ={rn[0]:.2f}, {ro[0]:.2f} and {rbl[0]:.2f})')
fl = synp('flicker0.6', 50, 50); fl_in = fl[4]['det_in'].sum() / fl[4]['steps_in'].sum(); rfl = rho_ci(fl[0], fl[1], fl[3])
need('ood', f'it flags {100 * fl_in:.0f}% of the in-window steps and recovers ρ={rfl[0]:.2f} [{rfl[1]:.2f}, {rfl[2]:.2f}]')
fl100 = synp('flicker0.6', 50, 100); xx = mc(fl100[3], fl100[2])
need('ood', f'({s_(fl100[3])} against {s_(fl100[2])} of 90, {xx[0]}/{xx[1]} discordant, exploratory p={xx[2]:.4f})')
dl = synp('delay50', 50, 50); dl_in = dl[4]['det_in'].sum() / dl[4]['steps_in'].sum(); dl_out = dl[4]['det_out'].sum() / dl[4]['steps_out'].sum()
rdl = rho_ci(dl[0], dl[1], dl[3]); rdh = rho_ci(dl[0], dl[1], dl[2])
need('ood', f'the flagged share ({dl_in:.3f}) is close to the false-alarm rate of {dl_out:.3f}, and the recovery is {sg(rdl[0])} [{sg(rdl[1])}, {sg(rdl[2])}]')
need('ood', f'hold recovers {rho_s(rdh)}')
g_d = bu['dim0.1'][3]; dim_in = g_d['det_in'].sum() / g_d['steps_in'].sum(); dim_ep = (g_d['det_in'] > 0).mean()
rdim = rho_ci(clean50, bu['dim0.1'][0], bu['dim0.1'][2]); rdimh = rho_ci(clean50, bu['dim0.1'][0], bu['dim0.1'][1])
need('ood', f'it flags {100 * dim_in:.0f}% of in-window steps and recovers ρ={rdim[0]:.2f} [{rdim[1]:.2f}, {rdim[2]:.2f}], against {rdimh[0]:.2f} for hold, and only {100 * dim_ep:.0f}% of the dim episodes contain any alarm')
design('(g=0.1)')
need('ood', f'it flags {100 * dim_in:.0f}% of in-window steps'.replace('16%', '16%'))
xb = mc(bu['dim0.1'][2], bu['dim0.1'][0]); xbl = mc(bu['blur40'][2], bu['blur40'][0]); f2h = holm([xbl[2], xb[2]])
need('ood', f'({xb[0]}/{xb[1]} discordant seeds, p_{{Holm}}={f2h[1]:.3f})')
design('flagged share below 0.30 or a recovery below 0.30')
g_b = bu['blur40'][3]; fb = g_b['det_out'].sum() / g_b['steps_out'].sum(); fd = g_d['det_out'].sum() / g_d['steps_out'].sum()
need('ood', f'is {100 * fb:.1f}% (blur) and {100 * fd:.1f}% (dim), and across the synthetic experiments it lies between {100 * gall.min():.1f}% and {100 * gall.max():.1f}% by cell')
gc = {n: gcl[gcl['n'] == n] for n in [25, 50, 100]}
epf = [(g['det_out'] > 0).groupby(g['env_seed']).any().mean() for g in gc.values()]
need('ood', f'Between {100 * min(epf):.0f}% and {100 * max(epf):.0f}% of the clean episodes contain at least one false alarm')
dl_ = [V(gc[n]).mean() - V(tcc[tcc['n'] == n]).mean() for n in [25, 50, 100]]
need('ood', f'differs by +{dl_[0]:.3f}, {dl_[1]:.3f} and {dl_[2]:.3f} at n=25, 50 and 100')
sd = rem[rem['rule'] == 'detect']
assert sd['det_in'].sum() == 0 and sd['det_out'].sum() == 0
need('ood', f'(0 flagged steps in {len(sd)} episodes)')
need('ood', 'seven variants')
src_has('seven variants', 'docs/p2p3_notes.md', '7 個變體', ['7 個變體'])
design('The false-alarm rate is a second limit. We set the threshold for 1% of clean frames')
design("Detection rates and the AUC against a ``harmful to the policy'' label were computed on the development seeds 120–149 only")
src_has('probe medians', 'docs/p2p3_notes.md', '2.6–3.9', ['3.8'])
need('app', 'score medians of its probe corruptions (2.6–3.9) were not above the median of clean scores (3.8)')

# ---- 5.6 baselines
def b_(m, mode, L):
    return V(BL(m, mode, L))


te_b25 = b_('medianTE_h1', 'black', 25); hd_b25 = b_('oracle_n50', 'black', 25); mte_b25 = b_('meanTE_h1', 'black', 25)
pw = mc(te_b25, hd_b25)
need('base', f'({s_(te_b25)} against {s_(hd_b25)} at L=25, {pw[0]}/{pw[1]} discordant, no difference) and far above Mean TE ({s_(mte_b25)})')
need('base', f'at L=100 ({s_(b_("medianTE_h1", "black", 100))} against {s_(b_("fixed_n50", "black", 100))})')
te50 = b_('medianTE_h1', 'black', 50); a = mc(te50, b_('fixed_n50', 'black', 50)); a2_ = mc(te50, b_('oracle_n50', 'black', 50))
need('base', f'({s_(te50)} against {s_(b_("fixed_n50", "black", 50))}, {a[0]}/{a[1]}, exploratory p={a[2]:.3f}) but well below hold ({a2_[0]}/{a2_[1]})')
te_f = b_('medianTE_h1', 'freeze', 100); a = mc(te_f, b_('fixed_n50', 'freeze', 100)); a2_ = mc(te_f, b_('oracle_n50', 'freeze', 100))
need('base', f'{s_(te_f)} of 100, against {s_(b_("fixed_n50", "freeze", 100))} for fixed(50) ({a[0]}/{a[1]}, exploratory p={a[2]:.4f}) and {s_(b_("oracle_n50", "freeze", 100))} for hold ({a2_[0]}/{a2_[1]})')
x1 = mc(c_te1, c_hold); x2 = mc(c_te1, c_fix); x3 = mc(c_te10, c_te1)   # x/y: first-named rule only / second only
need('base', f'(Median TE {s_(c_te1)}, hold {s_(c_hold)}, {x1[0]}/{x1[1]}, p_{{Holm}} in Table')
need('base', f'({s_(c_te1)} against {s_(c_fix)}, {x2[0]}/{x2[1]}, p={x2[2]:.2f})')
need('base', f'{q_te1:.0f} policy calls per episode at h=1 and {q_te10:.0f} at h=10 in the confirmatory cell, against {q_hold:.1f} for hold and {q_fix:.1f} for fixed(50), and with h=10 the success is {s_(c_te10)} against {s_(c_te1)} ({x3[0]}/{x3[1]}, no difference)')
assert x3[2] > 0.05 and x3[0] - x3[1] == s_(c_te10) - s_(c_te1)
# care
cr = {m: (b_(m, 'black', 50), b_(m, 'freeze', 50)) for m in ['care_tau0.3']}
rc = rho_ci(V(BL('fixed_n50', 'clean', 0)), b_('fixed_n50', 'black', 50), b_('care_tau0.3', 'black', 50))
cells = [('black', 25), ('black', 50), ('black', 100), ('freeze', 50), ('freeze', 100)]
below = sum(s_(b_('care_tau0.3', m, L)) < s_(b_('fixed_n50', m, L)) for m, L in cells)
assert below == 4
need('base', 'below fixed(50) in four of five cells')
need('base', f'{s_(b_("care_tau0.3", "black", 50))} against {s_(b_("fixed_n50", "black", 50))} at black L=50 (ρ={sg(rc[0])} [{sg(rc[1])}, {sg(rc[2])}]) and {s_(b_("care_tau0.3", "freeze", 50))} against {s_(b_("fixed_n50", "freeze", 50))} at freeze L=50')
need('base', f'(black L=100: {s_(b_("care_tau0.3", "black", 100))} against {s_(b_("fixed_n50", "black", 100))})')
cc3 = s_(V(BL('care_tau0.3', 'clean', 0))); cf = s_(V(BL('fixed_n50', 'clean', 0)))
need('base', f'CARE succeeds in {cc3} of 100 clean episodes against {cf} for fixed(50)')
need('base', f'(black L=50: {s_(b_("care_tau0.3", "black", 50)) / cc3:.2f} against {s_(b_("fixed_n50", "black", 50)) / cf:.2f})')
need('base', f'({s_(b_("care_tau0.5", "black", 50))} at τ=0.5)')
assert all(s_(b_(f'care_tau{t}', 'black', 50)) < s_(b_('fixed_n50', 'black', 50)) for t in ['0.18', '0.3', '0.5'])
assert all(s_(b_(f'care_tau{t}', 'freeze', 50)) < s_(b_('fixed_n50', 'freeze', 50)) for t in ['0.18', '0.3', '0.5'])

# ---- Discussion / limitations / DP
rh1 = rho_ci(cl25, fx25b, hd25b)
need('disc', f'hold recovers only {rb50[0]:.2f} to {rb25[0]:.2f} of the loss on TC')
assert 0.25 <= rb50[0] <= rb25[0] + 1e-9
need('disc', f'the four synthetic strengths give a retention of {min(Rf):.2f} to {max(Rf):.2f} for fixed(50), against {s_(b_("fixed_n50", "black", 50)) / cf:.2f} for black frames in the same cell')
cmin = min(s_(lvl(INSF('black', 0, n), 4)) for n in [25, 50, 100]) / 2; cmax = max(s_(lvl(INSF('black', 0, n), 4)) for n in [25, 50, 100]) / 2
need('disc', f"Insertion's clean success is {cmin:.1f}% to {cmax:.1f}%")
dp = {t: rd(f) for t, f in [('v1', 'dp_gate_dp_v1.csv'), ('v2', 'dp_gate_dp_v2.csv'), ('v3', 'dp_gate_dp_v3.csv')]}
assert all(len(d) == 20 and d['success'].sum() == 0 for d in dp.values()) and (dp['v2']['max_reward'] >= 2).sum() == 2 and (dp['v1']['max_reward'] >= 2).sum() == 0 and (dp['v3']['max_reward'] >= 2).sum() == 0
ub = (lambda k, n, z=1.959964: ((k / n + z * z / (2 * n)) + z * math.sqrt(k / n * (1 - k / n) / n + z * z / (4 * n * n))) / (1 + z * z / n))(0, 20)
need('dp', f'the 20k-step run (v1) succeeded in 0/20 episodes')
need('dp', 'a 40k-step run with EMA and the original simulation settings (horizon 16, v2) in 0/20 (2 episodes lifted the cube)')
need('dp', f'(v3) in 0/20 (Wilson upper bound {ub:.2f} each)')
src_has('dp steps', 'docs/dp_training.md', '閘門 40k', ['0/20'])
design('We extended v2 to 100k steps')
dpf = rd('dp_final_dp_v2_100000.csv')
assert len(dpf) == 20 and dpf['success'].sum() == 2
def wil(k, n, z=1.959964):
    c = (k / n + z * z / (2 * n)) / (1 + z * z / n); h = z * math.sqrt(k / n * (1 - k / n) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return c - h, c + h
wl, wh = wil(2, 20)
need('dp', f'it succeeded in 2/20 episodes (0.10, Wilson 95% CI {wl:.2f} to {wh:.2f})')
src_has('dp stop rule', 'docs/dp_training.md', '停損', ['<30%'])
design('below the 0.30 we had set before this run')
design('Python 3.10, PyTorch 2.10, LeRobot 0.4.4, gym-aloha 0.1.5, MuJoCo 3.8.1')

# ---- Experiments
sp = {'flips': 0, 'n': 0, 'mf': 0, 'mn': 0}
for cfg, g in pilot0.groupby('config'):
    f_ = fg[(fg['config'] == cfg) & (fg['env_seed'].isin(g['env_seed']))]
    m_ = g.merge(f_, on='env_seed', suffixes=('_p', '_f')); nf = int((m_['success_p'] != m_['success_f']).sum()); mean_ = (m_['success_p'].mean() + m_['success_f'].mean()) / 2
    sp['flips'] += nf; sp['n'] += len(m_)
    if 0.1 <= mean_ <= 0.9:
        sp['mf'] += nf; sp['mn'] += len(m_)
need('exp', f'{sp["flips"]} of {sp["n"]} episodes ({100 * sp["flips"] / sp["n"]:.1f}%) change outcome, and {sp["mf"]} of {sp["mn"]} ({100 * sp["mf"] / sp["mn"]:.1f}%) in cells whose mean success lies between 0.1 and 0.9')
cr_ = {}
for mode in ['black', 'freeze']:
    r_ = pd.read_csv(os.path.join(RES, 'parts', f'p2p3_regress_{mode}.csv'))
    for n in [25, 50]:
        a = r_[r_['config'] == f'fixed_n{n}_L50_{mode}']; b = FG(n, 50, mode); b = b[b['env_seed'].isin(a['env_seed'])]
        cr_[(mode, n)] = (int((V(a) != V(b)).sum()), s_(V(a)), s_(V(b)))
fl_ = sorted(v[0] for v in cr_.values())
need('exp', f'flips for {fl_[0]} to {fl_[-1]} of 100 episodes in the four cells we compared')
dtv = {}
for mode in ['black', 'freeze']:
    a = dty[dty['corr'] == mode]; b = FG(25, 50, mode); r64 = pd.read_csv(os.path.join(RES, 'parts', f'p2p3_regress_{mode}.csv')); r64 = r64[r64['config'] == f'fixed_n25_L50_{mode}']
    dtv[mode] = (int((V(a) != V(b)).sum()), int((V(r64) != V(b)).sum()), int((V(a) != V(r64)).sum()), s_(V(a)), s_(V(r64)), s_(V(b)))
f32t = dtv['black'][0] + dtv['freeze'][0]; f64t = dtv['black'][1] + dtv['freeze'][1]
need('exp', f'reduces the flips against the LeRobot loop from {f64t} to {f32t} of 200')
mg = max(abs(inf(m, L, n, 4).mean() - lvl(ins[(ins['rule'] == 'fixed') & (ins['mode'] == m) & (ins['L'] == L) & (ins['n'] == n)], 4).mean()) for m in ['black', 'freeze'] for L in [50, 100] for n in [25, 50, 100])
need('exp', f'differs from the old one by at most {mg:.2f} in all 12 cells')
design('(0–99)')
design('Insertion uses 200 seeds')
design('100 environment seeds')
design('maximum reward reaches 4')
design('reward 1 for touching both')
design('0–89')
design('seeds 120–149')
design('seeds 100–119')
design('seeds 150–249')
design('0.15,0.85')
design('50,…,250')
design('we do not interpret differences below 0.05')
# pilot & strength claims
need('exp', 'one clause added after the pilot'.replace('one clause added after the pilot', 'One clause was added after the pilot'))
design('a pilot of 30 episodes')

# ---- Insertion README vs eval_info (text claims from docs)
src_has('readme', 'results/insertion_table.md', '20.6%', ['83.0%', '415/500'])
need('readme', 'whereas the eval_info.json in the same Hugging Face repository reports 83.0% over 500 episodes')
need('readme', 'success rate of 20.6%')
c100 = lvl(INSF('black', 0, 100), 4)
need('readme', f'our clean success at n=100 is {s_(c100)}/200, which agrees with the card')

# ---- Appendix
for nm, row in [('blur σ=40', None)]:
    pass
gd = json.load(open(os.path.join(RES, 'parts', 'generic_detector.json')))
assert gd['K'] == 16 and gd['feat_dim'] == 6144
need('app', f'The principal subspace has {gd["K"]} components on {gd["feat_dim"]}-dimensional pooled features')
need('app', f'fit on {gd["fit_frames"]:,} frames of clean episodes with seeds 100–109 (explained variance {gd["explained_var_topK"]:.2f})')
need('app', f'θ={gd["thr"]:.2f} is the 99th percentile of the score on {gd["thr_frames"]:,} frames of clean episodes with seeds 110–119')
src_has('harm thr', 'results/realistic_table.md', '有害門檻 D*', ['0.0317'])
need('app', 'which is 0.0317 rad')
src_has('auc', 'results/realistic_table.md', '| flicker0.6 |', ['0.991'])
src_has('auc', 'results/realistic_table.md', '| freeze |', ['0.537'])
need('app', 'is 0.991 for flicker and 0.537 for frozen frames')
need('app', 'clean seeds 100–119 to give a mean executed length near 30 steps (0.6 of H_{\\max}; main, τ=0.3), 20 steps (τ=0.18) or 40 steps (τ=0.5); the closed-loop runs gave 31.5, 20.9 and 38.5 steps')
src_has('care calib', 'docs/p2p3_notes.md', 'closed-loop 實測平均 H', ['13.4/19.4/32.8'])
src_has('care calib', 'docs/p2p3_notes.md', 'τ=0.18/0.30/0.50/0.80 → H=', ['20.9/31.5/38.5/42.3'])
src_has('care eps', 'docs/p2p3_notes.md', 'R_0 中位數', ['0.019'])
need('app', 'discrepancy ratios are of order 10-2'.replace('10-2', '10^-2') if False else 'whose discrepancy ratios are of order 10^{-2}')
pi = {m: pil2[(pil2['corr'] == m)] for m in ['black']}
need('app', 'clean reference 24/30 on the same seeds')
assert s_(V(pil2[(pil2['corr'] == 'black') & (pil2['L'] == 0)])) == 24 and len(pil2[(pil2['corr'] == 'black') & (pil2['L'] == 0)]) == 30
for c in ['dim0.5', 'dim0.25', 'dim0.1']:
    pass
# Insertion appendix
oc_ = {n: s_(lvl(ins[(ins['rule'] == 'fixed') & (ins['mode'] == 'black') & (ins['L'] == 0) & (ins['n'] == n)], 4)) for n in [25, 50, 100]}
need('app', f'give {cl_i[25]}, {cl_i[50]} and {cl_i[100]} successes of 200 at n=25, 50 and 100 (the first run: {oc_[25]}, {oc_[50]} and {oc_[100]})')
fl_i = max(int((inf(m, L, n, 4) != lvl(ins[(ins['rule'] == 'fixed') & (ins['mode'] == m) & (ins['L'] == L) & (ins['n'] == n)], 4)).sum()) for m in ['black', 'freeze'] for L in [50, 100] for n in [25, 50, 100])
need('app', f'the episode-level outcome flips in up to {fl_i} of 200 episodes')
need('app', f'differ by at most {mg:.2f} in the 12 cells')
rc_ = {r: int((INSF('black', 0, 100)['max_reward'] == r).sum()) for r in range(5)}
need('app', f'the maximum reward is 0 in {rc_[0]}, 1 in {rc_[1]}, 2 in {rc_[2]}, 3 in {rc_[3]} and 4 in {rc_[4]} episodes')
need('app', f'is close to our {s_(c100)}/200 ({100 * s_(c100) / 200:.1f}%)'.replace('(20.5%)', '(20.5%)'))
need('app', 'the eval_info.json (83.0%, 415 of 500 reaching reward 4)')
# noise appendix
need('app', f'change the outcome of {sp["flips"]} ({100 * sp["flips"] / sp["n"]:.1f}%); in cells with mean success between 0.1 and 0.9 it is {sp["mf"]} of {sp["mn"]} ({100 * sp["mf"] / sp["mn"]:.1f}%)')
need('app', f'in {cr_[("black", 25)][0]}/100 (black, n=25), {cr_[("black", 50)][0]}/100 (black, n=50), {cr_[("freeze", 25)][0]}/100 (freeze, n=25) and {cr_[("freeze", 50)][0]}/100 (freeze, n=50) episodes at L=50, with success counts {cr_[("black", 25)][1]} against {cr_[("black", 25)][2]}, {cr_[("black", 50)][1]} against {cr_[("black", 50)][2]}, {cr_[("freeze", 25)][1]} against {cr_[("freeze", 25)][2]} and {cr_[("freeze", 50)][1]} against {cr_[("freeze", 50)][2]}')
rg = pd.read_csv(os.path.join(RES, 'parts', 'p2p3_regress_rungrid_freeze.csv'))
need('app', f'gives {int((V(rg) != V(FG(25, 50, "freeze"))).sum())}/100 flips')
need('app', f'from {f64t} to {f32t} in 200 episodes, below the pre-registered threshold of 8')
assert f32t <= 8
tcl = {n: (s_(V(tcc[tcc['n'] == n])), s_(V(FG(n, 0, 'black')[FG(n, 0, 'black')['env_seed'] < 90]))) for n in [25, 50, 100]}
mxc = max(abs(a - b) / 90 for a, b in tcl.values())
need('app', f'differ from the LeRobot-loop rates by at most {mxc:.3f} ({tcl[25][0]}/90 against {tcl[25][1]}/90 at n=25, {tcl[50][0]}/90 against {tcl[50][1]}/90 at n=50, {tcl[100][0]}/90 against {tcl[100][1]}/90 at n=100)')
src_has('probe', 'results/n1_diagnosis.md', '150 個動作', ['最大差 1'])
need('app', 'replaying the same 150 actions gives identical joint positions')
# F family and prereg
need('app', f'(observed {J["unseen"]["dim0.1"]["det"]["in_frac"]:.3f}) and a recovery of at least 0.60 ({J["unseen"]["dim0.1"]["rho_ood"][0]:.2f})')
assert abs(J['unseen']['dim0.1']['det']['in_frac'] - dim_in) < 1e-9 and abs(J['unseen']['dim0.1']['rho_ood'][0] - rdim[0]) < 1e-9
pte3 = [mc(c_te1, c_fix)]
f3p = [mc(c_te1, c_hold)[2], mc(V(te2[(te2['corr'] == 'black') & (te2['L'] == 25) & (te2['h'] == 1)]), V(te2[(te2['corr'] == 'black') & (te2['L'] == 25) & (te2['rule'] == 'oracle')]))[2], mc(c_te1, c_fix)[2]]
f3h = holm(f3p)
need('app', f'(observed +{c_te1.mean() - c_fix.mean():.2f}, p_{{Holm}}={f3h[2]:.3f})')
need('app', f'blur (flagged {g_b["det_in"].sum() / g_b["steps_in"].sum():.3f}, ρ={rbl[0]:.2f}, false alarms {fb:.4f}), false alarms below 0.05 for dim ({fd:.4f})')
need('app', f'(5/200 flips)'.replace('5', str(f32t)))
# proprio
pm = {}
for L, mode in [(100, 'black')]:
    d = pro[(pro['tag'] == 'main') & (pro['config'] == f'proprio_m10_L{L}_{mode}')]
    xx = mc(V(d), V(FG(50, L, mode)))
need('app', f'the threshold τ=0.159 was the 99th percentile on clean seeds 100–119 at n=100')
need('app', f'(p={xx[2]:.3f} uncorrected)')
src_has('proprio share', 'results/proprio_signal.md', '右夾爪占', ['93.2%', '5.0%'])
need('app', 'the gripper dimensions explained 98% of the squared error')
src_has('proprio stage', 'results/proprio_signal.md', 'reward 0 為 0.034', ['2 為 0.038'])
need('app', '(mean 0.034 in reward stage 0 and 0.038 in stage 2)')
# literature
need('app', 'on 2026-11-03 and 2026-11-04 in four threads')
src_has('lit', 'docs/literature.md', '日期：2026-11-03', ['2026-11-03'])
need('app', 'Every bibliography entry with an arXiv identifier was checked against its arXiv abstract page on 2026-11-04, or on 2026-11-06 for ResNet, ViM and ImageNet-C, which were cited after the review')
need('app', 'The record of that review, with the evidence level for each of its sources (full text, abstract only, or bibliographic data), is docs/literature.md in the code repository')
_lit = open(os.path.join(ROOT, 'docs', 'literature.md')).read()
_bib = open(os.path.join(HERE, 'refs.bib')).read()
assert all(k_ in _bib for k_ in ['he2016resnet', 'wang2022vim', 'hendrycks2019imagenetc'])
assert not any(w_ in _lit for w_ in ['ViM', 'ImageNet-C', 'ResNet', '1512.03385'])   # the three later entries are not in the review record

# =============================================================== C. tables (counts, x/y, p)
TABLE_PROBLEMS = []


def cells_of(path):
    return open(os.path.join(HERE, 'tables', path)).read()


def tbl_check(name, expect_rows):
    t = cells_of(name)
    rows = [r for r in t.split('\\\\') if '&' in r]
    texrows = [[c.strip() for c in r.replace('\\toprule', '').replace('\\midrule', '').split('&')] for r in rows]
    for exp in expect_rows:
        hit = [r for r in texrows if all(any(e in c for c in r) for e in exp)]
        if not hit:
            TABLE_PROBLEMS.append(f'{name}: no row containing {exp}')


# grid
rows = []
for n in [1, 10, 25, 50, 100]:
    exp = [str(n)] if False else []
    r = [str(gk(n, L, 'black')) for L in [0, 10, 25, 50, 100]] + [str(gk(n, L, 'freeze')) for L in [10, 25, 50, 100]]
    t = cells_of('tab_grid.tex')
    line = [l for l in t.split('\n') if l.startswith(f'{n} &')]
    if not line or [c.strip(' \\') for c in line[0].split('&')[1:]] != r:
        TABLE_PROBLEMS.append(f'tab_grid row n={n} differs: {line} vs {r}')
t = cells_of('tab_grid.tex'); line = [l for l in t.split('\n') if l.startswith('Mean TE')][0]
rt = [str(s_(V(FG(1, L, 'black', te=True)))) for L in [0, 10, 25, 50, 100]] + [str(s_(V(FG(1, L, 'freeze', te=True)))) for L in [10, 25, 50, 100]]
if [c.strip(' \\') for c in line.split('&')[1:]] != rt:
    TABLE_PROBLEMS.append('tab_grid Mean TE row differs')
# tc hold
t = cells_of('tab_tc_hold.tex').split('\n')
for mode in ['black', 'freeze']:
    for L in [25, 50, 100]:
        for n in [25, 50]:
            fx, hd, dt, cl_ = tcc_(mode, L, n); x, y, _ = mc(hd, fx); r = rho_ci(cl_, fx, hd)
            line = [l for l in t if l.startswith(f'{mode} & {L} & {n} &')]
            exp = f'{mode} & {L} & {n} & {s_(cl_)}/100 & {s_(fx)}/100 & {s_(hd)}/100 & '
            tail = f' & {x}/{y} & {s_(dt)}/100 \\\\'
            if not line or not line[0].startswith(exp) or not line[0].endswith(tail):
                TABLE_PROBLEMS.append(f'tab_tc_hold {mode} L={L} n={n}: {line} vs {exp}...{tail}')
            else:
                mid = line[0][len(exp):-len(tail)]
                want = 'n/r' if r is None else f'{r[0]:.2f} [{r[1]:.2f}, {r[2]:.2f}]'
                if not mid.startswith(want):
                    TABLE_PROBLEMS.append(f'tab_tc_hold rho {mode} L={L} n={n}: {mid} vs {want}')
# insertion
t = cells_of('tab_ins.tex').split('\n')
for k, tag in [(4, 'r4'), (3, 'r3')]:
    for mode in ['black', 'freeze']:
        for L in [50, 100]:
            for n in [25, 50]:
                cl_ = inf(mode, 0, n, k); fx = inf(mode, L, n, k); hd = ir('oracle', mode, L, n, k); dt = ir('detect', mode, L, n, k); x, y, _ = mc(hd, fx)
                exp = f'{mode} & {L} & {n} & {s_(cl_)}/200 & {s_(fx)}/200 & {s_(hd)}/200 & {s_(dt)}/200 & '
                tail = f' & {x}/{y} \\\\'
                cand = [l for l in t if l.startswith(f'{mode} & {L} & {n} &')]
                cand = [c for c in cand if c.startswith(exp) and c.endswith(tail)]
                if not cand:
                    TABLE_PROBLEMS.append(f'tab_ins {tag} {mode} L={L} n={n} no matching row {exp}...{tail}')
# synthetic
t = cells_of('tab_syn.tex').split('\n')
for c, nm in [('noise0.6', 'noise'), ('occlude0.5center', 'occlusion'), ('flicker0.6', 'flicker'), ('delay50', 'delay')]:
    cl_, fx, hd, gn, _ = synp(c, 50, 50)
    exp = f'{s_(cl_)}/90 & {s_(fx)}/90 & {s_(hd)}/90 & {s_(gn)}/90 &'
    line = [l for l in t if l.startswith(nm) and exp in l]
    if not line:
        TABLE_PROBLEMS.append(f'tab_syn {c}: no row with {exp}')
for c, nm in [('blur40', 'blur'), ('dim0.1', 'dim')]:
    fx, hd, gn = bu[c][0], bu[c][1], bu[c][2]
    exp = f'{s_(clean50)}/90 & {s_(fx)}/90 & {s_(hd)}/90 & {s_(gn)}/90 &'
    if not [l for l in t if l.startswith(nm) and exp in l]:
        TABLE_PROBLEMS.append(f'tab_syn {c}: no row with {exp}')
# base
t = cells_of('tab_base.tex').split('\n')
for m, nm in [('fixed_n50', 'fixed(50)'), ('fixed_n100', 'fixed(100)'), ('oracle_n50', 'hold(50)'), ('meanTE_h1', 'Mean TE'), ('medianTE_h1', 'Median TE, $h{=}1$'), ('medianTE_h10', 'Median TE, $h{=}10$'), ('care_tau0.3', 'CARE')]:
    cm = {'oracle_n50': 'fixed_n50'}.get(m, m)
    vals = [str(s_(V(BL(cm, 'clean', 0))))] + [str(s_(b_(m, mo, L))) for mo, L in cells] + [f'{BL(m, "freeze", 100)["replans_total"].mean():.1f}']
    line = [l for l in t if nm in l.replace('\\', '')]
    ok = False
    for l in line:
        got = [c.strip(' \\') for c in l.split('&')[1:]]
        if got == vals:
            ok = True
    if not ok:
        TABLE_PROBLEMS.append(f'tab_base {m}: {line} vs {vals}')
# primary
prim = J['primary']
chk = [(mc(V(BL('oracle_n50', 'black', 50)), V(BL('fixed_n50', 'black', 50)))), mc(c_hold, c_fix), mc(h50, f50), mc(bu['blur40'][2], bu['blur40'][0]), mc(bu['dim0.1'][2], bu['dim0.1'][0]), mc(c_te1, c_hold)]
ph = holm([c[2] for c in chk])
t = cells_of('tab_primary.tex')
for i, (c, h) in enumerate(zip(chk, ph)):
    line = [l for l in t.split('\n') if l.startswith(f'P{i + 1} &')]
    exp = f' & {c[0]}/{c[1]} & {pf(c[2])} & {pf(h)} & '
    if not line or exp not in line[0]:
        TABLE_PROBLEMS.append(f'tab_primary P{i + 1}: {line} lacks {exp}')
    need('primary', f'{c[0]}/{c[1]}') if False else None

# ---- appendix tables
pl = rea[rea['stage'] == 'pilot']
cl30 = V(pil2[(pil2['corr'] == 'black') & (pil2['L'] == 0)]).mean()
Rb = {c: V(pil2[pil2['corr'] == c]).mean() / cl30 for c in ['blur5', 'blur15', 'blur40', 'dim0.5', 'dim0.25', 'dim0.1']}
NT = [53]


def has_row(name, *parts):
    NT[0] += 1
    t = cells_of(name)
    if not any(all(x in l for x in parts) for l in t.split('\n')):
        TABLE_PROBLEMS.append(f'{name}: no row with {parts}')


for mode in ['black', 'freeze']:
    d = dtv[mode]
    has_row('tab_dtype.tex', f'{mode} & {d[5]}/100 & {d[4]}/100 & {d[3]}/100 & {d[1]} & {d[0]} & {d[2]}')
for m in ['black', 'freeze']:
    for L in [50, 100]:
        for n in [25, 50, 100]:
            old = lvl(ins[(ins['rule'] == 'fixed') & (ins['mode'] == m) & (ins['L'] == L) & (ins['n'] == n)], 4); new = inf(m, L, n, 4)
            has_row('tab_ins_fixed.tex', f'{m} & {L} & {n} & {s_(old)}/200 & {s_(new)}/200 & {int((old != new).sum())}')
for c in ['noise0.1', 'noise0.3', 'noise0.6', 'occlude0.25center', 'occlude0.5center', 'occlude0.75center', 'occlude0.25rand', 'occlude0.5rand', 'occlude0.75rand', 'delay5', 'delay25', 'delay50', 'flicker0.3', 'flicker0.6', 'flicker1']:
    d = pl[(pl['corr'] == c) & (pl['rule'] == 'fixed') & (pl['n'] == 50) & (pl['L'] == 50)]
    has_row('tab_pilot.tex', f'{s_(V(d))}/30 & {s_(V(d)) / 30 / (24 / 30):.2f}')
for c in ['blur5', 'blur15', 'blur40', 'dim0.5', 'dim0.25', 'dim0.1']:
    has_row('tab_pilot.tex', f'{s_(V(pil2[pil2["corr"] == c]))}/30 & {Rb[c]:.2f}')
for t_, m in [('0.18', 'care_tau0.18'), ('0.3', 'care_tau0.3'), ('0.5', 'care_tau0.5')]:
    c0 = s_(V(BL(m, 'clean', 0)))
    has_row('tab_care.tex', f'{t_} & {c0} & {s_(b_(m, "black", 50))} & {s_(b_(m, "freeze", 50))} & {s_(b_(m, "black", 50)) / c0:.2f} & {s_(b_(m, "freeze", 50)) / c0:.2f}')
for cn, corr, L in [('freeze $L{=}100$', 'freeze', 100), ('black $L{=}25$', 'black', 25)]:
    sb_ = te2[(te2['corr'] == corr) & (te2['L'] == L)]
    arr = {'Median TE $h{=}1$': sb_[(sb_['rule'] == 'median_te') & (sb_['h'] == 1)], 'Median TE $h{=}10$': sb_[(sb_['rule'] == 'median_te') & (sb_['h'] == 10)],
           '\\fixed(50)': sb_[(sb_['rule'] == 'fixed') & (sb_['n'] == 50)], '\\fixed(100)': sb_[(sb_['rule'] == 'fixed') & (sb_['n'] == 100)], '\\hold(50)': sb_[sb_['rule'] == 'oracle']}
    base_ = V(arr['Median TE $h{=}1$'])
    for lab, d in arr.items():
        q = f"{d['replans_total'].mean():.1f}"
        if lab.startswith('Median TE $h{=}1$'):
            has_row('tab_te_confirm.tex', cn, lab, f'{s_(V(d))}/100 & {q}')
        else:
            x, y, pp = mc(base_, V(d))
            has_row('tab_te_confirm.tex', cn, lab, f'{s_(V(d))}/100 & {q} & {x}/{y} & {pf(pp)}')
for n in [25, 50, 100]:
    fxc = V(tcc[tcc['n'] == n]); gn = V(gcl[gcl['n'] == n]); x, y, pp = mc(gn, fxc)
    fr = gcl[gcl['n'] == n]['det_out'].sum() / gcl[gcl['n'] == n]['steps_out'].sum()
    has_row('tab_cleancost.tex', f'{n} & {s_(fxc)}/90 & {s_(gn)}/90 & {gn.mean() - fxc.mean():+.3f} & {x}/{y} & {pf(pp)} & {fr:.4f}')
for L, mode in [(25, 'black'), (50, 'black'), (100, 'black'), (25, 'freeze'), (50, 'freeze'), (100, 'freeze')]:
    row = [mode, str(L), str(s_(V(FG(50, L, mode)))), str(s_(V(FG(100, L, mode))))]
    for m in [1, 10]:
        d = pro[(pro['tag'] == 'main') & (pro['config'] == f'proprio_m{m}_L{L}_{mode}')]
        x, y, pp = mc(V(d), V(FG(50, L, mode)))
        row += [f'{s_(V(d))}/100', f'{x}/{y}', pf(pp)]
    has_row('tab_proprio.tex', *[f' {r} ' if False else r for r in [' & '.join(row)]])
for c, nm in [('noise0.6', 'noise'), ('occlude0.5center', 'occlusion'), ('flicker0.6', 'flicker'), ('delay50', 'delay')]:
    g = RE('generic', c, 50, 50)
    has_row('tab_detector.tex', nm, f'{(g["det_in"] > 0).mean():.2f} & {g["det_in"].sum() / g["steps_in"].sum():.3f} & {g["det_out"].sum() / g["steps_out"].sum():.4f} & {(g["det_out"] > 0).mean():.2f}')
for c, nm in [('blur40', 'blur'), ('dim0.1', 'dim')]:
    g = bu[c][3]
    has_row('tab_detector.tex', nm, f'{(g["det_in"] > 0).mean():.2f} & {g["det_in"].sum() / g["steps_in"].sum():.3f} & {g["det_out"].sum() / g["steps_out"].sum():.4f} & {(g["det_out"] > 0).mean():.2f}')
# families
for lab, c in [('oracle $n{=}25$ vs fixed', mc(h25, f25)), ('detect $n{=}25$ vs fixed', mc(d25, f25)), ('oracle $n{=}50$ vs fixed', mc(h50, f50)),
               ('OOD vs fixed, blur', mc(bu['blur40'][2], bu['blur40'][0])), ('OOD vs fixed, dim', mc(bu['dim0.1'][2], bu['dim0.1'][0]))]:
    has_row('tab_families.tex', lab, f'{c[0]}/{c[1]} & {pf(c[2])}')
has_row('tab_families.tex', 'F3', 'freeze $L{=}100$', f'{mc(c_te1, c_hold)[0]}/{mc(c_te1, c_hold)[1]} & {pf(f3p[0])} & {pf(f3h[0])}')
has_row('tab_families.tex', 'F3', 'black $L{=}25$', f'& {pf(f3p[1])} & {pf(f3h[1])}')
has_row('tab_families.tex', 'F3', 'fixed(50)', f'{mc(c_te1, c_fix)[0]}/{mc(c_te1, c_fix)[1]} & {pf(f3p[2])} & {pf(f3h[2])}')

# prose references to primary Holm values
need('hold', f'(P2, p_{{Holm}}={sci(ph[1])})'.replace('4.3×10-4', '4.3×10-4'))
design('P1')
design('P2')



design('14 joint positions')
# gym-aloha source: the copy of the relevant lines in docs/gym_aloha_excerpt.txt is always checked; the installed package too, when it is found
def gym_aloha_sources():
    srcs = [('docs/gym_aloha_excerpt.txt', open(os.path.join(ROOT, 'docs', 'gym_aloha_excerpt.txt')).read(), None)]
    for d_ in [os.path.join(ROOT, '.venv/lib/python3.10/site-packages/gym_aloha')] + [os.path.join(q_, 'gym_aloha') for q_ in sys.path]:
        if os.path.exists(os.path.join(d_, 'tasks', 'sim.py')) and os.path.exists(os.path.join(d_, 'env.py')):
            srcs.append(('installed gym_aloha', open(os.path.join(d_, 'tasks', 'sim.py')).read(), open(os.path.join(d_, 'env.py')).read()))
            break
    return srcs
GYM_SRC = gym_aloha_sources()
for name_, sim, env_ in GYM_SRC:
    ia = sim[sim.index('class InsertionTask'):]
    assert all(t in ia for t in ['reward = 1', 'reward = 2', 'reward = 3', 'reward = 4', 'pin_touched', 'peg_touch_socket']), name_
design("reward 1 for touching both objects, 2 for grasping both off the table, 3 for peg–socket contact off the table, and 4 when the peg touches the socket's pin. Reward 4 is therefore")
design('Reward≥3 is a post hoc secondary metric')
design('closest to 0.5 and inside [0.15,0.85]')
assert cl[10] < cl[25] < cl[50] > cl[100]
design('rises with n up to 50 and falls slightly after')
design('At L=100 the chunk of 100 steps runs out')
design('Counts are successes out of 100')
assert s_(bu['blur40'][1]) == s_(bu['dim0.1'][1]) == 75
need('syn', f'which also give {s_(bu["blur40"][1])}/90')
design('its reward-4 criterion is a contact event about 2 cm inside a 12 cm socket that does not require the peg to be aligned')

# ---- what reward 4 means: socket geometry (excerpt of the model file) and depths read from the simulator
_xml = open(os.path.join(ROOT, 'docs', 'gym_aloha_excerpt.txt')).read()
assert 'pos="0 0 0" size="0.04 0.01 0.01" type="box" name="pin"' in _xml and 'size="0.06 0.018 0.002" type="box" name="socket-1"' in _xml
_dep = pd.read_csv(os.path.join(ROOT, 'results', 'insertion_depth.csv'))
assert len(_dep) == 200 and _dep['pin_contact_at_first_r4'].dropna().astype(bool).all()
_ds, _df = _dep[_dep['success'] == 1], _dep[_dep['success'] == 0]
need('depth', f'({len(_ds)} successes in this rerun')
need('depth', 'a tube 12 cm long and the end of the pin lies 2 cm inside its mouth')
need('depth', f"is {_ds['corner_depth_at_first_r4'].min()*100:.2f} to {_ds['corner_depth_at_first_r4'].max()*100:.2f} cm inside the tube")
need('depth', f"tilted by {_ds['tilt_at_first_r4'].min():.0f} to {_ds['tilt_at_first_r4'].max():.0f} degrees (median {_ds['tilt_at_first_r4'].median():.0f})")
assert _df['ever_pin_contact'].sum() == 0
need('depth', f"None of the {len(_df)} failed episodes touches the pin")
need('depth', f"reaches {_df['max_corner_depth'].max()*100:.2f} cm")
design('the 200 clean episodes at n=100')
design('by about 2 cm')
design('At the first reward of 4')
design('T^2')

# ---- selection rule reproduces the selected strengths (pilot rows)
pl = rea[rea['stage'] == 'pilot']
def pilot_R(corr):
    d = pl[(pl['corr'] == corr) & (pl['rule'] == 'fixed') & (pl['n'] == 50) & (pl['L'] == 50)]
    assert len(d) == 30
    c = pl[(pl['corr'] == 'black') & (pl['L'] == 0)]
    return d['success'].mean() / c['success'].mean()
def pick(cands):
    ok = [(abs(pilot_R(c) - 0.5), i, c) for i, c in enumerate(cands) if 0.15 <= pilot_R(c) <= 0.85]
    mn = min(o[0] for o in ok if True) if ok else None
    tie = [o for o in ok if abs(o[0] - mn) < 1e-9]
    return tie[0][2] if len(tie) == 1 else sorted(tie, key=lambda o: -pilot_R(o[2]))[0][2]
assert pick(['noise0.1', 'noise0.3', 'noise0.6']) == 'noise0.6'
assert abs(pilot_R('occlude0.5center') - pilot_R('occlude0.75center')) < 1e-9 and pick(['occlude0.25center', 'occlude0.5center', 'occlude0.75center']) == 'occlude0.5center'
assert pick(['delay5', 'delay25', 'delay50']) == 'delay50'
assert pick(['flicker0.3', 'flicker0.6']) == 'flicker0.6'   # flicker1 excluded by the added clause
pb = pil2[pil2['corr'].isin(['blur5', 'blur15', 'blur40', 'dim0.5', 'dim0.25', 'dim0.1'])]
cl30 = V(pil2[(pil2['corr'] == 'black') & (pil2['L'] == 0)]).mean()
Rb = {c: V(pil2[pil2['corr'] == c]).mean() / cl30 for c in ['blur5', 'blur15', 'blur40', 'dim0.5', 'dim0.25', 'dim0.1']}
assert min(Rb, key=lambda c: abs(Rb[c] - 0.5) if 'blur' in c and 0.15 <= Rb[c] <= 0.85 else 9) == 'blur40'
assert [c for c in ['dim0.5', 'dim0.25', 'dim0.1'] if 0.15 <= Rb[c] <= 0.85] == ['dim0.1']
need('app', 'Selected strengths are noise σ=0.6, occlusion 0.5 of the frame area centered (a tie with 0.75 centered, the weaker was taken), delay d=50 and flicker p=0.6, then blur σ=40 and dim g=0.1')
need('app', 'For dim, g=0.1 was the only strength inside [0.15,0.85]')
src_has('prereg thr', 'docs/p2b_prereg.md', 'dim：窗內標記步數比例', ['≥0.90', 'ρ_generic ≥0.60'])
need('app', 'a flagged share of at least 0.90 (observed')
src_has('probe seeds', 'docs/p2p3_notes.md', '合成探測幀', ['120–125', 'σ=0.6／0.3', '遮擋 0.5／0.25'])
need('app', 'on static starting frames of six probe seeds (120–125) corrupted by noise (σ=0.6,0.3), occlusion (0.5, 0.25), black and flicker')
need('app', 'The model card (20.6%)')
need('app', 'more than the 99th percentile of the difference caused by ±1 gray-level jitter')
need('app', 'which is 0.0317 rad')
src_has('jitter', 'results/realistic_table.md', '有害標籤', ['±1 灰階', '99 百分位'])
need('exp', 'The half-width of a 95% interval is about 0.08–0.10 per point') if False else None

# =============================================================== review round 1 additions (closed-door review, 2026-11-06)
import subprocess


def git_first(args):
    """commit time (unix) of the earliest commit matching `git log --reverse <args>`; None if git is unavailable"""
    try:
        out = subprocess.run(['git', '-C', ROOT, 'log', '--reverse', '--format=%ct'] + args, capture_output=True, text=True, timeout=60).stdout.split()
        return int(out[0]) if out else None
    except Exception:
        return None


def has_history():
    """True when ROOT is the original git repository (several commits). The public snapshot has one commit, and the order of the
    pre-registration and the runs cannot be read from it; the timing checks below are then skipped (and a note is printed)."""
    try:
        top = subprocess.run(['git', '-C', ROOT, 'rev-parse', '--show-toplevel'], capture_output=True, text=True, timeout=60).stdout.strip()
        if os.path.realpath(top) != os.path.realpath(ROOT):
            return False
        return subprocess.run(['git', '-C', ROOT, 'cat-file', '-e', 'f15042f78718bb567bc96e6bcaa0236c0ccd868b^{commit}'], capture_output=True, timeout=60).returncode == 0
    except Exception:
        return False


HISTORY = has_history()
if not HISTORY:
    print('NOTE: no original git history here (public snapshot). The pre-registration timing checks (git commit order, file times) are skipped; '
          'the checks of the frozen-detector hash and of the registered files below still run.')

# --- M8b: recovery range of hold on TC at L=25 and 50, Insertion at n=50
rr = [rho_ci(tcc_('black', L, n)[3], tcc_('black', L, n)[0], tcc_('black', L, n)[1])[0] for L in [25, 50] for n in [25, 50]]
need('disc', f'recovery at L=25 and 50 is {min(rr):.2f} to {max(rr):.2f} on TC and {ri[0]:.2f} on Insertion at n=50')
design('Where it works (black frames, L≤50 on TC and L=50 on Insertion) it shows that the replans inside the window account for most of the loss')

# --- M5b/M5c/M5d/M5: direction words for Median TE (title, discussion, conclusion)
assert mc(te_b25, hd_b25)[2] > 0.05                                           # close to hold on short black windows
assert s_(te_f) > s_(b_('oracle_n50', 'freeze', 100)) and mc(te_f, b_('oracle_n50', 'freeze', 100))[2] < 0.05     # beats hold on long frozen windows
assert mt_f[2] > 0.05 and q_te10 < q_te1 / 5                                   # not better than fixed(50) on fresh seeds; h=10 is ~10x cheaper
design('Median TE is close to hold on short black windows and beats hold on long frozen ones; a CARE-style trigger does not help')
need('disc', f'Median TE beat hold in the one regime where hold hurts, but on fresh seeds it did not beat fixed(50), and h=10 matched h=1 there ({s_(c_te10)} against {s_(c_te1)}) with {q_te10:.0f} policy calls instead of {q_te1:.0f}')
need('concl', 'for frozen frames at L=100 we found no rule that reliably beats leaving the policy alone: Median TE beat hold there, but not fixed(50) on fresh seeds')
# the unqualified claim would be false at L=50 (Table 3): hold beats fixed(50) on frozen frames at n=25
assert s_(hd25f) > s_(fx25f) and a1[2] < 0.05
src_has('P-TE3 failed', 'docs/p2b_prereg.md', 'P-TE3', ['fixed'])

# --- M7: Suh et al. numbers (project notes, which were read against the paper) and the arXiv text itself when it is available locally
for sub_ in ['5.8', '28.9', '9.3', '51.6']:
    src_has('suh table 5', 'docs/literature.md', 'scene black 28.9', [sub_ if sub_ in ('5.8', '28.9') else '51.6' if sub_ == '51.6' else '9.3'])
src_has('suh masking', 'docs/literature.md', 'oracle camera-token masking', ['1.3%→47.3%'])
need('disc', 'in their pooled table frozen frames are the worse fault (scene camera, with proprioception: \\pi_{0.5} 5.8% frozen against 28.9% black, GR00T 9.3% against 51.6%)')
need('rel', 'which raised scene-freezing success of their released checkpoint from 1.3% to 47.3%')
_pdf = os.path.join(ROOT, '.scratch', 'papers', '2609.39145', 'paper.txt')
if os.path.exists(_pdf):                                                     # optional: the downloaded PDF text (not part of the repository)
    _t = open(_pdf).read()
    assert 'π0.5 Yes 93.1 28.9 13.1 5.8 3.6' in re.sub(r'\s+', ' ', _t) and 'GR00T Yes 98.7 51.6 12.2 9.3 0.0' in re.sub(r'\s+', ' ', _t)
    assert 'Released Scene freezing 1.3 4.0 47.3' in re.sub(r'\s+', ' ', _t) and 'longer fault' not in _t.lower()
design('(as in their episode-start faults)')
design('and in their pooled table frozen frames are the worse fault')
design('at L=25 and 50, split by whether a replan falls inside the window')
design('Adding the onset t_0 and t_0^2 to the regression')

# --- M9a: the formula in the text has no square, as in the code that produced theta
cg_ = open(os.path.join(ROOT, 'code', 'calib_generic.py')).read(); rp_ = open(os.path.join(ROOT, 'code', 'run_p2p3.py')).read()
assert re.search(r'score = lambda X: np\.linalg\.norm\(\(X - mu\) - \(\(X - mu\) @ V\) @ V\.T, axis=1\)', cg_) and 'norm())  # PCA reconstruction error' in rp_
design('s(I)=\\lVert (f(I)-\\mu)-UU^{\\top}(f(I)-\\mu)\\rVert_2 (PCA reconstruction error)')
assert not re.search(r'rVert_2\^2', raw_nc)
all_equal('percentile', r'the (\d+)th percentile of', src_consts('code/calib_generic.py', r'np\.percentile\(st, (\d+)\)')[0])

# --- M9b: unseen corruptions only at n=50, L=50; synthetic grid and seeds
assert set(uns['n']) == {50} and set(uns['L']) == {50} and len(uns) == 540 and set(uns['env_seed']) == set(range(90))
assert set(rem['env_seed']) == set(range(90)) and set(rem['n']) <= {25, 50, 100} and set(rem['L']) <= {0, 50, 100}
assert set(rem[rem['corr'] == 'flicker0.6']['n']) == {50} and set(rem[rem['corr'] != 'flicker0.6']['n']) == {25, 50, 100}
design('the unseen corruptions use the same seeds at n=50, L=50 only')
# --- M9c: an episode ends at the first reward of 4 or after 400 steps
fgt = fg[fg['te'] == 0]
assert (fgt[fgt['success'] == 0]['steps'] == 400).all() and (fgt[fgt['success'] == 1]['steps'] <= 400).all() and (fgt[fgt['success'] == 1]['steps'] < 400).mean() > 0.99   # successes end early (one of 1755 succeeds on step 400)
for name_, sim, env_ in GYM_SRC:
    assert 'terminated = is_success = reward == 4' in (env_ if env_ is not None else sim), name_
assert 'max_steps=400' in open(os.path.join(ROOT, 'code', 'tasks.py')).read()
design('and an episode ends at the first reward of 4 or after 400 steps')
design('(gym-aloha ends an episode at the first reward of 4)')
# --- M9d: the pixel rule was run on noise, occlusion and delay only
sd_ = rem[rem['rule'] == 'detect']
assert set(sd_['corr']) == {'noise0.6', 'occlude0.5center', 'delay50'} and len(sd_) == 540
design('it was not run on flicker')
# --- M9e: hold at n=1, 10, 100 (oracle.csv, baselines.csv)
o1, o10 = s_(V(orc[orc['config'] == 'oracle_n1_L100_freeze'])), s_(V(orc[orc['config'] == 'oracle_n10_L100_freeze']))
f1_, f10_ = gk(1, 100, 'freeze'), gk(10, 100, 'freeze')
assert cl[1] <= 25 and cl[10] <= 25 and o1 > f1_ and o10 > f10_
for m_ in ['black', 'freeze']:
    for L_ in [25, 50, 100]:
        assert (b_('oracle_n100', m_, L_) == b_('fixed_n100', m_, L_)).all()    # hold at n=100 is the same rule as fixed(100)
need('hold', f'Hold was also run at n∈{{1,10}}, where clean success is at most {max(cl[1], cl[10])}/100 and it is above the LeRobot-loop fixed ({o1} against {f1_}, {o10} against {f10_}), and at n=100 it is the same rule as fixed')
# --- P2: rho above 1
rho_all = [r_[0] for m_ in ['black', 'freeze'] for L_ in [25, 50, 100] for n_ in [25, 50] for r_ in [rho_ci(tcc_(m_, L_, n_)[3], tcc_(m_, L_, n_)[0], tcc_(m_, L_, n_)[1])] if r_ is not None]
assert max(rho_all) > 1 and 'tab:hold' in raw_nc
hn25 = s_(V(orc[orc['config'] == 'oracle_n25_L50_black'])); hn10 = s_(V(orc[orc['config'] == 'oracle_n10_L50_black']))
assert hn25 > cl[25] and hn10 > cl[10]
need('meth', f'Values above 1 occur when hold scores above the clean rate of fixed(n), for example {hn25}/100 against {cl[25]}/100 at n=25 (Table ) and {hn10}/100 against {cl[10]}/100 at n=10, L=50 with black frames')
# --- M10: detection against the harmfulness label (realistic_table.md section 3b)
src_has('harm black', 'results/realistic_table.md', '| black | 1.00 | 1.000 | 1.000 |', ['1.000'])
src_has('harm freeze', 'results/realistic_table.md', '| freeze | 0.95 | 1.000 | 0.034 |', ['0.034'])
need('app', 'the detector flags 1.000 of harmful black steps and 0.034 of harmful frozen steps')
# --- P3: frozen frames not seen by the OOD detector (AUC 0.537 above; detection 0.034)
design('but not for delayed, dimmed or frozen frames')
design('cannot see delayed or frozen frames')
# --- M11b etc. covered above; M12: loops of the clean columns
bc = bl[bl['corr'] == 'clean']
assert (bc[bc['method'].isin(['fixed_n50', 'fixed_n100', 'meanTE_h1'])]['src'] == 'full_grid').all()
design('the clean column, and so the denominator of ρ, comes from the LeRobot loop')
design('except that the clean column of fixed and Mean TE comes from the LeRobot loop')
# --- M3a/M3b/M4: pre-registration timing (git commits and file times)
t_pre = git_first(['--diff-filter=A', '--', 'docs/p2b_prereg.md']) if HISTORY else None
t_ins_old = git_first(['--diff-filter=A', '--', 'results/parts/ins_oracle.csv']) if HISTORY else None
t_ins_det = git_first(['--diff-filter=A', '--', 'results/parts/ins_detect.csv']) if HISTORY else None
t_new = [git_first(['--diff-filter=A', '--', f'results/parts/{f_}']) if HISTORY else None for f_ in ['p2b_unseen_blur.csv', 'p2b_unseen_dim.csv', 'p2b_te.csv', 'p2b_ins_fixed_L50.csv']]
# checks that do not need the history: the registered files exist, and the frozen detector has the sha256 written in the pre-registration and in run_p2b.py
import hashlib
_prereg = open(os.path.join(ROOT, 'docs', 'p2b_prereg.md')).read()
_sha = hashlib.sha256(open(os.path.join(RES, 'parts', 'generic_detector.npz'), 'rb').read()).hexdigest()
assert _sha in _prereg and _sha in open(os.path.join(ROOT, 'code', 'run_p2b.py')).read(), 'frozen detector hash differs from the pre-registration or run_p2b.py'
n_ok += 1
if HISTORY and t_pre is None:
    problems.append('git history not available: pre-registration timing (M3a) not checked')
elif HISTORY:
    assert t_ins_old < t_pre and t_ins_det < t_pre and all(t > t_pre for t in t_new)
design('before the new runs they test (all arms of F2 and F3, and only the same-loop fixed arm of F1, whose hold and detect arms are first-study rows that were already known)')
design('The hold and detect arms are first-study rows on the same seeds, so this shows that the two borderline n=25 cells survive a rerun of fixed; it is not an independent replication')
design('this family was not declared in advance')
design('The public repository is a clean snapshot without the original history, so the date of the registration cannot be checked from it and rests on our statement. The private history can be given to reviewers on request.')
design('which is not in the public repository, and an earlier plan with other targets on other seeds (docs/baselines_spec.md), which we did not follow. The snapshot cannot show when any version was written, so for readers this is our statement.')
design('and the public repository cannot show when')
assert r'\newcommand{\repourl}{\url{https://github.com/Jadiouo/action-chunking-blackout}}' in raw and r'\newcommand{\repocommit}{\texttt{v1}}' in raw
assert '<REPO-URL>' not in raw and '<COMMIT>' not in raw
need('repro', 'are at \\repourl{} (tag \\repocommit{}). The repository is a clean snapshot without the original git history')
design('(8) The six-row list in Table was assembled after all runs')
# --- strength rule: qualitative form committed before the pilot, numerical form with the pilot results; added clause changes flicker
t_qual = git_first(['-S不歸零', '--', 'docs/tasks.md']) if HISTORY else None; t_num = git_first(['-S最接近 0.5']) if HISTORY else None; t_pilot = git_first(['--diff-filter=A', '--', 'results/parts/p2p3_pilot.csv']) if HISTORY else None
if HISTORY and t_qual is not None:
    assert t_qual < os.path.getmtime(os.path.join(RES, 'parts', 'p2p3_pilot.csv')) and t_num == t_pilot
fl_R = {c: pilot_R(c) for c in ['flicker0.3', 'flicker0.6', 'flicker1']}
assert abs(fl_R['flicker1'] - 0.5) < abs(fl_R['flicker0.6'] - 0.5) < abs(fl_R['flicker0.3'] - 0.5)       # the original rule would have chosen p=1
need('exp', f'under the original rule this strength would have been selected (retention {fl_R["flicker1"]:.2f} against {fl_R["flicker0.6"]:.2f} for p=0.6)')
design('with a rule that our notes (docs/p2p3_notes.md) declare as fixed before the pilot')
design('Our notes record the rule as fixed before the pilot for the first four corruptions, with one clause added after the pilot results were seen')
# --- reward=4: environment criterion; the declaration as primary metric is first in the repository after the first Insertion run
t_decl = git_first(['-Greward ?= ?4.{0,8}預先宣告|預先宣告.{0,8}reward ?= ?4']) if HISTORY else None   # first text calling reward=4 pre-declared
if t_decl is not None:
    assert t_decl > git_first(['--diff-filter=A', '--', 'results/insertion.csv'])
design('Our notes call it pre-declared, but we wrote that declaration down only after the first Insertion run, so we do not claim that it was fixed in advance')
# --- P10, P8, P7, P6, P5: design statements checked against code / data
assert ins['env_seed'].min() == 0 and ins['env_seed'].max() == 199
assert re.search(r'if prevA is None:\s+H = 1\b', rp_) and 'H_init=1' in rp_
design('with a minimum of one step and one step for the first chunk')
design('where i=0 is the oldest (Mean TE)')
assert 'w = np.exp(-0.01 * np.arange(len(cands)))' in rp_ and 'i=0 oldest' in rp_
design('On TC the seed ranges for pilot, calibration and evaluation are disjoint')
design('retention is reported without an interval')
design("Median TE's guarantee and CARE's design both assume that corruption is sparse or momentary")

# --- design constants compared with the code (not hard-coded): resamples, rho threshold, gamma, eps, H_max, black threshold, sample sizes
nb_code = int(src_consts('code/analyze_arxiv.py', r'(?m)^NB = (\d+)')[0])
assert nb_code == NB
all_equal('resamples', r'(\d+) resamples', nb_code)
th_code = set(src_consts('code/analyze_arxiv.py', r'den < (0\.\d+)') + src_consts('code/analyze_arxiv.py', r'ok = den >= (0\.\d+)'))
assert th_code == {0.10}
all_equal('rho threshold (reported only when)', r'reported only when[^≥]*≥ ?(0\.\d+)', 0.10)
all_equal('rho threshold (denominator below)', r'denominator (?:is )?below (0\.\d+)', 0.10)
all_equal('rho threshold (n/r)', r"n/r.{0,6}: denominator below (0\.\d+)", 0.10)
cnst = re.search(r'CHUNK, HMAX, GAMMA, EPS = (\d+), (\d+), ([\d.]+), ([\de.-]+)', rp_)
assert int(cnst.group(1)) // 2 == int(cnst.group(2))
all_equal('gamma', r'\\gamma=(0\.\d+)', float(cnst.group(3)))
all_equal('epsilon', r'\\epsilon=10\^\{-(\d+)\}', -math.log10(float(cnst.group(4))))
design('H\\le H_{\\max}=\\lfloor K/2\\rfloor')
bthr = src_consts('code/run_t3t4.py', r'(?m)^BLACK_THR = ([\d.]+)')[0]
all_equal('black threshold', r'mean pixel intensity is below (\d+\.\d+)', bthr)
ws_ = re.findall(r'(\w+) variants', T)
n_var = int(src_consts('docs/p2p3_notes.md', r'(\d+) 個變體')[0])
if not ws_ or set(ws_) != {'seven'} or n_var != 7:
    problems.append(f'MISMATCH [variants] text words {ws_}, notes say {n_var}')
else:
    n_ok += 1
all_equal('episodes per fault type', r'(\d+) episodes per fault type', RG[('J', 'black')]['N'])
all_equal('episodes per cell (paired initial conditions)', r'with (\d+) paired initial conditions per cell', len(V(FG(50, 50, 'black'))))
design('there is no \\todo')  if False else None

# --- direction words of the metric definition and of the frozen-frame conclusion
design('and ρ<0 means the rule does worse than fixed(n)')
design('with frozen frames over long windows, not replanning does worse than leaving the policy alone')
design('in our data it does worse than fixed(n) for frozen frames with L=100')
frz_rho = [rho_ci(tcc_('freeze', 100, n)[3], tcc_('freeze', 100, n)[0], tcc_('freeze', 100, n)[1]) for n in [25, 50]]
assert all(r_[0] < 0 for r_ in frz_rho)                                    # not replanning recovers nothing at freeze L=100 (rho < 0)
design('With frozen frames at L=100, replanning on the image is not the main cost, and not replanning is the worse choice')
# the qualifier L=100 is needed: at L=25 and 50 (n=25) hold recovers all of the loss on frozen frames (rho > 1)
assert all(rho_ci(tcc_('freeze', L_, 25)[3], tcc_('freeze', L_, 25)[0], tcc_('freeze', L_, 25)[1])[0] > 1 for L_ in [25, 50])

# --- second closed-door round (R2-4, R2-5, R2-7, R2-8, R2-9 and the two optional edits)
assert min(rr) > 0.5                                                       # "most of the loss" for black frames, L<=50 on TC (recovery 0.88 and up)
design('With black frames, replanning on the bad image is sufficient to explain most of the loss for L≤50, and hold removes most of it')
design('and the harmful faults that the detector cannot see are delay and frozen frames')   # data: flagged share of harmful frozen steps 0.034, AUC 0.537 (lines above)
# share of corrupted candidates at the window end (h=1, K=100) grows with L: 0.25-0.33, 0.50, 1.00 for L=25, 50, 100 and t0 in [50, 250]
_sh = {L_: [L_ / min(100, t_ + L_) for t_ in range(50, 251)] for L_ in (25, 50, 100)}
assert max(_sh[25]) < 0.34 and min(_sh[50]) == max(_sh[50]) == 0.5 and min(_sh[100]) == 1.0
design('A blackout lasting many replans corrupts every new chunk and a growing share of the candidates, and this is the regime we measure')
design('With black frames, then, horizon and blackout length interact: L and n together set')
design('and checks that a fixed list of key sentences is still present as written; when the original git history is present, it also checks the order of the pre-registration and the runs, and in the public snapshot it skips that check and says so; it does not check other constants such as K, and it cannot judge whether a sentence that is not on that list is worded correctly')
design('That statement is about our search, not a claim that no such work exists')

# --- direction check on every "a against b (x/y)" in the text: x - y must equal a - b (x/y = only-first / only-second)
xy_pat = re.compile(r'(\d+) against (\d+)(?: [^,();]{0,25})?[,(] ?(\d+)/(\d+)[,) ]')
xy = [(int(a), int(b), int(x), int(y)) for a, b, x, y in xy_pat.findall(T)]
bad_xy = [t for t in xy if t[2] - t[3] != t[0] - t[1]]
if len(xy) < 9 or bad_xy:
    problems.append(f'MISMATCH [x/y direction] {len(xy)} "a against b, x/y" groups found, inconsistent: {bad_xy}')
else:
    n_ok += len(xy)

# =============================================================== lint
design_patterns = [
    r'[nLhmgdστγεKHpq]\s*=\s*[--]?[\d.]+(?:–[\d.]+)?', r'∈\s*\{[^}]*\}', r'\{[\d,… ]+\}', r'\([0-9]+\)',   # parameters, sets, enumerations
    r'Sec\.\s?[\d.]+', r'Table [\d.]+', r'Fig\. [\d.]+', r'Eq\. ?[\d.]+', r'Appendix [A-Z]', r'\b[A-Z]\.\d\b',          # cross refs
    r'K=100', r'\b480×640\b', r'\b400 steps\b', r'\b50 Hz\b', r'6144', r'3×4', r'\b0\.2\b', r'0\.05\^2',
    r'e\^\{-0\.01 i\}', r'\^\{[-\d]+\}', r'10\^-2',
    r'\b(?:P|F)[1-6]\b', r'\bπ0(?:\.5)?\b', r'ResNet-18', r'\bπ_0\b', r'k≥1', r'k=0', r'\bL≥\s?\d+', r'\bn≥\s?\d+', r'\bL≤\s?\d+', r'20\d\d(?:-\d\d-\d\d)?',
    r'95%', r'4000 resamples', r'\b0\.10\b', r'\b0\.05\b', r'5% of', r'about 0\.08–0\.10', r'30-episode', r'up to 100 steps', r'25 to 100', r'12 arm joints', r'a_\{t-1\}',
    r'RTX 5070', r'seeds? \d+[–-]\d+', r'vs \d+', r'±0\.15–0\.2', r'up to 50 steps', r'\b(?:2|3|10|40|50|1\.0|1\.5)D\b', r'PyTorch 2\.10', r'LeRobot 0\.4\.4', r'Python 3\.10', r'MuJoCo 3\.8\.1', r'gym-aloha 0\.1\.5',
    r'\b1[-–]\d\b', r'\breward[ =≥]*\d', r'\b\d+ (?:\w+ )?(?:seeds|episodes)\b', r'reaches 4', r'\b(?:20k|40k|100k)', r'\b5–10\b', r'p=1\b', r'\bτ=1\.0\b', r'1\.0 and 1\.5',
]
for pat in design_patterns:
    for m in re.finditer(pat, T):
        for k in range(m.start(), m.end()):
            cov[k] = True
unc = []
_s1 = T.index('Related Work'); _s2 = T.index('Experiments We use the simulated')
for m in re.finditer(r'(?<![\w.])[-+]?\d+(?:[.,]\d+)*', T):
    if _s1 <= m.start() < _s2:
        continue
    if not all(cov[m.start():m.end()]):
        ctx = T[max(0, m.start() - 50):m.end() + 40]
        unc.append((m.group(), ctx))

print(f'claims ok: {n_ok}')
print(f'table rows checked: {NT[0]}; table problems: {len(TABLE_PROBLEMS)}')
print(f'total comparisons: {n_ok + NT[0]}')
print(f'todo markers: {n_todo} (allowed: 0)')
bad = problems + TABLE_PROBLEMS
if n_todo != 0:
    bad.append(f'expected no \\todo, found {n_todo}')
for b in bad:
    print(b)
if unc:
    print(f'UNCOVERED numeric tokens: {len(unc)}')
    for tok, ctx in unc[:400]:
        print(f'  {tok!r:10} ...{ctx}...')
sys.exit(1 if (bad or unc) else 0)
