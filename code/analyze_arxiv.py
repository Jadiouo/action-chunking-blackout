#!/usr/bin/env python3
"""Recompute every number used in paper/arxiv/main.tex from the result CSVs.

Run (system python3, no GPU):   python3 code/analyze_arxiv.py
Writes   results/arxiv_numbers.json
         paper/arxiv/figs/*.pdf       (figures)
         paper/arxiv/tables/*.tex     (tables, \\input by main.tex)

Rules (white review S8):  realistic.csv is always filtered to stage == 'main' (asserted);
pilot rows are only read by the strength-selection table, from the pilot rows explicitly.
Statistics: Wilson 95% CI; exact McNemar (two-sided binomial); Holm only inside the pre-listed
families; R and rho with paired seed bootstrap (4000 resamples, percentile 2.5/97.5, seed 0).
"""
import json, math, os, re, sys
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
RES = os.path.join(ROOT, 'results')
FIG = os.path.join(ROOT, 'paper', 'arxiv', 'figs')
TAB = os.path.join(ROOT, 'paper', 'arxiv', 'tables')
os.makedirs(FIG, exist_ok=True)
os.makedirs(TAB, exist_ok=True)
NB = 4000
OUT = {}


def rd(name):
    return pd.read_csv(os.path.join(RES, name))


# ----------------------------------------------------------------------------- statistics
def wilson(k, n, z=1.959964):
    if n == 0:
        return (float('nan'), float('nan'))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def mcnemar(a, b):
    """a, b: 0/1 arrays aligned on seed.  Returns (x, y, p): x = a-only wins, y = b-only wins."""
    a = np.asarray(a); b = np.asarray(b)
    x = int(((a == 1) & (b == 0)).sum()); y = int(((b == 1) & (a == 0)).sum())
    if x + y == 0:
        return x, y, 1.0
    return x, y, float(stats.binomtest(min(x, y), x + y, 0.5).pvalue)


def holm(ps):
    m = len(ps)
    order = np.argsort(ps)
    adj = [0.0] * m
    run = 0.0
    for rank, i in enumerate(order):
        run = max(run, (m - rank) * ps[i])
        adj[i] = min(1.0, run)
    return adj


def boot_idx(n, seed=0):
    rng = np.random.default_rng(seed)
    return rng.integers(0, n, size=(NB, n))


def boot_R(clean, rule, seed=0):
    idx = boot_idx(len(rule), seed)
    c = clean[idx].mean(1); r = rule[idx].mean(1)
    ok = c > 0
    v = r[ok] / c[ok]
    return float(rule.mean() / clean.mean()), float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))


def rho_point(clean, fixed, rule):
    den = clean.mean() - fixed.mean()
    if den < 0.10:
        return None
    return float((rule.mean() - fixed.mean()) / den)


def boot_rho(clean, fixed, rule, seed=0):
    """paired seed bootstrap; returns (point, lo, hi, frac_valid) or None if denominator < 0.10"""
    pt = rho_point(clean, fixed, rule)
    if pt is None:
        return None
    idx = boot_idx(len(rule), seed)
    c = clean[idx].mean(1); f = fixed[idx].mean(1); r = rule[idx].mean(1)
    den = c - f
    ok = den >= 0.10
    v = (r[ok] - f[ok]) / den[ok]
    return pt, float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5)), float(ok.mean())


def boot_diff(a, b, seed=0):
    idx = boot_idx(len(a), seed)
    d = a[idx].mean(1) - b[idx].mean(1)
    return float(a.mean() - b.mean()), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def ksched(n, L, t0):
    return -(-(t0 + L) // n) - -(-t0 // n)


def fmt_p(p):
    if p >= 0.1:
        if abs(p * 1000 - round(p * 1000)) < 1e-9 and round(p * 1000) % 10 == 5:
            return f'{p:.3f}'      # exact half (e.g. 0.125): two decimals would round it down to 0.12
        return f'{p:.2f}'
    if p >= 0.001:
        return f'{p:.3f}'
    return f'{p:.1e}'.replace('e-0', 'e-').replace('e+0', 'e+')


def cnt(a):
    return f'{int(np.sum(a))}/{len(a)}'


# ----------------------------------------------------------------------------- loaders
fg = rd('full_grid.csv')
orc = rd('oracle.csv')
det = rd('detect.csv')
bl = rd('baselines.csv')
re_all = rd('realistic.csv')
assert set(re_all.stage.unique()) == {'main', 'pilot'}
re_main = re_all[re_all.stage == 'main'].copy()          # S8: never aggregate pilot rows
assert len(re_main) == 5940
ins = rd('insertion.csv')
insf = rd('p2b_insertion_fixed.csv')
te2 = rd('p2b_te.csv')
unseen = rd('p2b_unseen.csv')
pil2 = rd('p2b_pilot.csv')
dty = rd('p2b_dtype.csv')
tcc = rd('p2b_tc_clean.csv')
gcl = rd('p2b_generic_clean.csv')
pro = rd('proprio.csv')
pilot0 = rd('pilot.csv')


def vec(df, col='success'):
    d = df.sort_values('env_seed')
    assert d.env_seed.is_unique, 'duplicate seeds'
    return d[col].to_numpy().astype(int)


def FG(n, L, mode, te=False):
    cfg = (f'TE_L0_black' if L == 0 else f'TE_L{L}_{mode}') if te else (f'n{n}_L0_black' if L == 0 else f'n{n}_L{L}_{mode}')
    d = fg[fg.config == cfg]
    assert len(d) == 100, cfg
    return d.sort_values('env_seed')


def BL(method, corr, L):
    d = bl[(bl.method == method) & (bl['corr'] == corr) & (bl.L == L)]
    assert len(d) in (90, 100), (method, corr, L, len(d))
    return d.sort_values('env_seed')


def RE(rule, corr, n, L):
    d = re_main[(re_main.rule == rule) & (re_main['corr'] == corr) & (re_main.n == n) & (re_main.L == L)]
    assert len(d) == 90, (rule, corr, n, L, len(d))
    return d.sort_values('env_seed')


def INS(rule, mode, L, n):
    """rule in fixed_old, oracle, detect from insertion.csv"""
    r = 'fixed' if rule == 'fixed_old' else rule
    d = ins[(ins.rule == r) & (ins['mode'] == mode) & (ins.L == L) & (ins.n == n) & (ins.te == 0)]
    assert len(d) == 200, (rule, mode, L, n, len(d))
    return d.sort_values('env_seed')


def INSF(mode, L, n):
    d = insf[(insf.L == L) & (insf.n == n) & ((insf['corr'] == mode) if L > 0 else (insf['corr'] == 'black'))]
    assert len(d) == 200, (mode, L, n, len(d))
    return d.sort_values('env_seed')


def succ_r(d, level):
    return (d.max_reward.to_numpy() >= level).astype(int)


def cell(a):
    a = np.asarray(a)
    k, n = int(a.sum()), len(a)
    lo, hi = wilson(k, n)
    return {'k': k, 'n': n, 'rate': k / n, 'lo': lo, 'hi': hi}


# ============================================================================ 1. fixed-n grid (TC)
N = {}
OUT['grid'] = {}
for mode in ['black', 'freeze']:
    for n in [1, 10, 25, 50, 100]:
        for L in [0, 10, 25, 50, 100]:
            if L == 0 and mode == 'freeze':
                continue
            OUT['grid'][f'fixed_n{n}_L{L}_{mode}'] = cell(vec(FG(n, L, mode)))
    for L in [0, 10, 25, 50, 100]:
        if L == 0 and mode == 'freeze':
            continue
        OUT['grid'][f'meanTE_L{L}_{mode}'] = cell(vec(FG(1, L, mode, te=True)))
# hold (oracle.csv) in the grid
for mode in ['black', 'freeze']:
    for n in [10, 25, 50]:
        for L in [10, 25, 50, 100]:
            d = orc[orc.config == f'oracle_n{n}_L{L}_{mode}']
            assert len(d) == 100
            OUT['grid'][f'hold_n{n}_L{L}_{mode}'] = cell(vec(d))

# ============================================================================ 2. mechanism
mech = {}
# 2a natural experiment: fixed n=100 (run_grid.py path), k_sched = 0 vs >= 1
for L in [10, 25, 50]:
    for mode in ['black', 'freeze']:
        d = FG(100, L, mode)
        ks = np.array([ksched(100, L, int(t)) for t in d.t0])
        s = vec(d)
        k0 = s[ks == 0]; k1 = s[ks >= 1]
        ft = stats.fisher_exact([[k0.sum(), len(k0) - k0.sum()], [k1.sum(), len(k1) - k1.sum()]])[1]
        mech[f'nat_n100_L{L}_{mode}'] = {'k0': [int(k0.sum()), len(k0)], 'k1': [int(k1.sum()), len(k1)], 'fisher_p': float(ft)}
# 2b regression with k x log2 n interaction (cluster-robust, cluster = env seed)


def logit_cluster(X, y, groups):
    beta = np.zeros(X.shape[1])
    for _ in range(200):
        p = 1 / (1 + np.exp(-(X @ beta)))
        W = p * (1 - p)
        H = X.T @ (X * W[:, None])
        g = X.T @ (y - p)
        step = np.linalg.solve(H, g)
        beta = beta + step
        if np.abs(step).max() < 1e-10:
            break
    p = 1 / (1 + np.exp(-(X @ beta)))
    W = p * (1 - p)
    H = X.T @ (X * W[:, None])
    Hinv = np.linalg.inv(H)
    sc = X * (y - p)[:, None]
    ug = np.unique(groups)
    S = np.zeros((X.shape[1], X.shape[1]))
    for g_ in ug:
        s = sc[groups == g_].sum(0)
        S += np.outer(s, s)
    G = len(ug); Nn = len(y); K = X.shape[1]
    V = Hinv @ S @ Hinv * (G / (G - 1)) * ((Nn - 1) / (Nn - K))
    return beta, V


reg = {}
for mode in ['black', 'freeze']:
    rows = []
    for n in [10, 25, 50, 100]:
        for L in [10, 25, 50, 100]:
            d = FG(n, L, mode).copy()
            d['ks'] = [ksched(n, L, int(t)) for t in d.t0]
            d['ln'] = math.log2(n)
            rows.append(d)
    D = pd.concat(rows)
    # I: logged count (outcome-dependent, kept for the sensitivity statement); J: scheduled count of Eq. (1), used in the
    # paper; JT: J plus onset t0 and t0^2 (task-phase adjustment, sensitivity analysis; t0 centred at 150, in units of 100 steps)
    for tag, kcol in [('I', 'replans_in_blackout'), ('J', 'ks'), ('JT', 'ks')]:
        k = D[kcol].to_numpy().astype(float); ln = D.ln.to_numpy(); Lc = D.L.to_numpy() / 10.0
        cols = [np.ones(len(D)), k, ln, k * ln, Lc]
        if tag == 'JT':
            tz = (D.t0.to_numpy().astype(float) - 150.0) / 100.0
            cols += [tz, tz ** 2]
        X = np.column_stack(cols)
        beta, V = logit_cluster(X, D.success.to_numpy().astype(float), D.env_seed.to_numpy())
        out = {'N': len(D), 'b_k': beta[1], 'se_k': math.sqrt(V[1, 1]), 'b_ln': beta[2], 'se_ln': math.sqrt(V[2, 2]),
               'b_int': beta[3], 'se_int': math.sqrt(V[3, 3]), 'z_int': beta[3] / math.sqrt(V[3, 3]),
               'p_int': float(2 * stats.norm.sf(abs(beta[3] / math.sqrt(V[3, 3]))))}
        sl = {}
        for n in [10, 25, 50, 100]:
            a = np.array([0, 1, 0, math.log2(n), 0.0])
            b_ = beta[1] + beta[3] * math.log2(n)
            var = V[1, 1] + math.log2(n) ** 2 * V[3, 3] + 2 * math.log2(n) * V[1, 3]
            se = math.sqrt(var)
            sl[str(n)] = {'slope': b_, 'se': se, 'lo': b_ - 1.959964 * se, 'hi': b_ + 1.959964 * se, 'or': math.exp(b_)}
        out['slope'] = sl
        reg[f'{tag}_{mode}'] = out
mech['reg'] = reg

# 2b' phase-only cells: cells where L is not a multiple of n, so that the scheduled count k takes two adjacent values
# (floor(L/n), ceil(L/n)) and which of the two is decided by the phase of t0 alone. Log odds ratio of success for one more
# scheduled replan, 0.5 added to each of the four counts (Haldane-Anscombe); raw counts are stored as well.
phase = {}
for mode in ['black', 'freeze']:
    for n in [10, 25, 50, 100]:
        for L in [10, 25, 50, 100]:
            if L % n == 0:
                continue
            d = FG(n, L, mode)
            ks = np.array([ksched(n, L, int(t)) for t in d.t0])
            s_ = vec(d)
            vals = sorted(set(ks.tolist()))
            assert len(vals) == 2 and vals[1] - vals[0] == 1, (n, L, vals)
            lo_, hi_ = s_[ks == vals[0]], s_[ks == vals[1]]
            a, b, c, e_ = hi_.sum() + .5, len(hi_) - hi_.sum() + .5, lo_.sum() + .5, len(lo_) - lo_.sum() + .5
            phase[f'{mode}_n{n}_L{L}'] = {'k_lo': int(vals[0]), 'k_hi': int(vals[1]),
                                          'lo': [int(lo_.sum()), len(lo_)], 'hi': [int(hi_.sum()), len(hi_)],
                                          'log_or': float(math.log(a / b) - math.log(c / e_))}
mech['phase_only'] = phase

# 2c stratified gain of hold over same-loop fixed n=100 by whether fixed-100 would replan in the window (pooled L = 25, 50)
strat = {}
for mode in ['black', 'freeze']:
    for hn in [25, 50]:
        res = {'k0': {'fx': 0, 'hd': 0, 'n': 0}, 'k1': {'fx': 0, 'hd': 0, 'n': 0}}
        for L in [25, 50]:
            fx = BL('fixed_n100', mode, L); hd = BL(f'oracle_n{hn}', mode, L)
            assert (fx.env_seed.to_numpy() == hd.env_seed.to_numpy()).all() and (fx.t0.to_numpy() == hd.t0.to_numpy()).all()
            ks = np.array([ksched(100, L, int(t)) for t in fx.t0])
            for key, m in [('k0', ks == 0), ('k1', ks >= 1)]:
                res[key]['fx'] += int(vec(fx)[m].sum()); res[key]['hd'] += int(vec(hd)[m].sum()); res[key]['n'] += int(m.sum())
        strat[f'{mode}_hold_n{hn}'] = res
mech['strat'] = strat
OUT['mech'] = mech

# ============================================================================ 3. TC intervention table (same-loop fixed)
tc = {}
for mode in ['black', 'freeze']:
    for L in [25, 50, 100]:
        for n in [25, 50]:
            clean = vec(FG(n, 0, 'black'))
            fx = vec(BL(f'fixed_n{n}', mode, L)); hd = vec(BL(f'oracle_n{n}', mode, L)); dt = vec(BL(f'detect_n{n}', mode, L))
            x, y, p = mcnemar(hd, fx)
            xd, yd, pd_ = mcnemar(dt, fx)
            xh, yh, ph = mcnemar(dt, hd)
            r_hd = boot_rho(clean, fx, hd); r_dt = boot_rho(clean, fx, dt)
            tc[f'{mode}_L{L}_n{n}'] = {
                'clean': cell(clean), 'fixed': cell(fx), 'hold': cell(hd), 'detect': cell(dt),
                'R_fixed': boot_R(clean, fx), 'R_hold': boot_R(clean, hd),
                'rho_hold': r_hd, 'rho_detect': r_dt,
                'hold_vs_fixed': {'x': x, 'y': y, 'p': p},
                'detect_vs_fixed': {'x': xd, 'y': yd, 'p': pd_},
                'detect_vs_hold': {'x': xh, 'y': yh, 'p': ph},
                'queries': {k_: float(BL(m_, mode, L).replans_total.mean()) for k_, m_ in
                            [('fixed', f'fixed_n{n}'), ('hold', f'oracle_n{n}'), ('detect', f'detect_n{n}')]},
            }
OUT['tc'] = tc
# detection latency and false alarms of the pixel rules (detect.csv, 100 episodes per cell)
dd = det[det.config != 'detect_n50_L0_black']
OUT['detect_pixel'] = {
    'episodes': int(len(dd)), 'detected_eps': int((dd.detect_latency.notna()).sum()),
    'max_latency': float(dd.detect_latency.max()), 'false_detect_total': int(dd.false_detect.sum()),
}
c0 = det[det.config == 'detect_n50_L0_black'].sort_values('env_seed')
f50 = FG(50, 0, 'black')
x, y, p = mcnemar(vec(c0), vec(f50))
OUT['detect_pixel']['L0_n50'] = {'detect': cell(vec(c0)), 'fixed': cell(vec(f50)), 'x': x, 'y': y, 'p': p, 'det_steps': int(c0.detected_steps.sum())}

# M2: the four hold rows of the synthetic corruptions are one experiment
m2 = {}
for n in [25, 50, 100]:
    for L in [50, 100]:
        arrs = {c: vec(RE('oracle', c, n, L)) for c in ['noise0.6', 'occlude0.5center', 'delay50']}
        if n == 50:
            arrs['flicker0.6'] = vec(RE('oracle', 'flicker0.6', n, L))
        names = list(arrs)
        diffs = {f'{a}|{b}': int((arrs[a] != arrs[b]).sum()) for i, a in enumerate(names) for b in names[i + 1:]}
        m2[f'n{n}_L{L}'] = {'succ': {c: int(v.sum()) for c, v in arrs.items()}, 'diffs': diffs}
OUT['hold_identity'] = m2

# ============================================================================ 4. synthetic corruptions (realistic.csv main)
syn = {}
CORR = ['noise0.6', 'occlude0.5center', 'delay50', 'flicker0.6']
for c in CORR:
    for L in [50, 100]:
        ns = [50] if c == 'flicker0.6' else [25, 50, 100]
        for n in ns:
            clean = vec(FG(n, 0, 'black'))[:90]
            clean_rows = FG(n, 0, 'black'); clean = vec(clean_rows[clean_rows.env_seed < 90])
            fx = vec(RE('fixed', c, n, L)); hd = vec(RE('oracle', c, n, L)); gn = vec(RE('generic', c, n, L))
            x1, y1, p1 = mcnemar(hd, fx); x2, y2, p2 = mcnemar(gn, fx); x3, y3, p3 = mcnemar(gn, hd)
            e = {'clean': cell(clean), 'fixed': cell(fx), 'hold': cell(hd), 'ood': cell(gn),
                 'R_fixed': boot_R(clean, fx), 'R_hold': boot_R(clean, hd), 'R_ood': boot_R(clean, gn),
                 'rho_hold': boot_rho(clean, fx, hd), 'rho_ood': boot_rho(clean, fx, gn),
                 'hold_vs_fixed': [x1, y1, p1], 'ood_vs_fixed': [x2, y2, p2], 'ood_vs_hold': [x3, y3, p3]}
            g = RE('generic', c, n, L)
            e['det'] = {'ep_detected': float((g.det_in > 0).mean()), 'in_frac': float(g.det_in.sum() / g.steps_in.sum()),
                        'false_rate': float(g.det_out.sum() / g.steps_out.sum()), 'ep_false': float((g.det_out > 0).mean())}
            syn[f'{c}_L{L}_n{n}'] = e
OUT['syn'] = syn

# simple pixel detector on synthetic corruptions: never fires (negative control)
sd = re_main[re_main.rule == 'detect']
OUT['simple_detect_synthetic'] = {'episodes': int(len(sd)), 'det_in_total': int(sd.det_in.sum()), 'det_out_total': int(sd.det_out.sum())}
# pooled false alarm of the OOD detector in the 18+18+18+6 main cells
g_all = re_main[re_main.rule == 'generic']
OUT['ood_false_main'] = {'rate': float(g_all.det_out.sum() / g_all.steps_out.sum()),
                         'min_cell': None, 'max_cell': None}
fr = g_all.groupby(['corr', 'n', 'L']).apply(lambda d: d.det_out.sum() / d.steps_out.sum(), include_groups=False)
OUT['ood_false_main'].update({'min_cell': float(fr.min()), 'max_cell': float(fr.max())})

# ============================================================================ 5. unseen corruptions + clean cost (P2b)
un = {}
pil_clean = pil2[(pil2['corr'] == 'black') & (pil2.L == 0)].sort_values('env_seed')
pc = vec(pil_clean)
un['pilot_clean'] = cell(pc)
for c in ['blur5', 'blur15', 'blur40', 'dim0.5', 'dim0.25', 'dim0.1']:
    d = pil2[pil2['corr'] == c].sort_values('env_seed')
    un[f'pilot_{c}'] = {**cell(vec(d)), 'R': float(vec(d).mean() / pc.mean())}
clean50 = vec(tcc[tcc.n == 50])
un['clean_n50'] = cell(clean50)
for c in ['blur40', 'dim0.1']:
    sel = unseen[unseen['corr'] == c]
    fx = vec(sel[sel.rule == 'fixed']); hd = vec(sel[sel.rule == 'oracle']); gn = vec(sel[sel.rule == 'generic'])
    g = sel[sel.rule == 'generic']
    x1, y1, p1 = mcnemar(hd, fx); x2, y2, p2 = mcnemar(gn, fx); x3, y3, p3 = mcnemar(gn, hd)
    un[c] = {'fixed': cell(fx), 'hold': cell(hd), 'ood': cell(gn),
             'R_fixed': boot_R(clean50, fx), 'R_hold': boot_R(clean50, hd), 'R_ood': boot_R(clean50, gn),
             'rho_hold': boot_rho(clean50, fx, hd), 'rho_ood': boot_rho(clean50, fx, gn),
             'hold_vs_fixed': [x1, y1, p1], 'ood_vs_fixed': [x2, y2, p2], 'ood_vs_hold': [x3, y3, p3],
             'det': {'in_frac': float(g.det_in.sum() / g.steps_in.sum()), 'ep_detected': float((g.det_in > 0).mean()),
                     'false_rate': float(g.det_out.sum() / g.steps_out.sum()), 'ep_false': float((g.det_out > 0).mean()),
                     'latency_mean': float(g.latency.mean())}}
# prereg F2: Holm over {blur, dim}
f2p = [un['blur40']['ood_vs_fixed'][2], un['dim0.1']['ood_vs_fixed'][2]]
f2h = holm(f2p)
un['F2'] = {'blur': {'p': f2p[0], 'p_holm': f2h[0]}, 'dim': {'p': f2p[1], 'p_holm': f2h[1]}}
# clean cost of the OOD detector
cc = {}
for n in [25, 50, 100]:
    fx = vec(tcc[tcc.n == n]); gn_d = gcl[gcl.n == n].sort_values('env_seed'); gn = vec(gn_d)
    x, y, p = mcnemar(gn, fx)
    cc[str(n)] = {'fixed': cell(fx), 'ood': cell(gn), 'x': x, 'y': y, 'p': p,
                  'delta': float(gn.mean() - fx.mean()),
                  'false_rate': float(gn_d.det_out.sum() / gn_d.steps_out.sum()), 'ep_false': float((gn_d.det_out > 0).mean())}
un['clean_cost'] = cc
un['false_range'] = [min([un['blur40']['det']['false_rate'], un['dim0.1']['det']['false_rate']] + [v['false_rate'] for v in cc.values()]),
                     max([un['blur40']['det']['false_rate'], un['dim0.1']['det']['false_rate']] + [v['false_rate'] for v in cc.values()])]
OUT['unseen'] = un

# ============================================================================ 6. Insertion
I = {}
for L in [50, 100]:
    for mode in ['black', 'freeze']:
        for n in [25, 50, 100]:
            for lvl, tag in [(4, 'r4'), (3, 'r3')]:
                clean = succ_r(INSF(mode, 0, n), lvl); nf = succ_r(INSF(mode, L, n), lvl); of = succ_r(INS('fixed_old', mode, L, n), lvl)
                e = {'clean': cell(clean), 'new_fixed': cell(nf), 'old_fixed': cell(of),
                     'flips': int((nf != of).sum())}
                if n in (25, 50):
                    hd = succ_r(INS('oracle', mode, L, n), lvl); dt = succ_r(INS('detect', mode, L, n), lvl)
                    xh, yh, ph = mcnemar(hd, nf); xd, yd, pd_ = mcnemar(dt, nf)
                    e.update({'hold': cell(hd), 'detect': cell(dt),
                              'hold_vs_fixed': [xh, yh, ph], 'detect_vs_fixed': [xd, yd, pd_],
                              'rho_hold': boot_rho(clean, nf, hd), 'rho_detect': boot_rho(clean, nf, dt),
                              'R_fixed': boot_R(clean, nf), 'R_hold': boot_R(clean, hd)})
                I[f'{tag}_{mode}_L{L}_n{n}'] = e
# F1 (pre-registered, Holm over 3)
f1 = [('oracle_n25', 'r4_black_L50_n25', 'hold_vs_fixed'), ('detect_n25', 'r4_black_L50_n25', 'detect_vs_fixed'),
      ('oracle_n50', 'r4_black_L50_n50', 'hold_vs_fixed')]
f1p = [I[k][t][2] for _, k, t in f1]
f1h = holm(f1p)
I['F1'] = {name: {'p': p_, 'p_holm': h_, 'x': I[k][t][0], 'y': I[k][t][1]} for (name, k, t), p_, h_ in zip(f1, f1p, f1h)}
# max marginal difference new vs old fixed (reward=4), 12 cells
I['max_marg_diff'] = max(abs(I[f'r4_{m}_L{L}_n{n}']['new_fixed']['rate'] - I[f'r4_{m}_L{L}_n{n}']['old_fixed']['rate'])
                         for m in ['black', 'freeze'] for L in [50, 100] for n in [25, 50, 100])
I['max_flips'] = max(I[f'r4_{m}_L{L}_n{n}']['flips'] for m in ['black', 'freeze'] for L in [50, 100] for n in [25, 50, 100])
# freeze L=50: |delta| of hold/detect vs new fixed (reward=4)
I['freeze50_max_abs_delta'] = max(abs(I[f'r4_freeze_L50_n{n}'][k]['rate'] - I[f'r4_freeze_L50_n{n}']['new_fixed']['rate'])
                                  for n in (25, 50) for k in ('hold', 'detect'))
# natural experiment (new fixed, n=100, L=50)
d = INSF('black', 50, 100); ks = np.array([ksched(100, 50, int(t)) for t in d.t0]); s = succ_r(d, 4)
dfz = INSF('freeze', 50, 100); sfz = succ_r(dfz, 4)
I['nat_black'] = {'k0': [int(s[ks == 0].sum()), int((ks == 0).sum())], 'k1': [int(s[ks >= 1].sum()), int((ks >= 1).sum())]}
I['nat_freeze'] = {'k0': [int(sfz[ks == 0].sum()), int((ks == 0).sum())], 'k1': [int(sfz[ks >= 1].sum()), int((ks >= 1).sum())]}
I['nat_black']['fisher_p'] = float(stats.fisher_exact([[I['nat_black']['k0'][0], I['nat_black']['k0'][1] - I['nat_black']['k0'][0]],
                                                       [I['nat_black']['k1'][0], I['nat_black']['k1'][1] - I['nat_black']['k1'][0]]])[1])
# reward distribution of clean n=100 (success definition)
c100 = INSF('black', 0, 100)
I['clean_n100_reward_counts'] = {str(r): int((c100.max_reward == r).sum()) for r in range(5)}
OUT['ins'] = I

# ============================================================================ 7. literature baselines
B = {}
cells = [('black', 25), ('black', 50), ('black', 100), ('freeze', 50), ('freeze', 100)]
for mode, L in cells:
    for m in ['fixed_n50', 'fixed_n100', 'oracle_n50', 'meanTE_h1', 'medianTE_h1', 'medianTE_h10', 'care_tau0.3']:
        d = BL(m, mode, L)
        B[f'{m}_{mode}_L{L}'] = {**cell(vec(d)), 'queries': float(d.replans_total.mean())}
for m in ['care_tau0.18', 'care_tau0.5']:
    d = BL(m, 'black', 50); B[f'{m}_black_L50'] = {**cell(vec(d)), 'queries': float(d.replans_total.mean())}
    d = BL(m, 'freeze', 50); B[f'{m}_freeze_L50'] = {**cell(vec(d)), 'queries': float(d.replans_total.mean())}
for m in ['fixed_n50', 'fixed_n100', 'meanTE_h1', 'medianTE_h1', 'medianTE_h10', 'care_tau0.18', 'care_tau0.3', 'care_tau0.5']:
    B[f'clean_{m}'] = cell(vec(BL(m, 'clean', 0)))
for L in [50, 100]:
    for m in ['fixed_n50', 'fixed_n100', 'meanTE_h1', 'medianTE_h1', 'medianTE_h10', 'care_tau0.3', 'oracle_n50']:
        B[f'{m}_noise0.6_L{L}'] = {**cell(vec(bl[(bl.method == m) & (bl['corr'] == 'noise0.6') & (bl.L == L)].sort_values('env_seed'))),
                                   'queries': float(bl[(bl.method == m) & (bl['corr'] == 'noise0.6') & (bl.L == L)].replans_total.mean())}
# rho of Median TE and CARE against fixed n=50 and n=100, with the clean reference of the *fixed* rule
for mode, L in cells:
    c50 = vec(BL('fixed_n50', 'clean', 0)); c100_ = vec(BL('fixed_n100', 'clean', 0))
    for m in ['medianTE_h1', 'medianTE_h10', 'care_tau0.3']:
        s = vec(BL(m, mode, L))
        B[f'rho_{m}_{mode}_L{L}'] = {'vs50': boot_rho(c50, vec(BL('fixed_n50', mode, L)), s),
                                      'vs100': boot_rho(c100_, vec(BL('fixed_n100', mode, L)), s)}
# pairwise tests (same seeds, same loop)
pw = {}
for mode, L in cells:
    for a_, b_ in [('medianTE_h1', 'fixed_n50'), ('medianTE_h1', 'fixed_n100'), ('medianTE_h1', 'oracle_n50'), ('medianTE_h10', 'medianTE_h1'),
                   ('care_tau0.3', 'fixed_n50'), ('care_tau0.3', 'oracle_n50'), ('oracle_n50', 'fixed_n50'), ('oracle_n50', 'fixed_n25'), ('medianTE_h1', 'meanTE_h1')]:
        x, y, p = mcnemar(vec(BL(a_, mode, L)), vec(BL(b_, mode, L)))
        pw[f'{a_}|{b_}|{mode}|L{L}'] = [x, y, p]
B['pairs'] = pw
B['oracle25_fixed25_freeze100'] = {'oracle': cell(vec(BL('oracle_n25', 'freeze', 100))), 'fixed': cell(vec(BL('fixed_n25', 'freeze', 100)))}
B['rho_oracle25_freeze100'] = boot_rho(vec(FG(25, 0, 'black')), vec(BL('fixed_n25', 'freeze', 100)), vec(BL('oracle_n25', 'freeze', 100)))
# CARE executed horizon: clean vs sustained blackout (mean H) is not stored per episode in baselines.csv for black; use realistic? skip.
OUT['base'] = B

# P2b confirmatory Median TE (seeds 150-249)
T = {}
for cellname, (corr, L) in {'freeze100': ('freeze', 100), 'black25': ('black', 25)}.items():
    sub = te2[(te2['corr'] == corr) & (te2.L == L)]
    arr = {}
    for nm, d in [('te1', sub[(sub.rule == 'median_te') & (sub.h == 1)]), ('te10', sub[(sub.rule == 'median_te') & (sub.h == 10)]),
                  ('fix50', sub[(sub.rule == 'fixed') & (sub.n == 50)]), ('fix100', sub[(sub.rule == 'fixed') & (sub.n == 100)]),
                  ('hold50', sub[sub.rule == 'oracle'])]:
        assert len(d) == 100, (cellname, nm)
        d = d.sort_values('env_seed')
        arr[nm] = (vec(d), float(d.replans_total.mean()))
        T[f'{cellname}_{nm}'] = {**cell(vec(d)), 'queries': float(d.replans_total.mean())}
    for nm in ['te10', 'fix50', 'fix100', 'hold50']:
        x, y, p = mcnemar(arr['te1'][0], arr[nm][0])
        T[f'{cellname}_te1_vs_{nm}'] = [x, y, p]
x, y, p = mcnemar(arr['hold50'][0], arr['fix50'][0]) if False else (0, 0, 1)
for cellname in ['freeze100']:
    sub = te2[(te2['corr'] == 'freeze') & (te2.L == 100)]
    h50 = vec(sub[sub.rule == 'oracle'].sort_values('env_seed')); f50_ = vec(sub[(sub.rule == 'fixed') & (sub.n == 50)].sort_values('env_seed'))
    T['freeze100_hold_vs_fix50'] = list(mcnemar(h50, f50_))
f3p = [T['freeze100_te1_vs_hold50'][2], mcnemar(vec(te2[(te2['corr'] == 'black') & (te2.L == 25) & (te2.h == 1)].sort_values('env_seed')),
                                               vec(te2[(te2['corr'] == 'black') & (te2.L == 25) & (te2.rule == 'oracle')].sort_values('env_seed')))[2],
       T['freeze100_te1_vs_fix50'][2]]
f3h = holm(f3p)
T['F3'] = {'p': f3p, 'p_holm': f3h}
OUT['te_confirm'] = T

# ============================================================================ 8. reproducibility / noise
Rp = {}
# same-path rerun: pilot.csv vs full_grid.csv (same code, same seeds)
fl = {}
tot = 0; flips = 0; mid_tot = 0; mid_flips = 0
for cfg, g in pilot0.groupby('config'):
    f_ = fg[(fg.config == cfg) & (fg.env_seed.isin(g.env_seed))]
    m_ = g.merge(f_, on='env_seed', suffixes=('_p', '_f'))
    nflip = int((m_.success_p != m_.success_f).sum())
    mean_ = (m_.success_p.mean() + m_.success_f.mean()) / 2
    tot += len(m_); flips += nflip
    if 0.1 <= mean_ <= 0.9:
        mid_tot += len(m_); mid_flips += nflip
Rp['same_path'] = {'flips': flips, 'n': tot, 'mid_flips': mid_flips, 'mid_n': mid_tot}
# cross-path: float64 loop vs run_grid.py (float32)
reg_dir = os.path.join(RES, 'parts')
cross = {}
for mode in ['black', 'freeze']:
    r_ = pd.read_csv(os.path.join(reg_dir, f'p2p3_regress_{mode}.csv'))
    for n in [25, 50]:
        a = r_[r_.config == f'fixed_n{n}_L50_{mode}'].sort_values('env_seed')
        b = FG(n, 50, mode)
        b = b[b.env_seed.isin(a.env_seed)]
        cross[f'f64_{mode}_n{n}'] = {'flips': int((vec(a) != vec(b)).sum()), 'f64': int(vec(a).sum()), 'grid': int(vec(b).sum())}
rg = pd.read_csv(os.path.join(reg_dir, 'p2p3_regress_rungrid_freeze.csv')).sort_values('env_seed')
cross['rungrid_rerun_freeze_n25'] = {'flips': int((vec(rg) != vec(FG(25, 50, 'freeze'))).sum())}
Rp['cross'] = cross
dt_ = {}
for mode in ['black', 'freeze']:
    a = dty[dty['corr'] == mode].sort_values('env_seed')
    b = FG(25, 50, mode)
    f64 = pd.read_csv(os.path.join(reg_dir, f'p2p3_regress_{mode}.csv')); f64 = f64[f64.config == f'fixed_n25_L50_{mode}'].sort_values('env_seed')
    dt_[mode] = {'f32_vs_grid': int((vec(a) != vec(b)).sum()), 'f64_vs_grid': int((vec(f64) != vec(b)).sum()),
                 'f32_vs_f64': int((vec(a) != vec(f64)).sum()), 'f32': int(vec(a).sum()), 'f64': int(vec(f64).sum()), 'grid': int(vec(b).sum())}
dt_['f32_total'] = dt_['black']['f32_vs_grid'] + dt_['freeze']['f32_vs_grid']
dt_['f64_total'] = dt_['black']['f64_vs_grid'] + dt_['freeze']['f64_vs_grid']
Rp['dtype'] = dt_
# TC clean reference, same loop vs run_grid
cl = {}
for n in [25, 50, 100]:
    a = tcc[tcc.n == n].sort_values('env_seed'); b = FG(n, 0, 'black'); b = b[b.env_seed < 90]
    cl[str(n)] = {'loop': cell(vec(a)), 'grid': cell(vec(b)), 'flips': int((vec(a) != vec(b)).sum()),
                  'delta': float(vec(a).mean() - vec(b).mean())}
Rp['tc_clean'] = cl
Rp['tc_clean_max_abs_delta'] = max(abs(v['delta']) for v in cl.values())
# Insertion: new fixed vs old fixed flips
Rp['ins_flips'] = {k: v['flips'] for k, v in I.items() if k.startswith('r4_') and 'flips' in v}
OUT['repro'] = Rp

# ============================================================================ 9. proprio (negative result)
P = {}
for L, mode in [(25, 'black'), (50, 'black'), (100, 'black'), (25, 'freeze'), (50, 'freeze'), (100, 'freeze')]:
    for m in [1, 10]:
        d = pro[(pro.tag == 'main') & (pro.config == f'proprio_m{m}_L{L}_{mode}')].sort_values('env_seed')
        assert len(d) == 100
        s = vec(d)
        e = {**cell(s), 'triggers_in_blackout': float(d.triggers_in_blackout.mean())}
        for fn in (50, 100):
            fx = vec(FG(fn, L, mode)); x, y, p = mcnemar(s, fx)
            e[f'vs{fn}'] = [x, y, p, int(fx.sum())]
        P[f'm{m}_{mode}_L{L}'] = e
d = pro[(pro.tag == 'main') & (pro.config == 'proprio_m1_L0_black')]
P['tau'] = float(d.tau.iloc[0])
OUT['proprio'] = P

# ============================================================================ 10. diffusion policy gates (counts only)
dp = {}
for tag, f in [('v1', 'dp_gate_dp_v1.csv'), ('v2', 'dp_gate_dp_v2.csv'), ('v3', 'dp_gate_dp_v3.csv')]:
    d = rd(f)
    dp[tag] = {'n': int(len(d)), 'succ': int(d.success.sum()), 'max_reward_ge2': int((d.max_reward >= 2).sum()), 'max_reward_max': float(d.max_reward.max())}
    dp[tag]['wilson'] = wilson(dp[tag]['succ'], dp[tag]['n'])
OUT['dp'] = dp

# ============================================================================ primary contrasts (<= 6) with Holm
prim = []
def add(name, desc, x, y, p, prereg):
    prim.append({'name': name, 'desc': desc, 'x': x, 'y': y, 'p': p, 'prereg': prereg})
t = tc['black_L50_n50']['hold_vs_fixed']
add('P1', 'TC, black, L=50, n=50: hold vs fixed (seeds 0-99)', t['x'], t['y'], t['p'], 'no')
add('P2', 'TC, freeze, L=100, n=50: hold vs fixed (seeds 150-249)', *T['freeze100_hold_vs_fix50'], 'no')
e = I['r4_black_L50_n50']['hold_vs_fixed']
add('P3', 'Insertion, black, L=50, n=50: hold vs new fixed (reward=4)', e[0], e[1], e[2], 'yes, F1')
e = un['blur40']['ood_vs_fixed']
add('P4', 'TC, unseen blur: OOD detector vs fixed (n=50, L=50)', e[0], e[1], e[2], 'yes, F2')
e = un['dim0.1']['ood_vs_fixed']
add('P5', 'TC, unseen dim: OOD detector vs fixed (n=50, L=50)', e[0], e[1], e[2], 'yes, F2')
t = T['freeze100_te1_vs_hold50']
add('P6', 'TC, freeze, L=100: Median TE h=1 vs hold n=50 (seeds 150-249)', t[0], t[1], t[2], 'yes, F3')
hp = holm([q['p'] for q in prim])
for q, h in zip(prim, hp):
    q['p_holm'] = h
OUT['primary'] = prim

# ============================================================================ pilot strength table (synthetic) - read pilot rows explicitly
pl = re_all[(re_all.stage == 'pilot')]
ps = {}
for (c, rule, n, L), g in pl.groupby(['corr', 'rule', 'n', 'L']):
    ps[f'{c}|{rule}|{n}|{L}'] = cell(vec(g))
OUT['syn_pilot'] = ps
OUT['syn_pilot_clean'] = cell(vec(pl[(pl['corr'] == 'black') & (pl.L == 0)]))

# ----------------------------------------------------------------------------- write JSON
def clean_json(o):
    if isinstance(o, dict):
        return {k: clean_json(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean_json(v) for v in o]
    if isinstance(o, (np.floating, float)):
        return None if (isinstance(o, float) and math.isnan(o)) else float(o)
    if isinstance(o, np.integer):
        return int(o)
    return o

with open(os.path.join(RES, 'arxiv_numbers.json'), 'w') as f:
    json.dump(clean_json(OUT), f, indent=1)

# ============================================================================ tables
def cs(c):
    return f"{c['k']}/{c['n']}"


def rho_s(r):
    if r is None:
        return 'n/r'
    star = '$^*$' if r[3] < 0.95 else ''
    return f'{r[0]:.2f} [{r[1]:.2f}, {r[2]:.2f}]{star}'


def tex_table(path, header, rows, colspec, caption_note=None):
    lines = ['\\begin{tabular}{' + colspec + '}', '\\toprule', ' & '.join(header) + ' \\\\', '\\midrule']
    for r in rows:
        if r == 'MID':
            lines.append('\\midrule')
        else:
            lines.append(' & '.join(r) + ' \\\\')
    lines += ['\\bottomrule', '\\end{tabular}']
    open(path, 'w').write('\n'.join(lines) + '\n')


# Tab 1: fixed-n grid
rows = []
hdr = ['$n$'] + [f'$L{{=}}{L}$' for L in [0, 10, 25, 50, 100]] + [f'$L{{=}}{L}$' for L in [10, 25, 50, 100]]
for n in [1, 10, 25, 50, 100]:
    r = [str(n)]
    for L in [0, 10, 25, 50, 100]:
        r.append(str(OUT['grid'][f'fixed_n{n}_L{L}_black']['k']))
    for L in [10, 25, 50, 100]:
        r.append(str(OUT['grid'][f'fixed_n{n}_L{L}_freeze']['k']))
    rows.append(r)
r = ['Mean TE']
for L in [0, 10, 25, 50, 100]:
    r.append(str(OUT['grid'][f'meanTE_L{L}_black']['k']))
for L in [10, 25, 50, 100]:
    r.append(str(OUT['grid'][f'meanTE_L{L}_freeze']['k']))
rows.append(r)
tex_table(os.path.join(TAB, 'tab_grid.tex'), hdr, rows, 'r' + 'r' * 5 + 'r' * 4)

# Tab 2: TC intervention
rows = []
for mode in ['black', 'freeze']:
    for L in [25, 50, 100]:
        for n in [25, 50]:
            e = tc[f'{mode}_L{L}_n{n}']
            rows.append([mode, str(L), str(n), cs(e['clean']), cs(e['fixed']), cs(e['hold']), rho_s(e['rho_hold']),
                         f"{e['hold_vs_fixed']['x']}/{e['hold_vs_fixed']['y']}", cs(e['detect'])])
    if mode == 'black':
        rows.append('MID')
tex_table(os.path.join(TAB, 'tab_tc_hold.tex'),
          ['mode', '$L$', '$n$', 'clean', '\\fixed', '\\hold', '$\\rho$ (hold)', 'hold/fixed', '\\detect'], rows, 'lrrrrrrrr')

# Tab 3: synthetic + unseen at n=50, L=50
rows = []
names = {'noise0.6': 'noise $\\sigma{=}0.6$', 'occlude0.5center': 'occlusion 0.5', 'flicker0.6': 'flicker $p{=}0.6$', 'delay50': 'delay $d{=}50$'}
for c in ['noise0.6', 'occlude0.5center', 'flicker0.6', 'delay50']:
    e = syn[f'{c}_L50_n50']
    rows.append([names[c], 'probe set' if c != 'delay50' else 'not in probe set', cs(e['clean']), cs(e['fixed']), cs(e['hold']), cs(e['ood']), rho_s(e['rho_hold']), rho_s(e['rho_ood']),
                 f"{e['det']['in_frac']:.2f}"])
for c, nm in [('blur40', 'blur $\\sigma{=}40$'), ('dim0.1', 'dim $g{=}0.1$')]:
    e = un[c]
    rows.append([nm, 'unseen, pre-reg.', cs(un['clean_n50']), cs(e['fixed']), cs(e['hold']), cs(e['ood']), rho_s(e['rho_hold']), rho_s(e['rho_ood']),
                 f"{e['det']['in_frac']:.2f}"])
tex_table(os.path.join(TAB, 'tab_syn.tex'),
          ['corruption', 'status', 'clean', '\\fixed', '\\hold', '\\ood', '$\\rho$ (hold)', '$\\rho$ (\\ood)', 'flagged'], rows, 'llrrrrrrr')

# Tab 4: Insertion
rows = []
for lvl_tag, lvl_name in [('r4', 'reward${=}4$ (primary)'), ('r3', 'reward${\\ge}3$ (post hoc)')]:
    rows.append([f'\\multicolumn{{9}}{{l}}{{\\emph{{{lvl_name}}}}} \\\\'.replace(' \\\\', '')])
    for mode in ['black', 'freeze']:
        for L in [50, 100]:
            for n in [25, 50]:
                e = I[f'{lvl_tag}_{mode}_L{L}_n{n}']
                rows.append([mode, str(L), str(n), cs(e['clean']), cs(e['new_fixed']), cs(e['hold']), cs(e['detect']),
                             rho_s(e['rho_hold']), f"{e['hold_vs_fixed'][0]}/{e['hold_vs_fixed'][1]}"])
    rows.append('MID')
rows = rows[:-1]
# multicolumn rows need no extra '&'
lines = ['\\begin{tabular}{lrrrrrrrr}', '\\toprule',
         'mode & $L$ & $n$ & clean & \\fixed & \\hold & \\detect & $\\rho$ (hold) & hold/fixed \\\\', '\\midrule']
for r in rows:
    if r == 'MID':
        lines.append('\\midrule')
    elif len(r) == 1:
        lines.append(r[0] + ' \\\\')
    else:
        lines.append(' & '.join(r) + ' \\\\')
lines += ['\\bottomrule', '\\end{tabular}']
open(os.path.join(TAB, 'tab_ins.tex'), 'w').write('\n'.join(lines) + '\n')

# Tab 5: baselines (main text)
meths = [('fixed_n50', '\\fixed(50)'), ('fixed_n100', '\\fixed(100)'), ('oracle_n50', '\\hold(50)'), ('meanTE_h1', 'Mean TE, $h{=}1$'),
         ('medianTE_h1', 'Median TE, $h{=}1$'), ('medianTE_h10', 'Median TE, $h{=}10$'), ('care_tau0.3', 'CARE-style, $\\tau{=}0.3$')]
cols = [('black', 25), ('black', 50), ('black', 100), ('freeze', 50), ('freeze', 100)]
rows = []
for m, nm in meths:
    r = [nm]
    cl_key = {'oracle_n50': 'fixed_n50'}.get(m, m)
    r.append(str(B[f'clean_{cl_key}']['k']))
    for mode, L in cols:
        r.append(str(B[f'{m}_{mode}_L{L}']['k']))
    r.append(f"{B[f'{m}_freeze_L100']['queries']:.1f}")
    rows.append(r)
tex_table(os.path.join(TAB, 'tab_base.tex'),
          ['rule', 'clean', 'black 25', 'black 50', 'black 100', 'freeze 50', 'freeze 100', 'calls/ep'], rows, 'lrrrrrrr')

# Tab 6: primary contrasts
def mathify(t):
    """put parameter assignments of the contrast descriptions in math mode (L=50 -> $L{=}50$) and use an en dash in seed ranges"""
    t = re.sub(r'\b([Lnh])=(\d+)', r'$\1{=}\2$', t)
    t = t.replace('reward=4', 'reward${=}4$')
    return re.sub(r'(\d+)-(\d+)\)', r'\1--\2)', t)


rows = []
for q in prim:
    rows.append([q['name'], mathify(q['desc'].replace('->', '$\\to$')), f"{q['x']}/{q['y']}", fmt_p(q['p']), fmt_p(q['p_holm']), q['prereg']])
tex_table(os.path.join(TAB, 'tab_primary.tex'), ['id', 'contrast', 'x/y', '$p$', '$p_{\\mathrm{Holm}}$', 'pre-registered?'], rows, 'lp{6.4cm}rrrp{3.2cm}')

# Appendix tables ------------------------------------------------------------------------------
# A1 strength pilot (synthetic)
rows = []
pn = [('noise0.1', 'noise $\\sigma{=}0.1$'), ('noise0.3', 'noise $\\sigma{=}0.3$'), ('noise0.6', 'noise $\\sigma{=}0.6$'),
      ('occlude0.25center', 'occlusion 0.25 center'), ('occlude0.5center', 'occlusion 0.5 center'), ('occlude0.75center', 'occlusion 0.75 center'),
      ('occlude0.25rand', 'occlusion 0.25 random'), ('occlude0.5rand', 'occlusion 0.5 random'), ('occlude0.75rand', 'occlusion 0.75 random'),
      ('delay5', 'delay $d{=}5$'), ('delay25', 'delay $d{=}25$'), ('delay50', 'delay $d{=}50$'),
      ('flicker0.3', 'flicker $p{=}0.3$'), ('flicker0.6', 'flicker $p{=}0.6$'), ('flicker1', 'flicker $p{=}1.0$')]
pc_ = OUT['syn_pilot_clean']
sel_syn = {'noise0.6', 'occlude0.5center', 'delay50', 'flicker0.6'}
for c, nm in pn:
    e = ps[f'{c}|fixed|50|50']
    rows.append([nm, cs(e), f"{e['rate'] / pc_['rate']:.2f}", 'selected' if c in sel_syn else ''])
for c, nm in [('blur5', 'blur $\\sigma{=}5$'), ('blur15', 'blur $\\sigma{=}15$'), ('blur40', 'blur $\\sigma{=}40$'), ('dim0.5', 'dim $g{=}0.5$'),
              ('dim0.25', 'dim $g{=}0.25$'), ('dim0.1', 'dim $g{=}0.1$')]:
    e = un[f'pilot_{c}']
    rows.append([nm + ' (unseen)', cs(e), f"{e['R']:.2f}", 'selected' if c in ('blur40', 'dim0.1') else ''])
tex_table(os.path.join(TAB, 'tab_pilot.tex'), ['corruption and strength', '\\fixed(50) success', '$R$', ''], rows, 'lrrl')

# A2 dtype ablation
rows = []
for mode in ['black', 'freeze']:
    e = Rp['dtype'][mode]
    rows.append([mode, str(e['grid']) + '/100', str(e['f64']) + '/100', str(e['f32']) + '/100', str(e['f64_vs_grid']), str(e['f32_vs_grid']), str(e['f32_vs_f64'])])
tex_table(os.path.join(TAB, 'tab_dtype.tex'),
          ['mode', 'run\\_grid (f32)', 'new loop f64', 'new loop f32', 'f64 vs grid', 'f32 vs grid', 'f32 vs f64'], rows, 'lrrrrrr')

# A3 detector metrics per corruption (n=50, L=50)
rows = []
for c, nm in [('noise0.6', 'noise'), ('occlude0.5center', 'occlusion'), ('flicker0.6', 'flicker'), ('delay50', 'delay')]:
    d_ = syn[f'{c}_L50_n50']['det']
    rows.append([nm, f"{d_['ep_detected']:.2f}", f"{d_['in_frac']:.3f}", f"{d_['false_rate']:.4f}", f"{d_['ep_false']:.2f}"])
for c, nm in [('blur40', 'blur (unseen)'), ('dim0.1', 'dim (unseen)')]:
    d_ = un[c]['det']
    rows.append([nm, f"{d_['ep_detected']:.2f}", f"{d_['in_frac']:.3f}", f"{d_['false_rate']:.4f}", f"{d_['ep_false']:.2f}"])
tex_table(os.path.join(TAB, 'tab_detector.tex'),
          ['corruption', 'episodes detected', 'steps flagged in window', 'false-alarm rate (steps)', 'episodes with $\\ge$1 false alarm'], rows, 'lrrrr')

# A4 clean cost
rows = []
for n in ['25', '50', '100']:
    e = cc[n]
    rows.append([n, cs(e['fixed']), cs(e['ood']), f"{e['delta']:+.3f}", f"{e['x']}/{e['y']}", fmt_p(e['p']), f"{e['false_rate']:.4f}"])
tex_table(os.path.join(TAB, 'tab_cleancost.tex'), ['$n$', '\\fixed', '\\ood', '$\\Delta$', 'ood/fixed', '$p$', 'false-alarm rate'], rows, 'lrrrrrr')

# A5 CARE sensitivity (black / freeze L=50)
rows = []
for tau, m in [('0.18', 'care_tau0.18'), ('0.3', 'care_tau0.3'), ('0.5', 'care_tau0.5')]:
    rows.append([tau, str(B[f'clean_{m}']['k']), str(B[f'{m}_black_L50']['k']), str(B[f'{m}_freeze_L50']['k']),
                 f"{B[f'{m}_black_L50']['k'] / B[f'clean_{m}']['k']:.2f}", f"{B[f'{m}_freeze_L50']['k'] / B[f'clean_{m}']['k']:.2f}"])
rows.append(['\\fixed(50)', str(B['clean_fixed_n50']['k']), str(B['fixed_n50_black_L50']['k']), str(B['fixed_n50_freeze_L50']['k']),
             f"{B['fixed_n50_black_L50']['k'] / B['clean_fixed_n50']['k']:.2f}", f"{B['fixed_n50_freeze_L50']['k'] / B['clean_fixed_n50']['k']:.2f}"])
tex_table(os.path.join(TAB, 'tab_care.tex'), ['$\\tau$', 'clean', 'black $L{=}50$', 'freeze $L{=}50$', '$R$ black', '$R$ freeze'], rows, 'lrrrrr')

# A6 Median TE confirmatory
rows = []
for cn, nm in [('freeze100', 'freeze $L{=}100$'), ('black25', 'black $L{=}25$')]:
    for k_, lab in [('te1', 'Median TE $h{=}1$'), ('te10', 'Median TE $h{=}10$'), ('fix50', '\\fixed(50)'), ('fix100', '\\fixed(100)'), ('hold50', '\\hold(50)')]:
        e = T[f'{cn}_{k_}']
        rel = '' if k_ == 'te1' else f"{T[f'{cn}_te1_vs_{k_}'][0]}/{T[f'{cn}_te1_vs_{k_}'][1]}"
        pp = '' if k_ == 'te1' else fmt_p(T[f'{cn}_te1_vs_{k_}'][2])
        rows.append([nm, lab, cs(e), f"{e['queries']:.1f}", rel, pp])
    rows.append('MID')
rows = rows[:-1]
tex_table(os.path.join(TAB, 'tab_te_confirm.tex'), ['cell', 'rule', 'success', 'calls/ep', 'TE/other', '$p$ (uncorrected)'], rows, 'llrrrr')

# A7 proprio
rows = []
for mode in ['black', 'freeze']:
    for L in [25, 50, 100]:
        e1 = P[f'm1_{mode}_L{L}']; e10 = P[f'm10_{mode}_L{L}']
        rows.append([mode, str(L), str(P[f'm1_{mode}_L{L}']['vs50'][3]), str(P[f'm1_{mode}_L{L}']['vs100'][3]), cs(e1), f"{e1['vs50'][0]}/{e1['vs50'][1]}",
                     fmt_p(e1['vs50'][2]), cs(e10), f"{e10['vs50'][0]}/{e10['vs50'][1]}", fmt_p(e10['vs50'][2])])
tex_table(os.path.join(TAB, 'tab_proprio.tex'),
          ['mode', '$L$', '\\fixed(50)', '\\fixed(100)', 'proprio $m{=}1$', 'vs 50', '$p$', 'proprio $m{=}10$', 'vs 50', '$p$'], rows, 'lrrrrrrrrr')

# A8 Insertion: old vs new fixed
rows = []
for mode in ['black', 'freeze']:
    for L in [50, 100]:
        for n in [25, 50, 100]:
            e = I[f'r4_{mode}_L{L}_n{n}']
            rows.append([mode, str(L), str(n), cs(e['old_fixed']), cs(e['new_fixed']), str(e['flips'])])
tex_table(os.path.join(TAB, 'tab_ins_fixed.tex'), ['mode', '$L$', '$n$', 'old \\fixed\\ (float32)', 'new \\fixed\\ (same loop)', 'flips'], rows, 'lrrrrr')

# A9 Primary F1-F3 as registered
rows = []
rows.append(['F1', 'oracle $n{=}25$ vs fixed (Ins.\\ black, $L{=}50$)', f"{I['F1']['oracle_n25']['x']}/{I['F1']['oracle_n25']['y']}", fmt_p(I['F1']['oracle_n25']['p']), fmt_p(I['F1']['oracle_n25']['p_holm'])])
rows.append(['F1', 'detect $n{=}25$ vs fixed', f"{I['F1']['detect_n25']['x']}/{I['F1']['detect_n25']['y']}", fmt_p(I['F1']['detect_n25']['p']), fmt_p(I['F1']['detect_n25']['p_holm'])])
rows.append(['F1', 'oracle $n{=}50$ vs fixed', f"{I['F1']['oracle_n50']['x']}/{I['F1']['oracle_n50']['y']}", fmt_p(I['F1']['oracle_n50']['p']), fmt_p(I['F1']['oracle_n50']['p_holm'])])
rows.append(['F2', 'OOD vs fixed, blur', f"{un['blur40']['ood_vs_fixed'][0]}/{un['blur40']['ood_vs_fixed'][1]}", fmt_p(f2p[0]), fmt_p(f2h[0])])
rows.append(['F2', 'OOD vs fixed, dim', f"{un['dim0.1']['ood_vs_fixed'][0]}/{un['dim0.1']['ood_vs_fixed'][1]}", fmt_p(f2p[1]), fmt_p(f2h[1])])
for nm, i_ in [('TE $h{=}1$ vs hold, freeze $L{=}100$', 0), ('TE $h{=}1$ vs hold, black $L{=}25$', 1), ('TE $h{=}1$ vs fixed(50), freeze $L{=}100$', 2)]:
    xy = [T['freeze100_te1_vs_hold50'], mcnemar(vec(te2[(te2['corr'] == 'black') & (te2.L == 25) & (te2.h == 1)].sort_values('env_seed')),
                                                 vec(te2[(te2['corr'] == 'black') & (te2.L == 25) & (te2.rule == 'oracle')].sort_values('env_seed'))),
          T['freeze100_te1_vs_fix50']][i_]
    rows.append(['F3', nm, f"{xy[0]}/{xy[1]}", fmt_p(f3p[i_]), fmt_p(f3h[i_])])
tex_table(os.path.join(TAB, 'tab_families.tex'), ['family', 'contrast', 'x/y', '$p$', '$p_{\\mathrm{Holm}}$ (within family)'], rows, 'llrrr')

# ============================================================================ figures
OI = {'black': '#000000', 'blue': '#0072B2', 'orange': '#E69F00', 'green': '#009E73', 'verm': '#D55E00', 'sky': '#56B4E9', 'purple': '#CC79A7', 'yellow': '#F0E442'}
plt.rcParams.update({'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False, 'figure.dpi': 150,
                     'pdf.fonttype': 42, 'axes.grid': True, 'grid.alpha': 0.25})
# Fig 1: success vs L
fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.7), sharey=True)
Ls = [0, 10, 25, 50, 100]
ncol = {10: OI['sky'], 25: OI['green'], 50: OI['blue'], 100: OI['verm']}
for ax, mode in zip(axes, ['black', 'freeze']):
    for n in [10, 25, 50, 100]:
        xs = [L for L in Ls if not (mode == 'freeze' and L == 0)]
        ys = [OUT['grid'][f'fixed_n{n}_L{L}_{mode}']['rate'] for L in xs]
        ax.plot(range(len(xs)) if False else xs, ys, '-o', ms=3.5, color=ncol[n], label=f'fixed $n$={n}')
    for n in [25, 50]:
        xs = [0, 10, 25, 50, 100] if mode == 'black' else [10, 25, 50, 100]
        ys = [OUT['grid']['fixed_n%d_L0_black' % n]['rate']] if mode == 'black' else []
        ys += [OUT['grid'][f'hold_n{n}_L{L}_{mode}']['rate'] for L in [10, 25, 50, 100]]
        ax.plot(xs, ys, '--s', ms=3.5, mfc='white', color=ncol[n], label=f'hold $n$={n}')
    ax.set_title(mode, fontsize=9)
    ax.set_xlabel('blackout length $L$ (steps)')
    ax.set_xticks(Ls)
axes[0].set_ylabel('success rate')
h_, l_ = axes[0].get_legend_handles_labels()
fig.legend(h_, l_, fontsize=7, ncol=6, frameon=False, loc='lower center')
fig.tight_layout(rect=(0, 0.08, 1, 1))
fig.savefig(os.path.join(FIG, 'fig_grid.pdf'))
plt.close(fig)

# Fig 2: mechanism
fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.7))
ax = axes[0]
groups = [(25, 'black'), (25, 'freeze'), (50, 'black'), (50, 'freeze')]
xpos = np.arange(len(groups))
w = 0.38
for j, (key, lab, col) in enumerate([('k0', 'no replan in window ($k$=0)', OI['green']), ('k1', '$\\geq$1 replan in window', OI['verm'])]):
    vals = []; los = []; his = []; txt = []
    for L, mode in groups:
        k_, n_ = mech[f'nat_n100_L{L}_{mode}'][key]
        lo, hi = wilson(k_, n_)
        vals.append(k_ / n_); los.append(k_ / n_ - lo); his.append(hi - k_ / n_); txt.append(f'{k_}/{n_}')
    ax.bar(xpos + (j - 0.5) * w, vals, w, color=col, label=lab, yerr=[los, his], error_kw={'lw': 0.8}, capsize=1.5)
    for xx, v, tt, h_ in zip(xpos + (j - 0.5) * w, vals, txt, his):
        ax.text(xx, v + h_ + 0.015, tt, ha='center', va='bottom', fontsize=6)
ax.set_xticks(xpos)
ax.set_xticklabels([f'$L$={L}\n{m}' for L, m in groups], fontsize=7)
ax.set_ylabel('success rate, fixed $n$=100')
ax.set_ylim(0, 1.32)
ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
ax.legend(fontsize=6.5, frameon=False, loc='upper center', ncol=2, columnspacing=0.8, handlelength=1.2)
ax.set_title('(a) same replan count, different image', fontsize=9)
ax = axes[1]
ns = [10, 25, 50, 100]
for mode, col, off in [('black', OI['black'], -0.04), ('freeze', OI['blue'], 0.04)]:
    sl = reg[f'J_{mode}']['slope']
    ys = [sl[str(n)]['slope'] for n in ns]
    lo = [sl[str(n)]['slope'] - sl[str(n)]['lo'] for n in ns]
    hi = [sl[str(n)]['hi'] - sl[str(n)]['slope'] for n in ns]
    ax.errorbar(np.log2(ns) + off, ys, yerr=[lo, hi], fmt='-o', ms=3.5, color=col, capsize=2, lw=1, label=mode)
ax.axhline(0, color='gray', lw=0.8)
ax.set_xticks(np.log2(ns))
ax.set_xticklabels([str(n) for n in ns])
ax.set_xlabel('execution horizon $n$')
ax.set_ylabel('log-odds change per replan in window')
ax.legend(fontsize=7, frameon=False)
ax.set_title('(b) log-odds change per scheduled replan', fontsize=9)
fig.tight_layout()
fig.savefig(os.path.join(FIG, 'fig_mech.pdf'))
plt.close(fig)

# Fig 3: rho forest
fig, ax = plt.subplots(figsize=(5.4, 2.9))
items = [('noise $\\sigma$=0.6', syn['noise0.6_L50_n50'], 'design'), ('occlusion 0.5', syn['occlude0.5center_L50_n50'], 'design'),
         ('flicker $p$=0.6', syn['flicker0.6_L50_n50'], 'design'), ('delay $d$=50', syn['delay50_L50_n50'], 'other'),
         ('blur $\\sigma$=40', un['blur40'], 'unseen'), ('dim $g$=0.1', un['dim0.1'], 'unseen')]
for i, (nm, e, st) in enumerate(items):
    y = len(items) - 1 - i
    for key, col, mk, off in [('rho_hold', OI['black'], 's', 0.14), ('rho_ood', OI['verm'], 'o', -0.14)]:
        r = e[key]
        ax.errorbar([r[0]], [y + off], xerr=[[r[0] - r[1]], [r[2] - r[0]]], fmt=mk, color=col, ms=4, capsize=2, lw=1,
                    label=('hold (told the window)' if key == 'rho_hold' else 'OOD detector (not told)') if i == 0 else None)
ax.set_yticks(range(len(items)))
ax.set_yticklabels([x[0] + ('' if x[2] == 'design' else ' *') for x in items][::-1])
ax.axvline(0, color='gray', lw=0.8); ax.axvline(1, color='gray', lw=0.8, ls=':')
ax.set_xlabel('recovery $\\rho$ ($n$=50, $L$=50)')
ax.legend(fontsize=7, frameon=False, loc='upper left')
fig.tight_layout()
fig.savefig(os.path.join(FIG, 'fig_corr.pdf'))
plt.close(fig)

print('wrote', os.path.join(RES, 'arxiv_numbers.json'))
