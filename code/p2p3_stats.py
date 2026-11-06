"""Shared statistics for P2/P3 tables (python3, numpy/scipy). Paired design: every series is a 0/1 vector indexed by env seed."""
import math
import numpy as np
from scipy.stats import binomtest

RERUN_NOISE = 0.05  # docs/rulers.md: re-run noise ~1 %; differences < 0.05 are not interpreted
RHO_MIN_DEN = 0.10  # rho = (rule - fixed) / (clean - fixed) is only reported when the denominator >= 0.10


def wilson(k, n, z=1.96):
    p = k / n; d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d; h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def rate_ci(x):
    k, n = int(np.sum(x)), len(x); lo, hi = wilson(k, n)
    return f"{k}/{n} = {k / n:.2f} [{lo:.2f}, {hi:.2f}]"


def mcnemar(a, b):
    """exact McNemar for paired 0/1 vectors: returns (a_only, b_only, p)."""
    a, b = np.asarray(a), np.asarray(b)
    x, y = int(((a == 1) & (b == 0)).sum()), int(((a == 0) & (b == 1)).sum())
    p = 1.0 if x + y == 0 else binomtest(x, x + y, 0.5).pvalue
    return x, y, p


def holm(ps):
    ps = np.asarray(ps, float); o = np.argsort(ps); m = len(ps); adj = np.empty(m); run = 0.0
    for r, i in enumerate(o):
        run = max(run, (m - r) * ps[i]); adj[i] = min(1.0, run)
    return adj


def _boot_idx(n, B, seed):
    return np.random.default_rng(seed).integers(0, n, size=(B, n))


def retention(x, clean, B=4000, seed=0):
    """R = mean(x) / mean(clean); bootstrap over seeds (paired resampling). Returns (R, lo, hi) or (nan,..) if clean mean = 0."""
    x, clean = np.asarray(x, float), np.asarray(clean, float)
    if clean.mean() == 0: return float("nan"), float("nan"), float("nan")
    idx = _boot_idx(len(x), B, seed)
    num, den = x[idx].mean(1), clean[idx].mean(1)
    ok = den > 0
    r = num[ok] / den[ok]
    return x.mean() / clean.mean(), float(np.percentile(r, 2.5)), float(np.percentile(r, 97.5))


def recovery(rule, fixed, clean, B=4000, seed=1):
    """rho = (rule - fixed) / (clean - fixed); None when point denominator < RHO_MIN_DEN. Bootstrap CI over seeds, draws whose
    denominator falls below RHO_MIN_DEN are dropped (their share is returned)."""
    rule, fixed, clean = (np.asarray(v, float) for v in (rule, fixed, clean))
    den0 = clean.mean() - fixed.mean()
    if den0 < RHO_MIN_DEN: return None
    idx = _boot_idx(len(rule), B, seed)
    den = clean[idx].mean(1) - fixed[idx].mean(1)
    ok = den >= RHO_MIN_DEN
    r = (rule[idx].mean(1) - fixed[idx].mean(1))[ok] / den[ok]
    return (rule.mean() - fixed.mean()) / den0, float(np.percentile(r, 2.5)), float(np.percentile(r, 97.5)), 1 - ok.mean()


def fmt_R(x, clean):
    r, lo, hi = retention(x, clean)
    return "n/a" if math.isnan(r) else f"{r:.2f} [{lo:.2f}, {hi:.2f}]"


def fmt_rho(rule, fixed, clean):
    v = recovery(rule, fixed, clean)
    if v is None: return "n/r"
    return f"{v[0]:.2f} [{v[1]:.2f}, {v[2]:.2f}]" + ("*" if v[3] > 0.05 else "")
