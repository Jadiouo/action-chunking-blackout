"""T5 final analysis: reads results/full_grid.csv, oracle.csv, proprio.csv; writes results/analysis.md and
results/figs/final_*.{png,pdf}.  Every number in analysis.md is produced here (f-strings), none typed by hand.

Run from anywhere:  python3 (system python with numpy/pandas/scipy/matplotlib; the venv has no matplotlib) code/analyze_final.py
CPU only.  No statsmodels in the venv: logistic regression (Newton/IRLS, cluster-robust sandwich SE),
Wilson CI, exact McNemar, Holm, Mantel-Haenszel are implemented below with numpy/scipy.
"""
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
FIGS = RES / "figs"
FIGS.mkdir(exist_ok=True)
NOISE = 0.05          # docs/rulers.md: differences below 0.05 are within rerun noise -> not interpretable
NS = ["1", "10", "25", "50", "100", "TE"]
LS = [0, 10, 25, 50, 100]
LB = [10, 25, 50, 100]
MODES = ["black", "freeze"]

# ------------------------------------------------------------------ data
F = pd.read_csv(RES / "full_grid.csv")
O = pd.read_csv(RES / "oracle.csv")
P = pd.read_csv(RES / "proprio.csv")
F["nl"] = np.where(F.te == 1, "TE", F.n.astype(str))
assert len(F) == 5400 and len(O) == 3200 and len(P) == 2600
assert (F.success == (F.max_reward == 4)).all() and (O.success == (O.max_reward == 4)).all()
# L=0 has a single run (mode column says black); treat as "no blackout"
FD = {}
for (nl, L, mode), g in F.groupby(["nl", "L", "mode"]):
    FD[(nl, L, mode)] = g.sort_values("episode").reset_index(drop=True)


def fcell(nl, L, mode):
    return FD[(str(nl), L, "black" if L == 0 else mode)]


OD = {(n, L, m): g.sort_values("episode").reset_index(drop=True) for (n, L, m), g in O.groupby(["n", "L", "mode"])}
PM = P[P.tag == "main"]
PD = {(m, L, mode): g.sort_values("episode").reset_index(drop=True) for (m, L, mode), g in PM.groupby(["m", "L", "mode"])}

# scheduled replans in blackout for fixed n (replan at t = 0, n, 2n, ...): independent of outcome (no early-stop censoring)
def k_sched(df, n):
    return (np.ceil((df.t0 + df.L) / n) - np.ceil(df.t0 / n)).astype(int)


# ------------------------------------------------------------------ statistics helpers
def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def fp(p):
    if p is None or (isinstance(p, float) and math.isnan(p)):
        return "NA"
    if p < 0.001:
        return f"{p:.1e}"
    return f"{p:.3f}"


def holm(ps):
    ps = np.asarray(ps, float)
    m = len(ps)
    order = np.argsort(ps)
    adj = np.empty(m)
    run = 0.0
    for rank, idx in enumerate(order):
        run = max(run, (m - rank) * ps[idx])
        adj[idx] = min(1.0, run)
    return adj


def exact_mcnemar(b, c):
    return 1.0 if b + c == 0 else float(stats.binomtest(b, b + c, 0.5).pvalue)


def rate_ci(s, n):
    lo, hi = wilson(s, n)
    return f"{s}/{n} = {s / n:.2f} [{lo:.2f}, {hi:.2f}]" if n else "-"


def paired(a, b):
    m = a.merge(b, on="episode", suffixes=("_a", "_b"))
    assert len(m) == len(a) == len(b)
    assert (m.env_seed_a == m.env_seed_b).all() and (m.t0_a == m.t0_b).all()
    sa, sb = m.success_a.to_numpy(), m.success_b.to_numpy()
    bb = int(((sa == 1) & (sb == 0)).sum())
    cc = int(((sa == 0) & (sb == 1)).sum())
    return dict(sa=int(sa.sum()), sb=int(sb.sum()), N=len(m), b=bb, c=cc, p=exact_mcnemar(bb, cc), delta=(bb - cc) / len(m))


def verdict(delta, p_raw, p_holm):
    if abs(delta) < NOISE:
        return "不可解讀 (差距<0.05)"
    if p_holm < 0.05:
        return "顯著 (Holm)"
    if p_raw < 0.05:
        return "僅原始 p 顯著"
    if abs(delta) < 0.10:
        return "無差異 (Δ<0.10, p>0.05)"
    return "未定 (Δ≥0.10, p>0.05)"


def md_table(header, rows):
    out = ["| " + " | ".join(header) + " |", "|" + "|".join(["---"] * len(header)) + "|"]
    for r in rows:
        out.append("| " + " | ".join(str(x) for x in r) + " |")
    return "\n".join(out)


def fisher(s1, n1, s2, n2):
    return float(stats.fisher_exact([[s1, n1 - s1], [s2, n2 - s2]])[1])


def cmh(tables):
    """Mantel-Haenszel common OR and CMH chi2 (continuity-corrected) p for 2x2 tables [[a,b],[c,d]] (rows: exposed / unexposed)."""
    num = den = 0.0
    sa = se = sv = 0.0
    used = 0
    for a, b, c, d in tables:
        n = a + b + c + d
        if n == 0 or (a + b) == 0 or (c + d) == 0:
            continue
        used += 1
        num += a * d / n
        den += b * c / n
        sa += a
        se += (a + b) * (a + c) / n
        if n > 1:
            sv += (a + b) * (c + d) * (a + c) * (b + d) / (n * n * (n - 1))
    orr = num / den if den > 0 else float("inf")
    if sv == 0:
        return orr, float("nan"), used
    chi = (abs(sa - se) - 0.5) ** 2 / sv
    return orr, float(stats.chi2.sf(chi, 1)), used


# ------------------------------------------------------------------ logistic regression (cluster-robust)
def logit_fit(cols, y, groups):
    """cols: list of (name, array). Intercept added. Newton-Raphson with step halving.
    Returns dict with coef, se_model, se_cluster (CR1 sandwich by groups), p values (normal), loglik, aic."""
    names = ["(intercept)"] + [c[0] for c in cols]
    X = np.column_stack([np.ones(len(y))] + [np.asarray(c[1], float) for c in cols])
    y = np.asarray(y, float)
    N, p = X.shape
    b = np.zeros(p)

    def ll(beta):
        eta = X @ beta
        return float(np.sum(y * eta - np.logaddexp(0, eta)))

    cur = ll(b)
    converged = False
    for _ in range(100):
        mu = 1 / (1 + np.exp(-(X @ b)))
        W = mu * (1 - mu)
        H = X.T @ (X * W[:, None]) + 1e-10 * np.eye(p)
        step = np.linalg.solve(H, X.T @ (y - mu))
        t = 1.0
        while t > 1e-6:
            new = ll(b + t * step)
            if new >= cur - 1e-12:
                break
            t /= 2
        b = b + t * step
        if abs(new - cur) < 1e-10 and np.max(np.abs(t * step)) < 1e-8:
            cur = new
            converged = True
            break
        cur = new
    mu = 1 / (1 + np.exp(-(X @ b)))
    W = mu * (1 - mu)
    H = X.T @ (X * W[:, None])
    Hinv = np.linalg.inv(H)
    S = X * (y - mu)[:, None]
    uniq, inv = np.unique(groups, return_inverse=True)
    G = len(uniq)
    cs = np.zeros((G, p))
    np.add.at(cs, inv, S)
    meat = cs.T @ cs
    cov_cl = (G / (G - 1)) * ((N - 1) / (N - p)) * Hinv @ meat @ Hinv
    se_m = np.sqrt(np.diag(Hinv))
    se_c = np.sqrt(np.diag(cov_cl))
    z = b / se_c
    sep = (not converged) or np.max(np.abs(b)) > 15 or np.max(se_m) > 50
    return dict(names=names, coef=b, se_m=se_m, se_c=se_c, cov_c=cov_cl, z=z, p_c=2 * stats.norm.sf(np.abs(z)),
                p_m=2 * stats.norm.sf(np.abs(b / se_m)), ll=cur, aic=-2 * cur + 2 * p, N=N, G=G, sep=sep)


# ============================================================================ build the report
MD = []


def W(s=""):
    MD.append(s)


FLAG = {"n": 0, "flagged": 0}
RECORDS = []   # all paired comparisons for Holm bookkeeping: dict(family, ...)


def add_record(family, **kw):
    kw["family"] = family
    RECORDS.append(kw)
    return kw


# ---------------------------------------------------------------- Section 1: main table
def sec1():
    W("## 1. 主表：完整網格成功率（n × L × mode）")
    W()
    W("資料：`results/full_grid.csv`（5400 行 = 54 格 × 100 個配對初始條件）。每格為 成功數/100 = 成功率 [Wilson 95% CI]。"
      "L=0 沒有失明，只跑一次（兩個 mode 共用同一欄）。TE = n=1 加 temporal ensembling（係數 0.01）。"
      "成功 = max_reward 為 4。docs/rulers.md 的重跑雜訊：同設定重跑成敗約有 1% 翻轉，小於 0.05 的差距不可解讀。")
    W()
    for mode in MODES:
        W(f"### {mode}")
        W()
        hdr = ["n \\ L"] + [f"L={L}" for L in LS]
        rows = []
        for nl in NS:
            row = [str(nl) if nl != "TE" else "1+TE"]
            for L in LS:
                g = fcell(nl, L, mode)
                s = int(g.success.sum())
                lo, hi = wilson(s, len(g))
                row.append(f"{s}/{len(g)} = {s / len(g):.2f} [{lo:.2f}, {hi:.2f}]")
            rows.append(row)
        W(md_table(hdr, rows))
        W()
    # headline numbers
    best = F[F.L == 0].groupby("nl").success.mean()
    l0 = {nl: fcell(nl, 0, "black").success.mean() for nl in NS}
    W("**事實（取自上表）**")
    W()
    W(f"- 無失明（L=0）：n=1 為 {l0['1']:.2f}、n=10 為 {l0['10']:.2f}、n=25 為 {l0['25']:.2f}、n=50 為 {l0['50']:.2f}、n=100 為 {l0['100']:.2f}、TE 為 {l0['TE']:.2f}。"
      f"n=1 在全部 {len(F[F.nl == '1'])} 個 episode 中成功 {int(F[F.nl == '1'].success.sum())} 次（T1 診斷的停滯，見 `results/n1_diagnosis.md`）。")
    for mode in MODES:
        vals = {nl: fcell(nl, 50, mode).success.mean() for nl in NS}
        W(f"- L=50、{mode}：" + "、".join(f"{'TE' if nl == 'TE' else 'n=' + nl} {vals[nl]:.2f}" for nl in NS) + "。")
    W()


# ---------------------------------------------------------------- Section 2: boundary hypothesis
REG = {}


def sec2():
    W("## 2. 邊界假說：成功率 ~ k（失明期間重規劃次數）")
    W()
    W("**用詞與設計**：k = 欄位 `replans_in_blackout`，是失明窗口 [t0, t0+L) 內發生的重規劃次數。固定 n 時重規劃發生在 t = 0, n, 2n, …，"
      "所以 k 由 n、L 與 t0 對 chunk 邊界的相位決定，是**中介變數，不是被獨立操縱的變因**；每個 episode 的 (n, L, t0) 決定它。"
      "單一 k 的效果只能在「同 (n, L) 格內，t0 相位造成 k 不同」的準自然實驗，或迴歸控制 L、n 之後觀察。")
    W()
    D0 = F[(F.te == 0) & (F.n >= 10) & (F.L > 0)].copy()
    D0["k"] = D0.replans_in_blackout
    D0["ks"] = k_sched(D0, D0.n)
    # exclusions
    n1 = F[(F.nl == "1")]
    te = F[(F.nl == "TE")]
    n1b = n1[n1.L > 0]
    teb = te[te.L > 0]
    W("**樣本與排除**")
    W()
    W(f"- 迴歸只用固定 n ∈ {{10, 25, 50, 100}}、L ∈ {{10, 25, 50, 100}}（每個 mode {len(D0[D0['mode'] == 'black'])} 個 episode，配對於 {D0.episode.nunique()} 個初始條件）。")
    W(f"- **排除 n=1**：n=1 全部 {len(n1)} 個 episode 成功 {int(n1.success.sum())} 次（失明格 {len(n1b)} 個也是 {int(n1b.success.sum())} 次，只因 T1 的停滯與 k 無關），"
      f"結果變數沒有變異，邏輯斯迴歸完全分離、係數發散，無法估計。")
    W(f"- **排除 TE**：TE 每步重規劃（失明格 k ≈ L，平均 {teb.replans_in_blackout.mean():.1f}），而且是 temporal ensembling 的另一種機制（不是改 n），"
      f"成功率 {teb.success.mean():.2f} 與「k 大必失敗」並不相容，混入同一個線性 k 項會改變 k 的意義；單獨列在主表。")
    W(f"- k 有 {D0.k.nunique()} 個不同值（{', '.join(str(int(v)) for v in sorted(D0.k.unique()))}）；k 與 log2(n) 的相關係數 {np.corrcoef(D0.k, np.log2(D0.n))[0, 1]:.2f}、"
      f"k 與 L 的相關係數 {np.corrcoef(D0.k, D0.L)[0, 1]:.2f}（共線性：k ≈ L/n），所以必須看控制後的係數。")
    mis = D0[D0.k != D0.ks]
    W(f"- 紀錄的 k 與排程 k_sched（由 t0, L, n 算出，不受成敗影響）一致比例 {np.mean(D0.k == D0.ks):.3f}；"
      f"不一致的 {len(mis)} 列全部是成功 episode（成功比例 {mis.success.mean() if len(mis) else float('nan'):.2f}，k 比 k_sched 少 {(mis.ks - mis.k).mean() if len(mis) else 0:.2f}）。"
      f"原因：成功即結束 episode，失明結束前就成功的 episode 少算了之後的重規劃（結果截斷 k）。因此另用 k_sched 做敏感度。")
    W()
    W("**模型**（邏輯斯迴歸，logit P(success)，每個 mode 各自擬合；L 單位為 10 步；n 取 log2，係數是「每加倍 n」；"
      "同一批 100 個初始條件在各格重複使用，所以 **SE 與 p 以初始條件（env seed）為 cluster 的 cluster-robust sandwich（CR1）為主**，原始 SE 並列；"
      "AIC 未做 cluster 校正，僅供描述性比較）：")
    W()
    W("A: k　B: log2 n　C: L　D: k+L　E: k+log2 n　F: k+L+log2 n　G: F + t0 分段（4 段：50–99、100–149、150–199、200–250）　H: 1{k>0}+L+log2 n　S: F 把 k 換成 k_sched　I: F + k×log2 n 交互項　J: S + k_sched×log2 n 交互項")
    W()
    models = {}
    for mode in MODES:
        d = D0[D0["mode"] == mode].reset_index(drop=True)
        y = d.success.to_numpy()
        grp = d.env_seed.to_numpy()
        k = d.k.to_numpy(float)
        L10 = d.L.to_numpy(float) / 10
        ln = np.log2(d.n.to_numpy(float))
        t0b = np.minimum((d.t0.to_numpy() - 50) // 50, 3)
        t0d = [(f"t0 in [{50 + 50 * j},{99 + 50 * j}]" if j < 3 else "t0 in [200,250]", (t0b == j).astype(float)) for j in range(1, 4)]
        spec = {
            "A": [("k", k)], "B": [("log2 n", ln)], "C": [("L/10", L10)],
            "D": [("k", k), ("L/10", L10)], "E": [("k", k), ("log2 n", ln)],
            "F": [("k", k), ("L/10", L10), ("log2 n", ln)],
            "G": [("k", k), ("L/10", L10), ("log2 n", ln)] + t0d,
            "H": [("1{k>0}", (k > 0).astype(float)), ("L/10", L10), ("log2 n", ln)],
            "S": [("k_sched", d.ks.to_numpy(float)), ("L/10", L10), ("log2 n", ln)],
            # review R1 M1: let the cost of one replan inside the blackout depend on how long its plan is then executed (n)
            "I": [("k", k), ("L/10", L10), ("log2 n", ln), ("k:log2 n", k * ln)],
            "J": [("k_sched", d.ks.to_numpy(float)), ("L/10", L10), ("log2 n", ln), ("k_sched:log2 n", d.ks.to_numpy(float) * ln)],
        }
        for key, cols in spec.items():
            models[(mode, key)] = logit_fit(cols, y, grp)
        models[(mode, "N")] = len(d)
    REG["models"] = models
    # summary table: k term (or its analogue) across models
    rows = []
    for mode in MODES:
        for key, term in [("A", "k"), ("D", "k"), ("E", "k"), ("F", "k"), ("G", "k"), ("H", "1{k>0}"), ("S", "k_sched"), ("B", "log2 n"), ("C", "L/10"), ("F", "log2 n"), ("F", "L/10")]:
            m = models[(mode, key)]
            i = m["names"].index(term)
            rows.append([mode, key, term, f"{m['coef'][i]:+.3f}", f"{math.exp(m['coef'][i]):.3g}", f"{m['se_c'][i]:.3f}", f"{m['se_m'][i]:.3f}",
                         fp(m['p_c'][i]), fp(m['p_m'][i]), m["N"], m["G"], f"{m['aic']:.1f}", "是" if m["sep"] else "否"])
    W(md_table(["mode", "model", "term", "coef (logit)", "OR", "SE cluster", "SE model", "p (cluster)", "p (model SE)", "N", "clusters", "AIC", "分離/未收斂"], rows))
    W()
    # pooled model with interaction
    d = D0.reset_index(drop=True)
    fz = (d["mode"] == "freeze").to_numpy(float)
    k = d.k.to_numpy(float)
    pooled = logit_fit([("k", k), ("freeze", fz), ("k:freeze", k * fz), ("L/10", d.L.to_numpy(float) / 10), ("log2 n", np.log2(d.n.to_numpy(float)))],
                       d.success.to_numpy(), d.env_seed.to_numpy())
    REG["pooled"] = pooled
    W("**合併兩個 mode 的交互作用模型**（success ~ k·mode + L + log2 n；cluster = 初始條件；同一初始條件的 black 與 freeze 兩列都在同一 cluster）：")
    W()
    W(md_table(["term", "coef", "SE cluster", "p (cluster)"],
               [[nm, f"{c:+.3f}", f"{s:.3f}", fp(p)] for nm, c, s, p in zip(pooled["names"], pooled["coef"], pooled["se_c"], pooled["p_c"])]))
    W(f"　N = {pooled['N']}，clusters = {pooled['G']}。")
    W()
    # full coefficient tables
    W("<details><summary>全部模型的完整係數表</summary>")
    W()
    rows = []
    for mode in MODES:
        for key in "ABCDEFGHSIJ":
            m = models[(mode, key)]
            for i, nm in enumerate(m["names"]):
                rows.append([mode, key, nm, f"{m['coef'][i]:+.3f}", f"{m['se_c'][i]:.3f}", f"{m['se_m'][i]:.3f}", fp(m["p_c"][i]), fp(m["p_m"][i])])
    W(md_table(["mode", "model", "term", "coef", "SE cluster", "SE model", "p cluster", "p model"], rows))
    W()
    W("</details>")
    W()

    # natural experiment: k=0 vs k>=1 within cell
    W("### 2.1 最乾淨的自然實驗：同一格（同 n、L、mode）內 k=0 對 k≥1")
    W()
    W("同一格內 n、L 固定，k 只由 t0 對 chunk 邊界的相位決定。表列各格 k=0 與 k≥1 的成功率（無配對，Fisher 精確檢定；兩組是不同 episode 子集）。"
      "左邊用紀錄的 k（`replans_in_blackout`），右邊用排程 k_sched（不受成敗截斷）。L=0 欄是同 n 無失明的成功率，作為上限參照。"
      "**限制**：k 由 t0 決定，t0 同時決定失明落在任務的哪個階段，所以 k 與「失明時機」糾纏（見 2.3 的 t0 控制）。")
    W()
    rows = []
    cells = []
    for n in [100, 50, 25, 10]:
        for L in LB:
            for mode in MODES:
                g = D0[(D0.n == n) & (D0.L == L) & (D0["mode"] == mode)]
                a, b = g[g.k == 0], g[g.k >= 1]
                a2, b2 = g[g.ks == 0], g[g.ks >= 1]
                if len(a) == 0 or len(b) == 0:
                    continue
                l0 = fcell(n, 0, "black").success.mean()
                p = fisher(int(a.success.sum()), len(a), int(b.success.sum()), len(b))
                p2 = fisher(int(a2.success.sum()), len(a2), int(b2.success.sum()), len(b2)) if len(a2) and len(b2) else float("nan")
                cells.append((n, L, mode, a, b, a2, b2))
                rows.append([n, L, mode, f"{l0:.2f}",
                             rate_ci(int(a.success.sum()), len(a)), rate_ci(int(b.success.sum()), len(b)), f"{a.success.mean() - b.success.mean():+.2f}", fp(p),
                             f"{int(a2.success.sum())}/{len(a2)} vs {int(b2.success.sum())}/{len(b2)}" if len(a2) and len(b2) else "-", fp(p2),
                             "不可解讀 (差距<0.05)" if abs(a.success.mean() - b.success.mean()) < NOISE else ("k=0 為截斷產物" if len(a2) == 0 else "")])
    W(md_table(["n", "L", "mode", "L=0 rate (same n)", "k=0: succ/N = rate [CI]", "k≥1: succ/N = rate [CI]", "Δ (k0−k≥1)", "Fisher p", "k_sched 0 vs ≥1", "Fisher p (k_sched)", "註"], rows))
    W()
    cens = [(c[0], c[1], c[2], len(c[3]), int(c[3].success.sum())) for c in cells if len(c[5]) == 0]
    W("**註（截斷）**：k_sched 欄為「-」的格，k_sched 對所有 episode 都 ≥1（該 n、L 下失明窗口必定跨過至少一個 chunk 邊界），"
      "但紀錄的 k=0 仍有 " + "、".join(f"n={n},L={L},{m}：{N} 個 episode（其中成功 {sc} 個）" for n, L, m, N, sc in cens)
      + "。這些是失明結束前就成功而提早結束的 episode（結果截斷了 k），不是真正沒遇到邊界，**不可當作 k=0 的證據**。")
    W()
    REG["cells"] = cells
    # Mantel-Haenszel over the cells (cells share the same 100 seeds => descriptive)
    mh_rows = []
    for mode in MODES:
        for label, sel in [("n=100,50 的格", lambda c: c[0] in (100, 50)), ("所有可比較的格 (n=10..100)", lambda c: True)]:
            tabs = []
            for (n, L, md, a, b, a2, b2) in cells:
                if md == mode and sel((n, L)):
                    tabs.append((int(b.success.sum()), len(b) - int(b.success.sum()), int(a.success.sum()), len(a) - int(a.success.sum())))
            orr, p, used = cmh(tabs)
            mh_rows.append([mode, label, used, f"{orr:.3g}", fp(p)])
    REG["mh"] = mh_rows
    W("分層合併（Mantel-Haenszel，各格為一層）：k≥1 對 k=0 的共同勝算比（OR < 1 表示 k≥1 較差）。各格共用同一批 100 個初始條件，格間不獨立，p 值僅供描述。")
    W()
    W(md_table(["mode", "層", "有效層數", "MH OR (k≥1 vs k=0)", "CMH p (continuity-corrected)"], mh_rows))
    W()

    # reading
    def cell(n, L, mode):
        for c in cells:
            if c[0] == n and c[1] == L and c[2] == mode:
                return c
    W("### 2.2 讀表（事實／推論分開）")
    W()
    W("**事實**")
    W()
    c1 = cell(100, 50, "black")
    c2 = cell(100, 50, "freeze")
    if c1 and c2:
        W(f"- n=100、L=50：黑畫面 k=0 為 {int(c1[3].success.sum())}/{len(c1[3])}，k≥1 為 {int(c1[4].success.sum())}/{len(c1[4])}；"
          f"凍結 k=0 為 {int(c2[3].success.sum())}/{len(c2[3])}，k≥1 為 {int(c2[4].success.sum())}/{len(c2[4])}。"
          f"同一個 k≥1 條件下，黑畫面幾乎全敗而凍結多數成功，所以傷害不在「重規劃發生」本身，而在重規劃**吃進了什麼影像**。")
    for mode in MODES:
        mA, mF, mG, mH = (models[(mode, key)] for key in "AFGH")
        kA, kF, kG = (m["names"].index("k") for m in (mA, mF, mG))
        sig = "仍顯著" if mF["p_c"][kF] < 0.05 else "不顯著"
        W(f"- {mode}：單變數 k 係數 {mA['coef'][kA]:+.3f}（cluster SE {mA['se_c'][kA]:.3f}，p={fp(mA['p_c'][kA])}，N={mA['N']}）；"
          f"控制 L 與 log2 n 後 k 係數 {mF['coef'][kF]:+.3f}（SE {mF['se_c'][kF]:.3f}，p={fp(mF['p_c'][kF])}，{sig}）；"
          f"再加 t0 分段後 {mG['coef'][kG]:+.3f}（SE {mG['se_c'][kG]:.3f}，p={fp(mG['p_c'][kG])}）；"
          f"二元 1{{k>0}} 控制後 {mH['coef'][1]:+.3f}（SE {mH['se_c'][1]:.3f}，p={fp(mH['p_c'][1])}）。")
    for mode in MODES:
        mB, mC, mA = models[(mode, "B")], models[(mode, "C")], models[(mode, "A")]
        mF = models[(mode, "F")]
        W(f"- {mode}：單變數模型 AIC：k {mA['aic']:.1f}、log2 n {mB['aic']:.1f}、L {mC['aic']:.1f}；三者齊全 {mF['aic']:.1f}（k 單獨解釋力 {'優於' if mA['aic'] < min(mB['aic'], mC['aic']) else '不優於'} 單用 n 或 L）。")
    for mode in MODES:
        mF = models[(mode, "F")]
        il, ik = mF["names"].index("log2 n"), mF["names"].index("L/10")
        W(f"- {mode}：模型 F 中 log2 n 係數 {mF['coef'][il]:+.3f}（SE {mF['se_c'][il]:.3f}，p={fp(mF['p_c'][il])}）、L/10 係數 {mF['coef'][ik]:+.3f}（SE {mF['se_c'][ik]:.3f}，p={fp(mF['p_c'][ik])}）；"
          f"單用 n 的模型 B 為 {models[(mode, 'B')]['coef'][1]:+.3f}（p={fp(models[(mode, 'B')]['p_c'][1])}）。")
    int_i = pooled["names"].index("k:freeze")
    W(f"- 交互作用：k:freeze 係數 {pooled['coef'][int_i]:+.3f}（SE {pooled['se_c'][int_i]:.3f}，p={fp(pooled['p_c'][int_i])}）。")
    W()
    W("**推論**")
    W()
    mFb, mFf = models[("black", "F")], models[("freeze", "F")]
    ib, i2 = mFb["names"].index("k"), mFf["names"].index("k")
    W(f"- 黑畫面：控制 L 與 n 後，每多一次失明期間重規劃 logit 變動 {mFb['coef'][ib]:+.2f}（勝算比 {math.exp(mFb['coef'][ib]):.2f}），"
      f"{'k 在控制 L 與 n 後仍有解釋力' if mFb['p_c'][ib] < 0.05 else 'k 在控制 L 與 n 後無法與零區分'}；"
      f"凍結：{mFf['coef'][i2]:+.2f}（勝算比 {math.exp(mFf['coef'][i2]):.2f}，p={fp(mFf['p_c'][i2])}），"
      f"{'仍顯著但幅度遠小於黑畫面' if mFf['p_c'][i2] < 0.05 and abs(mFf['coef'][i2]) < abs(mFb['coef'][ib]) else '見係數'}。這支持「失明期間的重規劃是主要傷害機制，且壞到什麼程度取決於影像內容」，"
      f"但 k 不是被操縱的變因，且與 t0 糾纏，只能稱為與機制一致的關聯證據；oracle（第 4 節）才是對 k 的介入。")
    W()
    W("### 2.3 限制")
    W()
    W("- k 是 (n, L, t0) 的函數；表中 k=0 與 k≥1 的 episode 失明落在任務的不同階段（t0 不同），G 模型的 t0 分段只做粗略控制。")
    W("- 記錄的 k 在成功時被截斷（成功 episode 提前結束），這會讓 k 與成功呈人為的負相關；k_sched 模型（S）不受此影響，結論是否一致見完整係數表。")
    W("- 各格共用同一批初始條件，已用 cluster-robust SE，但只有 100 個 cluster；AIC 與 MH p 未校正此相依。")
    W("- L 與 k 在 n 固定時幾乎共線，k 的獨立效果主要靠 n=50、100 的格內相位變異辨識。")
    W()
    sec2_interaction(models, D0)


def marg_slope(m, term_k, term_int, ln):
    """slope of logit P(success) in k at a given log2 n:  b_k + b_int*ln, with cluster-robust SE from the covariance matrix."""
    ik, ii = m["names"].index(term_k), m["names"].index(term_int)
    v = np.array([1.0, ln])
    b = np.array([m["coef"][ik], m["coef"][ii]])
    C = m["cov_c"][np.ix_([ik, ii], [ik, ii])]
    return float(v @ b), float(np.sqrt(v @ C @ v))


def sec2_interaction(models, D0):
    W("### 2.4 k 的代價隨 n 變大：k × log2 n 交互項（審查第一輪 M1）")
    W()
    W("模型 F 只放主效果，假設「多一次失明中重規劃」的代價與 n 無關。這裡加入 k×log2 n（模型 I，紀錄的 k；模型 J，排程 k_sched），"
      "SE 與 p 仍是以 env seed 為 cluster 的 CR1；AIC 未做 cluster 校正，只供描述。每次壞重規劃的 logit 斜率 = b_k + b_(k×log2 n)·log2 n。")
    W()
    rows = []
    for mode in MODES:
        for key, base, kn in [("F", None, "k"), ("I", "F", "k"), ("J", "S", "k_sched")]:
            m = models[(mode, key)]
            ik, il = m["names"].index(kn), m["names"].index("log2 n")
            ii = m["names"].index(f"{kn}:log2 n") if f"{kn}:log2 n" in m["names"] else None
            rows.append([mode, key, f"{m['coef'][ik]:+.2f} ({m['se_c'][ik]:.2f})", f"{m['coef'][il]:+.2f} ({m['se_c'][il]:.2f})",
                         "-" if ii is None else f"{m['coef'][ii]:+.3f} ({m['se_c'][ii]:.3f})", "-" if ii is None else f"{m['coef'][ii] / m['se_c'][ii]:+.1f}",
                         "-" if ii is None else fp(m["p_c"][ii]), f"{m['aic']:.1f}"])
    W(md_table(["mode", "model", "k 主效果 (cluster SE)", "log2 n 主效果 (SE)", "k×log2 n (SE)", "z", "p (cluster)", "AIC"], rows))
    W()
    mFb, mIb = models[("black", "F")], models[("black", "I")]
    W(f"- black：加入交互項後 AIC 由 {mFb['aic']:.1f} 降到 {mIb['aic']:.1f}（降 {mFb['aic'] - mIb['aic']:.1f}）；log2 n 主效果由 {mFb['coef'][mFb['names'].index('log2 n')]:+.3f} 變成 {mIb['coef'][mIb['names'].index('log2 n')]:+.3f}。")
    mFf, mIf = models[("freeze", "F")], models[("freeze", "I")]
    sl_f = [marg_slope(mIf, "k", "k:log2 n", math.log2(n)) for n in (25, 50, 100)]
    ci0 = all(sl - 1.96 * se < 0 < sl + 1.96 * se for sl, se in sl_f)
    W(f"- freeze：AIC {mFf['aic']:.1f} → {mIf['aic']:.1f}（降 {mFf['aic'] - mIf['aic']:.1f}）；交互項方向相反（正），n=25、50、100 的 k 斜率 95% CI {'都含 0' if ci0 else '不全含 0'}（見下表）。")
    W()
    W("**各 n 的 k 邊際效果**（每多一次失明中重規劃的 logit 斜率與勝算比；模型 I）：")
    W()
    rows = []
    for mode in MODES:
        m = models[(mode, "I")]
        for n in (10, 25, 50, 100):
            sl, se = marg_slope(m, "k", "k:log2 n", math.log2(n))
            rows.append([mode, n, f"{sl:+.2f}", f"{se:.2f}", f"{math.exp(sl):.2f}", f"[{sl - 1.96 * se:+.2f}, {sl + 1.96 * se:+.2f}]"])
    W(md_table(["mode", "n", "slope of k (logit)", "SE", "OR per replan", "95% CI"], rows))
    W()
    W("**固定 n × L 的 k=0 對 k≥1 成功數**（k 用排程 k_sched，不受成功即結束的截斷影響；L=0 欄是同 n 無失明的成功數；「-」＝該格沒有 k=0 的 episode）：")
    W()
    for mode in MODES:
        rows = []
        for n in (10, 25, 50, 100):
            row = [n, int(fcell(n, 0, "black").success.sum())]
            for L in (10, 25, 50):
                g = D0[(D0.n == n) & (D0.L == L) & (D0["mode"] == mode)]
                a, b = g[g.ks == 0], g[g.ks >= 1]
                row.append(f"{int(a.success.sum())}/{len(a)}" if len(a) else "-")
                row.append(f"{int(b.success.sum())}/{len(b)}")
            rows.append(row)
        W(f"{mode}：")
        W()
        W(md_table(["n", "L=0", "L=10 k=0", "L=10 k≥1", "L=25 k=0", "L=25 k≥1", "L=50 k=0", "L=50 k≥1"], rows))
        W()
    g10 = D0[(D0.n == 10) & (D0.L == 10) & (D0["mode"] == "black")]
    W(f"- n=10、L=10、black：{len(g10)} 個 episode 中排程 k≥1 的有 {int((g10.ks >= 1).sum())} 個（全部），成功 {int(g10.success.sum())}/{len(g10)}，等於無失明的 {int(fcell(10, 0, 'black').success.sum())}/100；"
      "同一個 episode 子集對無失明配對的不一致配對數見下行。")
    a0, a1 = fcell(10, 0, "black"), fcell(10, 10, "black")
    mm = a0.merge(a1, on="episode", suffixes=("_0", "_1"))
    bb = int(((mm.success_0 == 1) & (mm.success_1 == 0)).sum()); cc = int(((mm.success_0 == 0) & (mm.success_1 == 1)).sum())
    W(f"- n=10、L=10、black 對 L=0 的配對：只有無失明成功 {bb}、只有失明成功 {cc}（McNemar 精確 p={fp(exact_mcnemar(bb, cc))}）。")
    W("- n=100 遇到 k≥1 時，black 成功數隨 L：" + "、".join(
        f"L={L} {int(D0[(D0.n == 100) & (D0.L == L) & (D0['mode'] == 'black') & (D0.ks >= 1)].success.sum())}/{int(((D0.n == 100) & (D0.L == L) & (D0['mode'] == 'black') & (D0.ks >= 1)).sum())}" for L in (10, 25, 50)) + "。")
    W()
    W("**讀法**：n=10 時一次壞重規劃的計畫只被執行 10 步，之後就被下一次重規劃覆蓋；n=100 時同樣一次壞重規劃的計畫要被執行 100 步，幾乎必敗。"
      "所以「k 決定一切、n 沒有額外效果」只在沒放交互項的模型 F 成立；每次壞重規劃的代價隨 n 增加（斜率 n=10 約 −0.6 → n=100 約 −2.9）。"
      "TE（每步重規劃）把每個壞預測與較早的預測平均，單一壞預測的權重小，與此圖像一致（TE 未納入迴歸）。")
    W()


# ---------------------------------------------------------------- Section 3: McNemar families
def sec3():
    W("## 3. 關鍵對照：配對 McNemar 精確檢定")
    W()
    W("配對方式：同 env seed 與同 t0（程式內 assert 驗證）。b = 前者單贏、c = 後者單贏；p 為精確二項雙尾。"
      "Holm 校正對每個比較家族各做一次（家族內全部檢定），另列跨三個家族（共 84 個，見各表標頭）的全域 Holm。"
      "判讀欄：|Δ|<0.05 一律標「不可解讀」（docs/rulers.md 重跑雜訊）；其次看 Holm p<0.05；差距<0.10 且 p>0.05 為「無差異」（rulers.md 規則）。"
      "注意 oracle 與 proprio 用自寫執行迴圈 (`code/run_t3t4.py`)，fixed n 用 policy 佇列 (`code/run_grid.py`)，兩邊各帶重跑雜訊。")
    W()
    famA, famB, famC = [], [], []
    # A: black vs freeze
    for nl in NS:
        for L in LB:
            r = paired(fcell(nl, L, "black"), fcell(nl, L, "freeze"))
            famA.append(add_record("A", n=nl, L=L, **r))
    # B: oracle vs fixed n
    for n in [1, 10, 25, 50]:
        for mode in MODES:
            for L in LB:
                r = paired(OD[(n, L, mode)], fcell(n, L, mode))
                famB.append(add_record("B", n=n, L=L, mode=mode, **r))
    # C: proprio vs n=50 / n=100
    for m in [1, 10]:
        for (L, mode) in [(0, "black")] + [(L, md) for L in (25, 50, 100) for md in MODES]:
            for nf in [50, 100]:
                r = paired(PD[(m, L, mode)], fcell(nf, L, mode))
                famC.append(add_record("C", m=m, L=L, mode=mode, nf=nf, **r))
    allp = holm([r["p"] for r in RECORDS if r["family"] in "ABC"])
    for r, a in zip([r for r in RECORDS if r["family"] in "ABC"], allp):
        r["p_holm_all"] = a
    for fam in (famA, famB, famC):
        ad = holm([r["p"] for r in fam])
        for r, a in zip(fam, ad):
            r["p_holm"] = a
            r["verdict"] = verdict(r["delta"], r["p"], a)
            FLAG["n"] += 1
            FLAG["flagged"] += abs(r["delta"]) < NOISE

    def fmt(r, lead):
        return lead + [f"{r['sa']}", f"{r['sb']}", f"{r['delta']:+.2f}", f"{r['b']} / {r['c']}", fp(r["p"]), fp(r["p_holm"]), fp(r["p_holm_all"]), r["verdict"]]

    W("### 3.1 black 對 freeze（同 n、L；b = black 單贏，c = freeze 單贏）")
    W()
    W(md_table(["n", "L", "black succ", "freeze succ", "Δ (black−freeze)", "b / c", "p (raw)", f"p (Holm, 家族 {len(famA)})", f"p (Holm, 全域 {len(famA) + len(famB) + len(famC)})", "判讀"],
               [fmt(r, [r["n"], r["L"]]) for r in famA]))
    W()
    W("### 3.2 oracle 對 fixed n（同 n、L、mode；b = oracle 單贏，c = fixed 單贏）")
    W()
    W(md_table(["n", "mode", "L", "oracle succ", "fixed succ", "Δ (oracle−fixed)", "b / c", "p (raw)", f"p (Holm, 家族 {len(famB)})", f"p (Holm, 全域 {len(famA) + len(famB) + len(famC)})", "判讀"],
               [[r["n"], r["mode"], r["L"]] + fmt(r, [])[0:] for r in famB]))
    W()
    W("### 3.3 proprio（臂訊號 τ99）對固定 n=50、n=100（b = proprio 單贏，c = fixed 單贏）")
    W()
    W("L=0 無失明（proprio 與固定 n 都取 L=0 格）。m1 / m10 為最少執行步數。")
    W()
    W(md_table(["variant", "L", "mode", "vs fixed n", "proprio succ", "fixed succ", "Δ (proprio−fixed)", "b / c", "p (raw)", f"p (Holm, 家族 {len(famC)})", f"p (Holm, 全域 {len(famA) + len(famB) + len(famC)})", "判讀"],
               [[f"proprio-m{r['m']}", r["L"], "-" if r["L"] == 0 else r["mode"], r["nf"]] + fmt(r, [])[0:] for r in famC]))
    W()
    # facts
    def nsig(fam):
        return sum(r["p_holm"] < 0.05 and abs(r["delta"]) >= NOISE for r in fam)
    W("**事實**")
    W()
    big = [r for r in famA if r["L"] == 50 and r["n"] in ("50", "100")]
    W("- black 對 freeze：" + "；".join(f"n={r['n']}、L={r['L']}：black {r['sa']} 對 freeze {r['sb']}（b/c={r['b']}/{r['c']}，Holm p={fp(r['p_holm'])}）" for r in big)
      + f"。家族 {len(famA)} 個中 Holm 顯著且 |Δ|≥0.05 者 {nsig(famA)} 個。")
    sel = [r for r in famB if r["L"] == 50 and r["mode"] == "black" and r["n"] in (10, 25, 50)]
    W("- oracle 對 fixed n（L=50、black）：" + "；".join(f"n={r['n']}：{r['sa']} 對 {r['sb']}（Holm p={fp(r['p_holm'])}）" for r in sel)
      + f"。家族 {len(famB)} 個中 Holm 顯著且 |Δ|≥0.05 者 {nsig(famB)} 個，其中 oracle 較差者 {sum(r['p_holm'] < 0.05 and r['delta'] <= -NOISE for r in famB)} 個。")
    W(f"- proprio 對 n=50：{sum(r['p_holm'] < 0.05 and abs(r['delta']) >= NOISE for r in famC if r['nf'] == 50)} / {sum(r['nf'] == 50 for r in famC)} 個 Holm 顯著；對 n=100：{sum(r['p_holm'] < 0.05 and abs(r['delta']) >= NOISE for r in famC if r['nf'] == 100)} / {sum(r['nf'] == 100 for r in famC)} 個。"
      f"全域 Holm 顯著（含小差距）：A {sum(r['p_holm_all'] < 0.05 for r in famA)}、B {sum(r['p_holm_all'] < 0.05 for r in famB)}、C {sum(r['p_holm_all'] < 0.05 for r in famC)}。")
    W()
    return famA, famB, famC


# ---------------------------------------------------------------- Section 4: oracle distinguishing analysis
def sec4():
    W("## 4. 區分性分析：oracle 的改善來自「失明期間不吃壞觀測」，還是「等同較大 n」？")
    W()
    W("**oracle 的定義**：被**告知**失明區間的規則（失明期間不重規劃、沿用手上 chunk 剩餘動作，chunk 用完才被迫重規劃），不是策略自己察覺到影像壞掉；它是上限／介入實驗，不是可實作的方法。")
    W()
    W("兩個假說的可檢驗預測：")
    W("- H_blind（不吃壞觀測）：oracle 相對固定 n 的增益，集中在「固定 n 會在失明中重規劃」的 episode；在固定 n 本來就沒碰到壞影像的 episode 上增益應接近 0。")
    W("- H_n（等同較大 n）：增益與固定 n 是否碰到壞影像無關，且 oracle(n) 應與固定 n=100 相近（因為少重規劃 ≈ 大 n）。")
    W()
    # 4a oracle vs fixed 100
    W("### 4.1 oracle(n) 對 fixed n=100（同 L、mode）與對自己的無失明（L=0）參照")
    W()
    W("oracle n ∈ {10, 25, 50}（n=1 不納入：L=0 就停滯）。fixed n=100 取 T2 同格。L=0 參照為同 n 無失明的固定 n（有失明格用 oracle 的 n）。b/c 皆為 oracle 單贏／對照單贏。")
    W()
    rowsA = []
    famX, famY = [], []
    for n in [10, 25, 50]:
        for mode in MODES:
            for L in LB:
                r1 = paired(OD[(n, L, mode)], fcell(100, L, mode))
                r2 = paired(OD[(n, L, mode)], fcell(n, 0, mode))
                add_record("X", **r1)
                add_record("Y", **r2)
                famX.append((n, mode, L, r1))
                famY.append((n, mode, L, r2))
    for fam in (famX, famY):
        ad = holm([t[3]["p"] for t in fam])
        for (n, mode, L, r), a in zip(fam, ad):
            r["p_holm"] = a
            r["verdict"] = verdict(r["delta"], r["p"], a)
            FLAG["n"] += 1
            FLAG["flagged"] += abs(r["delta"]) < NOISE
    rows = []
    for (n, mode, L, r1), (_, _, _, r2) in zip(famX, famY):
        rows.append([n, mode, L, r1["sa"], r1["sb"], f"{r1['delta']:+.2f}", f"{r1['b']}/{r1['c']}", fp(r1["p"]), fp(r1["p_holm"]), r1["verdict"],
                     r2["sb"], f"{r2['delta']:+.2f}", f"{r2['b']}/{r2['c']}", fp(r2["p"]), fp(r2["p_holm"]), r2["verdict"]])
    W(md_table(["oracle n", "mode", "L", "oracle succ", "fixed-100 succ", "Δ vs fixed 100", "b/c", "p raw", f"p Holm({len(famX)})", "判讀",
                "fixed n, L=0 succ", "Δ vs L=0", "b/c", "p raw", f"p Holm({len(famX)})", "判讀"], rows))
    W()
    # 4.2 subset by scheduled k of fixed n=100
    W("### 4.2 以 fixed n=100 是否在失明中重規劃（k_sched）分層")
    W()
    W("k_sched(n=100) 由 t0 與 L 算出（重規劃發生於 t = 0, 100, 200, 300），不受成敗截斷。k100=0：固定 n=100 本來就沒有在失明中重規劃（它與 oracle 一樣沒吃壞影像，差別只剩 n）；"
      "k100≥1：固定 n=100 在失明中重規劃。b/c = oracle 單贏／fixed-100 單贏。")
    W()
    rows4 = []
    famS = []
    sub = {}
    for mode in MODES:
        for L in LB:
            base = fcell(100, L, mode)
            ks = k_sched(base, 100).to_numpy()
            for lab, mask in [("k100=0", ks == 0), ("k100≥1", ks >= 1)]:
                if mask.sum() == 0:
                    continue
                eps = base.episode.to_numpy()[mask]
                for n in [25, 50]:
                    a = OD[(n, L, mode)]
                    a = a[a.episode.isin(eps)].reset_index(drop=True)
                    bdf = base[base.episode.isin(eps)].reset_index(drop=True)
                    r = paired(a, bdf)
                    add_record("S", **r)
                    famS.append((mode, L, lab, n, r))
                    sub[(mode, L, lab, n)] = r
    ad = holm([t[4]["p"] for t in famS])
    for (mode, L, lab, n, r), a in zip(famS, ad):
        r["p_holm"] = a
        r["verdict"] = verdict(r["delta"], r["p"], a)
        FLAG["n"] += 1
        FLAG["flagged"] += abs(r["delta"]) < NOISE
        rows4.append([mode, L, lab, n, r["N"], r["sb"], r["sa"], f"{r['delta']:+.2f}", f"{r['b']}/{r['c']}", fp(r["p"]), fp(r["p_holm"]), r["verdict"]])
    W(md_table(["mode", "L", "子集", "oracle n", "N", "fixed-100 succ", "oracle succ", "Δ", "b/c", "p raw", f"p Holm({len(famS)})", "判讀"], rows4))
    W()
    # pooled summary over L=10,25,50
    W("合併 L ∈ {10, 25, 50}（同一批初始條件於各 L 重複，只作描述）：")
    W()
    prow = []
    pooled = {}
    for mode in MODES:
        for lab in ("k100=0", "k100≥1"):
            for n in (25, 50):
                s_o = s_f = N = 0
                for L in (10, 25, 50):
                    r = sub.get((mode, L, lab, n))
                    if r:
                        s_o += r["sa"]; s_f += r["sb"]; N += r["N"]
                pooled[(mode, lab, n)] = (s_f, s_o, N)
                prow.append([mode, lab, n, N, f"{s_f / N:.2f}" if N else "-", f"{s_o / N:.2f}" if N else "-", f"{(s_o - s_f) / N:+.2f}" if N else "-"])
    W(md_table(["mode", "子集", "oracle n", "episode-L 對數", "fixed-100 rate", "oracle rate", "oracle − fixed-100"], prow))
    W()
    # 4.3 forced
    W("### 4.3 oracle 內部：被迫重規劃（forced_replans_in_blackout）=0 對 ≥1（僅描述，易受截斷影響）")
    W()
    fc = O.groupby("L").forced_replans_in_blackout.agg(lambda s: int((s >= 1).sum()))
    f100 = O[(O.L == 100) & (O.n >= 10)]
    f0 = f100[f100.forced_replans_in_blackout == 0]
    early = float((f0.steps < f0.t0 + f0.L).mean()) if len(f0) else float("nan")
    W(f"forced≥1 的 episode 數按 L：" + "、".join(f"L={L}：{int(fc[L])}" for L in LB)
      + "。只有 L=100 同時有兩組，所以此比較**無法與 L 分開**；以下只在 L=100 內、各 (n, mode) 比較（unpaired，Fisher 精確檢定）。"
        "forced 由 t0 對 chunk 邊界的相位決定，與 t0 糾纏；而且 forced 是**紀錄值**，成功即結束 episode，所以失明結束前就成功的 episode 會被記成 forced=0："
        f"L=100、n≥10 的 forced=0 episode 共 {len(f0)} 個，其中在失明結束前（steps < t0+L）就結束的比例為 {early:.0%}。"
        "因此 forced=0 組的高成功率有一大部分是截斷產物，**這張表不能當作 oracle 內部「forced 傷害成功率」的證據**；第 4.2 節用不受截斷的 k_sched 分層才是主要證據。")
    W()
    rows = []
    famF = []
    for mode in MODES:
        for n in [1, 10, 25, 50]:
            g = OD[(n, 100, mode)]
            a, b = g[g.forced_replans_in_blackout == 0], g[g.forced_replans_in_blackout >= 1]
            if len(a) == 0 or len(b) == 0:
                rows.append([mode, n, rate_ci(int(a.success.sum()), len(a)), rate_ci(int(b.success.sum()), len(b)), "-", "-", ""])
                continue
            p = fisher(int(a.success.sum()), len(a), int(b.success.sum()), len(b))
            famF.append((mode, n, p))
            dd = a.success.mean() - b.success.mean()
            rows.append([mode, n, rate_ci(int(a.success.sum()), len(a)), rate_ci(int(b.success.sum()), len(b)), f"{dd:+.2f}", fp(p), "不可解讀 (差距<0.05)" if abs(dd) < NOISE else ""])
    W(md_table(["mode", "oracle n", "forced=0: succ/N = rate [CI]", "forced≥1: succ/N = rate [CI]", "Δ (0 − ≥1)", "Fisher p", "註"], rows))
    W()
    # facts / inference
    W("### 4.4 結論")
    W()
    W("**事實**")
    W()
    for mode in MODES:
        w = []
        for n in (25, 50):
            s_f0, s_o0, N0 = pooled[(mode, "k100=0", n)]
            s_f1, s_o1, N1 = pooled[(mode, "k100≥1", n)]
            w.append(f"oracle n={n}：k100=0 子集 {s_f0 / N0:.2f}→{s_o0 / N0:.2f}（{(s_o0 - s_f0) / N0:+.2f}，N={N0}），k100≥1 子集 {s_f1 / N1:.2f}→{s_o1 / N1:.2f}（{(s_o1 - s_f1) / N1:+.2f}，N={N1}）")
        W(f"- {mode}（L=10/25/50 合併，fixed-100→oracle）：" + "；".join(w) + "。")
    sel = {(n, mode, L): r for (n, mode, L, r) in famX}
    W("- oracle(n) 對 fixed n=100（L=50）：" + "；".join(f"{mode} n={n}：{sel[(n, mode, 50)]['sa']} 對 {sel[(n, mode, 50)]['sb']}" for mode in MODES for n in (25, 50)) + "。")
    sel2 = {(n, mode, L): r for (n, mode, L, r) in famY}
    W("- oracle(n) 與同 n 無失明（L=0）：" + "；".join(f"{mode} n={n}、L=50：oracle {sel2[(n, mode, 50)]['sa']} 對 L=0 {sel2[(n, mode, 50)]['sb']}（Δ={sel2[(n, mode, 50)]['delta']:+.2f}）" for mode in MODES for n in (25, 50)) + "。")
    W(f"- forced=0 組在 L=100 的成功率高，但 forced=0 的 episode 中有 {early:.0%} 在失明結束前（steps < t0+L）就已結束（截斷），故不採為證據（見 4.3）。")
    W()
    # compute decision for black
    gain0 = np.mean([(pooled[("black", "k100=0", n)][1] - pooled[("black", "k100=0", n)][0]) / pooled[("black", "k100=0", n)][2] for n in (25, 50)])
    gain1 = np.mean([(pooled[("black", "k100≥1", n)][1] - pooled[("black", "k100≥1", n)][0]) / pooled[("black", "k100≥1", n)][2] for n in (25, 50)])
    W("**推論**")
    W()
    W(f"- 黑畫面：oracle 相對 fixed-100 的增益在「fixed-100 會在失明中重規劃」的子集為平均 {gain1:+.2f}，在「不會」的子集為 {gain0:+.2f}。"
      + ("增益集中在會吃壞影像的子集，在固定 n 本來就沒吃到壞影像的子集上接近 0 或為負，**與 H_blind 一致、不支持 H_n**。"
         if gain1 - gain0 > 0.3 else "兩個子集的增益差距不大，無法區分兩個假說。"))
    gf0 = np.mean([(pooled[("freeze", "k100=0", n)][1] - pooled[("freeze", "k100=0", n)][0]) / pooled[("freeze", "k100=0", n)][2] for n in (25, 50)])
    gf1 = np.mean([(pooled[("freeze", "k100≥1", n)][1] - pooled[("freeze", "k100≥1", n)][0]) / pooled[("freeze", "k100≥1", n)][2] for n in (25, 50)])
    W(f"- 凍結：增益在會重規劃的子集為平均 {gf1:+.2f}、不會的子集 {gf0:+.2f}；兩者都小，因為凍結影像（最後一張好影像）本來就不傷，fixed-100 在 k100≥1 子集仍有 {np.mean([pooled[('freeze', 'k100≥1', n)][0] / pooled[('freeze', 'k100≥1', n)][2] for n in (25, 50)]):.2f}。")
    rng = []
    for mode in MODES:
        for n in (25, 50):
            ds = [sel2[(n, mode, L)]["delta"] for L in (10, 25, 50)]
            rng.append(f"{mode} n={n}：{min(ds):+.2f} 至 {max(ds):+.2f}")
    W(f"- 固定 n=100 在「失明中沒有重規劃」的子集（k100=0，L=10/25/50 合併）成功率 {pooled[('black', 'k100=0', 25)][0] / pooled[('black', 'k100=0', 25)][2]:.2f}，與其 L=0 的 {fcell(100, 0, 'black').success.mean():.2f} 相當；有重規劃的子集黑畫面 {pooled[('black', 'k100≥1', 25)][0] / pooled[('black', 'k100≥1', 25)][2]:.2f}、凍結 {pooled[('freeze', 'k100≥1', 25)][0] / pooled[('freeze', 'k100≥1', 25)][2]:.2f}。")
    W("- oracle(n=25/50) 在 L ≤ 50 與同 n 無失明（L=0）的差距（Δ，oracle − L=0）範圍：" + "；".join(rng)
      + f"。即只要失明期間不重規劃，失明的成本在這個範圍內大致消失；這同時說明增益不是「n 變大」造成：fixed n=100 在 L=50 黑畫面只有 {fcell(100, 50, 'black').success.mean():.2f}、凍結 {fcell(100, 50, 'freeze').success.mean():.2f}（主表）。")
    W("- oracle 在 L=100 並不是「完全不吃壞影像」：chunk（100 步）撐不過整段失明時被迫重規劃（forced），這時用的是過期影像；L=100 時 oracle 成功率明顯低於 L ≤ 50（見 4.1 表），與此一致，但無法排除 L=100 本身更難。")
    W()
    W("**限制**")
    W()
    W("- oracle 與 fixed 用不同執行迴圈（見檔頭說明）、各帶重跑雜訊；差距小於 0.05 的格不可解讀。")
    W("- k100 分層依 t0 相位，t0 也決定失明落在任務哪一階段，子集間任務難度並不相同。")
    W("- 沒有 oracle(n=100)（最直接的「相同 n、只改是否吃壞影像」對照）；k100=0 子集的 fixed-100 是最接近的代用品，但它的子集是事後依 t0 選出，不是隨機分派。")
    W("- forced 比較只能在 L=100 內做、為 unpaired，且受截斷影響（見 4.3）。")
    W()
    return famX, famY, famS


# ---------------------------------------------------------------- Section 5: reward stages
STAGE = {0: "no contact", 1: "right gripper touched cube", 2: "lifted (right)", 3: "left touched (attempt)", 4: "transferred (success)"}


def sec5():
    W("## 5. 獎勵分級：失明讓哪個階段失敗")
    W()
    W("gym-aloha 的 max_reward（`gym_aloha/tasks/sim.py` TransferCubeTask）：0 = 沒碰到方塊；1 = 右夾爪碰到；2 = 右夾爪舉起（方塊離桌）；3 = 左夾爪碰到（嘗試交接）；4 = 左夾爪持住且離桌（成功）。"
      f"三個資料集中 max_reward=3 的 episode 數：full_grid {int((F.max_reward == 3).sum())}、oracle {int((O.max_reward == 3).sum())}、proprio {int((P.max_reward == 3).sum())}"
      "（階段 3 只出現在交接當步，之後隨即成功，不會停在 3）。因此失敗只可能停在 0、1、2。")
    W()
    W("### 5.1 各格 max_reward 分布（次數 / 100）")
    W()
    rows = []
    for nl in NS:
        for mode in MODES:
            for L in LS:
                if L == 0 and mode == "freeze":
                    continue
                g = fcell(nl, L, mode)
                c = g.max_reward.value_counts()
                rows.append([nl, "-" if L == 0 else mode, L] + [int(c.get(float(r), 0)) for r in range(5)] + [f"{g.max_reward.mean():.2f}"])
    W(md_table(["n", "mode", "L", "r=0", "r=1", "r=2", "r=3", "r=4", "mean r"], rows))
    W()
    W("### 5.2 停在哪一階段（合併 n ∈ {25, 50, 100}，各 300 個 episode）")
    W()
    pool = F[(F.te == 0) & (F.n.isin([25, 50, 100]))]
    rows = []
    shares = {}
    for mode in MODES:
        for L in LS:
            if L == 0 and mode == "freeze":
                continue
            g = pool[(pool.L == L) & (pool["mode"] == "black")] if L == 0 else pool[(pool.L == L) & (pool["mode"] == mode)]
            N = len(g)
            c = g.max_reward.value_counts()
            sh = {r: c.get(float(r), 0) / N for r in range(5)}
            shares[(mode, L)] = sh
            reach1 = (g.max_reward >= 1).mean()
            reach2 = (g.max_reward >= 2).mean()
            p21 = ((g.max_reward >= 2).sum() / (g.max_reward >= 1).sum()) if (g.max_reward >= 1).sum() else float("nan")
            p42 = ((g.max_reward == 4).sum() / (g.max_reward >= 2).sum()) if (g.max_reward >= 2).sum() else float("nan")
            rows.append(["-" if L == 0 else mode, L, N] + [f"{sh[r]:.2f}" for r in (0, 1, 2, 4)] + [f"{reach1:.2f}", f"{p21:.2f}", f"{p42:.2f}"])
    W(md_table(["mode", "L", "N", "stop r=0", "stop r=1", "stop r=2", "reach 4 (success)", "P(≥1)", "P(≥2 given ≥1)", "P(=4 given ≥2)"], rows))
    W()
    W("### 5.3 black 對 freeze 的階段比較（配對 McNemar；指標＝max_reward 恰為該階段）")
    W()
    famR = []
    for L in (25, 50, 100):
        for n in (25, 50, 100):
            bl, fr = fcell(n, L, "black"), fcell(n, L, "freeze")
            m = bl.merge(fr, on="episode", suffixes=("_b", "_f"))
            for r in (0, 1, 2, 4):
                ib = (m.max_reward_b == r).to_numpy()
                iff = (m.max_reward_f == r).to_numpy()
                b = int((ib & ~iff).sum()); c = int((~ib & iff).sum())
                rec = dict(L=L, n=n, r=r, sb=int(ib.sum()), sf=int(iff.sum()), b=b, c=c, p=exact_mcnemar(b, c), delta=(b - c) / len(m))
                famR.append(rec)
    ad = holm([r["p"] for r in famR])
    for r, a in zip(famR, ad):
        r["p_holm"] = a
        r["verdict"] = verdict(r["delta"], r["p"], a)
        FLAG["n"] += 1
        FLAG["flagged"] += abs(r["delta"]) < NOISE
    W(md_table(["L", "n", "stage (max_reward=)", "black count", "freeze count", "Δ share (black−freeze)", "b / c", "p raw", f"p Holm({len(famR)})", "判讀"],
               [[r["L"], r["n"], r["r"], r["sb"], r["sf"], f"{r['delta']:+.2f}", f"{r['b']}/{r['c']}", fp(r["p"]), fp(r["p_holm"]), r["verdict"]] for r in famR]))
    W()
    W("**事實**")
    W()
    for mode in MODES:
        for L in (50, 100):
            s0, s = shares[("black", 0)], shares[(mode, L)]
            inc = {r: s[r] - s0[r] for r in (0, 1, 2)}
            top = max(inc, key=inc.get)
            W(f"- {mode}、L={L}（相對 L=0，n=25/50/100 合併）：停在 r=0 {s0[0]:.2f}→{s[0]:.2f}、r=1 {s0[1]:.2f}→{s[1]:.2f}、r=2 {s0[2]:.2f}→{s[2]:.2f}、成功 {s0[4]:.2f}→{s[4]:.2f}；增加最多的失敗階段為 r={top}（{STAGE[top]}，{inc[top]:+.2f}）。")
    W()
    tops = {(m, L): max((0, 1, 2), key=lambda r: shares[(m, L)][r] - shares[("black", 0)][r]) for m in MODES for L in (50, 100)}
    all2 = all(v == 2 for v in tops.values())
    W("**推論**：" + ("四個條件下增加最多的失敗階段都是 r=2（右手已舉起方塊、但沒有完成交接）" if all2 else "增加最多的失敗階段在不同條件下不同（見上列）")
      + f"；黑畫面 L=50 時 r=0（完全沒碰到）只由 {shares[('black', 0)][0]:.2f} 增到 {shares[('black', 50)][0]:.2f}。"
      "這與「失明期間的壞影像重規劃破壞了舉起之後的交接動作」一致，但本資料只有 episode 層級的 max_reward，沒有逐步事件，不能指出失敗在哪一步；階段也依賴任務順序（舉起一定先於交接），「失敗停在 r=2」包含所有沒走到交接的情況。")
    W()
    # review R1 M2: the shares above use all episodes as denominator; also give the shares among failures only
    W("### 5.4 以失敗 episode 為分母（審查第一輪 M2）")
    W()
    W("5.2 的比例分母是全部 episode（每組 300 個）。下表改以失敗 episode 為分母：失敗之中停在各階段的比例。")
    W()
    rows = []
    FS = {}
    for mode in MODES:
        for L in LS:
            if L == 0 and mode == "freeze":
                continue
            g = pool[(pool.L == L) & (pool["mode"] == "black")] if L == 0 else pool[(pool.L == L) & (pool["mode"] == mode)]
            fl = g[g.max_reward < 4]
            sh = {r: float((fl.max_reward == r).mean()) for r in (0, 1, 2)}
            FS[(mode, L)] = (len(fl), len(g), sh)
            rows.append(["-" if L == 0 else mode, L, len(g), len(fl)] + [f"{sh[r]:.2f}" for r in (0, 1, 2)])
    W(md_table(["mode", "L", "N (all)", "N (failures)", "failures stopping r=0", "r=1", "r=2"], rows))
    W()
    f0, f50 = FS[("black", 0)], FS[("black", 50)]
    W(f"- black L=50：全部 episode 中停在 r=2 的比例 {shares[('black', 0)][2]:.2f} → {shares[('black', 50)][2]:.2f}（分母 {f0[1]}），r=0 {shares[('black', 0)][0]:.2f} → {shares[('black', 50)][0]:.2f}；"
      f"只看失敗 episode，停在 r=2 的比例 {f0[2][2]:.2f} → {f50[2][2]:.2f}（失敗數 {f0[0]} → {f50[0]}），r=0 {f0[2][0]:.2f} → {f50[2][0]:.2f}。"
      "所以失明增加的失敗仍以「舉起後沒交接」為主（失敗中停在 r=2 的比例幾乎沒變），只是沒碰到方塊的失敗比例也上升。")
    n1 = {L: fcell("1", L, "black") for L in LS}
    r2 = {L: int((n1[L].max_reward == 2).sum()) for L in LS}
    W(f"- n=1（black）：停在 r=2 的 episode 數 L=0 為 {r2[0]}/100，L=10..100 為 " + "、".join(f"{r2[L]}/100" for L in LB)
      + f"（範圍 {min(r2[L] for L in LB)}–{max(r2[L] for L in LB)}）；成功仍為 0。壞影像讓 n=1 較常舉起方塊，與「停滯被壞影像部分打破」一致（推論）。")
    W()


# ---------------------------------------------------------------- Section 6/7: noise
def sec7():
    W("## 7. 重跑雜訊與可解讀性")
    W()
    W("依 docs/rulers.md：同設定重跑成敗約 1% 翻轉，單格成功數最大差 3/50，故**差距 < 0.05 一律標「不可解讀」**；差距 < 0.10 且 p > 0.05 為「無差異」。"
      f"本報告的配對比較表（3.1–3.3、4.1、4.2、5.3）共 {FLAG['n']} 個檢定，其中 |Δ|<0.05 者 {FLAG['flagged']} 個，已在判讀欄標出，不應據此下結論。"
      "第 1 節與圖中的單格成功率各帶 Wilson 95% CI（n=100 時半寬約 ±0.08–0.10）。")
    W()


# ------------------------------------------------------------------ figures
PAL = {"1": "#9a9a9a", "10": "#4c78a8", "25": "#f58518", "50": "#54a24b", "100": "#b279a2", "TE": "#e45756"}
plt.rcParams.update({"font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9, "legend.fontsize": 7.5, "figure.dpi": 100,
                     "axes.spines.top": False, "axes.spines.right": False, "font.family": "DejaVu Sans"})


def save(fig, name):
    fig.savefig(FIGS / f"{name}.png", dpi=300, bbox_inches="tight")
    fig.savefig(FIGS / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def fig_heatmap():
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.2), sharey=True)
    for ax, mode in zip(axes, MODES):
        M = np.array([[fcell(nl, L, mode).success.mean() for L in LS] for nl in NS])
        im = ax.imshow(M, vmin=0, vmax=1, cmap="YlGnBu", aspect="auto")
        for i in range(len(NS)):
            for j in range(len(LS)):
                ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", color="white" if M[i, j] > 0.55 else "black", fontsize=8)
        ax.set_xticks(range(len(LS)), [str(L) for L in LS])
        ax.set_yticks(range(len(NS)), ["1+TE" if nl == "TE" else str(nl) for nl in NS])
        ax.set_xlabel("Blackout length L (steps)")
        ax.set_title(f"{mode.capitalize()} blackout")
        ax.spines[:].set_visible(False)
    axes[0].set_ylabel("Execution steps n")
    cb = fig.colorbar(im, ax=axes, shrink=0.85, pad=0.02)
    cb.set_label("Success rate")
    save(fig, "final_heatmap_n_by_L")


def fig_curves():
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.3), sharey=True)
    for ax, mode in zip(axes, MODES):
        for nl in NS:
            ys = [fcell(nl, L, mode).success.mean() for L in LS]
            ax.plot(LS, ys, marker="o", ms=3.5, lw=1.4, color=PAL[nl], label="n=1+TE" if nl == "TE" else f"n={nl}", ls=":" if nl == "1" else "-")
        for n in (10, 25, 50):
            ys = [fcell(n, 0, mode).success.mean()] + [OD[(n, L, mode)].success.mean() for L in LB]
            ax.plot(LS, ys, marker="s", ms=3, lw=1.1, ls="--", color=PAL[str(n)], label=f"oracle n={n}", alpha=0.9)
        for m, mk in [(1, "^"), (10, "v")]:
            xs = [0, 25, 50, 100]
            ys = [PD[(m, L, "black" if L == 0 else mode)].success.mean() for L in xs]
            ax.plot(xs, ys, marker=mk, ms=4.5, lw=0, color="black", mfc="none", label=f"proprio m={m}")
        ax.set_xlabel("Blackout length L (steps)")
        ax.set_title(f"{mode.capitalize()} blackout")
        ax.set_ylim(-0.02, 1.02)
        ax.set_xticks(LS)
        ax.grid(alpha=0.25, lw=0.5)
    axes[0].set_ylabel("Success rate")
    axes[1].legend(loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=False, ncol=1)
    save(fig, "final_success_vs_L")


def fig_k(D0):
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.2), sharey=True)
    ks = sorted(D0.k.unique())
    pos = {k: i for i, k in enumerate(ks)}
    colors = {10: "#4c78a8", 25: "#f58518", 50: "#54a24b", 100: "#b279a2"}
    for ax, mode in zip(axes, MODES):
        d = D0[D0["mode"] == mode]
        for L in LB:
            dl = d[d.L == L]
            dl = dl[(dl.k > 0) | (dl.ks == 0)]   # k=0 from early-success censoring (k_sched>=1) is dropped here
            g = dl.groupby("k").success.agg(["sum", "count"])
            g = g[g["count"] >= 10]
            ax.plot([pos[k] for k in g.index], (g["sum"] / g["count"]).values, marker="o", ms=3, lw=0.9, color=colors[L], alpha=0.75, label=f"L={L}")
        g = d.groupby("k").success.agg(["sum", "count"])
        x = [pos[k] for k in g.index]
        y = (g["sum"] / g["count"]).values
        ci = np.array([wilson(int(s), int(c)) for s, c in zip(g["sum"], g["count"])])
        ax.errorbar(x, y, yerr=[y - ci[:, 0], ci[:, 1] - y], fmt="o-", color="black", ms=5, lw=1.6, capsize=2.5, label="pooled (Wilson 95% CI)", zorder=5)
        for xi, c in zip(x, g["count"]):
            ax.text(xi, 1.045, f"{int(c)}", ha="center", va="bottom", fontsize=6.5, color="#555")
        ax.set_xticks(range(len(ks)), [str(k) for k in ks])
        ax.set_xlabel("Replans during blackout, k")
        ax.set_title(f"{mode.capitalize()} blackout", pad=12)
        ax.set_ylim(-0.03, 1.03)
        ax.grid(alpha=0.25, lw=0.5)
    axes[0].set_ylabel("Success rate")
    axes[1].legend(loc="upper right", frameon=False, fontsize=7)
    save(fig, "final_success_vs_k")


def fig_nat(cells):
    fig, axes = plt.subplots(1, 2, figsize=(5.6, 2.9), sharey=True)
    for ax, n in zip(axes, (100, 50)):
        xs, labels = [], []
        i = 0
        for L in LB:
            for mode, col in zip(MODES, ["#333333", "#7fb7e6"]):
                for c in cells:
                    if c[0] == n and c[1] == L and c[2] == mode and len(c[5]) > 0:   # skip cells whose k=0 group is only censored successes
                        a, b = c[3], c[4]
                        for j, (g, hatch) in enumerate([(a, ""), (b, "//")]):
                            s, N = int(g.success.sum()), len(g)
                            lo, hi = wilson(s, N)
                            x = i + j * 0.38
                            ax.bar(x, s / N, width=0.34, color=col, hatch=hatch, edgecolor="white", lw=0.5)
                            ax.errorbar(x, s / N, yerr=[[s / N - lo], [hi - s / N]], color="black", lw=0.8, capsize=1.5)
                            ax.text(x, hi + 0.015, f"{N}", ha="center", fontsize=6.5, color="#555")
                        xs.append(i + 0.19)
                        labels.append(f"{L}\n{mode[:2]}")
                        i += 1
            i += 0.4
        ax.set_xticks(xs, labels, fontsize=8.5)
        ax.tick_params(axis="y", labelsize=8.5)
        ax.set_title(f"Fixed n={n}", fontsize=9.5)
        ax.set_ylim(0, 1.12)
        ax.set_xlabel("L; bl=black, fr=freeze", fontsize=8.5)
    axes[0].set_ylabel("Success rate", fontsize=9)
    from matplotlib.patches import Patch
    fig.legend(handles=[Patch(fc="#777", ec="white", label="k = 0"), Patch(fc="#777", ec="white", hatch="//", label="k ≥ 1")], frameon=False, loc="lower center", ncol=2, fontsize=8.5, bbox_to_anchor=(0.5, -0.30), title="Bar labels: number of episodes", title_fontsize=8)
    save(fig, "final_natural_experiment_k")


def fig_reward():
    cols = ["#7a1f1f", "#d9822b", "#e8c547", "#8fc1a9", "#2e7d4f"]
    ns = [10, 25, 50, 100]
    fig, axes = plt.subplots(2, 4, figsize=(7.6, 4.4), sharey=True)
    for i, mode in enumerate(MODES):
        for j, n in enumerate(ns):
            ax = axes[i, j]
            bottoms = np.zeros(len(LS))
            for r in range(5):
                if r == 3:
                    continue
                v = np.array([(fcell(n, L, mode).max_reward == r).mean() for L in LS])
                ax.bar(range(len(LS)), v, bottom=bottoms, color=cols[r], width=0.8, label=STAGE[r], edgecolor="white", lw=0.4)
                bottoms += v
            ax.set_xticks(range(len(LS)), [str(L) for L in LS], fontsize=7)
            ax.set_title(f"{mode.capitalize()}, n={n}", fontsize=8)
            if j == 0:
                ax.set_ylabel("Share of episodes")
            if i == 1:
                ax.set_xlabel("L (steps)")
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.04), fontsize=7)
    fig.subplots_adjust(hspace=0.35, wspace=0.12)
    save(fig, "final_reward_stages")


# ------------------------------------------------------------------ main
def main():
    W("# T5 最終分析（`code/analyze_final.py` 自動產生，請勿手改）")
    W()
    W(f"資料：`results/full_grid.csv`（{len(F)} 行）、`results/oracle.csv`（{len(O)} 行）、`results/proprio.csv`（{len(P)} 行，主實驗 {len(PM)} 行）。"
      "政策、環境與評估尺見 `docs/rulers.md`；每格 100 個配對初始條件（env seed 0–99，失明起點 t0 由 seed 決定，跨格相同）。"
      "全部數字由腳本產生；英文圖檔在 `results/figs/final_*.png|pdf`（300 dpi）。")
    W()
    sec1()
    sec2()
    sec3()
    sec4()
    sec5()
    W("## 6. 圖")
    W()
    W("- `results/figs/final_heatmap_n_by_L.png`：n × L 成功率熱圖（black、freeze 各一；L=0 兩者相同）。")
    W("- `results/figs/final_success_vs_L.png`：成功率對 L（各 n、oracle 虛線、proprio 空心符號）。")
    D1 = F[(F.te == 0) & (F.n >= 10) & (F.L > 0)]
    cens_n = int(((D1.replans_in_blackout == 0) & (k_sched(D1, D1.n) > 0)).sum())
    W(f"- `results/figs/final_success_vs_k.png`：成功率對 k（分 mode，黑線為合併＋Wilson 95% CI，彩線為各 L 分層，頂端數字為樣本數）。紀錄的 k=0 中有 {cens_n} 個 episode（兩個 mode 合計）是提早成功的截斷產物（k_sched≥1），合併點仍包含它們；彩線的 k=0 只用 k_sched=0 的 episode。")
    W("- `results/figs/final_natural_experiment_k.png`：固定 n=100、50 的同格 k=0 對 k≥1（k=0 只含截斷產物的格省略）。")
    W("- `results/figs/final_reward_stages.png`：max_reward 分布堆疊長條圖（black／freeze × n × L）。")
    W()
    sec7()
    W("## 8. 整體限制")
    W()
    W("- 單一 checkpoint、單一任務（Transfer Cube）、模擬環境；每格 100 個初始條件，CI 半寬 ±0.08–0.10。")
    W("- 格間共用同一批初始條件與 t0，格間比較已用配對或 cluster-robust SE，但不是獨立樣本；跨格合併的 p 值只作描述。")
    W("- 整條管線非位元級確定性（docs/rulers.md），oracle 與 proprio 另用自寫執行迴圈。")
    W("- k 與 t0 糾纏；oracle 是被告知失明區間的上限，不是可部署的方法。")
    (RES / "analysis.md").write_text("\n".join(MD) + "\n", encoding="utf-8")
    # figures
    D0 = F[(F.te == 0) & (F.n >= 10) & (F.L > 0)].copy()
    D0["k"] = D0.replans_in_blackout
    D0["ks"] = k_sched(D0, D0.n)
    fig_heatmap()
    fig_curves()
    fig_k(D0)
    fig_nat(REG["cells"])
    fig_reward()
    print("wrote", RES / "analysis.md", "and figures")


if __name__ == "__main__":
    main()
