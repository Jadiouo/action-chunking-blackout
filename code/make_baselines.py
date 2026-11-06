"""P3: literature-method comparison -> results/baselines.csv, results/baselines_table.md (python3, project root).
Methods: fixed n (full_grid.csv), Mean TE (= full_grid.csv '1+TE', ACT temporal ensembling coeff 0.01), oracle / simple-detect
(oracle.csv, detect.csv + new cells from run_p2p3.py), Median TE (h=1 main, h=10), CARE (calibrated tau, main + 2 sensitivity),
and on the P2 corruption also generic-detect.  All paired by env seed 0-99 (same t0).
Reference rows: R = success / the SAME method's clean (L=0) success; rho = (method - fixed n) / (clean fixed n - fixed n),
reported against fixed n=50 (best clean n) and fixed n=100 (ACT default), denominator < 0.10 -> n/r.
Holm family: every Median-TE / CARE(main) vs {fixed n=50, fixed n=100, Mean TE, oracle n=50, simple-detect n=50 (black/freeze) or
generic-detect n=50 (P2 corruption)} McNemar test in this file."""
import glob, json, os
import numpy as np, pandas as pd
from p2p3_stats import rate_ci, mcnemar, holm, fmt_R, fmt_rho, RERUN_NOISE

P = "results/parts/"
cal = json.load(open(P + "care_tau_calib.json")) if os.path.exists(P + "care_tau_calib.json") else {}
TAU = json.load(open(P + "care_tau_used.json")) if os.path.exists(P + "care_tau_used.json") else {}  # {"main": x, "low": x, "high": x}
sel = json.load(open(P + "p2p3_selected.json")) if os.path.exists(P + "p2p3_selected.json") else {}
P2C = sel.get("p3_corr")  # tag of the P2 corruption used for P3

rows = []  # unified long form
def add(df, **kw):
    for _, r in df.iterrows():
        rows.append(dict(episode=int(r.episode), env_seed=int(r.env_seed), t0=int(r.t0), success=int(r.success),
                         max_reward=float(r.max_reward), replans_total=int(r.replans_total), seconds=float(r.seconds), **kw))

g = pd.read_csv("results/full_grid.csv")
for (n, te, L, mode), d in g.groupby(["n", "te", "L", "mode"]):
    if te == 1: add(d, corr=mode, method="meanTE_h1" if L == 0 else "meanTE_grid_h1", L=L, src="full_grid")
    elif n in (25, 50, 100): add(d, corr=mode, method=f"fixed_n{n}" if L == 0 else f"fixed_grid_n{n}", L=L, src="full_grid")
for (n, L, mode), d in pd.read_csv("results/oracle.csv").groupby(["n", "L", "mode"]):
    if n in (25, 50): add(d, corr=mode, method=f"oracle_n{n}", L=L, src="oracle")
for (n, L, mode), d in pd.read_csv("results/detect.csv").groupby(["n", "L", "mode"]):
    add(d, corr=mode, method=f"detect_n{n}", L=L, src="detect")
# same-loop rows (run_p2p3.py): P3 cells, same-loop fixed rows (p3_fixed + regress100), and the P2 corruption used for P3
for f in sorted(glob.glob(P + "p2p3_p3_*.csv") + glob.glob(P + "p2p3_regress_black.csv") + glob.glob(P + "p2p3_regress_freeze.csv")
                + glob.glob(P + "p2p3_real_*.csv")):
    if "rungrid" in f: continue
    d = pd.read_csv(f)
    for (corr, rule, n, h, tau, L), dd in d.groupby(["corr", "rule", "n", "h", "tau", "L"], dropna=False):
        meth = {"fixed": f"fixed_n{n}", "oracle": f"oracle_n{n}", "detect": f"detect_n{n}", "generic": f"generic_n{n}",
                "median_te": f"medianTE_h{h}", "mean_te": f"meanTE_h{h}", "care": f"care_tau{float(tau):.4g}"}[rule]
        if "real_" in f and (P2C is None or corr != P2C): continue  # only the P3 corruption's rows enter baselines.csv
        add(dd, corr=corr, method=meth, L=L, src=os.path.basename(f))
D = pd.DataFrame(rows)
# L=0 rows belong to every corruption tag; (clean fixed / TE refs from full_grid; Median TE / CARE clean from their own L=0 cells)
D.loc[D.L == 0, "corr"] = "clean"
D = D.drop_duplicates(["corr", "method", "L", "episode", "src"])
D.sort_values(["corr", "method", "L", "episode"]).to_csv("results/baselines.csv", index=False)
NSLIM = {P2C: 90} if P2C else {}

def vec(corr, method, L):
    d = D[(D.method == method) & (D.L == L) & (D["corr"] == ("clean" if L == 0 else corr))].drop_duplicates("episode").sort_values("episode")
    d = d[d.episode < NSLIM.get(corr, 10**9)]
    return None if len(d) == 0 else d

def care_methods():
    return sorted({m for m in D.method.unique() if m.startswith("care_tau")}, key=lambda s: float(s[8:]))

out = []
W = out.append
W("# P3 文獻方法對照：Median TE、CARE 式觸發 vs fixed n／oracle／detect（Transfer Cube，seeds 0–99 配對）\n")
W("本檔由 `code/make_baselines.py` 從 `results/baselines.csv` 產生。方法與參數見 `docs/p2p3_notes.md`、`docs/baselines_spec.md`。"
  "R = 該方法失明成功率 / 該方法自己的乾淨（L=0）成功率；ρ = (方法 − fixed n)/(乾淨 fixed n − fixed n)，以 fixed n=50（乾淨最佳）與 fixed n=100（ACT 預設）為參照，分母 < 0.10 記 n/r。"
  f"McNemar 精確檢定，Holm 校正家族見腳本註解；差距 < {RERUN_NOISE} 標 noise，不做解讀。\n")
if TAU:
    W(f"CARE 的 τ（乾淨 seeds 100–119 校準，目標乾淨平均執行長度 0.6·H_max=30 為主）：main={TAU.get('main')}，low(0.4·H_max)={TAU.get('low')}，high(0.8·H_max)={TAU.get('high')}。\n")

comp_ref = ["fixed_n50", "fixed_n100", "meanTE_h1", "oracle_n50", "detect_n50"]
cells = [("black", L) for L in (25, 50, 100)] + [("freeze", L) for L in (25, 50, 100)]
if P2C: cells += [(P2C, L) for L in (50, 100)]
methods = ["fixed_n25", "fixed_n50", "fixed_n100", "fixed_grid_n25", "fixed_grid_n50", "fixed_grid_n100", "meanTE_h1", "meanTE_grid_h1", "oracle_n25", "oracle_n50", "oracle_n100", "detect_n25", "detect_n50", "detect_n100",
           "generic_n25", "generic_n50", "generic_n100", "medianTE_h1", "medianTE_h10"] + care_methods()
fam = []
for corr, L in cells:
    for m in ["medianTE_h1"] + ([f"care_tau{TAU['main']:.4g}"] if TAU else []):
        a = vec(corr, m, L)
        if a is None: continue
        for rname in comp_ref:
            r = rname if not (rname == "detect_n50" and corr == P2C) else "generic_n50"
            b = vec(corr, r, L)
            if b is None: continue
            x, y, p = mcnemar(a.success.values, b.success.values)
            fam.append(((corr, L, m, r), x, y, p))
adj = holm([f[3] for f in fam]) if fam else []
HP = {f[0]: (f[1], f[2], f[3], a_) for f, a_ in zip(fam, adj)}

W("## 成功率表（每格成功數/N = 率 [Wilson 95%]；R；ρ）\n")
W("R 的分母：n 規則（fixed／oracle／detect／generic）用同 n 的乾淨 fixed 成功率（full_grid.csv L=0）；TE／CARE 用自己的乾淨（L=0）成功率。"
  "ρ：n 規則用同 n 的 fixed（同格同 n）；Median／Mean TE 與 CARE 沒有 n，改對 fixed n=50（乾淨最佳）與 fixed n=100（ACT 預設）各算一個。"
  "`fixed_grid_*`／`meanTE_grid_*` 是 full_grid.csv（run_grid.py）的舊列，僅供對照、不做檢定（與同迴圈列逐 episode 約 6–14% 翻轉，見 p2p3_notes.md）。\n")
def nrule(m): return m.split("_")[0] in ("oracle", "detect", "generic")
for corr, L in cells:
    W(f"### {corr}, L={L}\n")
    W("| 方法 | 成功 | R | ρ（同 n fixed） | ρ vs fixed n=50 | ρ vs fixed n=100 | 乾淨（L=0）參照 | 查詢次數/episode |"); W("|---|---|---|---|---|---|---|---|")
    f50, f100 = vec(corr, "fixed_n50", L), vec(corr, "fixed_n100", L)
    for m in methods:
        a = vec(corr, m, L)
        if a is None: continue
        if m.startswith("fixed"): c0 = vec(corr, m.replace("_grid", ""), 0)
        elif nrule(m): c0 = vec(corr, "fixed_n" + m.split("_n")[1], 0)
        else: c0 = vec(corr, m.replace("_grid", ""), 0)
        R = fmt_R(a.success.values, c0.success.values) if c0 is not None else "-"
        same, r50, r100 = "-", "-", "-"
        if nrule(m):
            fn = vec(corr, "fixed_n" + m.split("_n")[1], L)
            if fn is not None and c0 is not None: same = fmt_rho(a.success.values, fn.success.values, c0.success.values)
        elif not m.startswith("fixed"):
            c50, c100 = vec(corr, "fixed_n50", 0), vec(corr, "fixed_n100", 0)
            if f50 is not None: r50 = fmt_rho(a.success.values, f50.success.values, c50.success.values)
            if f100 is not None: r100 = fmt_rho(a.success.values, f100.success.values, c100.success.values)
        W(f"| {m} | {rate_ci(a.success.values)} | {R} | {same} | {r50} | {r100} | {rate_ci(c0.success.values) if c0 is not None else '-'} | {a.replans_total.mean():.1f} |")
    W("")
    W("McNemar（Holm 校正；a 單贏／b 單贏）：" + "；".join(
        f"{k[2]} vs {k[3]}: {v[0]}/{v[1]} p_Holm={v[3]:.3g}" + (" (noise)" if abs((vec(corr, k[2], L).success.mean() - vec(corr, k[3], L).success.mean())) < RERUN_NOISE else "")
        for k, v in HP.items() if k[0] == corr and k[1] == L) + "\n")

# CARE diagnostics: realised mean H
W("## CARE 實際執行長度（clean 與失明下，平均每次查詢選的 H）\n")
W("| corr | L | " + " | ".join(care_methods()) + " |"); W("|---|---|" + "---|" * len(care_methods()))
cf = [pd.read_csv(f) for f in sorted(glob.glob(P + "p2p3_p3_care*.csv"))]
if cf:
    C = pd.concat(cf)
    for (corr, L), g_ in C.groupby(["corr", "L"]):
        W(f"| {'clean' if L == 0 else corr} | {L} | " + " | ".join(
            (lambda s: "-" if len(s) == 0 else f"H={pd.to_numeric(s.care_meanH).mean():.1f}, q={pd.to_numeric(s.care_queries).mean():.1f}")(
                g_[g_.tau.round(6) == round(float(m[8:]), 6)]) for m in care_methods()) + " |")
open("results/baselines_table.md", "w").write("\n".join(out) + "\n")
print("wrote results/baselines.csv", len(D), "rows; results/baselines_table.md")
