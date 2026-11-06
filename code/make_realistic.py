"""P2: merge results/parts/p2p3_pilot.csv + results/parts/p2p3_real_<corr>.csv -> results/realistic.csv, results/realistic_table.md.
Run with python3 from the project root. Pairing key = env_seed (same t0 / initial state in every cell, see run_p2p3.py).
Holm family: all rule-vs-fixed McNemar tests in the main table (all corruptions, n x L x {oracle, simple-detect, generic-detect}).
Differences |delta| < 0.05 are flagged 'noise' and not interpreted (docs/rulers.md re-run noise)."""
import glob, json, os
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
from p2p3_stats import rate_ci, mcnemar, holm, fmt_R, fmt_rho, RERUN_NOISE, wilson

P = "results/parts/"
RULES = ["fixed", "oracle", "detect", "generic"]
RNAME = {"fixed": "fixed", "oracle": "oracle", "detect": "simple-detect", "generic": "generic-detect"}
grid = pd.read_csv("results/full_grid.csv")
NS = 90  # main experiment seeds 0..89 (budget, see p2p3_selected.json); the clean reference uses the same seeds
clean = {n: grid[(grid.n == n) & (grid.L == 0) & (grid.te == 0) & (grid.episode < NS)].sort_values("episode").success.values for n in (25, 50, 100)}

pilot = pd.read_csv(P + "p2p3_pilot.csv"); pilot["stage"] = "pilot"
mains = {os.path.basename(f)[len("p2p3_real_"):-4]: pd.read_csv(f) for f in sorted(glob.glob(P + "p2p3_real_*.csv"))}
main = pd.concat(mains.values(), ignore_index=True) if mains else pd.DataFrame(); main["stage"] = "main"
pd.concat([pilot, main], ignore_index=True).to_csv("results/realistic.csv", index=False)
sel = json.load(open(P + "p2p3_selected.json")) if os.path.exists(P + "p2p3_selected.json") else {}

out = []
W = out.append
W("# P2 合成相機損壞（Transfer Cube）：gaussian noise、部分遮擋、延遲、間歇閃斷\n")
W("本檔由 `code/make_realistic.py` 從 `results/realistic.csv` 產生（程式 `code/run_p2p3.py`、`code/corrupt.py`；強度為我們自訂，見 `docs/p2p3_notes.md`）。"
  "損壞只作用在失明窗 [t0, t0+L)，窗外立即回到當前真影像；env seed 與 t0 與 T2 同（配對）。"
  "R = 失明下成功率 / 同 n 乾淨（full_grid.csv，L=0，同 seeds）成功率；ρ = (規則 − fixed)/(乾淨 − fixed)，分母 < 0.10 記 n/r（不報）；"
  "方括號為 seed 的 bootstrap 95% CI（4000 次）。ρ 後的 * 表示 bootstrap 有 >5% 的抽樣分母掉到 0.10 以下。"
  "McNemar 為精確二項檢定（規則單贏／fixed 單贏），p 為 Holm 校正後（家族＝本檔主表所有規則對 fixed 的比較）。"
  f"差距 < {RERUN_NOISE} 標 noise，不做解讀。\n")

# ------------------------------------------------------------------ pilot
W("## 1. 強度試跑（n=50、L=50、fixed、seeds 120–149 共 30 個；與評估 seeds 0–99、偵測器校準 seeds 100–119 不重疊）\n")
W("挑選規則（跑之前宣告）：在每種損壞的 3 個強度（遮擋為 3 面積 × 2 位置）中，取 fixed 成功率 / 乾淨成功率（R）最接近 0.5、且落在 [0.15, 0.85] 的強度；"
  "若沒有任何強度落在區間，該損壞記為「無害」（R > 0.85）或「過強」（R < 0.15），不進主實驗。\n")
W("| 損壞 | 強度 | fixed 成功 | R（相對乾淨 n=50） | 是否入選 |"); W("|---|---|---|---|---|")
pc = pilot[pilot.L == 0].sort_values("episode").success.values
W(f"| 乾淨（L=0） | - | {rate_ci(pc)} | 1.00 | - |")
for tg, g in pilot[pilot.L == 50].groupby("corr", sort=False):
    x = g.sort_values("episode").success.values
    W(f"| {tg} | | {rate_ci(x)} | {x.mean() / pc.mean():.2f} | {'**入選**' if tg in sel.get('chosen', []) else ''} |")
W("")
if sel.get("note"): W(sel["note"] + "\n")

# ------------------------------------------------------------------ main
S = {}  # (corr, rule, n, L) -> success vector (by episode order = env seed)
for (c, r, n, L), g in main.groupby(["corr", "rule", "n", "L"]):
    S[(c, r, n, L)] = g.sort_values("episode")
tests = []  # (key, a, b)
for c in mains:
    pass
W("## 2. 主實驗：fixed / oracle / simple-detect / generic-detect（每格 %d seeds，seeds 0–%d）" % (NS, NS - 1) + "\n")
rows_for_holm = []
for tgf, df in mains.items():
    tg = df["corr"].iloc[0]
    for L in sorted(df.L.unique()):
        for n in sorted(df.n.unique()):
            f = S.get((tg, "fixed", n, L))
            if f is None: continue
            for r in ("oracle", "detect", "generic"):
                g = S.get((tg, r, n, L))
                if g is None: continue
                assert (g.env_seed.values == f.env_seed.values).all() and (g.t0.values == f.t0.values).all()
                x, y, p = mcnemar(g.success.values, f.success.values)
                rows_for_holm.append(((tg, L, n, r), x, y, p))
adj = holm([v[3] for v in rows_for_holm]) if rows_for_holm else []
HP = {v[0]: (v[1], v[2], v[3], a) for v, a in zip(rows_for_holm, adj)}
for tgf, df in mains.items():
    tg = df["corr"].iloc[0]
    W(f"### {tg}\n")
    W("| L | n | 乾淨 | fixed | oracle | simple-detect | generic-detect |"); W("|---|---|---|---|---|---|---|")
    for L in sorted(df.L.unique()):
        for n in sorted(df.n.unique()):
            f = S.get((tg, "fixed", n, L))
            if f is None: continue
            cells = [rate_ci(clean[n]), rate_ci(f.success.values) + f" R={fmt_R(f.success.values, clean[n])}"]
            for r in ("oracle", "detect", "generic"):
                g = S.get((tg, r, n, L))
                if g is None: cells.append("-"); continue
                d = g.success.mean() - f.success.mean()
                x, y, p, pa = HP[(tg, L, n, r)]
                flag = " noise" if abs(d) < RERUN_NOISE else ""
                cells.append(f"{rate_ci(g.success.values)}<br>R={fmt_R(g.success.values, clean[n])}<br>ρ={fmt_rho(g.success.values, f.success.values, clean[n])}"
                             f"<br>Δ={d:+.2f}{flag}; {x}/{y}; p_Holm={pa:.3g}")
            W(f"| {L} | {n} | " + " | ".join(cells) + " |")
    W("")
W("格內：成功數/N = 率 [Wilson 95%]；Δ = 規則 − fixed；x/y = 規則單贏／fixed 單贏的 episode 數。\n")

# ------------------------------------------------------------------ detector tables
W("## 3. 偵測器表\n")
W("### 3a. 主實驗內的偵測率與誤報（偵測器各 episode 全程開著；真失明窗只用於記錄）\n")
W("episodes detected = 窗內至少偵測一步的 episode 比例；det/steps in = 窗內被判為壞的步數比例；false = 窗外被判為壞的步數比例（窗外影像全為真影像）；"
  "latency = 窗內第一次偵測步 − t0；forced = 窗內因 chunk 用完被迫重規劃的平均次數。\n")
W("| 損壞 | L | n | 偵測器 | episodes detected | det/steps in | episodes with false | false steps / steps out | latency mean | forced replans in window |")
W("|---|---|---|---|---|---|---|---|---|---|")
for tgf, df in mains.items():
    tg = df["corr"].iloc[0]
    for L in sorted(df.L.unique()):
        for n in sorted(df.n.unique()):
            for r in ("detect", "generic"):
                g = S.get((tg, r, n, L))
                if g is None: continue
                ed = (g.det_in > 0).mean(); frac_in = g.det_in.sum() / max(g.steps_in.sum(), 1)
                ef = (g.det_out > 0).mean(); frac_out = g.det_out.sum() / max(g.steps_out.sum(), 1)
                lat = pd.to_numeric(g.latency, errors="coerce")
                W(f"| {tg} | {L} | {n} | {RNAME[r]} | {ed:.2f} | {frac_in:.3f} | {ef:.2f} | {frac_out:.4f} | "
                  f"{'-' if lat.notna().sum() == 0 else f'{lat.mean():.1f}'} | {g.forced_replans_in_window.mean():.2f} |")
W("")

W("### 3b. 以「對策略有害」為標籤的偵測率與誤報率（診斷 episode：fixed n=50、L=50、seeds 120–149）\n")
W("有害標籤（每個時間步）：以損壞影像與同一步真影像各預測一個 chunk，取前 50 個動作的平均 L2 距離 D；門檻 = 乾淨影像加 ±1 灰階隨機抖動（約等於 docs/rulers.md 的渲染重跑雜訊）"
  "所造成的 D 之 99 百分位（取自所有診斷 episode 的窗外步），D > 門檻 → 有害。偵測率 = 有害步中偵測器報警的比例；"
  "窗內無害誤報 = 窗內無害步中報警的比例；乾淨誤報 = 窗外步中報警的比例。generic 的 AUC 以分數對有害標籤計算（窗內步）。\n")
W("| 損壞 | 窗內有害步比例 | simple 偵測率 | generic 偵測率 | generic AUC | simple 窗內無害誤報 | generic 窗內無害誤報 | simple 乾淨誤報 | generic 乾淨誤報 |")
W("|---|---|---|---|---|---|---|---|---|")
dirs = sorted(glob.glob(P + "p2p3_diag_*"))
recs = {}
for d in dirs:
    arrs = [np.load(f) for f in sorted(glob.glob(d + "/L50_*.npy"))]
    if arrs: recs[os.path.basename(d)[len("p2p3_diag_"):]] = np.concatenate(arrs)
if recs:
    jit = np.concatenate([a[a[:, 1] == 0][:, 3] for a in recs.values()])
    thr_h = np.percentile(jit, 99)
    gthr = float(np.load(P + "generic_detector.npz")["thr"])
    for tg, a in recs.items():
        win = a[:, 1] == 1; harm = a[:, 2] > thr_h
        hw = harm[win]; sd = a[:, 5] == 1; gd = a[:, 4] > gthr
        def rt(mask_sel, flag): return f"{flag[mask_sel].mean():.3f}" if mask_sel.sum() else "-"
        auc = roc_auc_score(hw, a[win][:, 4]) if 0 < hw.sum() < len(hw) else float("nan")
        W(f"| {tg} | {hw.mean():.2f} | {rt(win & harm, sd)} | {rt(win & harm, gd)} | {"-" if np.isnan(auc) else f"{auc:.3f}"} | {rt(win & ~harm, sd)} | {rt(win & ~harm, gd)} | {rt(~win, sd)} | {rt(~win, gd)} |")
    W(f"\n有害門檻 D* = {thr_h:.4f}（動作單位，弧度）；generic 門檻 = {gthr:.3f}（乾淨 seeds 110–119 分數的 99 百分位，`results/parts/generic_detector.json`）。\n")
open("results/realistic_table.md", "w").write("\n".join(out) + "\n")
print("wrote results/realistic.csv", len(pd.concat([pilot, main])), "rows; results/realistic_table.md")
