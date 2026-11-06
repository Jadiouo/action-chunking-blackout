"""P1: merge insertion parts -> results/insertion.csv and results/insertion_table.md. Run with python3 from project root.
Parts (code/run_insertion.sh): ins_noblack(_te), ins_fixed_black/freeze (run_grid.py), ins_oracle, ins_detect (run_t3t4.py).
Columns: task, rule (fixed|te|oracle|detect) + oracle.csv / detect.csv columns."""
import math
import numpy as np, pandas as pd
from scipy.stats import binomtest, fisher_exact

P = "results/parts/"
ref_cols = list(pd.read_csv("results/detect.csv", nrows=1).columns)
parts = {"fixed": ["ins_noblack", "ins_fixed_black", "ins_fixed_freeze"], "te": ["ins_noblack_te"],
         "oracle": ["ins_oracle"], "detect": ["ins_detect"]}
dfs = []
for rule, names in parts.items():
    for nm in names:
        d = pd.read_csv(P + nm + ".csv"); d["rule"] = rule; dfs.append(d)
D = pd.concat(dfs, ignore_index=True)
D["task"] = "insertion"
D["te"] = D["te"].fillna(0).astype(int) if "te" in D else 0
D.loc[D.rule == "te", "te"] = 1
D.loc[D.rule == "te", "n"] = 1
D.loc[D.rule == "te", "config"] = D.loc[D.rule == "te", "config"]  # already TE_L0_black
D["nl"] = np.where(D.te == 1, "TE", D.n.astype(str))
for c in ref_cols:
    if c not in D: D[c] = np.nan
D = D.sort_values(["rule", "mode", "L", "nl", "episode"]).reset_index(drop=True)
D[["task", "rule"] + ref_cols].to_csv("results/insertion.csv", index=False)

NS = 200  # seeds per cell (0..199)
# every cell must have exactly NS episodes
cnt = D.groupby(["rule", "nl", "L", "mode"]).episode.agg(["count", "min", "max"])
assert (cnt["count"] == NS).all() and (cnt["min"] == 0).all() and (cnt["max"] == NS - 1).all(), cnt[cnt["count"] != NS]
assert (D.success == (D.max_reward == 4)).all()
ncell = len(cnt)


def wilson(k, n, z=1.96):
    p = k / n; d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d; h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def rc(k, n):
    lo, hi = wilson(k, n); return f"{k}/{n} = {k / n:.2f} [{lo:.2f}, {hi:.2f}]"


def cell(rule, nl, L, mode):
    m = "black" if L == 0 else mode
    g = D[(D.rule == rule) & (D.nl == str(nl)) & (D.L == L) & (D["mode"] == m)]
    return g.sort_values("episode").reset_index(drop=True)


def holm(ps):
    o = np.argsort(ps); m = len(ps); adj = np.empty(m); run = 0.0
    for r, i in enumerate(o):
        run = max(run, (m - r) * ps[i]); adj[i] = min(1.0, run)
    return adj


out = []
W = out.append
W("# P1 第二任務 AlohaInsertion（gym_aloha/AlohaInsertion-v0，ACT `lerobot/act_aloha_sim_insertion_human`）\n")
W("本檔由 `code/make_insertion_table.py` 從 `results/insertion.csv`（%d 列 = %d 格 × 200）產生。" % (len(D), ncell))
W("設定：episode 長度 400 步（lerobot AlohaEnv 預設值，與 Transfer Cube 相同；gym_aloha 註冊的 300 被 `max_episode_steps=400` 覆蓋，同 T2）；"
  "成功 = max reward 4（insertion 的獎勵階段亦為 0–4：1 抓到 peg/socket、2 兩者離桌、3 接觸、4 插入）；"
  "失明起點 t0 ~ U{50..250}（episode 同為 400 步，沿用），env seed = episode（0–199）、t0 由 seed 決定，跨格、跨規則配對（與 T2 相同的程式）。"
  "權重內無 processor 檔，正規化用 safetensors 內的 mean/std buffer（同 Transfer Cube）。執行：`code/run_insertion.sh`。\n")

# ---------------- A. no blindness
W("## A. 無失明（L=0）n ∈ {1,10,25,50,100} 與 TE，各 200 episodes\n")
rowsA = ["| 設定 | 成功 | 成功率 [Wilson 95%] | max_reward 分布 0/1/2/3/4 | 平均 max_reward | 平均步數 | 平均重規劃次數 |", "|---|---|---|---|---|---|---|"]
for nl in ["1", "10", "25", "50", "100", "TE"]:
    g = cell("te" if nl == "TE" else "fixed", nl, 0, "black")
    dist = "/".join(str(int((g.max_reward.round() == r).sum())) for r in range(5))
    rowsA.append(f"| {'TE (coeff 0.01)' if nl == 'TE' else 'n=' + nl} | {int(g.success.sum())}/{NS} | {rc(int(g.success.sum()), NS)} | {dist} | {g.max_reward.mean():.2f} | {g.steps.mean():.0f} | {g.replans_total.mean():.1f} |")
W("\n".join(rowsA)); W("")
g100 = cell("fixed", 100, 0, "black"); k100 = int(g100.success.sum()); lo, hi = wilson(k100, NS)
b = binomtest(k100, NS, 0.206)
W("**與模型卡比較**：模型卡（`lerobot/act_aloha_sim_insertion_human` README）以 n_action_steps=100 評估 500 episodes，成功率 20.6%（原論文 ACT 20%）。"
  f"本實驗 n=100：{k100}/{NS} = {k100 / NS:.2f}，Wilson 95% CI [{lo:.2f}, {hi:.2f}]；差距 {k100 / NS - 0.206:+.3f}（對 20.6% 的精確二項檢定 p = {b.pvalue:.2f}）。"
  "另：同 repo 的 `eval_info.json` 列出 500 episodes 的 pc_success = 83.0%（avg_max_reward 3.546，415/500 達 reward 4），與 README 的 20.6% 互相矛盾；"
  "README 自述「成功在第二隻手夾爪與 peg 反向接觸時即登記」，原因不明，我們不採信其中任一個，只以本檔的實測為準。\n")

# ---------------- B. blindness cells
W("## B. 失明格成功率（每格 200 episodes；L=0 欄為同 n 無失明參照；R 見 E 節）\n")
rowsB = ["| mode | L | n | L=0 (同 n) | fixed | oracle | detect |", "|---|---|---|---|---|---|---|"]
for mode in ["black", "freeze"]:
    for L in [50, 100]:
        for n in [25, 50, 100]:
            f = cell("fixed", n, L, mode)
            l0 = cell("fixed", n, 0, "black")
            o = cell("oracle", n, L, mode) if n in (25, 50) else None
            d = cell("detect", n, L, mode) if n in (25, 50) else None
            rowsB.append(f"| {mode} | {L} | {n} | {rc(int(l0.success.sum()), NS)} | {rc(int(f.success.sum()), NS)} | "
                         + (rc(int(o.success.sum()), NS) if o is not None else "-") + " | " + (rc(int(d.success.sum()), NS) if d is not None else "-") + " |")
W("\n".join(rowsB)); W("")

# ---------------- C. natural experiment k=0 vs k>=1 (fixed rule)
W("## C. 自然實驗：同一格（同 n、L、mode）內，失明窗口內重規劃次數 k=0 對 k≥1\n")
W("固定 n 的重規劃發生在 t = 0, n, 2n, …，故排程 k_sched = ceil((t0+L)/n) − ceil(t0/n) 只由 t0 對 chunk 邊界的相位決定，不受成敗截斷影響（成功即結束 episode 會讓紀錄的 k 偏小，所以用 k_sched）。"
  "兩組是同格內不同 episode 子集（無配對），Fisher 精確檢定。**限制**：k 與失明落在任務的哪個階段（t0）糾纏。n=25、50 或 L=100 時失明窗口必跨過至少一個 chunk 邊界（k_sched≥1 對全部 episode），沒有 k=0 的對照組，只有 n=100、L=50 能做格內自然實驗。\n")
rowsC = ["| mode | L | n | k_sched=0: 成功/N = 率 [CI] | k_sched≥1: 成功/N = 率 [CI] | Δ (k0−k≥1) | Fisher p |", "|---|---|---|---|---|---|---|"]
pool = {m: [0, 0, 0, 0] for m in ["black", "freeze"]}
for mode in ["black", "freeze"]:
    for L in [50, 100]:
        for n in [100, 50, 25]:
            g = cell("fixed", n, L, mode)
            ks = (np.ceil((g.t0 + g.L) / n) - np.ceil(g.t0 / n)).astype(int)
            a, bb = g[ks == 0], g[ks >= 1]
            if len(a) == 0 or len(bb) == 0:
                rowsC.append(f"| {mode} | {L} | {n} | N={len(a)} | N={len(bb)} | - | - (某一組為空) |"); continue
            sa, sb = int(a.success.sum()), int(bb.success.sum())
            p = fisher_exact([[sa, len(a) - sa], [sb, len(bb) - sb]])[1]
            rowsC.append(f"| {mode} | {L} | {n} | {rc(sa, len(a))} | {rc(sb, len(bb))} | {sa / len(a) - sb / len(bb):+.2f} | {p:.3g} |")
            pool[mode][0] += sa; pool[mode][1] += len(a); pool[mode][2] += sb; pool[mode][3] += len(bb)
W("\n".join(rowsC)); W("")
W("合併（跨可比較的格直接加總，各格共用同一批 seed、非獨立，僅描述）：" + "；".join(
    f"{m}：k=0 {v[0]}/{v[1]} = {v[0] / max(v[1], 1):.2f}，k≥1 {v[2]}/{v[3]} = {v[2] / max(v[3], 1):.2f}" for m, v in pool.items()) + "。\n")

# ---------------- D. McNemar + Holm
W("## D. oracle／detect 對同格 fixed 的配對檢定（McNemar 精確 + Holm）\n")
W("配對單位 = 同 seed（env seed 與 t0 相同，程式內 assert）。b = 規則成功而 fixed 失敗，c = 反之；p = 精確二項檢定 binom(b, b+c, 0.5)。"
  "Holm 校正分別在 oracle 家族（8 格）與 detect 家族（8 格）內做，並另列 16 個檢定全體的 Holm。\n")
rows = []
for rule in ["oracle", "detect"]:
    for mode in ["black", "freeze"]:
        for L in [50, 100]:
            for n in [25, 50]:
                r, f = cell(rule, n, L, mode), cell("fixed", n, L, mode)
                assert len(r) == len(f) == NS and (r.env_seed == f.env_seed).all() and (r.t0 == f.t0).all()
                bb = int(((r.success == 1) & (f.success == 0)).sum()); cc = int(((r.success == 0) & (f.success == 1)).sum())
                p = binomtest(bb, bb + cc, 0.5).pvalue if bb + cc else 1.0
                rows.append(dict(rule=rule, mode=mode, L=L, n=n, kr=int(r.success.sum()), kf=int(f.success.sum()), b=bb, c=cc, p=p))
R = pd.DataFrame(rows)
R["p_holm_family"] = np.nan
for rule in ["oracle", "detect"]:
    ix = R.index[R.rule == rule]; R.loc[ix, "p_holm_family"] = holm(R.loc[ix, "p"].to_numpy())
R["p_holm_all"] = holm(R.p.to_numpy())
rowsD = ["| 規則 | mode | L | n | 規則成功 | fixed 成功 | Δ | b / c | McNemar p | Holm（家族內） | Holm（全部 16）|", "|---|---|---|---|---|---|---|---|---|---|---|"]
for _, x in R.iterrows():
    rowsD.append(f"| {x.rule} | {x['mode']} | {x.L} | {x.n} | {x.kr}/{NS} | {x.kf}/{NS} | {(x.kr - x.kf) / NS:+.2f} | {x.b} / {x.c} | {x.p:.3g} | {x.p_holm_family:.3g} | {x.p_holm_all:.3g} |")
W("\n".join(rowsD)); W("")
W("detect 診斷（失明窗口內偵測延遲與窗口外誤報）：")
dd = D[D.rule == "detect"]
W(f"- 有偵測延遲值的 episode：{int(dd.detect_latency.notna().sum())}/{len(dd)}；延遲平均 {dd.detect_latency.mean():.2f} 步（最大 {dd.detect_latency.max():.0f}）；失明窗口外誤報步數合計 {int(dd.false_detect.sum())}（{int((dd.false_detect > 0).sum())}/{len(dd)} 個 episode）。")
W(f"- oracle／fixed／detect 在失明中被迫重規劃（forced）平均：oracle {D[D.rule == 'oracle'].forced_replans_in_blackout.mean():.2f}，detect {dd.forced_replans_in_blackout.mean():.2f}。")

# ---------------- E. preregistered primary metrics: R and rho with seed-bootstrap CI
rng = np.random.default_rng(0); B = 5000
W("## E. 預先宣告的主指標：保留率 R 與恢復比例 ρ\n")
W("R = 失明成功率 / 同 n 無失明（L=0）成功率。ρ = (規則 − fixed) / (無失明 − fixed)（同 n、L、mode）：規則補回了 fixed 掉掉的成功率的幾成；"
  "分母（無失明 − fixed）< 0.10 時不報（分母太小、比值不穩）。95% CI 為對 seed 重抽 5000 次（同一個 seed 的無失明、fixed、規則三者一起抽，保留配對）的百分位區間；"
  "ρ 的 bootstrap 只用分母 ≥ 0.10 的重抽樣本（括號內為該比例）。ρ 可 <0 或 >1。\n")
rowsE = ["| mode | L | n | 無失明 | fixed | R_fixed [95% CI] | oracle | R_oracle [CI] | ρ_oracle [CI] | detect | R_detect [CI] | ρ_detect [CI] |", "|---|---|---|---|---|---|---|---|---|---|---|---|"]
def ci(x): return f"[{np.percentile(x, 2.5):.2f}, {np.percentile(x, 97.5):.2f}]"
for mode in ["black", "freeze"]:
    for L in [50, 100]:
        for n in [25, 50, 100]:
            z = cell("fixed", n, 0, "black").success.to_numpy(float); f = cell("fixed", n, L, mode).success.to_numpy(float)
            idx = rng.integers(0, NS, size=(B, NS))
            zb, fb = z[idx].mean(1), f[idx].mean(1)
            Rf = f.mean() / z.mean(); Rfb = fb / np.where(zb > 0, zb, np.nan)
            cols = [f"{z.mean():.2f}", f"{f.mean():.2f}", f"{Rf:.2f} {ci(Rfb[~np.isnan(Rfb)])}"]
            for rule in ["oracle", "detect"]:
                if n not in (25, 50):
                    cols += ["-", "-", "-"]; continue
                r = cell(rule, n, L, mode).success.to_numpy(float); rb = r[idx].mean(1)
                Rr = r.mean() / z.mean(); Rrb = rb / np.where(zb > 0, zb, np.nan)
                den = z.mean() - f.mean(); denb = zb - fb; ok = denb >= 0.10
                if den >= 0.10 and ok.sum() > 100:
                    rho = (r.mean() - f.mean()) / den; rhob = (rb - fb)[ok] / denb[ok]
                    rs = f"{rho:.2f} {ci(rhob)} ({ok.mean():.0%})"
                else:
                    rs = f"不報（分母 {den:.2f} < 0.10）"
                cols += [f"{r.mean():.2f}", f"{Rr:.2f} {ci(Rrb[~np.isnan(Rrb)])}", rs]
            rowsE.append(f"| {mode} | {L} | {n} | " + " | ".join(cols) + " |")
W("\n".join(rowsE)); W("")

# ---------------- F. rerun noise
try:
    rr = pd.read_csv(P + "ins_rerun_n50.csv").sort_values("episode").reset_index(drop=True)
    o0 = cell("fixed", 50, 0, "black"); o0 = o0[o0.episode < len(rr)].reset_index(drop=True)
    assert len(rr) == 50 and (rr.env_seed == o0.env_seed).all() and (rr.t0 == o0.t0).all()
    flip = int((rr.success != o0.success).sum()); rflip = int((rr.max_reward != o0.max_reward).sum()); sflip = int((rr.steps != o0.steps).sum())
    W("## F. 重跑雜訊檢查（n=50、L=0、seeds 0–49 各再跑一次，同 env seed）\n")
    W(f"- 成敗翻轉 {flip}/50 = {flip / 50:.2f}（第一次 {int(o0.success.sum())}/50 成功，重跑 {int(rr.success.sum())}/50）；max_reward 不同 {rflip}/50；步數不同 {sflip}/50。"
      "對照 Transfer Cube（docs/rulers.md）：成敗不同 8/900 = 0.9%。\n")
except FileNotFoundError:
    W("## F. 重跑雜訊檢查：尚未執行\n")
open("results/insertion_table.md", "w").write("\n".join(out) + "\n")
print(len(D), "rows,", ncell, "cells"); print("\n".join(out))
