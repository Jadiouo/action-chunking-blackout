"""T4b: merge detect parts -> results/detect.csv, build results/detect_table.md. Run with python3 from project root."""
import glob, math
import numpy as np, pandas as pd
from scipy.stats import binomtest

o = pd.concat([pd.read_csv(p) for p in sorted(glob.glob("results/parts/detect_n*.csv"))], ignore_index=True)
o = o.sort_values(["mode", "L", "n", "episode"]).reset_index(drop=True)
o["te"] = 0
orc = pd.read_csv("results/oracle.csv")
fg = pd.read_csv("results/full_grid.csv"); fg = fg[fg.te == 0]
cols = list(orc.columns) + ["detected_steps", "detect_latency", "false_detect"]
o[cols].to_csv("results/detect.csv", index=False)


def wilson(k, n, z=1.96):
    p = k / n; d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d; h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def mcnemar(a, b):
    x = int(((a == 1) & (b == 0)).sum()); y = int(((a == 0) & (b == 1)).sum())
    return x, y, (binomtest(x, x + y, 0.5).pvalue if x + y else 1.0)


def pair(g, ref, tag):
    mg = g.merge(ref, on="episode", suffixes=("", "_r"))
    assert len(mg) == 100 and (mg.env_seed == mg.env_seed_r).all() and (mg.t0 == mg.t0_r).all(), tag
    return mcnemar(mg.success, mg.success_r)


H1 = ["| mode | L | n | detect succ | detect rate [Wilson 95%] | oracle succ | oracle rate | T2 fixed-n succ | T2 rate | det only / orc only | McNemar p (det vs orc) | det only / T2 only | McNemar p (det vs T2) |",
      "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
H2 = ["| mode | L | n | episodes detected in blackout | latency mean (min..max) | detected steps / L (mean) | episodes with false detect | total false-detect steps | mean forced replans in blackout |",
      "|---|---|---|---|---|---|---|---|---|"]
for (mode, L, n), g in o[o.L > 0].groupby(["mode", "L", "n"]):
    assert len(g) == 100, (mode, L, n)
    r1 = orc[(orc["mode"] == mode) & (orc.L == L) & (orc.n == n)]
    r2 = fg[(fg["mode"] == mode) & (fg.L == L) & (fg.n == n)]
    assert len(r1) == 100 and len(r2) == 100
    k = int(g.success.sum()); lo, hi = wilson(k, 100)
    b1, c1, p1 = pair(g, r1, "oracle"); b2, c2, p2 = pair(g, r2, "T2")
    H1.append(f"| {mode} | {L} | {n} | {k}/100 | {k/100:.2f} [{lo:.2f}, {hi:.2f}] | {int(r1.success.sum())}/100 | {r1.success.mean():.2f} | "
              f"{int(r2.success.sum())}/100 | {r2.success.mean():.2f} | {b1} / {c1} | {p1:.3f} | {b2} / {c2} | {p2:.3f} |")
    lat = g.detect_latency.dropna()
    inb = g.detected_steps - g.false_detect
    H2.append(f"| {mode} | {L} | {n} | {len(lat)}/100 | " + (f"{lat.mean():.2f} ({int(lat.min())}..{int(lat.max())})" if len(lat) else "n/a") +
              f" | {(inb / L).mean():.3f} | {int((g.false_detect > 0).sum())}/100 | {int(g.false_detect.sum())} | {g.forced_replans_in_blackout.mean():.2f} |")

g0 = o[o.L == 0]
assert len(g0) == 100 and (g0.n == 50).all()
r0 = fg[(fg.L == 0) & (fg.n == 50)]; assert len(r0) == 100
b0, c0, p0 = pair(g0, r0, "L0"); k0 = int(g0.success.sum()); lo0, hi0 = wilson(k0, 100)
L0 = (f"- n=50, L=0（無失明）100 episodes：detect 規則成功 {k0}/100 = {k0/100:.2f} [{lo0:.2f}, {hi0:.2f}]；T2 固定 n=50 {int(r0.success.sum())}/100；"
      f"detect 單贏 {b0} / T2 單贏 {c0}，McNemar p = {p0:.3f}。\n"
      f"- 誤觸發：偵測步數合計 {int(g0.detected_steps.sum())}（{int((g0.detected_steps > 0).sum())}/100 個 episode 有任何偵測）；"
      f"每 episode 平均偵測步數 {g0.detected_steps.mean():.3f}，replans_total 平均 {g0.replans_total.mean():.2f}。")

# calibration (seeds 100-119, no blackout): per-step arrays (t, img_mean, det_black, det_freeze, r, need, blind)
cal = [np.load(f) for f in sorted(glob.glob("results/parts/detect_calib/seed*.npy"))]
C = ["## 偵測器校準（無失明，seeds 100–119，n=100 固定規則，%d episodes）" % len(cal), ""]
if cal:
    A = np.concatenate(cal); means = A[:, 1]
    adj_eq = int(A[:, 3].sum())
    C += [f"- 影像（uint8 尺度 0–255）全畫面均值分布：min {means.min():.2f}，p1 {np.percentile(means, 1):.2f}，中位 {np.median(means):.2f}，p99 {np.percentile(means, 99):.2f}，max {means.max():.2f}（{len(means)} 步）。黑畫面門檻 {1.0}（即 1/255 滿刻度；全黑均值 = 0）。最小均值與門檻的比值 {means.min() / 1.0:.0f}x，無誤報：black 偵測 {int(A[:, 2].sum())} 步。",
          f"- 相鄰兩步影像逐像素完全相同的步數：{adj_eq} / {len(means)}（freeze 偵測誤報）。",
          "- 同一批 episode 的規則層面誤報（black 或 freeze 任一）= 上面兩項之聯集；見下方「誤觸發（L=0）」一節的 100-episode 統計。", ""]

hdr = ("# T4b 影像偵測式規則（detect）：策略只從收到的影像自行判斷是否「看不見」\n\n"
       "規則（`code/run_t3t4.py --rule detect`）：兩個偵測器**同時開**（與 mode 無關）。black：所有相機影像的像素均值 < 1.0（uint8 尺度，約 1/255 滿刻度）；"
       "freeze：當前影像與策略上一步收到的影像逐像素完全相同。偵測到時不重規劃、繼續執行手上 chunk；chunk（100 步）用完才被迫重規劃（同 oracle）；"
       "未偵測到時照固定 n 規則（k >= n 就重規劃）。真實失明區間只用於記錄（latency、false_detect），不參與控制。"
       "env seed / t0 與 T2、oracle 同（程式內 assert）。McNemar 為精確二項檢定（「det only」= detect 成功而對照失敗的 episode 數）。\n\n"
       "「detected steps / L」< 1 是因為 episode 在失明結束前成功而提前結束（steps < t0+L）。延遲 = 失明區間內第一次偵測的步數 − t0（0 代表失明第一步就偵測到）；false detect = 失明區間外的偵測步數。\n\n")
txt = (hdr + "## 成功率與配對比較\n\n" + "\n".join(H1) + "\n\n## 偵測延遲與誤報\n\n" + "\n".join(H2) + "\n\n## 誤觸發（L=0）\n\n" + L0 + "\n\n" + "\n".join(C) + "\n")
open("results/detect_table.md", "w").write(txt)
print(len(o), "rows"); print(txt)
