"""T3: merge oracle parts -> results/oracle.csv, build results/oracle_table.md (paired vs T2 full_grid). Run with python3."""
import glob, math, sys
import pandas as pd
from scipy.stats import binomtest

parts = sorted(glob.glob("results/parts/t3_n*.csv"))
o = pd.concat([pd.read_csv(p) for p in parts], ignore_index=True)
o = o.sort_values(["mode", "L", "n", "episode"]).reset_index(drop=True)
o["te"] = 0
cols = list(pd.read_csv("results/full_grid.csv", nrows=1).columns) + ["forced_replans_in_blackout"]
o[cols].to_csv("results/oracle.csv", index=False)
fg = pd.read_csv("results/full_grid.csv")
fg = fg[(fg.te == 0)]


def wilson(k, n, z=1.96):
    p = k / n; d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d; h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


rows = ["| mode | L | n | oracle succ | oracle rate [Wilson 95%] | T2 fixed-n succ | T2 rate | diff (oracle-T2) | oracle only / T2 only | McNemar p | mean forced replans in blackout | mean replans in blackout (T2) | mean max_reward oracle / T2 |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
for (mode, L, n), g in o.groupby(["mode", "L", "n"]):
    f = fg[(fg["mode"] == mode) & (fg.L == L) & (fg.n == n)]
    assert len(g) == 100 and len(f) == 100, (mode, L, n, len(g), len(f))
    mg = g.merge(f, on="episode", suffixes=("_o", "_f"))
    assert (mg.env_seed_o == mg.env_seed_f).all() and (mg.t0_o == mg.t0_f).all()
    b = int(((mg.success_o == 1) & (mg.success_f == 0)).sum())
    c = int(((mg.success_o == 0) & (mg.success_f == 1)).sum())
    p = binomtest(b, b + c, 0.5).pvalue if b + c > 0 else 1.0
    ko, kf = int(g.success.sum()), int(f.success.sum())
    lo, hi = wilson(ko, 100)
    rows.append(f"| {mode} | {L} | {n} | {ko}/100 | {ko/100:.2f} [{lo:.2f}, {hi:.2f}] | {kf}/100 | {kf/100:.2f} | {(ko-kf)/100:+.2f} | {b} / {c} | {p:.3f} | "
                f"{g.forced_replans_in_blackout.mean():.2f} | {f.replans_in_blackout.mean():.2f} | {g.max_reward.mean():.2f} / {f.max_reward.mean():.2f} |")
hdr = ("# T3 Oracle（知道自己看不見）與 T2 固定 n 的同格配對比較\n\n"
       "規則：失明期間（t0 <= t < t0+L）不重新規劃，繼續執行手上 chunk 剩餘動作；chunk（100 步）用完時仍在失明中，被迫用當下（壞的）影像重規劃（forced）。失明外照固定 n 規則。"
       "env seed 與 t0 與 T2 完全相同（程式內以 assert 驗證）。McNemar 為精確二項檢定（oracle 單贏 b 對 T2 單贏 c）。"
       "oracle 為 `code/run_t3t4.py --rule oracle`（自寫執行迴圈取 `predict_action_chunk` 的完整 100 步），T2 為 `code/run_grid.py`（policy 動作佇列）；"
       "兩者在無失明時等價（見 `.scratch` 煙霧測試與報告回報），但整條管線並非位元級確定性（docs/rulers.md），單格成功數差 <= 3/50 屬重跑雜訊量級。\n\n")
open("results/oracle_table.md", "w").write(hdr + "\n".join(rows) + "\n")
print(len(o), "rows"); print("\n".join(rows))
