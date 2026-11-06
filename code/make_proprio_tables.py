"""T4: merge parts -> results/proprio.csv, print/emit tables for results/proprio_signal.md. python3, project root.
Writes results/parts/proprio_tables.md (tables only; the narrative is in results/proprio_signal.md)."""
import json, math
import pandas as pd
from scipy.stats import binomtest

def wilson(k, n, z=1.96):
    p = k / n; d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d; h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h

P = "results/parts/"
tau_arm = json.load(open(P + "proprio_calib_stats_arm.json"))
tau_all = json.load(open(P + "proprio_calib_stats_all.json"))
parts = [("main", "arm", tau_arm["tau99"], "t4_main_L0.csv"), ("main", "arm", tau_arm["tau99"], "t4_main_black.csv"),
         ("main", "arm", tau_arm["tau99"], "t4_main_freeze.csv"),
         ("sens_arm_p95", "arm", tau_arm["tau95"], "t4_sens_arm95.csv"),
         ("sens_all_p99", "all", tau_all["tau99"], "t4_sens_all99.csv")]
dfs = []
for tag, sig, tau, fn in parts:
    d = pd.read_csv(P + fn); d.insert(0, "sig", sig); d.insert(0, "tag", tag); dfs.append(d)
pr = pd.concat(dfs, ignore_index=True)
for tag, sig, tau, fn in parts:  # tau column must equal the declared value
    d = pr[(pr.tag == tag)]
    assert (abs(d.tau - tau) < 1e-9).all(), tag
pr.to_csv("results/proprio.csv", index=False)
fg = pd.read_csv("results/full_grid.csv"); fg = fg[fg.te == 0]
orc = pd.read_csv("results/oracle.csv")


def mcn(a, b):
    m = a.merge(b, on="episode", suffixes=("_a", "_b"))
    assert len(m) == 100 and (m.env_seed_a == m.env_seed_b).all() and (m.t0_a == m.t0_b).all()
    x = int(((m.success_a == 1) & (m.success_b == 0)).sum()); y = int(((m.success_a == 0) & (m.success_b == 1)).sum())
    return x, y, (binomtest(x, x + y, 0.5).pvalue if x + y else 1.0)


def fixed_cell(n, L, mode):
    return fg[(fg.n == n) & (fg.L == (0 if L == 0 else L)) & ((fg["mode"] == mode) if L else (fg["mode"] == "black"))]


out = []
for tag in ["main", "sens_arm_p95", "sens_all_p99"]:
    d = pr[pr.tag == tag]
    out.append(f"\n### {tag}（signal={d.sig.iloc[0]}, tau={d.tau.iloc[0]:.4f}）\n")
    out.append("| variant | L | mode | succ | rate [Wilson 95%] | mean max_reward | mean triggers/ep | mean triggers in blackout | mean replans total | mean replans in blackout | vs fixed n=25 (b/c, p) | vs n=50 | vs n=100 |")
    out.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for (L, mode, m), g in d.groupby(["L", "mode", "m"]):
        k = int(g.success.sum()); lo, hi = wilson(k, len(g))
        cmp = []
        for n in (25, 50, 100):
            b, c, p = mcn(g, fixed_cell(n, L, mode))
            cmp.append(f"{b}/{c}, p={p:.3f}")
        out.append(f"| proprio-m{int(m)} | {L} | {mode if L else '-'} | {k}/{len(g)} | {k/len(g):.2f} [{lo:.2f}, {hi:.2f}] | {g.max_reward.mean():.2f} | {g.triggers.mean():.2f} | {g.triggers_in_blackout.mean():.2f} | {g.replans_total.mean():.2f} | {g.replans_in_blackout.mean():.2f} | " + " | ".join(cmp) + " |")
# fixed-n reference rows (T2) for the same cells
out.append("\n### T2 固定 n 同格參照\n")
out.append("| L | mode | n=10 | n=25 | n=50 | n=100 | oracle n=25 | oracle n=50 |")
out.append("|---|---|---|---|---|---|---|---|")
for L, mode in [(0, "black"), (25, "black"), (25, "freeze"), (50, "black"), (50, "freeze"), (100, "black"), (100, "freeze")]:
    row = [f"{int(fixed_cell(n, L, mode).success.sum())}/100" for n in (10, 25, 50, 100)]
    for n in (25, 50):
        o = orc[(orc.n == n) & (orc.L == L) & (orc["mode"] == mode)]
        row.append(f"{int(o.success.sum())}/100" if len(o) else "-")
    out.append(f"| {L} | {mode if L else '-'} | " + " | ".join(row) + " |")
open(P + "proprio_tables.md", "w").write("\n".join(out) + "\n")
print(len(pr), "rows"); print("\n".join(out))
