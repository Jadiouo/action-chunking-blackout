import pandas as pd, math, sys
def wilson(k, n, z=1.96):
    if n == 0: return (float("nan"),) * 2
    p = k / n; d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d; h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h
df = pd.read_csv(sys.argv[1] if len(sys.argv) > 1 else "results/pilot.csv")
rows = ["| config | n | L | mode | episodes | success | rate | Wilson 95% CI | mean max_reward | mean replans in blackout | mean |d2 action| | mean s/episode |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|"]
df["_o"] = df.te * 1000 + df.n
for (cfg, n, te, L, mode), g in df.sort_values(["mode", "L", "_o"]).groupby(["config", "n", "te", "L", "mode"], sort=False):
    k, m = int(g.success.sum()), len(g); lo, hi = wilson(k, m)
    rows.append(f"| {cfg} | {'1+TE' if te else n} | {L} | {mode} | {m} | {k} | {k/m:.2f} | [{lo:.2f}, {hi:.2f}] | {g.max_reward.mean():.2f} | {g.replans_in_blackout.mean():.2f} | {g.mean_abs_d2.mean():.4f} | {g.seconds.mean():.1f} |")
open(sys.argv[2] if len(sys.argv) > 2 else "results/pilot_table.md", "w").write("\n".join(rows) + "\n"); print("\n".join(rows))
