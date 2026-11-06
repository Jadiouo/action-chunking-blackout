"""Merge T2 parts into results/full_grid.csv, validate size, and compare seeds 0-49 with pilot.csv (cells present in both)."""
import glob, pandas as pd
parts = sorted(glob.glob("results/parts/t2_*.csv"))
df = pd.concat([pd.read_csv(p) for p in parts], ignore_index=True)
cfgs = ["n1", "n10", "n25", "n50", "n100", "TE"]
df = df.sort_values(["te", "n", "mode", "L", "episode"]).reset_index(drop=True)
df.to_csv("results/full_grid.csv", index=False)
cells = df.groupby("config").size()
print("rows:", len(df), "cells:", len(cells), "rows/cell min,max:", cells.min(), cells.max(), "duplicates:", int(df.duplicated(["config", "episode"]).sum()))
pil = pd.read_csv("results/pilot.csv")
m = pil.merge(df, on=["config", "episode"], suffixes=("_p", "_f"))
print("pilot rows:", len(pil), "matched in full grid:", len(m), "cells matched:", m.config.nunique())
print("env_seed/t0 identical:", bool((m.env_seed_p == m.env_seed_f).all() and (m.t0_p == m.t0_f).all()))
dsucc = m[m.success_p != m.success_f]; dmr = m[m.max_reward_p != m.max_reward_f]
print(f"success mismatches: {len(dsucc)}/{len(m)}; max_reward mismatches: {len(dmr)}/{len(m)}; steps mismatches: {(m.steps_p != m.steps_f).sum()}")
print(dsucc[["config", "episode", "success_p", "success_f", "max_reward_p", "max_reward_f", "steps_p", "steps_f"]].to_string(index=False))
g = m.groupby("config").agg(N=("episode", "size"), succ_pilot=("success_p", "sum"), succ_full=("success_f", "sum"), flips=("success_p", lambda x: 0))
g["flips"] = m.assign(f=(m.success_p != m.success_f)).groupby("config").f.sum()
print(g.to_string())
dsucc.to_csv("results/pilot_vs_full_mismatch.csv", index=False)
