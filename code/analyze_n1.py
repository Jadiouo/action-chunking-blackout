"""T1 analysis of n=1 vs n=25 vs TE trajectories recorded by diagnose_n1.py. Prints tables, writes figures to results/figs/
and numbers to results/n1_diag_numbers.json."""
import glob, json, os, re, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

D = ".scratch/diag"; FIG = "results/figs"; os.makedirs(FIG, exist_ok=True)
CFGS = ["n1", "n25", "te"]; LAB = {"n1": "n=1", "n25": "n=25", "te": "1+TE"}; COL = {"n1": "#d62728", "n25": "#1f77b4", "te": "#2ca02c"}
RIGHT = slice(7, 13)  # right arm joints (6); index 13 = right gripper; 6 = left gripper
runs = {}
for f in sorted(glob.glob(f"{D}/*.npz")):
    m = re.match(r"(n1|n25|te)_s(\d+)\.npz", os.path.basename(f))
    if m: runs[(m.group(1), int(m.group(2)))] = dict(np.load(f))
seeds = sorted({s for _, s in runs})
nrm = lambda x: np.linalg.norm(x, axis=-1)


def metrics(d):
    q, a, r, fr = d["q"], d["a"], d["r"], d["fresh"]
    dist = nrm(d["rg"] - d["cube"])
    out = dict(T=len(q), max_reward=r.max(), steps_r_ge1=int((r >= 1).sum()), steps_r_ge2=int((r >= 2).sum()),
               first_touch=int(np.argmax(r >= 1)) if (r >= 1).any() else np.nan,
               min_dist=dist.min(), dist_end=dist[-1], t_dist_lt_005=int(np.argmax(dist < 0.05)) if (dist < 0.05).any() else np.nan,
               cube_moved=nrm(d["cube"][-1] - d["cube"][0]),
               right_path=nrm(np.diff(q[:, RIGHT], axis=0)).sum(),
               speed_0_100=nrm(np.diff(q[:101, RIGHT], axis=0)).mean(), speed_100_400=nrm(np.diff(q[100:, RIGHT], axis=0)).mean() if len(q) > 101 else np.nan,
               d_first=nrm(fr[1:, 0] - fr[:-1, 0]).mean(),            # |c_t[0]-c_{t-1}[0]| adjacent first-step difference
               d_cons=nrm(fr[1:, 0] - fr[:-1, 1]).mean(),             # |c_t[0]-c_{t-1}[1]| fresh first step vs previous plan's next step
               act_step=nrm(a[1:] - a[:-1]).mean(),
               cmd_gap=nrm(a - q).mean(),                              # |a_t - q_t|
               grip_cmd_end=a[-1, 13], grip_q_end=q[-1, 13], grip_cmd_min=a[:, 13].min(), grip_cmd_max=a[:, 13].max())
    for k in (0, 5, 10, 25, 50, 99): out[f"lead{k}"] = nrm(fr[:, k] - q).mean()  # how far ahead chunk element k points from current qpos
    return out


rows = [dict(cfg=c, seed=s, **metrics(d)) for (c, s), d in runs.items()]
M = pd.DataFrame(rows).sort_values(["cfg", "seed"])
M.to_csv("results/n1_diag_metrics.csv", index=False)
pd.set_option("display.width", 250, "display.max_columns", 50)
summ = M.groupby("cfg").agg(N=("seed", "size"), success=("max_reward", lambda x: int((x >= 4).sum())), mean_max_reward=("max_reward", "mean"),
                            touch_frac=("first_touch", lambda x: x.notna().mean()), min_dist=("min_dist", "mean"), dist_end=("dist_end", "mean"),
                            cube_moved=("cube_moved", "mean"), right_path=("right_path", "mean"), speed_0_100=("speed_0_100", "mean"),
                            speed_100_400=("speed_100_400", "mean"), d_first=("d_first", "mean"), d_cons=("d_cons", "mean"),
                            act_step=("act_step", "mean"), cmd_gap=("cmd_gap", "mean"), lead0=("lead0", "mean"), lead10=("lead10", "mean"),
                            lead25=("lead25", "mean"), lead50=("lead50", "mean"), T=("T", "mean"))
print(summ.T.round(4).to_string())
print(M[["cfg", "seed", "max_reward", "first_touch", "t_dist_lt_005", "min_dist", "dist_end", "cube_moved", "grip_cmd_min", "grip_cmd_max", "T"]].round(3).to_string(index=False))
summ.T.to_csv("results/n1_diag_summary.csv")

# ---------- extra: chunk-vs-chunk structure
# (i) TE action vs fresh first step
te_gap = [nrm(runs[("te", s)]["a"] - runs[("te", s)]["fresh"][:, 0]).mean() for s in seeds if ("te", s) in runs]
# (ii) self-consistency profile: |c_t[k] - c_{t-1}[k+1]| over k  (how much a fresh plan shifts vs the previous plan)
prof = {c: np.mean([nrm(runs[(c, s)]["fresh"][1:, :-1] - runs[(c, s)]["fresh"][:-1, 1:]).mean(0) for s in seeds if (c, s) in runs], axis=0) for c in CFGS}
# (iii) signed lead along motion: project (c_t[k]-c_t[0]) on the direction of actual 10-step motion
nums = dict(summary=summ.round(5).to_dict(), te_action_vs_first_step=float(np.mean(te_gap)))

# ---------- Fig 1: seed-0 time series
s0 = seeds[0]
fig, ax = plt.subplots(2, 3, figsize=(15, 7), sharex=True)
for (j, name), a_ in zip([(7, "right waist (q7)"), (8, "right shoulder (q8)"), (9, "right elbow (q9)"), (11, "right wrist angle (q11)")], ax.flat[:4]):
    for c in CFGS:
        if (c, s0) in runs: a_.plot(runs[(c, s0)]["q"][:, j], color=COL[c], label=LAB[c], lw=1.4)
    a_.set_title(name); a_.set_ylabel("rad")
for c in CFGS:
    if (c, s0) in runs:
        ax.flat[4].plot(runs[(c, s0)]["a"][:, 13], color=COL[c], label=f"{LAB[c]} command"); ax.flat[4].plot(runs[(c, s0)]["q"][:, 13], color=COL[c], ls=":", lw=1)
        ax.flat[5].plot(runs[(c, s0)]["r"] + {"n1": 0.04, "n25": 0, "te": -0.04}[c], color=COL[c], label=LAB[c])
ax.flat[4].set_title("right gripper: command (solid) / measured (dotted)"); ax.flat[5].set_title("reward level (offset slightly for visibility)")
for a_ in ax[1]: a_.set_xlabel("env step")
ax.flat[0].legend(); fig.suptitle(f"Seed {s0}: right-arm joints, gripper and reward, n=1 vs n=25 vs 1+TE (L=0)"); fig.tight_layout()
fig.savefig(f"{FIG}/n1_traj_seed{s0}.png", dpi=130); plt.close(fig)

# ---------- Fig 2: gripper-cube distance, all seeds
fig, ax = plt.subplots(1, 3, figsize=(15, 4), sharey=True)
for a_, c in zip(ax, CFGS):
    for s in seeds:
        if (c, s) in runs:
            d = runs[(c, s)]; ok = d["r"].max() >= 4
            a_.plot(nrm(d["rg"] - d["cube"]), color=COL[c], alpha=0.8 if ok else 0.4, ls="-" if ok else "--", lw=1)
    a_.set_title(f"{LAB[c]}: right gripper - cube distance (m)"); a_.set_xlabel("env step")
ax[0].set_ylabel("distance (m)"); fig.tight_layout(); fig.savefig(f"{FIG}/n1_gripper_cube_dist.png", dpi=130); plt.close(fig)

# ---------- Fig 3: prediction differences + chunk lead profile
fig, ax = plt.subplots(1, 3, figsize=(16, 4.2))
for c in CFGS:
    v1 = np.concatenate([nrm(runs[(c, s)]["fresh"][1:, 0] - runs[(c, s)]["fresh"][:-1, 0]) for s in seeds if (c, s) in runs])
    v2 = np.concatenate([nrm(runs[(c, s)]["fresh"][1:, 0] - runs[(c, s)]["fresh"][:-1, 1]) for s in seeds if (c, s) in runs])
    v3 = np.concatenate([nrm(np.diff(runs[(c, s)]["a"], axis=0)) for s in seeds if (c, s) in runs])
    ax[0].boxplot([v1, v2, v3], positions=np.arange(3) + {"n1": -0.25, "n25": 0, "te": 0.25}[c], widths=0.22, patch_artist=True, showfliers=False,
                  boxprops=dict(facecolor=COL[c], alpha=0.6), medianprops=dict(color="k"))
ax[0].set_xticks(range(3)); ax[0].set_xticklabels(["|c_t[0]-c_{t-1}[0]|", "|c_t[0]-c_{t-1}[1]|", "|a_t-a_{t-1}|"]); ax[0].set_ylabel("L2 over 14 dims (rad)")
ax[0].set_title("adjacent-step differences (red n=1, blue n=25, green TE)")
for c in CFGS:
    ax[1].plot(np.mean([nrm(runs[(c, s)]["fresh"] - runs[(c, s)]["q"][:, None, :]).mean(0) for s in seeds if (c, s) in runs], axis=0), color=COL[c], label=LAB[c])
ax[1].set_xlabel("chunk index k"); ax[1].set_ylabel("mean |c_t[k] - q_t|"); ax[1].set_title("how far ahead of the current pose each chunk element points"); ax[1].legend()
for c in CFGS: ax[2].plot(prof[c], color=COL[c], label=LAB[c])
ax[2].set_xlabel("k"); ax[2].set_ylabel("mean |c_t[k] - c_{t-1}[k+1]|"); ax[2].set_title("plan revision between consecutive steps"); ax[2].legend()
fig.tight_layout(); fig.savefig(f"{FIG}/n1_pred_diffs.png", dpi=130); plt.close(fig)

# ---------- Fig 4: right-arm speed vs time (rolling 20 steps)
fig, ax = plt.subplots(figsize=(7, 4))
for c in CFGS:
    sp = np.array([np.convolve(nrm(np.diff(runs[(c, s)]["q"][:240, RIGHT], axis=0)), np.ones(20) / 20, "valid") for s in seeds if (c, s) in runs])
    ax.plot(sp.mean(0), color=COL[c], label=LAB[c]); ax.fill_between(range(sp.shape[1]), sp.min(0), sp.max(0), color=COL[c], alpha=0.12)
ax.set_xlabel("env step"); ax.set_ylabel("right-arm joint speed (rad/step, 20-step mean)"); ax.legend(); ax.set_title("arm speed over time (band = min/max over seeds)")
fig.tight_layout(); fig.savefig(f"{FIG}/n1_speed.png", dpi=130); plt.close(fig)


# ---------- extra diagnostics for the stall hypothesis
ex = {}
leadk = lambda d, t: nrm(d["fresh"][t] - d["q"][t])            # (100,) |c_t[k]-q_t|
kk = [0, 2, 5, 10, 15, 20, 25, 30, 50, 99]
ex["lead_at_t0"] = {c: np.mean([leadk(runs[(c, s)], 0)[kk] for s in seeds if (c, s) in runs], axis=0).round(4).tolist() for c in CFGS}
ex["lead_k_index"] = kk
# first chunk index whose pose is >0.05 rad (L2, 14-d) from the current pose  ("pause length" of the plan)
pause = lambda d, t: int(np.argmax(leadk(d, t) > 0.05)) if (leadk(d, t) > 0.05).any() else 100
ex["pause_len_by_t"] = {f"{c}@t{t}": [pause(runs[(c, s)], t) for s in seeds if (c, s) in runs and len(runs[(c, s)]["q"]) > t] for c in ("n1", "n25") for t in (25, 50, 100)}
ex["lead_profile_by_t"] = {f"{c}@t{t}": np.mean([leadk(runs[(c, s)], t)[kk] for s in seeds if (c, s) in runs and len(runs[(c, s)]["q"]) > t], axis=0).round(4).tolist() for c in ("n1", "n25") for t in (25, 50, 100, 300)}
# is the t=0 observation (and hence the first chunk) identical between configs for the same seed?
ex["t0_chunk_same_across_cfgs_maxabs"] = float(max(np.abs(runs[("n1", s)]["fresh"][0] - runs[("n25", s)]["fresh"][0]).max() for s in seeds))
# fixed-point test for n=1: chunk at the end of the episode vs chunk at t=0, and pose drift
T1 = lambda d: len(d["q"]) - 1
ex["n1_chunk_end_vs_t0_mean_abs"] = float(np.mean([nrm(runs[("n1", s)]["fresh"][T1(runs[("n1", s)])] - runs[("n1", s)]["fresh"][0]).mean() for s in seeds]))
ex["n1_chunk_end_vs_t0_rel"] = float(np.mean([nrm(runs[("n1", s)]["fresh"][T1(runs[("n1", s)])] - runs[("n1", s)]["fresh"][0]).mean() / nrm(runs[("n1", s)]["fresh"][0] - runs[("n1", s)]["q"][0]).mean() for s in seeds]))
ex["n1_pose_drift_end_vs_t0_L2"] = [float(nrm(runs[("n1", s)]["q"][-1] - runs[("n1", s)]["q"][0])) for s in seeds]
ex["n1_pose_drift_by_t"] = {t: float(np.mean([nrm(runs[("n1", s)]["q"][t] - runs[("n1", s)]["q"][0]) for s in seeds])) for t in (10, 25, 50, 100, 200, 399)}
ex["n25_pose_drift_by_t"] = {t: float(np.mean([nrm(runs[("n25", s)]["q"][min(t, len(runs[("n25", s)]["q"]) - 1)] - runs[("n25", s)]["q"][0]) for s in seeds])) for t in (10, 25, 50, 100, 200, 399)}
# command-vs-pose gap in steady stall (last 100 steps), per arm joint index: is it a constant offset?
ex["n1_last100_mean_gap_per_joint"] = np.mean([np.abs(runs[("n1", s)]["a"][-100:] - runs[("n1", s)]["q"][-100:]).mean(0) for s in seeds], axis=0).round(4).tolist()
ex["n1_last100_q_std_per_joint"] = np.mean([runs[("n1", s)]["q"][-100:].std(0) for s in seeds], axis=0).round(5).tolist()
nums["extra"] = ex
print(json.dumps(ex, indent=1))
fig, ax = plt.subplots(1, 3, figsize=(16, 4.2))
for s in seeds:
    ax[0].plot(leadk(runs[("n1", s)], 0), color="#d62728", alpha=0.5)
ax[0].axhline(0.05, color="k", ls=":"); ax[0].set_xlabel("chunk index k"); ax[0].set_ylabel("|c_0[k] - q_0|"); ax[0].set_title("first plan at t=0 (n=1 runs, 8 seeds)")
for t, col in [(0, "k"), (50, "#888"), (200, "#d62728"), (399, "#7f0000")]:
    ax[1].plot(np.mean([leadk(runs[("n1", s)], t) for s in seeds], axis=0), color=col, label=f"t={t}")
ax[1].set_xlabel("chunk index k"); ax[1].set_ylabel("mean |c_t[k] - q_t|"); ax[1].legend(); ax[1].set_title("n=1: plan at different times (stalled arm)")
for c in ("n1", "n25", "te"):
    ax[2].plot(np.mean([nrm(runs[(c, s)]["q"][:240] - runs[(c, s)]["q"][0]) if len(runs[(c, s)]["q"]) >= 240 else np.pad(nrm(runs[(c, s)]["q"] - runs[(c, s)]["q"][0]), (0, 240 - len(runs[(c, s)]["q"])), mode="edge") for s in seeds], axis=0), color=COL[c], label=LAB[c])
ax[2].set_xlabel("env step"); ax[2].set_ylabel("|q_t - q_0| (14-d, rad)"); ax[2].legend(); ax[2].set_title("pose displacement from start")
fig.tight_layout(); fig.savefig(f"{FIG}/n1_stall_evidence.png", dpi=130); plt.close(fig)

# ---------- offset intervention
if os.path.exists("results/n1_offset_runs.csv"):
    o = pd.read_csv("results/n1_offset_runs.csv")
    g = o.groupby("cfg").agg(N=("seed", "size"), k=("success", "sum"), mr=("max_reward", "mean"))
    print(g); nums["offset"] = g.to_dict()
    fig, ax = plt.subplots(figsize=(5, 3.6)); xs = ["off2", "off5", "off10", "off25"]
    ax.bar([x for x in xs if x in g.index], [g.loc[x, "k"] / g.loc[x, "N"] for x in xs if x in g.index], color="#888")
    ax.set_ylabel("success rate"); ax.set_title("replan every step, execute chunk element j"); fig.tight_layout(); fig.savefig(f"{FIG}/n1_offset.png", dpi=130); plt.close(fig)
json.dump(nums, open("results/n1_diag_numbers.json", "w"), indent=1, default=float)
