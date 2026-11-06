"""T4 step 1-2: e_t time series (L=0, n=100) and threshold tau. Run with python3 from project root.
usage: python3 code/proprio_calib.py <sig: all|arm> -> figs + results/parts/proprio_calib_stats.json
Rec columns (run_t3t4.py): t, e_all, reward, replan, blind, diff[14]
tau is computed ONLY from seeds 100-119 (held out from evaluation seeds 0-99); seeds 0-19 are for the figure / description."""
import glob, json, sys, os
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

sig = sys.argv[1] if len(sys.argv) > 1 else "all"
mask = np.ones(14)
if sig == "arm":
    mask[[6, 13]] = 0


def load(d):
    out = {}
    for p in sorted(glob.glob(f"{d}/seed*.npy"), key=lambda s: int(os.path.basename(s)[4:-4])):
        r = np.load(p); out[int(os.path.basename(p)[4:-4])] = r
    return out


def e_of(r, msk=mask):
    return np.linalg.norm(r[:, 5:19] * msk, axis=1)


A, B = load(".scratch/proprio/s0"), load(".scratch/proprio/s100")
assert len(A) == 20 and len(B) == 20, (len(A), len(B))
os.makedirs("results/figs", exist_ok=True)

# --- figure: e_t of 20 seeds (seeds 0-19), reward level changes and replans marked
fig, axs = plt.subplots(5, 4, figsize=(16, 12), sharex=True)
for ax, (s, r) in zip(axs.ravel(), A.items()):
    e = e_of(r); t = r[:, 0]; rew = r[:, 2]
    ax.plot(t, e, lw=0.8, color="C0")
    lvl = np.maximum.accumulate(rew)
    for i in np.where(np.diff(lvl) > 0)[0] + 1:
        ax.axvline(t[i], color="C2", ls="--", lw=0.8)
        ax.text(t[i], ax.get_ylim()[1] * 0.9, f"r={int(lvl[i])}", fontsize=6, color="C2")
    for i in np.where(r[:, 3] == 1)[0]:
        ax.axvline(t[i], color="C3", lw=0.8, alpha=0.6)
    ax.set_title(f"seed {s} (max r={int(rew.max())})", fontsize=8)
for ax in axs[-1]: ax.set_xlabel("t")
for ax in axs[:, 0]: ax.set_ylabel(f"e_t ({sig})")
fig.suptitle("e_t = ||a_{t-1} - q_t||_2 (n=100, L=0, seeds 0-19). red: replan; green dashed: reward level reached", fontsize=11)
fig.tight_layout(); fig.savefig(f"results/figs/proprio_et_series_{sig}.png", dpi=110); plt.close(fig)

# --- phase-averaged view and per-dim contribution
allA = np.concatenate([e_of(r)[1:] for r in A.values()])
allB = np.concatenate([e_of(r)[1:] for r in B.values()])
q = [50, 90, 95, 99, 99.9]
stats = dict(sig=sig,
             pct_seeds0_19={str(p): float(np.percentile(allA, p)) for p in q},
             pct_seeds100_119={str(p): float(np.percentile(allB, p)) for p in q},
             tau95=float(np.percentile(allB, 95)), tau99=float(np.percentile(allB, 99)),
             n_steps_B=int(len(allB)), max_B=float(allB.max()))
# fraction of steps above tau on seeds 0-19, by episode
for name in ("tau95", "tau99"):
    tau = stats[name]
    stats[name + "_frac_steps_above_on_seeds0_19"] = float((allA > tau).mean())
    stats[name + "_episodes_with_any_above_seeds0_19"] = int(sum((e_of(r)[1:] > tau).any() for r in A.values()))
    stats[name + "_steps_above_per_episode_seeds0_19"] = float(np.mean([(e_of(r)[1:] > tau).sum() for r in A.values()]))
# per-dimension share of squared error (seeds 100-119)
d2 = np.mean(np.concatenate([r[1:, 5:19] ** 2 for r in B.values()]), axis=0)
stats["per_dim_mean_sq_share"] = (d2 / d2.sum()).round(4).tolist()
# e_t in reward phases (seeds 0-19): by running max reward
ph = {}
for lv in range(5):
    v = np.concatenate([e_of(r)[1:][np.maximum.accumulate(r[:, 2])[1:] == lv] for r in A.values()])
    if len(v): ph[str(lv)] = dict(n=int(len(v)), mean=float(v.mean()), p99=float(np.percentile(v, 99)))
stats["by_running_max_reward_seeds0_19"] = ph
# e_t just before/at replans vs elsewhere (n=100 replans at t=0,100,..)
rp = np.concatenate([e_of(r)[1:][r[1:, 3] == 1] for r in A.values()])
stats["e_at_replan_steps_mean"] = float(rp.mean()) if len(rp) else None
stats["e_other_steps_mean"] = float(allA.mean())
json.dump(stats, open(f"results/parts/proprio_calib_stats_{sig}.json", "w"), indent=1)
print(json.dumps(stats, indent=1))

# histogram figure
fig, ax = plt.subplots(figsize=(6, 3.5))
ax.hist(allB, bins=80, color="C0", alpha=0.8); ax.set_yscale("log")
for name, c in (("tau95", "C1"), ("tau99", "C3")):
    ax.axvline(stats[name], color=c, label=f"{name}={stats[name]:.3f}")
ax.set_xlabel(f"e_t ({sig})"); ax.set_ylabel("steps (log)"); ax.set_title("e_t distribution, seeds 100-119, L=0, n=100"); ax.legend()
fig.tight_layout(); fig.savefig(f"results/figs/proprio_et_hist_{sig}.png", dpi=110)
