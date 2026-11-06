"""P2 (realistic corruptions) and P3 (literature baselines) runner, Transfer Cube.  Own execution loop like run_t3t4.py
(full 100-step chunk from policy.predict_action_chunk, unnormalised with the checkpoint buffers), plus

rules
  fixed    : replan when k >= n.                                           (== run_grid.py / run_t3t4.py fixed)
  oracle   : like run_t3t4 oracle: never replans inside the true corruption window [t0, t0+L) unless the chunk (100) runs out.
  detect   : T4b "simple" detector: black (mean pixel < 1.0) or bit-identical-to-previous-frame; replan only if k>=n and not detected.
  generic  : generic detector: PCA reconstruction error of the policy-visible frame's ResNet18 (ACT backbone) pooled features w.r.t.
             the clean-frame subspace (PCA K=16, fitted on clean seeds 100-109, threshold = 99th percentile of per-step scores on clean
             seeds 110-119). Detected -> keep executing the stale chunk; replan only if k>=n and not detected (chunk exhausted -> forced).
  median_te / mean_te : temporal ensembling with query stride h (Jiang 2026 Median TE; mean = ACT/lerobot weights exp(-0.01 i), i=0 oldest).
                        action(t) = agg over all chunks queried at t_i = i*h with 0 <= t - t_i < 100 of A_i[t - t_i].
  care     : CARE-style consistency-triggered execution length (Yamabe & Sakuma 2026, Alg. 1) with H_max=50, gamma=0.85, H_init=1,
             tau given by --tau (calibrated on clean seeds 100-119, docs/p2p3_notes.md).
Corruption (code/corrupt.py) is applied to the policy-visible image only; env seed / t0 are the same as in T2 (t0_for).
"""
import argparse, csv, os, time, json
os.environ.setdefault("MUJOCO_GL", "egl")
import numpy as np, torch, gymnasium as gym, gym_aloha  # noqa
from lerobot.policies.act.modeling_act import ACTPolicy
from safetensors.torch import load_file
from huggingface_hub import hf_hub_download
from lerobot.envs.utils import preprocess_observation
from run_grid import t0_for
from run_t3t4 import detect_blind
from tasks import TASKS
from corrupt import Corruptor, KINDS, parse_param, tag

CHUNK, HMAX, GAMMA, EPS = 100, 50, 0.85, 1e-8
FIELDS = ["config", "rule", "corr", "n", "h", "tau", "L", "episode", "env_seed", "t0", "success", "max_reward",
          "replans_in_window", "replans_total", "steps", "seconds", "forced_replans_in_window",
          "det_in", "det_out", "steps_in", "steps_out", "latency", "care_meanH", "care_queries"]


def care_curve(A_new, A_prev, H_prev):
    """Cumulative CARE discrepancy R_i = sum_{j<=i} gamma^j r_j, i = 0..HMAX-1 (Eq. 4-5). A_prev's action for the time of
    A_new[i] is A_prev[H_prev + i] (ACT chunk index 0 = current step)."""
    a = A_new[:HMAX]; b = A_prev[H_prev:H_prev + HMAX]
    r = np.linalg.norm(a - b, axis=1) / (np.linalg.norm(a, axis=1) + np.linalg.norm(b, axis=1) + EPS)
    return np.cumsum(GAMMA ** np.arange(HMAX) * r)


def care_H(curve, tau):
    """Algorithm 1: H = 1 + number of leading steps with R <= tau, but the first step always runs (H >= 1)."""
    ok = curve <= tau
    if not ok[0]:
        return 1
    bad = np.where(~ok)[0]
    return int(bad[0]) if len(bad) else HMAX


class Generic:
    def __init__(self, path, device):
        z = np.load(path)
        self.mu = torch.tensor(z["mu"], device=device, dtype=torch.float32)
        self.V = torch.tensor(z["V"], device=device, dtype=torch.float32)
        self.isq = torch.tensor(1.0 / np.sqrt(z["lam"]), device=device, dtype=torch.float32)
        self.thr = float(z["thr"])

    def score(self, feat):  # feat (6144,) torch
        d = feat - self.mu
        return float((d - (d @ self.V) @ self.V.T).norm())  # PCA reconstruction error (distance to the clean-frame subspace)


def run_episode(env, plan, featfn, gen, rule, n, h, tau, corr, par, L, env_seed, t0, diag=None, curve_log=None, feat_log=None):
    cor = Corruptor(corr, par, t0, L, env_seed)
    obs, _ = env.reset(seed=env_seed)
    chunk, k, H_cur, prevA, prevH = None, 0, 1, None, 0
    buf = []  # (t_i, chunk) for TE
    maxr, t, done = 0.0, 0, False
    replans = replans_w = forced_w = 0
    prev_pix = None
    det_in = det_out = st_in = st_out = 0
    first_det = None
    Hs = []
    rec = []
    while not done:
        o = cor(t, obs)
        blind = cor.active(t)
        det = False
        if rule in ("detect", "generic") or diag is not None:
            d_black, d_freeze = detect_blind(o["pixels"], prev_pix)
            prev_pix = {kk: v.copy() for kk, v in o["pixels"].items()}
            sdet = d_black or d_freeze
        score = float("nan")
        if rule == "generic" or feat_log is not None or (diag is not None and gen is not None):
            feat = featfn(o)
            if feat_log is not None: feat_log.append(feat.cpu().numpy().astype(np.float16))
            if gen is not None: score = gen.score(feat)
        if rule == "detect":
            det = sdet
        elif rule == "generic":
            det = score > gen.thr
        if rule in ("detect", "generic"):
            if blind: st_in += 1
            else: st_out += 1
            if det:
                if blind:
                    det_in += 1
                    if first_det is None: first_det = t
                else:
                    det_out += 1
        if diag is not None:  # diagnostic: harm label = ||chunk(corrupted) - chunk(true image)|| over first 50 actions
            obs_true = {**o, "pixels": obs["pixels"]}
            Ac, At = plan(o), plan(obs_true)
            jit = {kk: np.clip(v.astype(np.int16) + np.random.default_rng([31, env_seed, t]).integers(-1, 2, v.shape), 0, 255).astype(np.uint8)
                   for kk, v in obs["pixels"].items()}
            Aj = plan({**obs, "pixels": jit})
            d = lambda A, B: float(np.linalg.norm(A[:50] - B[:50], axis=1).mean())
            rec.append((t, int(blind), d(Ac, At), d(At, Aj), score, int(sdet),
                        float(np.mean([v.mean() for v in o["pixels"].values()])), int(cor.flicker_on)))
        # ---------------- choose action
        if rule in ("median_te", "mean_te"):
            if t % h == 0:
                A = plan(o); replans += 1; replans_w += int(blind)
                buf.append((t, A))
                buf = [(ti, Ai) for ti, Ai in buf if t - ti < CHUNK]
            cands = np.stack([Ai[t - ti] for ti, Ai in buf])
            if rule == "median_te":
                a = np.median(cands, axis=0)
            else:
                w = np.exp(-0.01 * np.arange(len(cands))); a = (w[:, None] * cands).sum(0) / w.sum()
        elif rule == "care":
            if chunk is None or k >= H_cur:
                A = plan(o); replans += 1; replans_w += int(blind)
                if prevA is None:
                    H = 1
                else:
                    H = care_H(care_curve(A, prevA, prevH), tau)
                Hs.append(H); prevA, prevH, chunk, k, H_cur = A, H, A, 0, H
            a = chunk[k]; k += 1
        else:
            exhausted = chunk is None or k >= CHUNK
            need = exhausted
            if not exhausted:
                if rule == "fixed": need = k >= n
                elif rule == "oracle": need = (k >= n) and not blind
                else: need = (k >= n) and not det
            if need:
                new_chunk = plan(o)
                if curve_log is not None and chunk is not None:
                    curve_log.append(care_curve(new_chunk, chunk, k))  # k = steps executed from the old chunk
                chunk = new_chunk
                replans += 1
                if blind:
                    replans_w += 1
                    if exhausted and rule in ("oracle", "detect", "generic"): forced_w += 1
                k = 0
            a = chunk[k]; k += 1
        obs, r, term, trunc, info = env.step(a)
        maxr = max(maxr, float(r)); t += 1
        done = term or trunc
    out = dict(success=int(maxr >= 4), max_reward=maxr, replans_in_window=replans_w, replans_total=replans, steps=t,
               forced_replans_in_window=forced_w, det_in=det_in, det_out=det_out, steps_in=st_in, steps_out=st_out,
               latency=("" if first_det is None or L == 0 else first_det - t0),
               care_meanH=(float(np.mean(Hs)) if Hs else ""), care_queries=(len(Hs) if Hs else ""))
    return (out, np.array(rec)) if diag is not None else out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rule", choices=["fixed", "oracle", "detect", "generic", "median_te", "mean_te", "care"], required=True)
    ap.add_argument("--n", type=int, nargs="+", default=[100], help="fixed/oracle/detect/generic only")
    ap.add_argument("--h", type=int, nargs="+", default=[1], help="TE query stride")
    ap.add_argument("--tau", type=float, nargs="+", default=[float("inf")], help="care")
    ap.add_argument("--L", type=int, nargs="+", default=[0])
    ap.add_argument("--corr", default="black", choices=list(KINDS))
    ap.add_argument("--param", default="", help="intensity: sigma | area[:center|rand] | d | p")
    ap.add_argument("--episodes", type=int, default=100)
    ap.add_argument("--seed0", type=int, default=0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--generic", default="results/parts/generic_detector.npz")
    ap.add_argument("--diag_dir", default=None, help="save per-step diagnostic arrays (3 forward passes/step; rule fixed only)")
    ap.add_argument("--curve_dir", default=None, help="CARE tau calibration: save cumulative discrepancy curves (rule fixed)")
    ap.add_argument("--feat_dir", default=None, help="generic calibration: save per-step pooled backbone features (rule fixed)")
    a = ap.parse_args()
    T = TASKS["transfer_cube"]; REPO = T["repo"]
    device = "cuda"
    policy = ACTPolicy.from_pretrained(REPO).to(device).eval()
    st = load_file(hf_hub_download(REPO, "model.safetensors"))
    g = lambda kk: st[kk].to(device)
    norm = {"observation.images.top": (g("normalize_inputs.buffer_observation_images_top.mean"), g("normalize_inputs.buffer_observation_images_top.std")),
            "observation.state": (g("normalize_inputs.buffer_observation_state.mean"), g("normalize_inputs.buffer_observation_state.std"))}
    amean, astd = g("unnormalize_outputs.buffer_action.mean"), g("unnormalize_outputs.buffer_action.std")

    def prep(o):
        batch = preprocess_observation(o)
        batch = {kk: v.to(device) for kk, v in batch.items()}
        return {kk: ((v - norm[kk][0]) / (norm[kk][1] + 1e-8) if kk in norm else v) for kk, v in batch.items()}

    def plan(o):
        batch = prep(o)
        with torch.inference_mode():
            ch = policy.predict_action_chunk(batch)
        return (ch * astd + amean).squeeze(0).cpu().numpy().astype(np.float64)

    def featfn(o):
        batch = prep(o)
        with torch.inference_mode():
            fm = policy.model.backbone(batch["observation.images.top"])["feature_map"]
            return torch.nn.functional.adaptive_avg_pool2d(fm, (3, 4)).flatten()

    gen = Generic(a.generic, device) if (a.rule == "generic" or a.diag_dir) and os.path.exists(a.generic) else None
    if a.rule == "generic" and gen is None:
        raise SystemExit("generic detector file missing: " + a.generic)
    par = parse_param(a.corr, a.param)
    env = gym.make(T["env_id"], obs_type="pixels_agent_pos", max_episode_steps=T["max_steps"])
    new = not os.path.exists(a.out)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    f = open(a.out, "a", newline="")
    w = csv.DictWriter(f, FIELDS)
    if new: w.writeheader()
    for d_ in (a.diag_dir, a.curve_dir, a.feat_dir):
        if d_: os.makedirs(d_, exist_ok=True)
    if a.rule in ("median_te", "mean_te"):
        grid = [dict(n=0, h=hh, tau="") for hh in a.h]
    elif a.rule == "care":
        grid = [dict(n=0, h=0, tau=tt) for tt in a.tau]
    else:
        grid = [dict(n=nn, h=0, tau="") for nn in a.n]
    for L in a.L:
        for gg in grid:
            for ep in range(a.episodes):
                es, t0 = a.seed0 + ep, t0_for(ep, a.seed0, T["t0_lo"], T["t0_hi"])
                t_s = time.time()
                curve_log = [] if a.curve_dir else None
                feat_log = [] if a.feat_dir else None
                res = run_episode(env, plan, featfn, gen, a.rule, gg["n"] or CHUNK, gg["h"], gg["tau"] if gg["tau"] != "" else 0.0,
                                  a.corr, par, L, es, t0, diag=({} if a.diag_dir else None), curve_log=curve_log, feat_log=feat_log)
                if a.diag_dir:
                    res, rec = res
                    np.save(os.path.join(a.diag_dir, f"L{L}_n{gg['n']}_seed{es}.npy"), rec)
                if a.feat_dir:
                    np.save(os.path.join(a.feat_dir, f"seed{es}.npy"), np.stack(feat_log))
                if a.curve_dir and curve_log:
                    np.save(os.path.join(a.curve_dir, f"n{gg['n']}_seed{es}.npy"), np.array(curve_log))
                nm = {"fixed": f"fixed_n{gg['n']}", "oracle": f"oracle_n{gg['n']}", "detect": f"detect_n{gg['n']}",
                      "generic": f"generic_n{gg['n']}", "median_te": f"medianTE_h{gg['h']}", "mean_te": f"meanTE_h{gg['h']}",
                      "care": f"care_tau{gg['tau'] if gg['tau'] == '' else format(gg['tau'], '.4g')}"}[a.rule]
                row = dict(config=f"{nm}_L{L}_{tag(a.corr, par)}", rule=a.rule, corr=tag(a.corr, par), n=gg["n"], h=gg["h"],
                           tau=gg["tau"], L=L, episode=ep, env_seed=es, t0=t0, seconds=round(time.time() - t_s, 2), **res)
                w.writerow(row); f.flush(); print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
