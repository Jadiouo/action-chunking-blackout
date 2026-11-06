"""T3 oracle and T4 proprioception-triggered replanning.

Own execution loop (no policy action queue): at each replan we take the full 100-step chunk from
policy.predict_action_chunk (unnormalised with the checkpoint's buffers, same as run_grid.py) and index into it.

rules
  fixed   : replan when k >= n                       (reproduces run_grid.py; used for sanity check and calibration)
  oracle  : replan when k >= n and not blind; while blind keep executing the stale chunk; if the chunk (100) runs
            out while blind, replan on the (bad) current image (counted as forced_replans_in_blackout)
  proprio : replan when k >= m and e_t > tau, or when the chunk (100) runs out.  Blindness is NOT known to the rule.
  detect  : T4b.  Like oracle, but "blind" is inferred ONLY from the image the policy receives (both detectors always on):
            black  = mean pixel value of every camera < BLACK_THR (uint8 units),
            freeze = image identical (bit-for-bit) to the image the policy received at the previous step.
            When detected: keep executing the stale chunk; replan only when k >= n and not detected, or chunk exhausted.
            Ground-truth blackout interval is used only for logging (detect_latency, false_detect), never for control.
k = number of actions already executed from the current chunk.  e_t = ||mask * (a_{t-1} - q_t)||_2.
"""
import argparse, csv, os, time, json
os.environ.setdefault("MUJOCO_GL", "egl")
import numpy as np, torch, gymnasium as gym, gym_aloha  # noqa
from lerobot.policies.act.modeling_act import ACTPolicy
from safetensors.torch import load_file
from huggingface_hub import hf_hub_download
from lerobot.envs.utils import preprocess_observation
from blackout import Blackout
from run_grid import t0_for, REPO
from tasks import TASKS

CHUNK = 100
BLACK_THR = 1.0  # T4b: black detector threshold on mean uint8 pixel value (= 1/255 of full scale)
FIELDS = ["config", "rule", "n", "m", "tau", "L", "mode", "episode", "env_seed", "t0", "success", "max_reward",
          "replans_in_blackout", "replans_total", "mean_abs_d2", "steps", "seconds",
          "forced_replans_in_blackout", "triggers", "triggers_in_blackout", "e_mean"]


DETECT_FIELDS = ["detected_steps", "detect_latency", "false_detect"]


def detect_blind(pix, prev_pix):
    """T4b detector on the policy-visible image dict. Returns (black, freeze). prev_pix may be None."""
    black = all(float(v.mean()) < BLACK_THR for v in pix.values())
    freeze = prev_pix is not None and all(np.array_equal(v, prev_pix[k]) for k, v in pix.items())
    return black, freeze


def sig_mask(sig):
    m = np.ones(14)
    if sig == "arm":
        m[[6, 13]] = 0
    elif sig == "all":
        pass
    else:
        raise ValueError(sig)
    return m


def run_episode(env, plan, rule, n, m, tau, mask, L, mode, env_seed, t0, record=False):
    bo = Blackout(t0, L, mode)
    obs, _ = env.reset(seed=env_seed)
    chunk, k, prev_a = None, 0, None
    maxr, acts, t, done = 0.0, [], 0, False
    replans = replans_bo = forced_bo = trig = trig_bo = 0
    es, rec = [], []
    prev_pix, det_steps, first_det, false_det = None, 0, None, 0
    while not done:
        o = bo(t, obs)
        blind = bo.active(t)
        q = np.asarray(obs["agent_pos"], dtype=np.float64)
        diff = np.zeros(14) if prev_a is None else (prev_a - q)
        e = float(np.linalg.norm(mask * diff))
        es.append(e)
        det = False
        if rule == "detect":
            d_black, d_freeze = detect_blind(o["pixels"], prev_pix)
            prev_pix = {kk: v.copy() for kk, v in o["pixels"].items()}
            det = d_black or d_freeze
            if det:
                det_steps += 1
                if first_det is None and blind:  # first detection inside the true blackout (latency = first_det - t0)
                    first_det = t
                if not blind:
                    false_det += 1
            if record:
                m_img = float(np.mean([v.mean() for v in o["pixels"].values()]))
        exhausted = chunk is None or k >= CHUNK
        need, trig_now = exhausted, False
        if not exhausted:
            if rule == "fixed":
                need = k >= n
            elif rule == "oracle":
                need = (k >= n) and not blind
            elif rule == "detect":
                need = (k >= n) and not det
            elif rule == "proprio":
                need = (k >= m) and (e > tau)
                trig_now = need
        if need:
            chunk = plan(o)
            replans += 1
            if blind:
                replans_bo += 1
                if rule in ("oracle", "detect") and exhausted:
                    forced_bo += 1
            if trig_now:
                trig += 1
                trig_bo += int(blind)
            k = 0
        a = chunk[k]; k += 1
        prev_a = a
        acts.append(a)
        obs, r, term, trunc, info = env.step(a)
        if record and rule == "detect":
            rec.append((t, m_img, int(d_black), int(d_freeze), float(r), int(need), int(blind)))
        elif record:
            rec.append((t, e, float(r), int(need), int(blind), *diff))
        maxr = max(maxr, float(r)); t += 1
        done = term or trunc
    acts = np.array(acts)
    d2 = np.abs(acts[2:] - 2 * acts[1:-1] + acts[:-2]).mean() if len(acts) > 2 else float("nan")
    out = dict(success=int(maxr >= 4), max_reward=maxr, replans_in_blackout=replans_bo, replans_total=replans,
               mean_abs_d2=float(d2), steps=t, forced_replans_in_blackout=forced_bo, triggers=trig,
               triggers_in_blackout=trig_bo, e_mean=float(np.mean(es)))
    if rule == "detect":
        out.update(detected_steps=det_steps, detect_latency=("" if (first_det is None or L == 0) else first_det - t0),
                   false_detect=false_det)
    return (out, np.array(rec)) if record else out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rule", choices=["fixed", "oracle", "proprio", "detect"], required=True)
    ap.add_argument("--n", type=int, nargs="+", default=[100])
    ap.add_argument("--m", type=int, nargs="+", default=[1], help="proprio: min steps before trigger allowed")
    ap.add_argument("--tau", type=float, default=float("inf"))
    ap.add_argument("--sig", default="all")
    ap.add_argument("--L", type=int, nargs="+", default=[0])
    ap.add_argument("--mode", nargs="+", default=["black"])
    ap.add_argument("--episodes", type=int, default=100)
    ap.add_argument("--seed0", type=int, default=0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--task", default="transfer_cube", choices=list(TASKS))
    ap.add_argument("--record_dir", default=None, help="save per-step arrays (npz) per episode (calibration)")
    a = ap.parse_args()
    T = TASKS[a.task]; REPO = T["repo"]
    device = "cuda"
    policy = ACTPolicy.from_pretrained(REPO).to(device).eval()
    st = load_file(hf_hub_download(REPO, "model.safetensors"))
    g = lambda k: st[k].to(device)
    norm = {"observation.images.top": (g("normalize_inputs.buffer_observation_images_top.mean"), g("normalize_inputs.buffer_observation_images_top.std")),
            "observation.state": (g("normalize_inputs.buffer_observation_state.mean"), g("normalize_inputs.buffer_observation_state.std"))}
    amean, astd = g("unnormalize_outputs.buffer_action.mean"), g("unnormalize_outputs.buffer_action.std")

    def plan(o):
        batch = preprocess_observation(o)
        batch = {k: v.to(device) for k, v in batch.items()}
        batch = {k: ((v - norm[k][0]) / (norm[k][1] + 1e-8) if k in norm else v) for k, v in batch.items()}
        with torch.inference_mode():
            ch = policy.predict_action_chunk(batch)  # (1, 100, 14) normalised
        return (ch * astd + amean).squeeze(0).cpu().numpy().astype(np.float64)

    env = gym.make(T["env_id"], obs_type="pixels_agent_pos", max_episode_steps=T["max_steps"])
    mask = sig_mask(a.sig)
    new = not os.path.exists(a.out)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    if a.record_dir:
        os.makedirs(a.record_dir, exist_ok=True)
    f = open(a.out, "a", newline="")
    w = csv.DictWriter(f, FIELDS + (DETECT_FIELDS if a.rule == "detect" else []))
    if new: w.writeheader()
    if a.rule == "proprio":
        grid = [(None, mm) for mm in a.m]
    else:
        grid = [(nn, None) for nn in a.n]
    for L in a.L:
        for mode in a.mode:
            for nn, mm in grid:
                for ep in range(a.episodes):
                    es, t0 = a.seed0 + ep, t0_for(ep, a.seed0, T["t0_lo"], T["t0_hi"])
                    t_s = time.time()
                    res = run_episode(env, plan, a.rule, nn if nn else CHUNK, mm if mm else 0, a.tau, mask, L, mode,
                                      es, t0, record=bool(a.record_dir))
                    if a.record_dir:
                        res, rec = res
                        np.save(os.path.join(a.record_dir, f"seed{es}.npy"), rec)
                    cfg = {"fixed": f"fixed_n{nn}", "oracle": f"oracle_n{nn}", "proprio": f"proprio_m{mm}", "detect": f"detect_n{nn}"}[a.rule]
                    row = dict(config=f"{cfg}_L{L}_{mode}", rule=a.rule, n=nn if nn else CHUNK, m=mm if mm else "",
                               tau=a.tau if a.rule == "proprio" else "", L=L, mode=mode, episode=ep, env_seed=es,
                               t0=t0, seconds=round(time.time() - t_s, 2), **res)
                    w.writerow(row); f.flush(); print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
