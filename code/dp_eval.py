"""Evaluate a LeRobot Diffusion Policy checkpoint on gym_aloha/AlohaTransferCube-v0 (same env settings, seeds and
blackout protocol as run_grid.py).  Writes one CSV row per episode and a JSON summary (success rate + Wilson 95% CI).

  .venv/bin/python code/dp_eval.py --ckpt runs/dp/<run>/checkpoints/last/pretrained_model --episodes 50 --out results/dp_noblack.csv

--n overrides policy.config.n_action_steps (default: the value stored in the checkpoint config, i.e. DP's own setting).
--L/--mode add the Blackout of code/blackout.py (default L=0 = no blackout).  This script does not change run_t3t4.py.
"""
import argparse, csv, json, os, sys, time
os.environ.setdefault("MUJOCO_GL", "egl")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, torch, gymnasium as gym, gym_aloha  # noqa
from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
from lerobot.policies.factory import make_pre_post_processors
from lerobot.envs.utils import preprocess_observation
from blackout import Blackout
from run_grid import t0_for

FIELDS = ["config", "n", "L", "mode", "episode", "env_seed", "t0", "success", "max_reward",
          "replans_in_blackout", "replans_total", "mean_abs_d2", "steps", "seconds"]


def wilson(k, n, z=1.96):
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return float(max(0.0, c - h)), float(min(1.0, c + h))


def run_episode(env, policy, pre, post, L, mode, env_seed, t0, device):
    policy.reset()
    bo = Blackout(t0, L, mode)
    obs, _ = env.reset(seed=env_seed)
    maxr, acts, replans, replans_bo, t, done = 0.0, [], 0, 0, 0, False
    while not done:
        o = bo(t, obs)
        batch = preprocess_observation(o)
        batch = {k: v.to(device) for k, v in batch.items()}
        batch = pre(batch)
        replan = len(policy._queues["action"]) == 0  # queue empty -> this select_action call runs the diffusion sampler
        with torch.inference_mode():
            a = policy.select_action(batch)
        a = post(a).squeeze(0).cpu().numpy()
        if replan:
            replans += 1
            replans_bo += int(bo.active(t))
        acts.append(a)
        obs, r, term, trunc, info = env.step(a)
        maxr = max(maxr, float(r)); t += 1
        done = term or trunc
    acts = np.array(acts)
    d2 = np.abs(acts[2:] - 2 * acts[1:-1] + acts[:-2]).mean() if len(acts) > 2 else float("nan")
    return dict(success=int(maxr >= 4), max_reward=maxr, replans_in_blackout=replans_bo, replans_total=replans,
                mean_abs_d2=float(d2), steps=t)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="pretrained_model dir of a lerobot checkpoint")
    ap.add_argument("--n", type=int, default=None, help="n_action_steps override (default: from checkpoint)")
    ap.add_argument("--infer", default="default", help="default (checkpoint's sampler) | ddim<k> | ddpm<k>, e.g. ddpm100")
    ap.add_argument("--no_ema", action="store_true", help="ignore ema.safetensors even if present (default: use EMA weights when available)")
    ap.add_argument("--L", type=int, default=0)
    ap.add_argument("--mode", default="black")
    ap.add_argument("--episodes", type=int, default=50)
    ap.add_argument("--seed0", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed_torch", type=int, default=0, help="torch seed for the diffusion sampling noise")
    a = ap.parse_args()
    dev = a.device
    policy = DiffusionPolicy.from_pretrained(a.ckpt).to(dev).eval()
    ema_file = os.path.join(a.ckpt, "ema.safetensors")
    use_ema = os.path.exists(ema_file) and not a.no_ema
    if use_ema:  # EMA weights (written by code/dp_train_main.py next to model.safetensors)
        from safetensors.torch import load_file
        ema = load_file(ema_file, device=dev); ema.pop("__ema_step__", None)
        print(policy.load_state_dict(ema, strict=True), "EMA weights loaded", flush=True)
    if a.n is not None:
        policy.config.n_action_steps = a.n
    n = policy.config.n_action_steps
    if a.infer != "default":  # diagnostic: change the sampler at inference only (weights unchanged)
        from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
        from diffusers.schedulers.scheduling_ddim import DDIMScheduler
        c = policy.config
        kw = dict(num_train_timesteps=c.num_train_timesteps, beta_start=c.beta_start, beta_end=c.beta_end,
                  beta_schedule=c.beta_schedule, clip_sample=c.clip_sample, clip_sample_range=c.clip_sample_range,
                  prediction_type=c.prediction_type)
        kind, steps = a.infer[:4], int(a.infer[4:])
        policy.diffusion.noise_scheduler = (DDPMScheduler if kind == "ddpm" else DDIMScheduler)(**kw)
        policy.diffusion.num_inference_steps = steps
    pre, post = make_pre_post_processors(policy.config, pretrained_path=a.ckpt,
                                         preprocessor_overrides={"device_processor": {"device": dev}})
    env = gym.make("gym_aloha/AlohaTransferCube-v0", obs_type="pixels_agent_pos", max_episode_steps=400)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    new = not os.path.exists(a.out)
    f = open(a.out, "a", newline="")
    w = csv.DictWriter(f, FIELDS)
    if new:
        w.writeheader()
    rows = []
    for ep in range(a.episodes):
        es, t0 = a.seed0 + ep, t0_for(ep, a.seed0)
        torch.manual_seed(a.seed_torch * 100000 + es)
        t_s = time.time()
        res = run_episode(env, policy, pre, post, a.L, a.mode, es, t0, dev)
        row = dict(config=f"dp_n{n}_L{a.L}_{a.mode}" + ("" if a.infer == "default" else f"_{a.infer}"), n=n, L=a.L, mode=a.mode, episode=ep, env_seed=es, t0=t0,
                   seconds=round(time.time() - t_s, 2), **res)
        w.writerow(row); f.flush(); rows.append(row); print(json.dumps(row), flush=True)
    k, N = sum(r["success"] for r in rows), len(rows)
    lo, hi = wilson(k, N)
    summ = dict(ckpt=a.ckpt, ema=use_ema, infer=a.infer, n_action_steps=n, L=a.L, mode=a.mode, episodes=N, successes=k, rate=k / N,
                wilson95=[lo, hi], mean_max_reward=float(np.mean([r["max_reward"] for r in rows])))
    json.dump(summ, open(os.path.splitext(a.out)[0] + "_summary.json", "w"), indent=1)
    print("SUMMARY", json.dumps(summ), flush=True)


if __name__ == "__main__":
    main()
