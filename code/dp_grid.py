"""Blackout grid for a LeRobot Diffusion Policy checkpoint (own loop; does not touch run_t3t4.py).

Same protocol as run_grid.py / run_t3t4.py: gym_aloha/AlohaTransferCube-v0, 400 steps, success = max reward 4,
env seed = seed0+episode, blackout start t0 ~ U{50..250} from t0_for(), image-only blackout (code/blackout.py,
modes black|freeze, agent_pos untouched).

Execution rules (k = number of actions already executed from the current chunk; K = horizon - n_obs_steps + 1 is the
number of executable actions one diffusion sample provides, 15 for horizon 16):
  fixed  : replan when k >= n.
  oracle : replan when k >= n and the camera is NOT blind; while blind keep executing the stale chunk; if the chunk (K)
           runs out while blind, replan on the (bad) current observation (counted as forced_replans_in_blackout).
DP difference to ACT: the policy conditions on n_obs_steps=2 consecutive observations.  The observation queue is fed
EVERY env step with what the policy-visible camera gives (blind frames included), as lerobot's select_action does.
Rows are appended to --out; (config, episode) pairs already present are skipped (resumable).
"""
import argparse, csv, json, os, sys, time
os.environ.setdefault("MUJOCO_GL", "egl")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, torch, gymnasium as gym, gym_aloha  # noqa
from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.utils import populate_queues
from lerobot.envs.utils import preprocess_observation
from lerobot.utils.constants import OBS_IMAGES
from blackout import Blackout
from run_grid import t0_for

FIELDS = ["config", "rule", "n", "L", "mode", "episode", "env_seed", "t0", "success", "max_reward",
          "replans_in_blackout", "replans_total", "forced_replans_in_blackout", "steps", "seconds"]


def load(ckpt, dev, no_ema=False):
    policy = DiffusionPolicy.from_pretrained(ckpt).to(dev).eval()
    ef = os.path.join(ckpt, "ema.safetensors")
    if os.path.exists(ef) and not no_ema:
        from safetensors.torch import load_file
        ema = load_file(ef, device=dev); ema.pop("__ema_step__", None)
        policy.load_state_dict(ema, strict=True)
    pre, post = make_pre_post_processors(policy.config, pretrained_path=ckpt,
                                         preprocessor_overrides={"device_processor": {"device": dev}})
    return policy, pre, post


def run_episode(env, policy, pre, post, rule, n, L, mode, env_seed, t0, dev):
    K = policy.config.horizon - policy.config.n_obs_steps + 1
    policy.config.n_action_steps = K  # one sample -> K executable actions; the loop below decides how many to use
    policy.reset()
    bo = Blackout(t0, L, mode)
    obs, _ = env.reset(seed=env_seed)
    chunk, k, maxr, t, done = None, 0, 0.0, 0, False
    replans = replans_bo = forced_bo = 0
    while not done:
        o = bo(t, obs)
        blind = bo.active(t)
        b = preprocess_observation(o)
        b = pre({kk: v.to(dev) for kk, v in b.items()})
        b.pop("action", None)  # processor may add action=None (lerobot select_action pops it too)
        b[OBS_IMAGES] = torch.stack([b[kk] for kk in policy.config.image_features], dim=-4)
        policy._queues = populate_queues(policy._queues, b)  # observation queue sees every step
        exhausted = chunk is None or k >= K
        need = exhausted
        if not exhausted:
            need = (k >= n) if rule == "fixed" else ((k >= n) and not blind)
        if need:
            with torch.inference_mode():
                ch = policy.predict_action_chunk(b)  # (1, K, 14) normalised
            chunk = post(ch)[0].cpu().numpy().astype(np.float64)
            replans += 1
            if blind:
                replans_bo += 1
                if rule == "oracle" and exhausted:
                    forced_bo += 1
            k = 0
        a = chunk[k]; k += 1
        obs, r, term, trunc, _ = env.step(a)
        maxr = max(maxr, float(r)); t += 1
        done = term or trunc
    return dict(success=int(maxr >= 4), max_reward=maxr, replans_in_blackout=replans_bo, replans_total=replans,
                forced_replans_in_blackout=forced_bo, steps=t)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--rule", choices=["fixed", "oracle"], nargs="+", default=["fixed"])
    ap.add_argument("--n", type=int, nargs="+", default=[8])
    ap.add_argument("--L", type=int, nargs="+", default=[0])
    ap.add_argument("--mode", nargs="+", default=["black"])
    ap.add_argument("--episodes", type=int, default=50)
    ap.add_argument("--seed0", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--no_ema", action="store_true")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    dev = a.device
    policy, pre, post = load(a.ckpt, dev, a.no_ema)
    env = gym.make("gym_aloha/AlohaTransferCube-v0", obs_type="pixels_agent_pos", max_episode_steps=400)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    done_keys = set()
    new = not os.path.exists(a.out)
    if not new:
        done_keys = {(r["config"], int(r["episode"])) for r in csv.DictReader(open(a.out))}
    f = open(a.out, "a", newline="")
    w = csv.DictWriter(f, FIELDS)
    if new:
        w.writeheader()
    for L in a.L:
        for mode in (a.mode if L > 0 else a.mode[:1]):  # L=0: blackout mode is irrelevant, run once
            for rule in a.rule:
                for n in a.n:
                    cfg = f"dp_{rule}_n{n}_L{L}_{mode}"
                    if L == 0 and rule == "oracle":  # identical to fixed when never blind
                        continue
                    for ep in range(a.episodes):
                        if (cfg, ep) in done_keys:
                            continue
                        es, t0 = a.seed0 + ep, t0_for(ep, a.seed0)
                        torch.manual_seed(es)
                        t_s = time.time()
                        res = run_episode(env, policy, pre, post, rule, n, L, mode, es, t0, dev)
                        row = dict(config=cfg, rule=rule, n=n, L=L, mode=mode, episode=ep, env_seed=es, t0=t0,
                                   seconds=round(time.time() - t_s, 2), **res)
                        w.writerow(row); f.flush(); print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
