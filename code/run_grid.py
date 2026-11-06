import argparse, csv, os, time, json
os.environ.setdefault("MUJOCO_GL", "egl")
import numpy as np, torch, gymnasium as gym, gym_aloha  # noqa
from lerobot.policies.act.modeling_act import ACTPolicy
from safetensors.torch import load_file
from huggingface_hub import hf_hub_download
from lerobot.envs.utils import preprocess_observation
from blackout import Blackout
from tasks import TASKS

REPO = "lerobot/act_aloha_sim_transfer_cube_human"
FIELDS = ["config", "n", "te", "L", "mode", "episode", "env_seed", "t0", "success", "max_reward",
          "replans_in_blackout", "replans_total", "mean_abs_d2", "steps", "seconds"]


def t0_for(ep, seed0, lo=50, hi=250):
    return int(np.random.default_rng(10_000 + seed0 + ep).integers(lo, hi + 1))  # uniform on [lo,hi] (default [50,250])


def run_episode(env, policy, pre, post, n, te, L, mode, env_seed, t0, device):
    policy.config.n_action_steps = n
    policy.reset()
    bo = Blackout(t0, L, mode)
    obs, _ = env.reset(seed=env_seed)
    maxr, acts, replans, replans_bo, t, done = 0.0, [], 0, 0, 0, False
    while not done:
        o = bo(t, obs)
        batch = preprocess_observation(o)
        batch = {k: v.to(device) for k, v in batch.items()}
        batch = pre(batch)
        # replan happens when the queue is empty (or every step with temporal ensembling)
        if te:
            replan = True
        else:
            replan = len(policy._action_queue) == 0
        with torch.inference_mode():
            a = policy.select_action(batch)
        a = post(a).squeeze(0).cpu().numpy()
        if replan:
            replans += 1
            if bo.active(t):
                replans_bo += 1
        acts.append(a)
        obs, r, term, trunc, info = env.step(a)
        maxr = max(maxr, float(r)); t += 1
        done = term or trunc
    acts = np.array(acts)
    d2 = np.abs(acts[2:] - 2 * acts[1:-1] + acts[:-2]).mean() if len(acts) > 2 else float("nan")
    return dict(success=int(maxr >= 4), max_reward=maxr, replans_in_blackout=replans_bo,
                replans_total=replans, mean_abs_d2=float(d2), steps=t)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, nargs="+", default=[1, 10, 25, 50, 100])
    ap.add_argument("--L", type=int, nargs="+", default=[0, 50])
    ap.add_argument("--mode", default="black")
    ap.add_argument("--te", action="store_true", help="temporal ensembling (coeff 0.01, n=1); ignores --n")
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--seed0", type=int, default=0)
    ap.add_argument("--out", default="results/pilot.csv")
    ap.add_argument("--task", default="transfer_cube", choices=list(TASKS))
    a = ap.parse_args()
    T = TASKS[a.task]; REPO = T["repo"]
    device = "cuda"
    policy = ACTPolicy.from_pretrained(REPO).to(device).eval()
    # This legacy checkpoint ships no processor files; normalization stats live in the safetensors buffers.
    st = load_file(hf_hub_download(REPO, "model.safetensors"))
    g = lambda k: st[k].to(device)
    norm = {"observation.images.top": (g("normalize_inputs.buffer_observation_images_top.mean"), g("normalize_inputs.buffer_observation_images_top.std")),
            "observation.state": (g("normalize_inputs.buffer_observation_state.mean"), g("normalize_inputs.buffer_observation_state.std"))}
    amean, astd = g("unnormalize_outputs.buffer_action.mean"), g("unnormalize_outputs.buffer_action.std")
    pre = lambda b: {k: ((v - norm[k][0]) / (norm[k][1] + 1e-8) if k in norm else v) for k, v in b.items()}
    post = lambda a: a * astd + amean
    if a.te:  # rebuild policy with temporal ensembling
        cfg = policy.config; cfg.temporal_ensemble_coeff = 0.01; cfg.n_action_steps = 1
        sd = policy.state_dict(); policy = ACTPolicy(cfg).to(device).eval(); policy.load_state_dict(sd)
        ns = [1]
    else:
        ns = a.n
    env = gym.make(T["env_id"], obs_type="pixels_agent_pos", max_episode_steps=T["max_steps"])
    new = not os.path.exists(a.out)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    f = open(a.out, "a", newline=""); w = csv.DictWriter(f, FIELDS)
    if new: w.writeheader()
    for L in a.L:
        for n in ns:
            for ep in range(a.episodes):
                es, t0 = a.seed0 + ep, t0_for(ep, a.seed0, T["t0_lo"], T["t0_hi"])
                t_s = time.time()
                r = run_episode(env, policy, pre, post, n, a.te, L, a.mode, es, t0, device)
                row = dict(config=f"{'TE' if a.te else 'n'+str(n)}_L{L}_{a.mode}", n=n, te=int(a.te), L=L,
                           mode=a.mode, episode=ep, env_seed=es, t0=t0, seconds=round(time.time() - t_s, 2), **r)
                w.writerow(row); f.flush(); print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
