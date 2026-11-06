"""T1 diagnosis: record per-step trajectories and every fresh action-chunk prediction for n=1 / n=25 / TE
(and an 'offset' intervention: replan every step but execute chunk element j instead of element 0).
No blackout (L=0). Raw arrays go to .scratch/diag/*.npz; per-episode summary to results/n1_diag_runs.csv."""
import argparse, csv, os, sys, time
os.environ.setdefault("MUJOCO_GL", "egl")
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np, torch, gymnasium as gym, gym_aloha  # noqa
from lerobot.policies.act.modeling_act import ACTPolicy
from safetensors.torch import load_file
from huggingface_hub import hf_hub_download
from lerobot.envs.utils import preprocess_observation
from run_grid import REPO


def pos(env):
    p = env.unwrapped._env.physics
    x = p.named.data.xpos
    return (np.array(x["box"]), np.array(x["vx300s_right/gripper_link"]), np.array(x["vx300s_left/gripper_link"]))


def run(env, policy, pre, post, cfgname, n, seed, offset, device, outdir):
    policy.config.n_action_steps = n
    policy.reset()
    orig = policy.predict_action_chunk
    box = {}

    def wrapped(batch):
        out = orig(batch); box["c"] = out; return out
    policy.predict_action_chunk = wrapped
    obs, _ = env.reset(seed=seed)
    Q, A, R, CUBE, RG, LG, FRESH, REPL = [], [], [], [], [], [], [], []
    done, t = False, 0
    while not done:
        batch = preprocess_observation(obs)
        batch = {k: v.to(device) for k, v in batch.items()}
        batch = pre(batch)
        box.clear()
        with torch.inference_mode():
            if offset is None:
                a = policy.select_action(batch)
                if "c" in box:
                    fresh, repl = box["c"], 1
                else:
                    fresh, repl = orig(batch), 0
            else:  # intervention: replan every step, execute element `offset` of the fresh chunk
                fresh, repl = orig(batch), 1
                a = fresh[:, offset]
        a = post(a).squeeze(0).cpu().numpy()
        fresh = post(fresh).squeeze(0).cpu().numpy()  # (100,14)
        if offset is None and repl and n == 1 and cfgname.startswith("n1"):
            assert np.allclose(a, fresh[0], atol=1e-4), "select_action != fresh chunk[0]"
        Q.append(np.array(obs["agent_pos"])); A.append(a); FRESH.append(fresh.astype(np.float32)); REPL.append(repl)
        c, rg, lg = pos(env); CUBE.append(c); RG.append(rg); LG.append(lg)
        obs, r, term, trunc, _ = env.step(a)
        R.append(float(r)); t += 1; done = term or trunc
    policy.predict_action_chunk = orig
    d = dict(q=np.array(Q), a=np.array(A), r=np.array(R), cube=np.array(CUBE), rg=np.array(RG), lg=np.array(LG),
             fresh=np.array(FRESH), replan=np.array(REPL))
    if outdir:
        os.makedirs(outdir, exist_ok=True)
        np.savez_compressed(f"{outdir}/{cfgname}_s{seed}.npz", **d)
    return float(max(R)), t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", nargs="+", default=["n1", "n25", "te"], help="n1 n25 te off<j>")
    ap.add_argument("--seeds", type=int, nargs="+", default=list(range(8)))
    ap.add_argument("--outdir", default=".scratch/diag")
    ap.add_argument("--csv", default="results/n1_diag_runs.csv")
    ap.add_argument("--save", type=int, default=1)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--maxt", type=int, default=400)
    a = ap.parse_args()
    device = a.device
    base = ACTPolicy.from_pretrained(REPO).to(device).eval()
    st = load_file(hf_hub_download(REPO, "model.safetensors"))
    g = lambda k: st[k].to(device)
    norm = {"observation.images.top": (g("normalize_inputs.buffer_observation_images_top.mean"), g("normalize_inputs.buffer_observation_images_top.std")),
            "observation.state": (g("normalize_inputs.buffer_observation_state.mean"), g("normalize_inputs.buffer_observation_state.std"))}
    amean, astd = g("unnormalize_outputs.buffer_action.mean"), g("unnormalize_outputs.buffer_action.std")
    pre = lambda b: {k: ((v - norm[k][0]) / (norm[k][1] + 1e-8) if k in norm else v) for k, v in b.items()}
    post = lambda x: x * astd + amean
    import copy
    cfg = copy.deepcopy(base.config); cfg.temporal_ensemble_coeff = 0.01; cfg.n_action_steps = 1
    te_policy = ACTPolicy(cfg).to(device).eval(); te_policy.load_state_dict(base.state_dict())
    cfg2 = copy.deepcopy(base.config); cfg2.temporal_ensemble_coeff = None
    base2 = ACTPolicy(cfg2).to(device).eval(); base2.load_state_dict(base.state_dict())
    env = gym.make("gym_aloha/AlohaTransferCube-v0", obs_type="pixels_agent_pos", max_episode_steps=a.maxt)
    new = not os.path.exists(a.csv)
    f = open(a.csv, "a", newline=""); w = csv.writer(f)
    if new: w.writerow(["cfg", "seed", "max_reward", "success", "steps", "seconds"])
    for c in a.cfg:
        for s in a.seeds:
            t_s = time.time()
            if c == "te": pol, n, off = te_policy, 1, None
            elif c == "n1": pol, n, off = base2, 1, None
            elif c == "n25": pol, n, off = base2, 25, None
            elif c.startswith("off"): pol, n, off = base2, 1, int(c[3:])
            mr, T = run(env, pol, pre, post, c, n, s, off, device, a.outdir if a.save else None)
            w.writerow([c, s, mr, int(mr >= 4), T, round(time.time() - t_s, 1)]); f.flush()
            print(c, s, mr, T, flush=True)


if __name__ == "__main__":
    main()
