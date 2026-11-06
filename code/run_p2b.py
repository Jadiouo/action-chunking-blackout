"""P2b runner (white-team supplements, docs/p2b_prereg.md).  Thin wrapper around code/run_p2p3.py: the SAME run_episode loop
(own chunk indexing, full 100-step chunk from predict_action_chunk), plus
  --task insertion|transfer_cube   (run_p2p3 is hard-wired to Transfer Cube)
  --act_dtype float64|float32      (dtype of the action handed to env.step; run_p2p3 = float64, run_grid.py = float32)
  extra corruptions blur / dim     (code/corrupt_p2b.py)
The generic detector is frozen: its sha256 is checked at start-up."""
import argparse, csv, hashlib, json, os, time
os.environ.setdefault("MUJOCO_GL", "egl")
import numpy as np, torch, gymnasium as gym, gym_aloha  # noqa
from lerobot.policies.act.modeling_act import ACTPolicy
from safetensors.torch import load_file
from huggingface_hub import hf_hub_download
from lerobot.envs.utils import preprocess_observation
import run_p2p3 as P
from run_grid import t0_for
from tasks import TASKS
import corrupt_p2b as C2

DET_SHA = "5193a96b546ffd97d4632bf2c1eb8b01b40b049748c77e62241cf600e244c2f2"
FIELDS = ["task", "act_dtype"] + P.FIELDS
P.Corruptor = C2.Corruptor2  # run_episode looks the class up in its module globals


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="transfer_cube", choices=list(TASKS))
    ap.add_argument("--rule", choices=["fixed", "oracle", "detect", "generic", "median_te", "mean_te"], required=True)
    ap.add_argument("--n", type=int, nargs="+", default=[100])
    ap.add_argument("--h", type=int, nargs="+", default=[1])
    ap.add_argument("--L", type=int, nargs="+", default=[0])
    ap.add_argument("--corr", default="black", choices=list(C2.KINDS2) + ["black", "freeze", "noise", "occlude", "delay", "flicker"])
    ap.add_argument("--param", default="")
    ap.add_argument("--episodes", type=int, default=100)
    ap.add_argument("--seed0", type=int, default=0)
    ap.add_argument("--act_dtype", default="float64", choices=["float64", "float32"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--generic", default="results/parts/generic_detector.npz")
    a = ap.parse_args()
    if a.rule == "generic":
        sha = hashlib.sha256(open(a.generic, "rb").read()).hexdigest()
        if sha != DET_SHA:
            raise SystemExit(f"generic detector changed! sha256 {sha} != {DET_SHA}")
    T = TASKS[a.task]; REPO = T["repo"]
    device = "cuda"
    policy = ACTPolicy.from_pretrained(REPO).to(device).eval()
    st = load_file(hf_hub_download(REPO, "model.safetensors"))
    g = lambda kk: st[kk].to(device)
    norm = {"observation.images.top": (g("normalize_inputs.buffer_observation_images_top.mean"), g("normalize_inputs.buffer_observation_images_top.std")),
            "observation.state": (g("normalize_inputs.buffer_observation_state.mean"), g("normalize_inputs.buffer_observation_state.std"))}
    amean, astd = g("unnormalize_outputs.buffer_action.mean"), g("unnormalize_outputs.buffer_action.std")
    npdt = np.float64 if a.act_dtype == "float64" else np.float32

    def prep(o):
        batch = preprocess_observation(o)
        batch = {kk: v.to(device) for kk, v in batch.items()}
        return {kk: ((v - norm[kk][0]) / (norm[kk][1] + 1e-8) if kk in norm else v) for kk, v in batch.items()}

    def plan(o):
        batch = prep(o)
        with torch.inference_mode():
            ch = policy.predict_action_chunk(batch)
        return (ch * astd + amean).squeeze(0).cpu().numpy().astype(npdt)

    def featfn(o):
        batch = prep(o)
        with torch.inference_mode():
            fm = policy.model.backbone(batch["observation.images.top"])["feature_map"]
            return torch.nn.functional.adaptive_avg_pool2d(fm, (3, 4)).flatten()

    gen = P.Generic(a.generic, device) if a.rule == "generic" else None
    par = C2.parse_param(a.corr, a.param)
    env = gym.make(T["env_id"], obs_type="pixels_agent_pos", max_episode_steps=T["max_steps"])
    new = not os.path.exists(a.out)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    f = open(a.out, "a", newline="")
    w = csv.DictWriter(f, FIELDS)
    if new: w.writeheader()
    grid = [dict(n=0, h=hh) for hh in a.h] if a.rule in ("median_te", "mean_te") else [dict(n=nn, h=0) for nn in a.n]
    for L in a.L:
        for gg in grid:
            for ep in range(a.episodes):
                es, t0 = a.seed0 + ep, t0_for(ep, a.seed0, T["t0_lo"], T["t0_hi"])
                t_s = time.time()
                res = P.run_episode(env, plan, featfn, gen, a.rule, gg["n"] or P.CHUNK, gg["h"], 0.0, a.corr, par, L, es, t0)
                nm = {"fixed": f"fixed_n{gg['n']}", "oracle": f"oracle_n{gg['n']}", "detect": f"detect_n{gg['n']}",
                      "generic": f"generic_n{gg['n']}", "median_te": f"medianTE_h{gg['h']}", "mean_te": f"meanTE_h{gg['h']}"}[a.rule]
                row = dict(task=a.task, act_dtype=a.act_dtype, config=f"{nm}_L{L}_{C2.tag(a.corr, par)}", rule=a.rule,
                           corr=C2.tag(a.corr, par), n=gg["n"], h=gg["h"], tau="", L=L, episode=ep, env_seed=es, t0=t0,
                           seconds=round(time.time() - t_s, 2), **res)
                w.writerow(row); f.flush(); print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
