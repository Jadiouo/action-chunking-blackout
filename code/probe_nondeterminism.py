"""Locate the source of run-to-run differences: (a) same observation -> policy chunk repeated; (b) env render/reset repeated;
(c) same rollout of fixed (recorded) actions repeated -> observation/state differences."""
import os, sys
os.environ.setdefault("MUJOCO_GL", "egl")
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np, torch, gymnasium as gym, gym_aloha  # noqa
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.envs.utils import preprocess_observation
from safetensors.torch import load_file
from huggingface_hub import hf_hub_download
from run_grid import REPO
dev = "cuda"
pol = ACTPolicy.from_pretrained(REPO).to(dev).eval()
st = load_file(hf_hub_download(REPO, "model.safetensors")); g = lambda k: st[k].to(dev)
nm = {"observation.images.top": (g("normalize_inputs.buffer_observation_images_top.mean"), g("normalize_inputs.buffer_observation_images_top.std")),
      "observation.state": (g("normalize_inputs.buffer_observation_state.mean"), g("normalize_inputs.buffer_observation_state.std"))}
pre = lambda b: {k: ((v - nm[k][0]) / (nm[k][1] + 1e-8) if k in nm else v) for k, v in b.items()}
env = gym.make("gym_aloha/AlohaTransferCube-v0", obs_type="pixels_agent_pos", max_episode_steps=400)
obs, _ = env.reset(seed=1)
b = pre({k: v.to(dev) for k, v in preprocess_observation(obs).items()})
with torch.inference_mode():
    outs = [pol.predict_action_chunk(b).cpu().numpy() for _ in range(20)]
print("(a) same obs, 20 forward passes: max abs diff between chunks =", max(np.abs(o - outs[0]).max() for o in outs))
# (b) env reset twice with same seed
o2, _ = env.reset(seed=1)
print("(b) reset(seed=1) twice: pixel max abs diff =", int(np.abs(o2["pixels"]["top"].astype(int) - obs["pixels"]["top"].astype(int)).max()),
      "agent_pos diff =", float(np.abs(o2["agent_pos"] - obs["agent_pos"]).max()))
# (c) replay a fixed action sequence twice
rng = np.random.default_rng(0)
def rollout(acts):
    o, _ = env.reset(seed=1); P = []
    for a in acts:
        o, r, te, tr, _ = env.step(a); P.append((o["agent_pos"].copy(), o["pixels"]["top"].copy()))
    return P
# use a plausible action sequence: policy n=25 rollout actions recorded once
pol.config.n_action_steps = 25; pol.reset()
astd, amean = g("unnormalize_outputs.buffer_action.std"), g("unnormalize_outputs.buffer_action.mean")
o, _ = env.reset(seed=1); acts = []
for t in range(150):
    bb = pre({k: v.to(dev) for k, v in preprocess_observation(o).items()})
    with torch.inference_mode(): a = (pol.select_action(bb) * astd + amean).squeeze(0).cpu().numpy()
    acts.append(a); o, *_ = env.step(a)
P1, P2 = rollout(acts), rollout(acts)
print("(c) replay identical 150 actions twice: max agent_pos diff =", max(float(np.abs(x[0] - y[0]).max()) for x, y in zip(P1, P2)),
      "| max pixel diff =", max(int(np.abs(x[1].astype(int) - y[1].astype(int)).max()) for x, y in zip(P1, P2)))
