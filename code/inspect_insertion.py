"""Look at what "reward 4" means on Insertion (open question of the paper, Sec. 4 and Limitation 4).

Runs ten clean Insertion episodes (no blackout, fixed n=100, same environment, policy, action dtype and execution loop as
code/run_p2b.py) and stores the top-camera image at the first step whose reward is 4 and at the last step, with that step's reward.
Seeds are taken from results/insertion.csv (rule fixed, n=100, L=0, black, te=0; LeRobot loop): the first 6 successes by seed, and
4 failures chosen to show different outcomes (the first 2 failures with max reward 3, the first 2 with max reward <= 2).  Episodes are
not bit-reproducible across loops, so every row also says whether this run reproduced the success of the reference rows.

Outputs (all under the project folder):
  results/insertion_frames/seed<S>_first_r4_step<T>.png   top image returned by the env.step that first gave reward 4 (successes only)
  results/insertion_frames/seed<S>_best_step<T>.png       top image at the first step with the episode's maximum reward (failures only)
  results/insertion_frames/seed<S>_final_step<T>.png      top image returned by the last env.step
  results/insertion_frames/contact_sheet.png              one row per episode: [first reward-4 (or best-reward) frame | final frame]
  results/insertion_inspect.csv                           seed, reproduced success, first_r4_step, final_reward, ...

Run (optionally via the local queue wrapper gpujob, see README.md):
  MUJOCO_GL=egl HF_HOME=$PWD/.hf HF_HUB_OFFLINE=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 gpujob systemd-run --user --scope -q \
      -p MemoryMax=12G -p MemorySwapMax=0 .venv/bin/python code/inspect_insertion.py
"""
import csv
import os
import sys

os.environ.setdefault("MUJOCO_GL", "egl")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import torch
import gymnasium as gym
import gym_aloha  # noqa: F401
from PIL import Image, ImageDraw
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.envs.utils import preprocess_observation
from safetensors.torch import load_file
from huggingface_hub import hf_hub_download

import run_p2p3 as P
import corrupt_p2b as C2
from run_grid import t0_for
from tasks import TASKS

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
OUT_DIR = os.path.join(ROOT, "results", "insertion_frames")
OUT_CSV = os.path.join(ROOT, "results", "insertion_inspect.csv")
N_FIXED = 100          # execution horizon of the clean reference (fixed n=100, whole chunk)
TASK = "insertion"


def read_rows(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def pick_seeds():
    """6 successes and 4 failures of fixed n=100 without blackout in results/insertion.csv (rule documented in the module docstring)."""
    rows = [r for r in read_rows(os.path.join(ROOT, "results", "insertion.csv"))
            if r["rule"] == "fixed" and r["te"] == "0" and int(r["n"]) == N_FIXED and int(r["L"]) == 0 and r["mode"] == "black"]
    assert len(rows) == 200, len(rows)
    rows.sort(key=lambda r: int(r["env_seed"]))
    succ = [r for r in rows if int(r["success"]) == 1]
    f3 = [r for r in rows if int(r["success"]) == 0 and float(r["max_reward"]) == 3]
    flow = [r for r in rows if int(r["success"]) == 0 and float(r["max_reward"]) <= 2]
    chosen = succ[:6] + f3[:2] + flow[:2]
    assert len(chosen) == 10
    ref_same_loop = {int(r["env_seed"]): int(r["success"]) for r in read_rows(os.path.join(ROOT, "results", "p2b_insertion_fixed.csv"))
                     if int(r["n"]) == N_FIXED and int(r["L"]) == 0}
    return [dict(seed=int(r["env_seed"]), ref_success=int(r["success"]), ref_max_reward=float(r["max_reward"]),
                 ref_same_loop_success=ref_same_loop.get(int(r["env_seed"]), "")) for r in chosen]


def build_policy():
    T = TASKS[TASK]
    device = "cuda"
    policy = ACTPolicy.from_pretrained(T["repo"]).to(device).eval()
    st = load_file(hf_hub_download(T["repo"], "model.safetensors"))
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
        return (ch * astd + amean).squeeze(0).cpu().numpy().astype(np.float64)   # float64 actions = run_p2b default (same loop)

    return plan


def top_image(obs):
    return np.asarray(obs["pixels"]["top"]).copy()


def run_clean_episode(env, plan, seed):
    """fixed n=100, no blackout; same structure as run_p2p3.run_episode for rule 'fixed' (chunk of 100, replan when exhausted)."""
    T = TASKS[TASK]
    t0 = t0_for(seed, 0, T["t0_lo"], T["t0_hi"])
    cor = C2.Corruptor2("black", C2.parse_param("black", ""), t0, 0, seed)       # L=0: passes the image through unchanged
    obs, _ = env.reset(seed=seed)
    chunk, k, t, done = None, 0, 0, False
    maxr, best_t, best_img = 0.0, -1, None
    first_r4, first_img = None, None
    last_r, last_img = 0.0, None
    while not done:
        o = cor(t, obs)
        if chunk is None or k >= P.CHUNK or k >= N_FIXED:
            chunk = plan(o)
            k = 0
        a = chunk[k]
        k += 1
        obs, r, term, trunc, _ = env.step(a)
        r = float(r)
        img = top_image(obs)
        if r > maxr:                                     # first step with the new maximum reward
            maxr, best_t, best_img = r, t, img
        if r >= 4 and first_r4 is None:
            first_r4, first_img = t, img
        last_r, last_img = r, img
        t += 1
        done = term or trunc
    return dict(success=int(maxr >= 4), max_reward=maxr, steps=t, first_r4_step=first_r4, first_r4_img=first_img,
                best_step=best_t, best_img=best_img, final_step=t - 1, final_reward=last_r, final_img=last_img)


def save_png(arr, name):
    Image.fromarray(arr).save(os.path.join(OUT_DIR, name))


def contact_sheet(results, path, thumb=(320, 240)):
    bar = 18
    W, H = thumb
    sheet = Image.new("RGB", (2 * W, len(results) * (H + bar)), "white")
    d = ImageDraw.Draw(sheet)
    for i, r in enumerate(results):
        y = i * (H + bar)
        first = r["first_r4_img"] if r["first_r4_step"] is not None else r["best_img"]
        lab1 = (f"seed {r['seed']}  first reward-4 step {r['first_r4_step']} (reward 4.0)" if r["first_r4_step"] is not None
                else f"seed {r['seed']}  no reward 4; best reward {r['max_reward']:.0f} at step {r['best_step']}")
        lab2 = f"final step {r['final_step']} reward {r['final_reward']:.0f}  success {r['success']} (ref {r['ref_success']})"
        for j, (img, lab) in enumerate([(first, lab1), (r["final_img"], lab2)]):
            d.text((j * W + 3, y + 3), lab, fill="black")
            if img is not None:
                sheet.paste(Image.fromarray(img).resize(thumb), (j * W, y + bar))
    sheet.save(path)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    seeds = pick_seeds()
    print("seeds:", [(s["seed"], s["ref_success"], s["ref_max_reward"]) for s in seeds], flush=True)
    plan = build_policy()
    T = TASKS[TASK]
    env = gym.make(T["env_id"], obs_type="pixels_agent_pos", max_episode_steps=T["max_steps"])
    results = []
    for s in seeds:
        res = run_clean_episode(env, plan, s["seed"])
        res.update(s)
        results.append(res)
        if res["first_r4_img"] is not None:
            save_png(res["first_r4_img"], f"seed{s['seed']}_first_r4_step{res['first_r4_step']}.png")
        elif res["best_img"] is not None:
            save_png(res["best_img"], f"seed{s['seed']}_best_step{res['best_step']}.png")
        save_png(res["final_img"], f"seed{s['seed']}_final_step{res['final_step']}.png")
        print(f"seed {s['seed']}: ref success {s['ref_success']} -> this run {res['success']}, first_r4_step {res['first_r4_step']}, "
              f"max reward {res['max_reward']}, final step {res['final_step']} reward {res['final_reward']}", flush=True)
    with open(OUT_CSV, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["seed", "ref_success_insertion_csv", "ref_max_reward", "ref_success_same_loop_csv", "reproduced_success",
                    "first_r4_step", "final_step", "final_reward", "max_reward", "steps"])
        for r in results:
            w.writerow([r["seed"], r["ref_success"], r["ref_max_reward"], r["ref_same_loop_success"], r["success"],
                        "" if r["first_r4_step"] is None else r["first_r4_step"], r["final_step"], r["final_reward"], r["max_reward"], r["steps"]])
    contact_sheet(results, os.path.join(OUT_DIR, "contact_sheet.png"))
    print("wrote", OUT_CSV, "and", os.path.join(OUT_DIR, "contact_sheet.png"), flush=True)


if __name__ == "__main__":
    main()
