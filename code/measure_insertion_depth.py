"""How far is the peg inside the socket when Insertion gives reward 4?  Measured from the simulator state, not from images.

The top camera cannot show this: once the peg is inside the socket, its tip is hidden by the socket's upper wall, so a peg that has
entered and a peg that stops at the mouth look alike (see results/insertion_frames/contact_sheet.png).  This script runs the 200 clean episodes (same
seeds, policy, action dtype and loop as code/inspect_insertion.py: fixed n=100, no blackout) and after every env.step reads
the pose of the peg and of the socket.

Geometry of gym_aloha/assets/bimanual_viperx_insertion.xml, read from the loaded model and asserted below: the socket is a tube of
half-length 0.06 m along its x axis, open at both ends, with a square cavity of half-width 0.018 m; the pin is centered in the tube with
half-length 0.04 m; the peg is a box of half-length 0.06 m and half-width 0.01 m.  So the pin's end lies 0.02 m inside each mouth.

depth = for each end of the peg's axis that lies laterally inside the cavity (|y|, |z| < 0.018 in the socket frame), 0.06 - |x| in the
socket frame; the larger of the two ends; negative when the end is short of the mouth; empty when neither end is laterally inside.
corner_depth = the same 0.06 - |x| for the deepest of the peg's 8 corners that lies laterally inside the cavity (a tilted peg reaches the
pin with a corner before the end of its axis does).  tilt = angle in degrees between the peg's axis and the socket's axis, modulo direction.

Output: results/insertion_depth.csv, one row per episode.

Run (optionally via the local queue wrapper gpujob, see README.md):
  MUJOCO_GL=egl HF_HOME=$PWD/.hf HF_HUB_OFFLINE=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 gpujob systemd-run --user --scope -q \
      -p MemoryMax=12G -p MemorySwapMax=0 .venv/bin/python code/measure_insertion_depth.py
"""
import csv
import os
import sys

os.environ.setdefault("MUJOCO_GL", "egl")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import gymnasium as gym

import inspect_insertion as I
import corrupt_p2b as C2
import run_p2p3 as P
from run_grid import t0_for
from tasks import TASKS

OUT_CSV = os.path.join(I.ROOT, "results", "insertion_depth.csv")
N_EPISODES = 200
TUBE_HALF, CAVITY_HALF, PIN_HALF, PEG_HALF, PEG_SIDE = 0.06, 0.018, 0.04, 0.06, 0.01
WALLS = ("socket-1", "socket-2", "socket-3", "socket-4")


def check_geometry(physics):
    """Stop if the model differs from the numbers stated in the docstring."""
    size = physics.named.model.geom_size
    assert np.allclose(size["pin"], [PIN_HALF, 0.01, 0.01]), size["pin"]
    assert np.allclose(size["red_peg"], [PEG_HALF, 0.01, 0.01]), size["red_peg"]
    assert np.allclose(size["socket-1"], [TUBE_HALF, CAVITY_HALF, 0.002]), size["socket-1"]
    assert np.allclose(size["socket-3"], [TUBE_HALF, 0.002, CAVITY_HALF]), size["socket-3"]
    assert np.allclose(physics.named.model.geom_pos["pin"], 0), physics.named.model.geom_pos["pin"]
    assert np.allclose(np.abs(physics.named.model.geom_pos["socket-1"]), [0, 0, 0.02]), physics.named.model.geom_pos["socket-1"]
    assert np.allclose(np.abs(physics.named.model.geom_pos["socket-3"]), [0, 0.02, 0]), physics.named.model.geom_pos["socket-3"]


def peg_depth(physics):
    """Insertion depth in metres (definition in the module docstring); None if no peg end is laterally inside the cavity."""
    d = physics.named.data
    pos_s, rot_s = d.xpos["socket"].copy(), d.xmat["socket"].reshape(3, 3)
    pos_p, axis_p = d.geom_xpos["red_peg"].copy(), d.geom_xmat["red_peg"].reshape(3, 3)[:, 0]
    best = None
    for sign in (1, -1):
        end = rot_s.T @ (pos_p + sign * PEG_HALF * axis_p - pos_s)
        if abs(end[1]) < CAVITY_HALF and abs(end[2]) < CAVITY_HALF:
            depth = TUBE_HALF - abs(end[0])
            best = depth if best is None else max(best, depth)
    return best


def corner_depth_and_tilt(physics):
    """Deepest laterally-inside corner (metres; None if none) and the tilt of the peg against the socket axis (degrees)."""
    d = physics.named.data
    pos_s, rot_s = d.xpos["socket"].copy(), d.xmat["socket"].reshape(3, 3)
    pos_p, rot_p = d.geom_xpos["red_peg"].copy(), d.geom_xmat["red_peg"].reshape(3, 3)
    best = None
    for sx in (1, -1):
        for sy in (1, -1):
            for sz in (1, -1):
                c = rot_s.T @ (pos_p + rot_p @ np.array([sx * PEG_HALF, sy * PEG_SIDE, sz * PEG_SIDE]) - pos_s)
                if abs(c[1]) < CAVITY_HALF and abs(c[2]) < CAVITY_HALF:
                    depth = TUBE_HALF - abs(c[0])
                    best = depth if best is None else max(best, depth)
    tilt = np.degrees(np.arccos(np.clip(abs(rot_s[:, 0] @ rot_p[:, 0]), 0, 1)))
    return best, float(tilt)


def contacts(physics):
    """(peg touches pin, peg touches a socket wall) in the current state."""
    pin, wall = False, False
    names = physics.named.model.geom_pos.axes.row.names
    for i in range(physics.data.ncon):
        c = physics.data.contact[i]
        pair = {names[c.geom1], names[c.geom2]}
        if "red_peg" in pair:
            pin = pin or "pin" in pair
            wall = wall or bool(pair & set(WALLS))
    return pin, wall


def run_episode(env, plan, seed, physics):
    T = TASKS[I.TASK]
    t0 = t0_for(seed, 0, T["t0_lo"], T["t0_hi"])
    cor = C2.Corruptor2("black", C2.parse_param("black", ""), t0, 0, seed)       # L=0: image passes through unchanged
    obs, _ = env.reset(seed=seed)
    chunk, k, t, done = None, 0, 0, False
    maxr, first_r4, depth_r4, pin_r4 = 0.0, None, None, None
    max_depth, ever_pin, ever_wall, first_r3, depth_r3 = None, False, False, None, None
    corner_r4, tilt_r4, max_corner = None, None, None
    while not done:
        o = cor(t, obs)
        if chunk is None or k >= P.CHUNK or k >= I.N_FIXED:
            chunk = plan(o)
            k = 0
        a = chunk[k]
        k += 1
        obs, r, term, trunc, _ = env.step(a)
        r = float(r)
        maxr = max(maxr, r)
        depth = peg_depth(physics)
        pin, wall = contacts(physics)
        ever_pin, ever_wall = ever_pin or pin, ever_wall or wall
        if depth is not None:
            max_depth = depth if max_depth is None else max(max_depth, depth)
        corner, tilt = corner_depth_and_tilt(physics)
        if corner is not None:
            max_corner = corner if max_corner is None else max(max_corner, corner)
        if r >= 3 and first_r3 is None:
            first_r3, depth_r3 = t, depth
        if r >= 4 and first_r4 is None:
            first_r4, depth_r4, pin_r4 = t, depth, pin
            corner_r4, tilt_r4 = corner, tilt
        t += 1
        done = term or trunc
    return dict(seed=seed, success=int(maxr >= 4), max_reward=maxr, steps=t, first_r4_step=first_r4, depth_at_first_r4=depth_r4,
                pin_contact_at_first_r4=pin_r4, corner_depth_at_first_r4=corner_r4, tilt_at_first_r4=tilt_r4, max_corner_depth=max_corner, first_r3_step=first_r3, depth_at_first_r3=depth_r3, max_depth=max_depth,
                ever_pin_contact=int(ever_pin), ever_wall_contact=int(ever_wall))


def main():
    plan = I.build_policy()
    T = TASKS[I.TASK]
    env = gym.make(T["env_id"], obs_type="pixels_agent_pos", max_episode_steps=T["max_steps"])
    env.reset(seed=0)
    physics = env.unwrapped._env.physics
    check_geometry(physics)
    rows = []
    for seed in range(N_EPISODES):
        res = run_episode(env, plan, seed, physics)
        rows.append(res)
        print(res, flush=True)
    with open(OUT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        for r in rows:
            w.writerow({k: "" if v is None else v for k, v in r.items()})
    print("wrote", OUT_CSV, flush=True)


if __name__ == "__main__":
    main()
