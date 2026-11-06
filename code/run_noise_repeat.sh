#!/bin/bash
# Re-run the 8 episodes whose success differed between pilot and full grid, 3 times each, default thread settings (as pilot).
export HF_HOME=$PWD/.hf HF_HUB_DISABLE_IMPLICIT_TOKEN=1 MUJOCO_GL=egl
PY=.venv/bin/python; O=results/parts/noise_repeat.csv
for rep in 1 2 3; do
  for s in 5 30; do $PY code/run_grid.py --n 10 --L 0 --mode black --seed0 $s --episodes 1 --out $O; done
  for s in 39 41 43; do $PY code/run_grid.py --n 25 --L 50 --mode black --seed0 $s --episodes 1 --out $O; done
  $PY code/run_grid.py --n 50 --L 50 --mode black --seed0 15 --episodes 1 --out $O
  for s in 5 18; do $PY code/run_grid.py --n 10 --L 50 --mode freeze --seed0 $s --episodes 1 --out $O; done
done
