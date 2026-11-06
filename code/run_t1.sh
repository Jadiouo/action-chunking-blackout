#!/bin/bash
# T1: determinism check (two identical runs) + n in {1,2,5} at L=0, 50 seeds.
export HF_HOME=$PWD/.hf HF_HUB_DISABLE_IMPLICIT_TOKEN=1 MUJOCO_GL=egl
PY=.venv/bin/python
$PY code/run_grid.py --n 25 --L 0 --mode black --episodes 5 --out results/parts/t1_det_a.csv &&
$PY code/run_grid.py --n 25 --L 0 --mode black --episodes 5 --out results/parts/t1_det_b.csv &&
$PY code/run_grid.py --n 1 2 5 --L 0 --mode black --episodes 50 --out results/parts/t1_n125.csv
