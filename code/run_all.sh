#!/bin/bash
# Full pilot. Run from project root: gpujob systemd-run --user --scope -q -p MemoryMax=12G -p MemorySwapMax=0 bash code/run_all.sh
export HF_HOME=$PWD/.hf HF_HUB_DISABLE_IMPLICIT_TOKEN=1 MUJOCO_GL=egl
PY=.venv/bin/python
$PY code/run_grid.py --n 1 10 25 50 100 --L 0 50 --mode black --episodes 50 &&
$PY code/run_grid.py --n 1 10 25 50 100 --L 50 --mode freeze --episodes 50 &&
$PY code/run_grid.py --te --L 0 50 --mode black --episodes 50 &&
$PY code/run_grid.py --te --L 50 --mode freeze --episodes 50
