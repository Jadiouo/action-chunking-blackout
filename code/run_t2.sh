#!/bin/bash
# T2 full grid, split into jobs. Usage: bash code/run_t2.sh <job>   (run from project root)
export HF_HOME=$PWD/.hf HF_HUB_DISABLE_IMPLICIT_TOKEN=1 MUJOCO_GL=egl OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
PY=.venv/bin/python
E=100
case "$1" in
  A) # black (incl. L=0), n in 10 25 50 100
     $PY code/run_grid.py --n 10 25 50 100 --L 0 10 25 50 100 --mode black --episodes $E --out results/parts/t2_A.csv ;;
  B) # freeze, n in 10 25 50 100
     $PY code/run_grid.py --n 10 25 50 100 --L 10 25 50 100 --mode freeze --episodes $E --out results/parts/t2_B.csv ;;
  C) # n=1 black(+L0) then freeze
     $PY code/run_grid.py --n 1 --L 0 10 25 50 100 --mode black --episodes $E --out results/parts/t2_C1.csv &&
     $PY code/run_grid.py --n 1 --L 10 25 50 100 --mode freeze --episodes $E --out results/parts/t2_C2.csv ;;
  D) # temporal ensembling black(+L0) then freeze
     $PY code/run_grid.py --te --L 0 10 25 50 100 --mode black --episodes $E --out results/parts/t2_D1.csv &&
     $PY code/run_grid.py --te --L 10 25 50 100 --mode freeze --episodes $E --out results/parts/t2_D2.csv ;;
esac
