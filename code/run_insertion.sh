#!/bin/bash
# P1 AlohaInsertion (200 seeds; fixed n in {25,50,100} x L in {50,100}; oracle/detect n in {25,50} x L in {50,100})
# P1 AlohaInsertion jobs. Usage: bash code/run_insertion.sh <job>   (run from the repository root; the optional local queue wrapper gpujob is described in README.md)
# episode length 400, t0 ~ U{50..250} (same as Transfer Cube), success = max reward 4 (code/tasks.py)
export HF_HOME=$PWD/.hf HF_HUB_DISABLE_IMPLICIT_TOKEN=1 MUJOCO_GL=egl OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
PY=.venv/bin/python
E=200   # coordinator update: 200 seeds (0-199)
G="--task insertion --episodes $E"
case "$1" in
  noblack) # step 1: no blindness, n in {1,10,25,50,100} + TE
     $PY code/run_grid.py $G --n 1 10 25 50 100 --L 0 --mode black --out results/parts/ins_noblack.csv &&
     $PY code/run_grid.py $G --te --L 0 --mode black --out results/parts/ins_noblack_te.csv ;;
  fixed_black)  $PY code/run_grid.py $G --n 25 50 100 --L 50 100 --mode black  --out results/parts/ins_fixed_black.csv ;;
  fixed_freeze) $PY code/run_grid.py $G --n 25 50 100 --L 50 100 --mode freeze --out results/parts/ins_fixed_freeze.csv ;;
  rerun) # run-to-run noise check: n=50, L=0, seeds 0-49 once more
     $PY code/run_grid.py --task insertion --episodes 50 --n 50 --L 0 --mode black --out results/parts/ins_rerun_n50.csv ;;
  oracle) $PY code/run_t3t4.py --task insertion --rule oracle --n 25 50 --L 50 100 --mode black freeze --episodes $E --out results/parts/ins_oracle.csv ;;
  detect) $PY code/run_t3t4.py --task insertion --rule detect --n 25 50 --L 50 100 --mode black freeze --episodes $E --out results/parts/ins_detect.csv ;;
  *) echo "unknown job $1"; exit 1 ;;
esac
