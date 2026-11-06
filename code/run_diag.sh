#!/bin/bash
# T1 diagnosis jobs (see code/diagnose_n1.py)
export HF_HOME=$PWD/.hf HF_HUB_DISABLE_IMPLICIT_TOKEN=1 MUJOCO_GL=egl OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
PY=.venv/bin/python
case "$1" in
  traj) $PY code/diagnose_n1.py --cfg n1 n25 te --seeds 0 1 2 3 4 5 6 7 ;;
  off)  $PY code/diagnose_n1.py --cfg off2 off5 off10 off25 --seeds $(seq 0 19) --save 0 --csv results/n1_offset_runs.csv ;;
esac
