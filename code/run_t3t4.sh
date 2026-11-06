#!/bin/bash
# T3/T4 jobs. Usage: bash code/run_t3t4.sh <job> [extra args]   (run from project root, via gpujob)
export HF_HOME=$PWD/.hf HF_HUB_DISABLE_IMPLICIT_TOKEN=1 MUJOCO_GL=egl OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
PY=.venv/bin/python
J=$1; shift
case "$J" in
  smoke) $PY code/run_t3t4.py --rule fixed --n 25 --L 0 --episodes 10 --out .scratch/smoke/fixed.csv
         $PY code/run_t3t4.py --rule oracle --n 10 --L 50 --mode black freeze --episodes 10 --out .scratch/smoke/oracle.csv ;;
  oracle) # $1 = n
         $PY code/run_t3t4.py --rule oracle --n $1 --L 10 25 50 100 --mode black freeze --episodes 100 --out results/parts/t3_n$1.csv ;;
  calib_a) $PY code/run_t3t4.py --rule fixed --n 100 --L 0 --episodes 20 --seed0 0 --out results/parts/t4_calib_s0.csv --record_dir .scratch/proprio/s0 ;;
  calib_b) $PY code/run_t3t4.py --rule fixed --n 100 --L 0 --episodes 20 --seed0 100 --out results/parts/t4_calib_s100.csv --record_dir .scratch/proprio/s100 ;;
  proprio_L0) T=$(python3 -c "import json;print(json.load(open('results/parts/proprio_calib_stats_arm.json'))['tau99'])")
         $PY code/run_t3t4.py --rule proprio --sig arm --tau $T --m 1 10 --L 0 --mode black --episodes 100 --out results/parts/t4_main_L0.csv ;;
  proprio_main) # $1 = black|freeze ; primary: arm-only signal, tau = 99th pct (seeds 100-119)
         T=$(python3 -c "import json;print(json.load(open('results/parts/proprio_calib_stats_arm.json'))['tau99'])")
         $PY code/run_t3t4.py --rule proprio --sig arm --tau $T --m 1 10 --L 25 50 100 --mode $1 --episodes 100 --out results/parts/t4_main_$1.csv ;;
  sens_arm95) T=$(python3 -c "import json;print(json.load(open('results/parts/proprio_calib_stats_arm.json'))['tau95'])")
         $PY code/run_t3t4.py --rule proprio --sig arm --tau $T --m 1 10 --L 0 --mode black --episodes 100 --out results/parts/t4_sens_arm95.csv
         $PY code/run_t3t4.py --rule proprio --sig arm --tau $T --m 1 10 --L 50 --mode black freeze --episodes 100 --out results/parts/t4_sens_arm95.csv ;;
  sens_all99) T=$(python3 -c "import json;print(json.load(open('results/parts/proprio_calib_stats_all.json'))['tau99'])")
         $PY code/run_t3t4.py --rule proprio --sig all --tau $T --m 1 10 --L 0 --mode black --episodes 100 --out results/parts/t4_sens_all99.csv
         $PY code/run_t3t4.py --rule proprio --sig all --tau $T --m 1 10 --L 50 --mode black freeze --episodes 100 --out results/parts/t4_sens_all99.csv ;;
  detect_smoke) $PY code/run_t3t4.py --rule detect --n 25 --L 50 --mode black freeze --episodes 10 --out .scratch/detect/smoke.csv ;;
  detect_calib) # no blackout, seeds 100-119: image mean / adjacent-frame equality distribution + false detects
         $PY code/run_t3t4.py --rule detect --n 100 --L 0 --episodes 20 --seed0 100 --out results/parts/detect_calib.csv --record_dir results/parts/detect_calib ;;
  detect_main) # $1 = n (25|50); n=50 also runs the L=0 false-trigger cell
         $PY code/run_t3t4.py --rule detect --n $1 --L 50 100 --mode black freeze --episodes 100 --out results/parts/detect_n$1.csv
         if [ "$1" = 50 ]; then $PY code/run_t3t4.py --rule detect --n 50 --L 0 --mode black --episodes 100 --out results/parts/detect_n50.csv; fi ;;
  *) echo "unknown job $J"; exit 1 ;;
esac
