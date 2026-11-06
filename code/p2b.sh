#!/bin/bash
# P2b jobs (docs/p2b_prereg.md). Usage (project root, via gpujob):  bash code/p2b.sh <job>
# Submit like code/p2p3.sh:  gpujob env OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 MUJOCO_GL=egl HF_HOME=$PWD/.hf HF_HUB_DISABLE_IMPLICIT_TOKEN=1 HF_HUB_OFFLINE=1 \
#                              systemd-run --user --scope -q -p MemoryMax=12G -p MemorySwapMax=0 bash code/p2b.sh <job>
# Outputs: results/parts/p2b_*.csv (smoke: .scratch/p2b).  Never touches old CSVs.
export HF_HOME=$PWD/.hf HF_HUB_DISABLE_IMPLICIT_TOKEN=1 HF_HUB_OFFLINE=1 MUJOCO_GL=egl OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
PY=.venv/bin/python
R="$PY code/run_p2b.py"
P=results/parts
S=.scratch/p2b
mkdir -p $S
# memory guard: need >= 8 GB available (user rule); wait up to 20 min, then give up (exit 3) so the queue is not blocked
for i in $(seq 40); do
  av=$(awk '/MemAvailable/{print int($2/1048576)}' /proc/meminfo)
  [ "$av" -ge 8 ] && break
  [ "$i" = 40 ] && { echo "memory guard: available ${av}GB < 8GB for 20 min" >&2; exit 3; }
  sleep 30
done
sel() { python3 -c "import json,sys; print(json.load(open('$P/p2b_selected.json'))['$1'])"; }  # chosen intensity of an unseen corruption
case "$1" in
  smoke) # wrapper regression vs p2p3 (float64, same loop) + one episode of every new feature
     $R --rule fixed --n 25 --L 50 --corr black --episodes 5 --out $S/reg_black.csv &&
     $R --rule fixed --n 25 --L 50 --corr freeze --act_dtype float32 --episodes 2 --out $S/f32.csv &&
     $R --task insertion --rule fixed --n 25 --L 50 --corr black --episodes 2 --out $S/ins.csv &&
     $R --rule generic --n 50 --L 50 --corr blur --param 15 --episodes 2 --out $S/blur.csv &&
     $R --rule generic --n 50 --L 50 --corr dim --param 0.25 --episodes 2 --out $S/dim.csv &&
     $R --rule median_te --h 1 --L 25 --corr black --seed0 150 --episodes 1 --out $S/te.csv ;;
  tc_clean) # TC same-loop clean reference (white #1.3)
     $R --rule fixed --n 25 50 100 --L 0 --episodes 90 --out $P/p2b_tc_clean_fixed.csv ;;
  dtype) # dtype ablation: float32 actions in the new loop, TC n=25 L=50, seeds 0-99 (white #1.2)
     $R --rule fixed --n 25 --L 50 --corr black  --act_dtype float32 --episodes 100 --out $P/p2b_dtype.csv &&
     $R --rule fixed --n 25 --L 50 --corr freeze --act_dtype float32 --episodes 100 --out $P/p2b_dtype.csv ;;
  pilot) # unseen-corruption intensity pilot: fixed n=50 L=50 (+ L=0 reference), seeds 120-149
     S3="--seed0 120 --episodes 30"
     $R --rule fixed --n 50 --L 0 $S3 --out $P/p2b_pilot.csv &&
     for p in 5 15 40; do $R --rule fixed --n 50 --L 50 --corr blur --param $p $S3 --out $P/p2b_pilot.csv || exit 1; done &&
     for p in 0.5 0.25 0.1; do $R --rule fixed --n 50 --L 50 --corr dim --param $p $S3 --out $P/p2b_pilot.csv || exit 1; done ;;
  pilot_ext) # pre-declared stronger 4th intensity (only if all three R > 0.85): pilot_ext blur 80 | dim 0.03
     $R --rule fixed --n 50 --L 50 --corr $2 --param $3 --seed0 120 --episodes 30 --out $P/p2b_pilot.csv ;;
  unseen_main) # usage: unseen_main <blur|dim>; param from p2b_selected.json; fixed/oracle/generic n=50 L=50 seeds 0-89
     c=$2; q=$(sel $c)
     for r in fixed oracle generic; do $R --rule $r --n 50 --L 50 --corr $c --param $q --episodes 90 --out $P/p2b_unseen_$c.csv || exit 1; done ;;
  clean_cost) # generic-detect L=0, n in {25,50,100}, 90 seeds (white #2 clean cost)
     $R --rule generic --n 25 50 100 --L 0 --episodes 90 --out $P/p2b_generic_clean.csv ;;
  te) # Median TE confirmatory re-run, new seeds 150-249: freeze L=100 and black L=25 (white #3)
     for cfg in "freeze 100" "black 25"; do
       set -- $cfg
       $R --rule median_te --h 1 10 --L $2 --corr $1 --seed0 150 --episodes 100 --out $P/p2b_te.csv &&
       $R --rule fixed --n 50 100 --L $2 --corr $1 --seed0 150 --episodes 100 --out $P/p2b_te.csv &&
       $R --rule oracle --n 50 --L $2 --corr $1 --seed0 150 --episodes 100 --out $P/p2b_te.csv || exit 1
     done ;;
  ins_L50) # Insertion same-loop fixed, L=50 (white #1.1), 200 seeds
     $R --task insertion --rule fixed --n 25 50 100 --L 50 --corr black  --episodes 200 --out $P/p2b_ins_fixed_L50.csv &&
     $R --task insertion --rule fixed --n 25 50 100 --L 50 --corr freeze --episodes 200 --out $P/p2b_ins_fixed_L50.csv ;;
  ins_clean) # Insertion same-loop clean reference
     $R --task insertion --rule fixed --n 25 50 100 --L 0 --episodes 200 --out $P/p2b_ins_fixed_L0.csv ;;
  ins_L100)
     $R --task insertion --rule fixed --n 25 50 100 --L 100 --corr black  --episodes 200 --out $P/p2b_ins_fixed_L100.csv &&
     $R --task insertion --rule fixed --n 25 50 100 --L 100 --corr freeze --episodes 200 --out $P/p2b_ins_fixed_L100.csv ;;
  *) echo "unknown job $1"; exit 1 ;;
esac
