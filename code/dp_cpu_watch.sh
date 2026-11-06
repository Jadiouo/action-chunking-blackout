#!/bin/bash
# Usage: bash code/dp_cpu_watch.sh <run_name> [steps...]   (default 10000 20000 30000)
# CPU-only evaluation (n=8 from checkpoint config, EMA weights, seeds 0-9) of each intermediate checkpoint as it appears.
# Writes results/dp_mid_<run>_<step>.csv / _summary.json.  Stop: touch .scratch/stop_watch
cd "$(dirname "$0")/.."
export CUDA_VISIBLE_DEVICES= HF_HOME=$PWD/.hf HF_HUB_DISABLE_IMPLICIT_TOKEN=1 MUJOCO_GL=egl OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
NAME=$1; shift; STEPS=${@:-10000 20000 30000}
while [ ! -e .scratch/stop_watch ]; do
  left=0
  for s in $STEPS; do
    g=$(printf "%06d" $s); ck=runs/dp/$NAME/checkpoints/$g; out=results/dp_mid_${NAME}_$g
    [ -e ${out}_summary.json ] && continue
    left=1
    if [ -e $ck/training_state/training_step.json ]; then
      sleep 20
      systemd-run --user --scope -q -p MemoryMax=6G -p MemorySwapMax=0 nice -n 10 .venv/bin/python code/dp_eval.py --ckpt $ck/pretrained_model --episodes 10 --device cpu --out $out.csv > .scratch/dp_mid_${NAME}_$g.log 2>&1
      bash code/dp_prune.sh $NAME 2 ${DP_GATE:-40000}
    fi
  done
  [ $left -eq 0 ] && break
  sleep 60
done
