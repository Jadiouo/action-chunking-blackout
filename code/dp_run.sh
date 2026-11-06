#!/bin/bash
# One gpujob = go/no-go gate + (resumable) training.  Usage: bash code/dp_run.sh <run_name> <total_steps> <batch_size> [gate_step=20000]
#   1. train with --steps=<total_steps> (resumes automatically from runs/dp/<run>/checkpoints/last if present)
#   2. when checkpoint <gate_step> is written (first time only), stop training, evaluate 20 seeds (0-19) with dp_eval.py
#      -> results/dp_gate_<run>_summary.json ; if success rate < 10% -> exit 3 (no further training)
#   3. otherwise resume and train to <total_steps>, stopping after DP_JOB_SECONDS (default 5.67h; rerun the same command to resume).
# If the training process dies (e.g. OOM-killed) it is restarted from the last checkpoint, at most 4 times.
set -u
cd "$(dirname "$0")/.."
DEADLINE=$((SECONDS + ${DP_JOB_SECONDS:-20400}))
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 MUJOCO_GL=egl HF_HOME=$PWD/.hf HF_HUB_DISABLE_IMPLICIT_TOKEN=1
NAME=$1; TOTAL=$2; BS=$3; GATE=${4:-20000}
OUT=runs/dp/$NAME
G=$(printf "%06d" "$GATE")
GATECK="$OUT/checkpoints/$G/training_state/training_step.json"
GATEJSON=results/dp_gate_${NAME}_summary.json
mkdir -p runs/dp results
# resource log: epoch seconds, GPU memory used (MiB, whole GPU incl. ~1.3GB display), RSS (GB) of all python processes of this job
( while true; do
    R=$(ps -eo rss,cmd | grep "[d]p_train_main.py" | awk '{s+=$1} END {printf "%.2f", s/1e6}')
    echo "$(date +%s) $(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits) $R" >> "runs/dp/${NAME}_res.log"; sleep 30
  done ) &
VP=$!; trap 'kill $VP 2>/dev/null; bash code/dp_prune.sh "$NAME" 2 "$GATE"' EXIT   # keep disk small (see docs/dp_training.md 清理紀錄)

if [ ! -e "$GATEJSON" ] && [ ! -e "$GATECK" ]; then
  for TRY in 1 2 3 4; do
    bash code/dp_train.sh "$NAME" "$TOTAL" "$BS" &
    TP=$!
    while [ ! -e "$GATECK" ] && kill -0 $TP 2>/dev/null; do sleep 20; done
    [ -e "$GATECK" ] && break
    wait $TP; echo "training died before gate (try $TRY), restarting from last checkpoint"
  done
  [ -e "$GATECK" ] || { echo "no gate checkpoint"; exit 5; }
  sleep 15   # let the checkpoint finish writing
  pkill -TERM -P $TP; kill -TERM $TP 2>/dev/null; sleep 10; pkill -KILL -f "dp_train_main.py.*$NAME" 2>/dev/null
  wait $TP 2>/dev/null
fi
if [ ! -e "$GATEJSON" ]; then
  .venv/bin/python code/dp_eval.py --ckpt "$OUT/checkpoints/$G/pretrained_model" --episodes 20 --out results/dp_gate_${NAME}.csv || exit 4
fi
RATE=$(.venv/bin/python -c "import json;print(json.load(open('$GATEJSON'))['rate'])")
echo "GATE rate=$RATE"
if ! .venv/bin/python -c "import sys;sys.exit(0 if $RATE>=0.10 else 1)"; then
  echo "GATE FAILED (<10%): stopping, no further training"; exit 3
fi
for TRY in 1 2 3 4; do
  REM=$((DEADLINE - SECONDS)); [ $REM -le 60 ] && { echo "job time limit reached; resubmit the same command to resume"; exit 0; }
  timeout -s TERM $REM bash code/dp_train.sh "$NAME" "$TOTAL" "$BS"
  RC=$?
  [ $RC -eq 0 ] && exit 0
  [ $RC -eq 124 ] && { echo "job time limit reached; resubmit the same command to resume"; exit 0; }
  echo "training died rc=$RC (try $TRY), restarting from last checkpoint"
done
exit 6
