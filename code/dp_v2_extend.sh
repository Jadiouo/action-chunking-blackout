#!/bin/bash
# Last DP attempt: continue v2 (human data, EMA) from its 40k checkpoint to 100k, the step count its LR schedule was set for.
# Bypasses dp_run.sh because the 40k gate already failed. Stop rule: docs/dp_training.md.
cd "$(dirname "$0")/.."
export DP_HORIZON=16 DP_NACT=8 DP_EMA_DECAY=0.999
timeout -s TERM 20400 bash code/dp_train.sh dp_v2 100000 64
RC=$?; echo "train rc=$RC"
LAST=$(readlink -f runs/dp/dp_v2/checkpoints/last); S=$(basename "$LAST")
.venv/bin/python code/dp_eval.py --ckpt "$LAST/pretrained_model" --episodes 20 --out results/dp_final_dp_v2_$S.csv
bash code/dp_prune.sh dp_v2 2 40000
