#!/bin/bash
# Usage: bash code/dp_train.sh <run_name> <steps> <batch_size> [extra lerobot-train args...]
# Resumable: if runs/dp/<run_name>/checkpoints/last exists, resumes from it (saved config; steps taken from saved config).
# Config rationale: docs/dp_training.md.  Env knobs (defaults = v1): DP_REPO, DP_HORIZON=64, DP_NACT=32, DP_EMA_DECAY (unset = no EMA)
# v2: DP_HORIZON=16 DP_NACT=8 DP_EMA_DECAY=0.999 (human data);  v3: same + DP_REPO=lerobot/aloha_sim_transfer_cube_scripted
set -e
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 MUJOCO_GL=egl HF_HOME=$PWD/.hf HF_HUB_DISABLE_IMPLICIT_TOKEN=1
NAME=$1; STEPS=$2; BS=$3; shift 3
REPO=${DP_REPO:-lerobot/aloha_sim_transfer_cube_human}; H=${DP_HORIZON:-64}; NA=${DP_NACT:-32}; DROP=$((H - NA - 2 + 1))
OUT=runs/dp/$NAME
if [ -e "$OUT/checkpoints/last/pretrained_model/train_config.json" ]; then
  exec .venv/bin/python code/dp_train_main.py --config_path="$OUT/checkpoints/last/pretrained_model/train_config.json" --resume=true --steps="$STEPS"
fi
exec .venv/bin/python code/dp_train_main.py \
  --dataset.repo_id="$REPO" \
  --policy.type=diffusion --policy.push_to_hub=false --policy.device=cuda \
  --policy.n_obs_steps=2 --policy.horizon=$H --policy.n_action_steps=$NA --policy.drop_n_last_frames=$DROP \
  --policy.resize_shape='[240,320]' --policy.crop_ratio=0.9 --policy.crop_is_random=true \
  --policy.down_dims='[256,512,1024]' \
  --policy.noise_scheduler_type=DDIM --policy.num_inference_steps=8 \
  --policy.scheduler_warmup_steps=500 \
  --batch_size="$BS" --steps="$STEPS" --num_workers=4 --seed=1000 \
  --save_freq=${DP_SAVE_FREQ:-10000} --eval_freq=0 --log_freq=50 --wandb.enable=false \
  --output_dir="$OUT" --job_name="$NAME" "$@"
