#!/bin/bash
# Usage: bash code/dp_prune.sh <run_name> [keep_newest=2] [gate_step=40000]
# Disk saver (disk was ~12G free).  Deletes numbered checkpoint dirs under runs/dp/<run>/checkpoints EXCEPT:
#   - the dir that `last` points to, the <keep_newest> newest numbered dirs, the gate checkpoint (<gate_step>),
#   - any checkpoint in $DP_PROTECT (default "10000 20000 30000" = CPU-eval steps of dp_cpu_watch.sh) whose
#     results/dp_mid_<run>_<step>_summary.json does not exist yet (evaluation pending or running).
# Every deletion is appended to the "清理紀錄" section at the end of docs/dp_training.md.
cd "$(dirname "$0")/.."
NAME=$1; KEEP=${2:-2}; GATE=${3:-40000}
D=runs/dp/$NAME/checkpoints; [ -d "$D" ] || exit 0
PROT=${DP_PROTECT:-"10000 20000 30000"}
last=$(readlink -f "$D/last" 2>/dev/null | xargs -r basename)
ls -1 "$D" | grep -E '^[0-9]+$' | sort > .scratch/prune_all.txt
keep=$(tail -n "$KEEP" .scratch/prune_all.txt; printf "%06d\n" "$GATE"; echo "$last")
for s in $PROT; do
  g=$(printf "%06d" $s)
  [ -e results/dp_mid_${NAME}_${g}_summary.json ] || keep="$keep"$'\n'"$g"
done
for c in $(cat .scratch/prune_all.txt); do
  echo "$keep" | grep -qx "$c" && continue
  sz=$(du -sh "$D/$c" | cut -f1)
  rm -rf "$D/$c"
  echo "- $(date '+%F %T') 清理 runs/dp/$NAME/checkpoints/$c（$sz）；保留：$(echo "$keep" | sort -u | tr '\n' ' ')" >> docs/dp_training.md
done
