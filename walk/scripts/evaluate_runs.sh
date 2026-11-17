#!/bin/sh
# Evaluate finished training runs: the simulation gates on the last checkpoint, the f0 trajectory, an export and two
# videos (0.10 and 0.20 m/s) for visual inspection.
#   scripts/evaluate_runs.sh runs/ppo/<run> [...]      (F0_VX="0.10" traces f0 at 0.10 m/s only; default 0.10 0.20)
# Outputs: <run>/eval_model_<last>.json, <run>/f0_trajectory.json, runs/export/<run>/ (policy.npz, config, eval),
# runs/video/<run>_0p10.mp4 and _0p20.mp4.
set -e
PY="env WANDB_MODE=disabled ${PYTHON:-python} -B"
for RUN in "$@"; do
  RUN=${RUN%/}
  NAME=$(basename "$RUN")
  ARM=$($PY -c "import json,sys;print(json.load(open('$RUN/config.json'))['task']['arm'])")
  LAST=$(ls "$RUN" | sed -n 's/^model_\([0-9]*\)\.pt$/\1/p' | sort -n | tail -1)
  CK="$RUN/model_$LAST.pt"
  $PY scripts/evaluate.py --ckpt "$CK" --arm "$ARM"
  $PY tools/f0_trajectory.py "$RUN" --every 100 --vx ${F0_VX:-0.10 0.20} --json "$RUN/f0_trajectory.json" | tail -n +1 > "$RUN/f0_trajectory.log"
  $PY scripts/export_policy.py "$CK" "runs/export/$NAME"
  for VX in 0.10 0.20; do
    MUJOCO_GL=egl $PY scripts/render.py --ckpt "runs/export/$NAME" --arm "$ARM" --vx $VX --seconds 10 \
      --out "runs/video/${NAME}_$(echo $VX | tr . p).mp4"
  done
done
