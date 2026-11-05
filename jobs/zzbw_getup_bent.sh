#!/bin/bash
# Fine-tune Mjlab-GetUpBent-Booster-K1 (joint-speed capped, K1_GETUP_VCAP rad/s) on zz-bw GPU0.
# Warm start: $WARM_CKPT (relative to mjlab/), default getup_v1; same obs/action dims.
# Run inside the getup_bent_20261104 snapshot (mjlab_k1/ overlay + getup_panel.py).
#   ./zzbw_getup_bent.sh smoke | train
set -euo pipefail
Z=/export/scratch/thakk100/k1_mjlab
RUN_ROOT=${RUN_ROOT:-$Z/getup_bent_20261104}
export PYTHONPATH=$RUN_ROOT
export K1_XML=$Z/src/k1_description/assets/robots/K1/K1_22dof.xml
export K1_BASE_CKPT=$Z/base/walk_still_v4.pt
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} MUJOCO_GL=egl WANDB_MODE=disabled
cd "$Z/mjlab"
WARM_CKPT=${WARM_CKPT:-logs/rsl_rl/k1_getup/2026-10-29_14-12-04_getup_v1/model_5999.pt}
RUN_NAME=${RUN_NAME:-gb_v2}
DST=logs/rsl_rl/k1_getup_bent/0000-00-00_00-00-00_warmstart_$RUN_NAME
mkdir -p "$DST"; cp "$WARM_CKPT" "$DST/model_0.pt"
if [[ ${1:-train} == smoke ]]; then
  .venv/bin/k1-train Mjlab-GetUpBent-Booster-K1 --env.scene.num-envs 64 --agent.run-name ${RUN_NAME}_smoke \
    --agent.logger tensorboard --agent.max-iterations 3 --agent.save-interval 1 \
    --agent.resume True --agent.load-run ".*warmstart_$RUN_NAME" > "$RUN_ROOT/smoke.log" 2>&1
  touch "$RUN_ROOT/SMOKE_OK"; exit
fi
test -f "$RUN_ROOT/SMOKE_OK"
.venv/bin/k1-train Mjlab-GetUpBent-Booster-K1 --env.scene.num-envs 4096 --agent.run-name $RUN_NAME \
  --agent.logger tensorboard --agent.max-iterations 3000 --agent.save-interval 100 \
  --agent.resume True --agent.load-run ".*warmstart_$RUN_NAME" > "$RUN_ROOT/train.log" 2>&1
