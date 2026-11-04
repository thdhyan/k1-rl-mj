#!/bin/bash
# grid.sh <ckpt relative to mjlab/> <out.mp4>: 16-env curriculum render
Z=/export/scratch/thakk100/k1_mjlab
cd $Z/mjlab
export CUDA_VISIBLE_DEVICES=0 WANDB_MODE=disabled MUJOCO_GL=egl K1_PIK_BASE=$Z/base/walk_still_v2.pt K1_PIK_BODY_W=6.0
.venv/bin/python ../scripts/push_ik_render.py grid "$1" "$2"
echo "GRID_DONE rc=$?"
