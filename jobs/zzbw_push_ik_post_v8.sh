#!/bin/bash
# After v8 training ends: eval and multi-size render with the final checkpoint.
Z=/export/scratch/thakk100/k1_mjlab
while tmux has-session -t pik8 2>/dev/null; do sleep 60; done
cd $Z/mjlab
export CUDA_VISIBLE_DEVICES=0 WANDB_MODE=disabled MUJOCO_GL=egl K1_PIK_BASE=$Z/base/walk_still_v2.pt K1_PIK_BODY_W=6.0
RUN=$(ls -d logs/rsl_rl/k1_push_ik/*_push_ik_v8 | tail -1)
CK=$RUN/$(ls $RUN | grep -E '^model_[0-9]+\.pt$' | sort -t_ -k2 -n | tail -1)
echo "ckpt $CK"
.venv/bin/python ../scripts/push_ik_eval.py $CK 512 $Z/eval_v8.json
.venv/bin/python ../scripts/push_ik_render.py push $CK $Z/push_v8_sizes.mp4
echo POST_DONE
