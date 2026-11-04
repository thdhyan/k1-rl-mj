#!/bin/bash
# On dl: when carry_vel_v1 ends, velocity grid per box mass on its last checkpoint.
cd ~/k1_mjlab/mjlab || exit 1
while tmux has-session -t carryvel 2>/dev/null; do sleep 60; done
export MUJOCO_GL=egl CUDA_VISIBLE_DEVICES=1 WANDB_MODE=disabled K1_BASE_CKPT=$HOME/k1_mjlab/base/walk_still_v4.pt K1_CV_BASE=$HOME/k1_mjlab/base/walk_still_v4.pt
RUN=$(ls -d logs/rsl_rl/k1_carry_vel/*_carry_vel_v1 | tail -1)
CK=$RUN/$(ls $RUN | grep -E '^model_[0-9]+\.pt$' | sort -t_ -k2 -n | tail -1)
.venv/bin/python ../scripts/velocity_grid.py Mjlab-CarryVel-Booster-K1 "$CK" ../grid_carry_vel_v1.json 0.5,1,2,3,4,5 > ../grid_carry_vel_v1.log 2>&1
echo GRID_DONE
