#!/bin/bash
# After v8's eval and renders: train v9 on tall boxes (height 0.75-2x K1 height 0.95 m), then eval and render it.
Z=/export/scratch/thakk100/k1_mjlab
while tmux has-session -t grid8 2>/dev/null; do sleep 60; done
latest() { echo $(ls $1 | grep -E '^model_[0-9]+\.pt$' | sort -t_ -k2 -n | tail -1); }
cd $Z/mjlab
export CUDA_VISIBLE_DEVICES=0 WANDB_MODE=disabled MUJOCO_GL=egl K1_PIK_BASE=$Z/base/walk_still_v2.pt K1_PIK_BODY_W=6.0 K1_PIK_HALF_Z=0.356,0.95
V8=$(ls -d logs/rsl_rl/k1_push_ik/*_push_ik_v8)
.venv/bin/k1-train Mjlab-PushIK-Booster-K1 --env.scene.num-envs 4096 --agent.run-name push_ik_v9 --agent.logger tensorboard --agent.max-iterations 3000 --agent.save-interval 200 --agent.resume True --agent.load-run '.*push_ik_v8' --agent.load-checkpoint $(latest $V8) > $Z/push_ik_v9.log 2>&1
V9=$(ls -d logs/rsl_rl/k1_push_ik/*_push_ik_v9)
CK=$V9/$(latest $V9)
echo "ckpt $CK"
.venv/bin/python ../scripts/push_ik_eval.py $CK 512 $Z/eval_v9.json
.venv/bin/python ../scripts/push_ik_render.py grid $CK $Z/push_v9_grid16.mp4
echo V9_DONE
