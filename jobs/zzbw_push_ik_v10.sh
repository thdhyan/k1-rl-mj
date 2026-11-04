#!/bin/bash
# On zz-bw, in tmux pik10: PushIK v10 from v8. Two-hand reward credit, near-goal and at-face resets. Then evals and renders.
Z=/export/scratch/thakk100/k1_mjlab
cd $Z/mjlab || exit 1
export CUDA_VISIBLE_DEVICES=0 WANDB_MODE=disabled MUJOCO_GL=egl K1_PIK_BASE=$Z/base/walk_still_v2.pt K1_PIK_BODY_W=6.0
# the v8 baseline is evaluated with the same script, first, so the hand columns exist for both
V8=$(ls -d logs/rsl_rl/k1_push_ik/*_push_ik_v8 | tail -1)/model_9497.pt
.venv/bin/python ../scripts/push_ik_eval.py "$V8" 512 $Z/eval_v8_hands.json > $Z/eval_v8_hands.log 2>&1
export K1_PIK_ONE_HAND=0.3 K1_PIK_BOTH_W=1 K1_PIK_NEAR_FRAC=0.2 K1_PIK_FACE_FRAC=0.2
.venv/bin/k1-train Mjlab-PushIK-Booster-K1 --env.scene.num-envs 4096 --agent.run-name push_ik_v10 --agent.logger tensorboard \
  --agent.max-iterations 3000 --agent.save-interval 200 --agent.resume True --agent.load-run '.*push_ik_v8' \
  --agent.load-checkpoint model_9497.pt > $Z/push_ik_v10.log 2>&1
RUN=$(ls -d logs/rsl_rl/k1_push_ik/*_push_ik_v10 | tail -1)
CK=$RUN/$(ls $RUN | grep -E '^model_[0-9]+\.pt$' | sort -t_ -k2 -n | tail -1)
.venv/bin/python ../scripts/push_ik_eval.py "$CK" 512 $Z/eval_v10.json > $Z/eval_v10.log 2>&1
.venv/bin/python ../scripts/push_ik_render.py push "$CK" $Z/push_v10_sizes.mp4 > $Z/render_v10.log 2>&1
.venv/bin/python ../scripts/push_ik_render.py grid "$CK" $Z/push_v10_grid16.mp4 >> $Z/render_v10.log 2>&1
echo V10_DONE
