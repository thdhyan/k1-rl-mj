#!/bin/bash
# On dl, in tmux walk4: walk_still_v4 from v3 with sideways shoves, training videos, then the shove table for v3 and v4.
cd ~/k1_mjlab/mjlab || exit 1
export MUJOCO_GL=egl CUDA_VISIBLE_DEVICES=${1:-1} WANDB_MODE=disabled K1_STAND_W=1 K1_STAND_FRAC=0.25 K1_FLAT_W=5 K1_SYM_W=1 K1_FEET_W=20 K1_FEET_MIN=0.15 K1_VY_W=1
export K1_ACT_RATE_W=0.3 K1_JOINT_ACC_W=1e-6 K1_ANG_VEL_W=0.1 K1_LAND_W=5e-5 K1_TRACK_W=3 K1_VX_MAX=1.3
V3=logs/rsl_rl/k1_velocity/2026-10-31_12-55-56_walk_still_v3
T=Mjlab-Velocity-Height-Flat-Booster-K1
.venv/bin/python ../scripts/shove_eval.py $T $V3/model_13997.pt ../shove_v3.json > ../shove_v3.log 2>&1
K1_SHOVE_N=100 K1_SHOVE_RAMP_ITERS=2000 .venv/bin/k1-train $T --env.scene.num-envs 4096 --agent.run-name walk_still_v4 --agent.logger tensorboard \
  --agent.resume True --agent.load-run 2026-10-31_12-55-56_walk_still_v3 --agent.load-checkpoint model_13997.pt --agent.max-iterations 5000 \
  --agent.save-interval 200 --video True --video-interval 12000 --video-length 400 > ../walk_still_v4.log 2>&1
RUN=$(ls -d logs/rsl_rl/k1_velocity/*_walk_still_v4 | tail -1)
CK=$RUN/$(ls $RUN | grep -E '^model_[0-9]+\.pt$' | sort -t_ -k2 -n | tail -1)
.venv/bin/python ../scripts/shove_eval.py $T "$CK" ../shove_v4.json > ../shove_v4.log 2>&1
echo V4_DONE
