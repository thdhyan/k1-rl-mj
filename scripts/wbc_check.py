"""Closed-loop parity: wbc.DecoupledWBC leg targets == the training-time FrozenBaseAction targets.

Runs Mjlab-Kick-Booster-K1 (frozen base under a constant task action); each step feeds the same robot state to the
WBC and compares its 12 leg targets with the action term's. usage: python scripts/wbc_check.py [num_envs] [steps]
"""
import sys

import torch

import mjlab_k1  # noqa: F401
from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg
from mjlab.tasks.velocity import mdp
from k1_runtime import wbc
from mjlab_k1.tasks_kick import BASE_CKPT

n = int(sys.argv[1]) if len(sys.argv) > 1 else 16
steps = int(sys.argv[2]) if len(sys.argv) > 2 else 300
cfg = load_env_cfg("Mjlab-Kick-Booster-K1")
cfg.scene.num_envs = n
env = ManagerBasedRlEnv(cfg=cfg, device="cuda:0")
env.reset()
robot = env.scene["robot"]
order = [robot.joint_names.index(j) for j in wbc.JOINTS]
w = wbc.DecoupledWBC(BASE_CKPT, n, env.device)
base = env.action_manager.get_term("base")
a = torch.zeros(n, env.action_manager.total_action_dim, device=env.device)
a[:, 0], a[:, 2] = 0.5, 0.3
err, falls = 0.0, 0
for _ in range(steps):
  d = robot.data
  p = wbc.Proprio(mdp.builtin_sensor(env, "robot/imu_ang_vel"), mdp.projected_gravity(env),
                  d.joint_pos[:, order], d.joint_vel[:, order])
  q = w.step(p, wbc.LocoCmd.from_action(a))
  _, _, term, trunc, _ = env.step(a)
  err = max(err, (q[:, wbc.LEGS] - base._target).abs().max().item())
  done = (term | trunc).nonzero().flatten()
  falls += int(term.sum())
  w.reset(done)
print(f"WBC parity: max |leg target diff| = {err:.2e} over {steps} steps x {n} envs, terminations={falls}")
