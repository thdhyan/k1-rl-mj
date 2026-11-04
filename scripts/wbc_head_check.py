"""Closed-loop parity of wbc.HeadModule against the trained head policy inside Mjlab-HeadTrack-Booster-K1.

The env is stepped with the head policy's own actions (from the rsl-rl checkpoint, normalised obs); each step
HeadModule gets the same detection + robot state and its head targets are compared with the action term's.
usage: python scripts/wbc_head_check.py <head ckpt> [num_envs] [steps]
"""
import sys

import torch

import mjlab_k1  # noqa: F401
from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg
from mjlab.tasks.velocity import mdp
from k1_runtime import wbc
from mjlab_k1.tasks_head import ball_detection

ckpt = sys.argv[1]
n = int(sys.argv[2]) if len(sys.argv) > 2 else 16
steps = int(sys.argv[3]) if len(sys.argv) > 3 else 300
cfg = load_env_cfg("Mjlab-HeadTrack-Booster-K1")
cfg.scene.num_envs = n
env = ManagerBasedRlEnv(cfg=cfg, device="cuda:0")
obs, _ = env.reset()
robot = env.scene["robot"]
order = [robot.joint_names.index(j) for j in wbc.JOINTS]
head = wbc.HeadModule(ckpt, n, env.device)
policy = wbc.PolicyModule(ckpt, env.device)
term = env.action_manager.get_term("head")
err, uerr, falls, seen = 0.0, 0.0, 0, 0.0
oerr = torch.zeros(obs["actor"].shape[1], device=env.device)
for _ in range(steps):
  d = robot.data
  p = wbc.Proprio(mdp.builtin_sensor(env, "robot/imu_ang_vel"), mdp.projected_gravity(env),
                  d.joint_pos[:, order], d.joint_vel[:, order])
  oerr = torch.maximum(oerr, (head.obs(p, ball_detection(env)) - obs["actor"]).abs().max(0).values)
  q = head(p, ball_detection(env)).tensor()
  a = policy(obs["actor"])
  obs, _, done, trunc, _ = env.step(a)
  err = max(err, (q - term._processed_actions).abs().max().item())
  uerr = max(uerr, (head.home + head.SCALE * head.last - term._processed_actions).abs().max().item())
  seen += ball_detection(env)[:, 0].mean().item() / steps
  falls += int(done.sum())
  head.reset((done | trunc).nonzero().flatten())
print("per-dim max |obs diff|:", oerr)
print(f"unclamped target diff {uerr:.2e}")
print(f"HeadModule parity: max |head target diff| = {err:.2e}, ball in view {seen:.0%}, "
      f"{steps} steps x {n} envs, terminations={falls}")
