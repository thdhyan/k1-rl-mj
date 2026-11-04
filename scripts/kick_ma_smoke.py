"""Smoke test for Mjlab-KickMA-Booster-K1: build, dims, phase switch, goal, OOB, and a shot-free random run.

usage: python scripts/kick_ma_smoke.py [num_envs] [steps]
"""
import sys

import torch

import mjlab_k1  # noqa: F401
from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg
from mjlab_k1 import tasks_kick_ma as M

TASK = "Mjlab-KickMA-Booster-K1"
n = int(sys.argv[1]) if len(sys.argv) > 1 else 8
steps = int(sys.argv[2]) if len(sys.argv) > 2 else 300
cfg = load_env_cfg(TASK)
cfg.scene.num_envs = n
env = ManagerBasedRlEnv(cfg=cfg, device="cuda:0")
obs, _ = env.reset()
u = env.unwrapped if hasattr(env, "unwrapped") else env
print("SMOKE obs", {k: tuple(v.shape) for k, v in obs.items()}, "act", env.action_manager.total_action_dim)
print("SMOKE action terms", list(env.action_manager.active_terms))
print("SMOKE reward terms", list(env.reward_manager.active_terms))
print("SMOKE term terms", list(env.termination_manager.active_terms))

st = M._st(env)
ball = env.scene["ball"].data
print("SMOKE ball start", ball.root_link_pos_w[0].tolist(), "pass point", st.pass_w[0].tolist())
print("SMOKE robot_a", env.scene["robot_a"].data.root_link_pos_w[0].tolist(), "robot_b", env.scene["robot_b"].data.root_link_pos_w[0].tolist())

sw_step = {}
for t in range(steps):
  a = 0.3 * torch.randn(n, env.action_manager.total_action_dim, device=env.device)
  env.step(a)
  for i in torch.nonzero(st.switched).flatten().tolist():
    sw_step.setdefault(i, t)
print(f"SMOKE switched envs {len(sw_step)}/{n} at steps {sorted(sw_step.values())[:8]}")
print("SMOKE received", int(st.received.sum()), "goal", int(M.goal_scored(env).sum()), "ball_oob", int(M.ball_oob(env).sum()))
print("SMOKE robot_b held cmd (should be 0 before switch):", env.action_manager.get_term("base_robot_b").command[~st.switched].abs().max().item() if (~st.switched).any() else "all switched")
