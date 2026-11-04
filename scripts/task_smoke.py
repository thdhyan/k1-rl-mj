"""Build an mjlab K1 task, step it with random actions, print dims, reward terms and terminations.

usage: python scripts/task_smoke.py <task_id> [num_envs] [steps]
"""
import sys

import torch

import mjlab_k1  # noqa: F401
from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg

task = sys.argv[1]
n = int(sys.argv[2]) if len(sys.argv) > 2 else 16
steps = int(sys.argv[3]) if len(sys.argv) > 3 else 300
cfg = load_env_cfg(task)
cfg.scene.num_envs = n
env = ManagerBasedRlEnv(cfg=cfg, device="cuda:0")
obs, _ = env.reset()
dim = env.action_manager.total_action_dim
print(f"SMOKE {task} obs={ {k: tuple(v.shape) for k, v in obs.items()} } act={dim}")
rew = {k: 0.0 for k in env.reward_manager.active_terms}
terms = {k: 0 for k in env.termination_manager.active_terms}
for _ in range(steps):
  a = 0.3 * torch.randn(n, dim, device=env.device)
  a[:, 0] = 0.5  # walk forward a bit
  env.step(a)
  for k in rew:
    rew[k] += env.reward_manager._step_reward[:, env.reward_manager.active_terms.index(k)].mean().item()
  for k in terms:
    terms[k] += int(env.termination_manager.get_term(k).sum())
z = env.scene["robot"].data.root_link_pos_w[:, 2].mean().item()
print(f"SMOKE rewards/step { {k: round(v / steps, 4) for k, v in rew.items()} }")
print(f"SMOKE terminations {terms} trunk_z={z:.3f}")
