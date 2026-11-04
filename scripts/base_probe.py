"""Sanity: frozen base inside a task env must walk at the commanded speed. Usage: python scripts/base_probe.py <task> [vx_norm]"""
import sys, torch
import mjlab_k1  # noqa: F401
from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg

task, vx = sys.argv[1], float(sys.argv[2]) if len(sys.argv) > 2 else 0.4
cfg = load_env_cfg(task, play=True)
cfg.scene.num_envs = 32
env = ManagerBasedRlEnv(cfg=cfg, device="cuda:0")
env.reset()
print('EVENTS', list(env.event_manager.active_terms) if hasattr(env.event_manager,'active_terms') else cfg.events.keys())
dim = env.action_manager.total_action_dim
act = torch.zeros(32, dim, device="cuda:0"); act[:, 0] = vx
robot = env.scene["robot"]
v, falls = [], 0
for i in range(300):
  _, _, term, trunc, _ = env.step(act)
  falls += int(term.sum())
  if i in (5, 50, 150): print('TERMS@%d' % i, {k: int(env.termination_manager.get_term(k).sum()) for k in env.termination_manager.active_terms})
  if i > 100:
    v.append(robot.data.root_link_lin_vel_b[:, 0].mean().item())
print(f"cmd vx={vx} mean body vx={sum(v)/len(v):.3f} terminations={falls} trunk_z={robot.data.root_link_pos_w[:,2].mean().item():.3f}")
