"""Servo convergence on a still ball: robot stands, ball placed at a known range and bearing, residual u, v after 3 s."""
import math

import torch
import mjlab_k1  # noqa: F401
from mjlab.envs import ManagerBasedRlEnv
from mjlab_k1 import tasks_kick_strike as S
from mjlab_k1.ball_gaze import detect

CASES = [(r, b) for r in (0.8, 1.2, 2.0, 3.5) for b in (0.0, 0.5, -0.5)]
cfg = S.kick_strike_env_cfg(play=True)
cfg.scene.num_envs = len(CASES)
cfg.actions["upper_body"].frozen_prob = 1.0
cfg.actions["gaze"].noise, cfg.actions["gaze"].dropout = 0.0, 0.0
env = ManagerBasedRlEnv(cfg, device="cuda:0")
env.reset()
r, ball = env.scene["robot"], env.scene["ball"]
act = torch.zeros(env.num_envs, env.action_manager.total_action_dim, device=env.device)
trace = []
with torch.no_grad():
  for t in range(150):
    for i, (rng, b) in enumerate(CASES):
      a = r.data.heading_w[i].item() + b
      st = ball.data.default_root_state[[i]].clone()
      st[0, 0] = r.data.root_link_pos_w[i, 0] + rng * math.cos(a)
      st[0, 1] = r.data.root_link_pos_w[i, 1] + rng * math.sin(a)
      st[0, 7:] = 0.0
      ball.write_root_state_to_sim(st, env_ids=torch.tensor([i], device=env.device))
    env.step(act)
    trace.append(detect(env)[0][:, :2].clone())
g = env.action_manager.get_term("gaze")
for i, c in enumerate(CASES):
  settle = next((t for t in range(150) if all(trace[k][i].norm() < 0.1 for k in range(t, 150))), None)
  print(f"GAZE range {c[0]} bearing {c[1]:+.1f} u {trace[-1][i,0]:+.3f} v {trace[-1][i,1]:+.3f} settle_s {settle and settle * env.step_dt} "
        f"target {g._target[i,0]:+.2f} {g._target[i,1]:+.2f} joint {r.data.joint_pos[i,g._ids[0]]:+.2f} {r.data.joint_pos[i,g._ids[1]]:+.2f}")
