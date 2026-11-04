"""Ball in view of the head camera against a fixed head pitch and pan, geometry from rollouts of a KickStrike checkpoint. <teacher.pt> [envs]
The head and arms are frozen at the default pose in the rollout; camera poses are recomputed for each candidate head angle."""
import json
import math
import sys

import torch
import mjlab_k1  # noqa: F401
from mjlab.envs import ManagerBasedRlEnv
from mjlab_k1 import tasks_kick as K, tasks_kick_strike as S
from mjlab_k1.tasks_head import HFOV, VFOV
from k1_runtime.wbc import PolicyModule

ckpt, n = sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 256
cfg = S.kick_strike_env_cfg(play=True)
cfg.scene.num_envs = n
cfg.actions["upper_body"].frozen_prob = 1.0
env = ManagerBasedRlEnv(cfg, device="cuda:0")
policy = PolicyModule(ckpt, env.device)
obs, _ = env.reset()
r = env.scene["robot"]
head = r.find_bodies(("Head_2",))[0][0]
PITCHES = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
hits = {p: [0.0, 0.0, 0.0, 0.0] for p in PITCHES}  # near seen, near n, far seen, far n
pan_hits = {p: [0.0, 0.0] for p in PITCHES}  # near seen with the head yaw following the ball, near n
with torch.no_grad():
  for t in range(400):
    obs, *_ = env.step(policy(obs["actor"]))
    h = r.data.body_link_pos_w[:, head]
    rel = S.K._yaw_rel(env, K._ball(env).root_link_pos_w - h)
    dist = (K._ball(env).root_link_pos_w - r.data.root_link_pos_w)[:, :2].norm(dim=-1)
    near = dist < 0.8
    for p in PITCHES:  # rotate into a camera pitched down by p, about the yaw-frame y axis
      c, s = math.cos(p), math.sin(p)
      x, z = c * rel[:, 0] - s * rel[:, 2], s * rel[:, 0] + c * rel[:, 2]
      vy = torch.atan2(z, x).abs() < VFOV / 2
      vis = (x > 0.05) & (torch.atan2(rel[:, 1], x).abs() < HFOV / 2) & vy
      hits[p][0] += (vis & near).sum().item()
      hits[p][1] += near.sum().item()
      hits[p][2] += (vis & ~near).sum().item()
      hits[p][3] += (~near).sum().item()
      # head yaw free: only the vertical test, with range along the bearing
      rng = rel[:, :2].norm(dim=-1)
      x2, z2 = c * rng - s * rel[:, 2], s * rng + c * rel[:, 2]
      pan = (x2 > 0.05) & (torch.atan2(z2, x2).abs() < VFOV / 2)
      pan_hits[p][0] += (pan & near).sum().item()
      pan_hits[p][1] += near.sum().item()
out = {str(p): dict(near_view=v[0] / max(v[1], 1), far_view=v[2] / max(v[3], 1), near_view_pan=pan_hits[p][0] / max(pan_hits[p][1], 1)) for p, v in hits.items()}
print("SWEEP", json.dumps(out))
