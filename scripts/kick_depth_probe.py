"""Head-camera depth on KickStrike: render speed, encoder speed, and how often the ball is in view. <teacher.pt> <resnet.pth> <out_dir> [envs]"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import mjlab_k1  # noqa: F401
from mjlab.envs import ManagerBasedRlEnv
from mjlab.sensor import CameraSensorCfg
from mjlab_k1 import tasks_kick as K, tasks_kick_strike as S
from mjlab_k1.depth_encoder import FrozenDepthEncoder
from mjlab_k1.tasks_head import CAM_OFFSET, HFOV, VFOV
from k1_runtime.wbc import PolicyModule
from mjlab.utils.lab_api.math import quat_apply, quat_apply_inverse

ckpt, weights, out = sys.argv[1:4]
n = int(sys.argv[4]) if len(sys.argv) > 4 else 64
W, H = 128, 96
cfg = S.kick_strike_env_cfg(play=True)
cfg.scene.num_envs = n
cam = CameraSensorCfg(name="head_cam", parent_body="robot/Head_2", pos=CAM_OFFSET, quat=(0.5, 0.5, -0.5, -0.5),
                      fovy=float(np.degrees(VFOV)), width=W, height=H, data_types=("depth",))
cfg.scene.sensors = (*cfg.scene.sensors, cam)
env = ManagerBasedRlEnv(cfg, device="cuda:0")
policy = PolicyModule(ckpt, env.device)
enc = FrozenDepthEncoder(weights).to(env.device)
sensor = env.scene["head_cam"]
robot = env.scene["robot"]
head = robot.find_bodies(("Head_2",))[0][0]
obs, _ = env.reset()
Path(out).mkdir(parents=True, exist_ok=True)

seen, near_seen, near_n, steps = 0.0, 0.0, 0.0, 0
t_sim = t_cam = t_enc = 0.0
frames = []
with torch.no_grad():
  for t in range(300):
    t0 = time.time()
    obs, *_ = env.step(policy(obs["actor"]))
    t1 = time.time()
    depth = sensor.data.depth[..., 0]
    t2 = time.time()
    feat = enc(depth)
    torch.cuda.synchronize()
    t3 = time.time()
    t_sim, t_cam, t_enc = t_sim + t1 - t0, t_cam + t2 - t1, t_enc + t3 - t2
    q = robot.data.body_link_quat_w[:, head]
    c = robot.data.body_link_pos_w[:, head] + quat_apply(q, torch.tensor(CAM_OFFSET, device=env.device).expand(n, 3))
    rel = quat_apply_inverse(q, K._ball(env).root_link_pos_w - c)
    vis = (rel[:, 0] > 0.05) & (torch.atan2(rel[:, 1], rel[:, 0]).abs() < HFOV / 2) & (torch.atan2(rel[:, 2], rel[:, 0]).abs() < VFOV / 2)
    near = (K._ball(env).root_link_pos_w - robot.data.root_link_pos_w)[:, :2].norm(dim=-1) < 0.8
    seen += vis.float().mean().item()
    near_seen += (vis & near).float().sum().item()
    near_n += near.float().sum().item()
    steps += 1
    if t % 75 == 0:
      frames.append(depth[0].cpu().numpy())
np.savez(Path(out) / "depth_frames.npz", frames=np.stack(frames))
r = dict(envs=n, res=[W, H], feat=list(feat.shape), in_view=seen / steps, in_view_within_0p8m=near_seen / max(near_n, 1), near_frac=near_n / (n * steps),
         step_ms_env_incl_render=1000 * t_sim / steps, read_ms=1000 * t_cam / steps, encode_ms=1000 * t_enc / steps,
         depth_min=float(depth.min()), depth_max=float(depth.max()), ball_in_frame_pixels_env0=int(((depth[0] > 0) & (depth[0] < 1.5)).sum()))
print("PROBE", json.dumps(r))
json.dump(r, open(Path(out) / "probe.json", "w"), indent=1)
