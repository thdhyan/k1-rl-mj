"""Run the bbox head servo on KickStrike and measure centring; save RGB frames with projected boxes for the YOLO check.
<teacher.pt> <out_dir> [envs] [steps]"""
import json
import sys
from pathlib import Path

import numpy as np
import torch
import mjlab_k1  # noqa: F401
from mjlab.envs import ManagerBasedRlEnv
from mjlab.sensor import CameraSensorCfg
from mjlab_k1 import tasks_kick as K, tasks_kick_strike as S
from mjlab_k1.ball_gaze import detect
from mjlab_k1.tasks_head import CAM_OFFSET, VFOV
from k1_runtime.wbc import PolicyModule

ckpt, out = sys.argv[1:3]
n = int(sys.argv[3]) if len(sys.argv) > 3 else 64
steps = int(sys.argv[4]) if len(sys.argv) > 4 else 500
W, H = 320, 240
cfg = S.kick_strike_env_cfg(play=True)
cfg.scene.num_envs = n
cfg.actions["upper_body"].frozen_prob = 1.0
cam = CameraSensorCfg(name="head_cam", parent_body="robot/Head_2", pos=CAM_OFFSET, quat=(0.5, 0.5, -0.5, -0.5),
                      fovy=float(np.degrees(VFOV)), width=W, height=H, data_types=("rgb",))
cfg.scene.sensors = (*cfg.scene.sensors, cam)
env = ManagerBasedRlEnv(cfg, device="cuda:0")
policy = PolicyModule(ckpt, env.device)
obs, _ = env.reset()
sensor, r = env.scene["head_cam"], env.scene["robot"]
frames, boxes, seens, dists = [], [], [], []
stat = {k: [0.0, 0.0] for k in ("near", "mid", "far", "behind")}
err_sum = {k: 0.0 for k in stat}
with torch.no_grad():
  for t in range(steps):
    obs, *_ = env.step(policy(obs["actor"]))
    box, seen = detect(env)
    d = (K._ball(env).root_link_pos_w - r.data.root_link_pos_w)[:, :2].norm(dim=-1)
    p = K.ball_pos_b(env)
    front = torch.atan2(p[:, 1], p[:, 0]).abs() < 1.0  # within the head's yaw reach
    for k, m in (("near", front & (d < 0.8)), ("mid", front & (d >= 0.8) & (d < 2.5)), ("far", front & (d >= 2.5)), ("behind", ~front)):
      stat[k][0] += (seen & m).sum().item()
      stat[k][1] += m.sum().item()
      err_sum[k] += box[seen & m][:, :2].norm(dim=-1).sum().item()
    if t >= 50 and t % 10 == 0:
      k = min(16, n)
      frames.append(sensor.data.rgb[:k].cpu().numpy().astype(np.uint8))
      boxes.append(box[:k].cpu().numpy())
      seens.append(seen[:k].cpu().numpy())
      dists.append(d[:k].cpu().numpy())
Path(out).mkdir(parents=True, exist_ok=True)
np.savez_compressed(Path(out) / "frames.npz", rgb=np.concatenate(frames), box=np.concatenate(boxes), seen=np.concatenate(seens), dist=np.concatenate(dists))
res = {k: dict(in_view=v[0] / max(v[1], 1), share=v[1] / (n * steps), mean_offset=err_sum[k] / max(v[0], 1)) for k, v in stat.items()}
print("SERVO", json.dumps(res))
json.dump(res, open(Path(out) / "servo.json", "w"), indent=1)
