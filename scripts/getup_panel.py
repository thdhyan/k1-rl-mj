"""4x4 panel video of 16 get-up envs: scripts/getup_panel.py <out.mp4> [ckpt|none] [seconds] [task]

ckpt = an rsl_rl model_*.pt (frozen actor via k1_runtime PolicyModule); none = zero actions (robot held at the goal
pose targets, shows the random fallen starts). Camera follows each trunk. Tile text: height, uprightness, stood flag.
"""
import sys

import imageio.v2 as imageio
import mujoco
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

import mjlab_k1  # noqa: F401
from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg
from mjlab_k1.tasks_getup import _height, _stood, _up

out = sys.argv[1]
ckpt = sys.argv[2] if len(sys.argv) > 2 and sys.argv[2] != "none" else None
secs = float(sys.argv[3]) if len(sys.argv) > 3 else 8.0
task = sys.argv[4] if len(sys.argv) > 4 else "Mjlab-GetUpBent-Booster-K1"
N, ROWS, TW, TH = 16, 4, 400, 300

cfg = load_env_cfg(task, play=True)
cfg.scene.num_envs = N
env = ManagerBasedRlEnv(cfg, device="cuda:0")
obs, _ = env.reset()
policy = None
if ckpt:
  try:
    from k1_runtime.wbc import PolicyModule
  except ImportError:  # older zz-bw snapshots
    from mjlab_k1.wbc import PolicyModule
  policy = PolicyModule(ckpt, env.device)
dim = env.action_manager.total_action_dim

m, d = env.sim.mj_model, mujoco.MjData(env.sim.mj_model)
m.vis.global_.offwidth, m.vis.global_.offheight = 1280, 720
r = mujoco.Renderer(m, TH, TW)
cam = mujoco.MjvCamera()
free = env.scene["robot"].indexing.free_joint_q_adr.cpu().numpy()
FONT, SMALL = ImageFont.load_default(size=24), ImageFont.load_default(size=15)
w = imageio.get_writer(out, fps=25, codec="libx264", quality=8, macro_block_size=8)


def tile(i, text, color):
  d.qpos[:] = env.sim.data.qpos[i].cpu().numpy()
  d.qpos[free[:3]] -= env.scene.env_origins[i].cpu().numpy()  # the renderer draws static geometry at the model origin
  mujoco.mj_kinematics(m, d)
  cam.lookat[:], cam.azimuth, cam.elevation, cam.distance = (*d.qpos[free[:2]], 0.3), 135.0, -20.0, 2.2
  r.update_scene(d, cam)
  img = Image.fromarray(r.render())
  ImageDraw.Draw(img).text((6, 4), text, font=SMALL, fill=color, stroke_width=2, stroke_fill="black")
  return img


stood_n = 0
with torch.no_grad():
  for t in range(int(secs / env.step_dt)):
    a = policy(obs["actor"]) if policy else torch.zeros(N, dim, device=env.device)
    if t % 2 == 0:
      h, up, st = _height(env).cpu(), _up(env).cpu(), _stood(env).cpu()
      sheet = Image.new("RGB", (ROWS * TW, ROWS * TH + 40), "black")
      for i in range(N):
        sheet.paste(tile(i, f"#{i}  h {h[i]:.2f}  up {up[i]:+.2f}  {'STOOD' if st[i] else ''}",
                         (90, 255, 120) if st[i] else "white"), (i % ROWS * TW, 40 + i // ROWS * TH))
      ImageDraw.Draw(sheet).text((12, 6), f"{task}  16 envs  {'policy' if policy else 'zero action'}  "
                                 f"t {t * env.step_dt:.1f} s  stood {int(st.sum())}/{N}", font=FONT, fill="white")
      w.append_data(np.asarray(sheet))
    obs, *_ = env.step(a)
    stood_n = int(_stood(env).sum())
w.close()
print(f"PANEL {out} stood {stood_n}/{N}", flush=True)
