"""9 parallel envs from MuJoCo Warp physics, no episode picked: scripts/kick_grid_render.py strike|ma <ckpt> <out.mp4> [seconds] [title]
Each tile shows the outcome of the env's last episode. Strike: camera follows the ball. Ma: camera sees the whole pitch, blue marker
robot_a, red marker robot_b. Red post = pass target (strike) or pass point (ma)."""
import sys

import imageio.v2 as imageio
import mujoco
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
import mjlab_k1  # noqa: F401
from mjlab.envs import ManagerBasedRlEnv
from mjlab_k1 import tasks_kick_dir as D, tasks_kick_ma as MA, tasks_kick_strike as S
from k1_runtime.wbc import PolicyModule

mode, ckpt, out = sys.argv[1:4]
secs = float(sys.argv[4]) if len(sys.argv) > 4 else 20.0
title = sys.argv[5] if len(sys.argv) > 5 else ("kick_ma: pass then shoot" if mode == "ma" else "kick_strike teacher, level 1")
N, TW, TH = 9, 480, 270
IS_MA = mode == "ma"
cfg = MA.kick_ma_env_cfg(play=True) if IS_MA else S.kick_strike_env_cfg(play=True)
cfg.scene.num_envs = N
env = ManagerBasedRlEnv(cfg, device="cuda:0")
if not IS_MA:
  D._st(env).level, env._strike_level_fixed = 1.0, True
obs, _ = env.reset()
policy = PolicyModule(ckpt, env.device)
m, d = env.sim.mj_model, mujoco.MjData(env.sim.mj_model)
m.vis.global_.offwidth, m.vis.global_.offheight = 1280, 720
r = mujoco.Renderer(m, TH, TW)
cam = mujoco.MjvCamera()
ball = env.scene["ball"]
adr = ball.indexing.free_joint_q_adr.cpu().numpy()
FREE = [env.scene[k].indexing.free_joint_q_adr.cpu().numpy() for k in (("robot_a", "robot_b", "ball") if IS_MA else ("robot", "ball"))]
FONT, SMALL = ImageFont.load_default(size=24), ImageFont.load_default(size=15)
w = imageio.get_writer(out, fps=25, codec="libx264", quality=8, macro_block_size=8)
tm = env.termination_manager
TERMS = ("goal_scored", "fell_a", "fell_b", "ball_oob") if IS_MA else ("strike_done", "fell")


def target(i):
  return (MA._st(env).pass_w if IS_MA else D._st(env).target)[i, :2].cpu().numpy()


def tile(i, lines, color):
  d.qpos[:] = env.sim.data.qpos[i].cpu().numpy()
  o = env.scene.env_origins[i].cpu().numpy()
  for k in FREE:  # the renderer draws static geometry once, at the model origin
    d.qpos[k[:3]] -= o
  mujoco.mj_kinematics(m, d)
  xy = d.qpos[adr[:2]]
  if IS_MA:
    cam.lookat[:], cam.azimuth, cam.elevation, cam.distance = (0.0, 0.0, 0.0), 90.0, -55.0, 10.0
  else:
    cam.lookat[:], cam.azimuth, cam.elevation, cam.distance = (*xy, 0.3), 120.0, -35.0, 5.0
  r.update_scene(d, cam)
  mujoco.mjv_initGeom(r.scene.geoms[r.scene.ngeom], mujoco.mjtGeom.mjGEOM_CYLINDER, np.array((0.04, 0.04, 0.4)),
                      np.array((*(target(i) - o[:2]), 0.4)), np.eye(3).flatten(), np.array((0.95, 0.1, 0.1, 1), dtype=np.float32))
  r.scene.ngeom += 1
  if IS_MA:
    for k, rgba in zip(FREE[:2], ((0.2, 0.5, 1.0, 1.0), (1.0, 0.15, 0.15, 1.0))):
      mujoco.mjv_initGeom(r.scene.geoms[r.scene.ngeom], mujoco.mjtGeom.mjGEOM_SPHERE, np.array((0.16, 0, 0)), d.qpos[k[:3]] + [0, 0, 0.65],
                          np.eye(3).flatten(), np.array(rgba, dtype=np.float32))
      r.scene.ngeom += 1
  img = Image.fromarray(r.render())
  draw = ImageDraw.Draw(img)
  for k, s in enumerate(lines):
    draw.text((6, 4 + 18 * k), s, font=SMALL, fill=color if k == 1 else "white", stroke_width=2, stroke_fill="black")
  return img


last, count = ["running"] * N, {}
with torch.no_grad():
  for t in range(int(secs / env.step_dt)):
    a = policy(obs["actor"].reshape(2 * N, -1)).reshape(N, -1) if IS_MA else policy(obs["actor"])
    if t % 2 == 0:
      sheet = Image.new("RGB", (3 * TW, 3 * TH + 40), "black")
      for i in range(N):
        good = last[i] in ("goal_scored", "strike_done")
        sheet.paste(tile(i, [f"#{i}", f"last ep: {last[i]}"], (90, 255, 120) if good else "white" if last[i] == "running" else (255, 110, 90)),
                    (i % 3 * TW, 40 + i // 3 * TH))
      ImageDraw.Draw(sheet).text((12, 6), f"{title}   9 envs, none picked   t {t * env.step_dt:.1f} s",
                                 font=FONT, fill="white")
      w.append_data(np.asarray(sheet))
    obs, _, term, tout, _ = env.step(a)
    for i in (term | tout).nonzero().flatten().tolist():
      last[i] = next((k for k in TERMS if tm.get_term(k)[i]), "timeout")
      count[last[i]] = count.get(last[i], 0) + 1
w.close()
print("EPISODES", count, flush=True)
