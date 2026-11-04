"""carry_vel_v1 clips, MuJoCo render: <ckpt> <out_dir>
Fixed mass: env 0 of 8 at each mass, same command schedule inside the trained range, no episode picked.
Drop: grid cells where 5 kg drops, 5 kg next to 0.5 kg; the 5 kg side is the first of 8 envs to lose the box, so it is picked. Counts are printed
and shown. Fell and box_lost are detected but do not end the episode, so the fall stays in frame."""
import json
import sys
from pathlib import Path

import imageio.v2 as imageio
import mujoco
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
import mjlab_k1  # noqa: F401
from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg
from k1_runtime.wbc import PolicyModule

ckpt, out = sys.argv[1:3]
Path(out).mkdir(parents=True, exist_ok=True)
TASK, N, W, H = "Mjlab-CarryVel-Booster-K1", 8, 1280, 720
MASSES = (0.5, 1.0, 2.0, 3.0, 4.0, 5.0)
SCHEDULE = ((1.5, 0.0, 0.0), (4.0, 0.6, 0.0), (4.0, 1.0, 0.0), (4.0, 0.6, 0.8), (4.0, 0.6, -0.8))  # seconds, vx, wz
DROP_CELLS = ((3.0, 0.0), (2.7, -0.7854))
DROP_SECS = 10.0

cfg = load_env_cfg(TASK, play=True)
cfg.scene.num_envs = N
cfg.episode_length_s = 60.0
cfg.commands["twist"].resampling_time_range = (1e6, 1e6)
for name in ("fell", "box_lost"):
  def probe(env, _real=cfg.terminations[name].func, _name=name, **kw):
    setattr(env, f"_probe_{_name}", _real(env, **kw).bool())
    return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
  cfg.terminations[name].func = probe
env = ManagerBasedRlEnv(cfg, device="cuda:0")
policy = PolicyModule(ckpt, env.device)
clip = load_rl_cfg(TASK).clip_actions
twist, robot = env.command_manager.get_term("twist"), env.scene["robot"]
FREE = [env.scene[k].indexing.free_joint_q_adr.cpu().numpy() for k in ("robot", "box")]
ORIGINS = env.scene.env_origins.cpu().numpy()
m, d = env.sim.mj_model, mujoco.MjData(env.sim.mj_model)
m.vis.global_.offwidth, m.vis.global_.offheight = W, H
r = mujoco.Renderer(m, H, W)
cam = mujoco.MjvCamera()
FONT = ImageFont.load_default(size=28)


def run(mass, steps, command):
  """All N envs. Returns qpos[T, N, nq], measured vx and wz of each env, first fell and box_lost step per env (-1 = never)."""
  env.box_mass_fixed = mass
  obs, _ = env.reset()
  env.episode_length_buf[:] = 0  # the 1 s weld hand-off starts now in every env
  qs, vs = [], []
  fell, lost = -np.ones(N, int), -np.ones(N, int)
  now = torch.zeros((N, 3), device=env.device)
  for t in range(steps):
    target = torch.tensor(command(t * env.step_dt), device=env.device, dtype=torch.float32)
    limit = torch.tensor((1.0, 0.0, 2.0), device=env.device) * env.step_dt  # same ramp as scripts/velocity_grid.py
    now = now + torch.maximum(torch.minimum(target - now, limit), -limit)
    twist.vel_command_b[:] = now
    twist.is_standing_env[:] = bool((target == 0).all())
    qs.append(env.sim.data.qpos.cpu().numpy().astype(np.float32))
    vs.append(torch.stack((robot.data.root_link_lin_vel_b[:, 0], robot.data.root_link_ang_vel_b[:, 2]), -1).cpu().numpy())
    with torch.no_grad():
      a = policy(obs["actor"])
    if clip:
      a = a.clamp(-clip, clip)
    obs, *_ = env.step(a)
    for store, flag in ((fell, env._probe_fell), (lost, env._probe_box_lost)):
      hit = flag.cpu().numpy() & (store < 0)
      store[hit] = t
  return np.stack(qs), np.stack(vs), fell, lost


def frame(q, i, lines, look=None):
  d.qpos[:] = q[i]
  for k in FREE:  # the renderer draws static geometry once, at the model origin
    d.qpos[k[:3]] -= ORIGINS[i]
  mujoco.mj_kinematics(m, d)
  xy = d.qpos[FREE[0][:2]] if look is None else look
  cam.lookat[:], cam.azimuth, cam.elevation, cam.distance = (*xy, 0.55), 135.0, -18.0, 3.2
  r.update_scene(d, cam)
  img = Image.fromarray(r.render())
  draw = ImageDraw.Draw(img)
  for k, s in enumerate(lines):
    draw.text((16, 12 + 36 * k), s, font=FONT, fill="white", stroke_width=3, stroke_fill="black")
  return img


def writer(path):
  return imageio.get_writer(path, fps=25, codec="libx264", quality=8, macro_block_size=8)


def scheduled(t):
  for secs, vx, wz in SCHEDULE:
    if t < secs:
      return vx, 0.0, wz
    t -= secs
  return SCHEDULE[-1][1], 0.0, SCHEDULE[-1][2]


steps = int(sum(s[0] for s in SCHEDULE) / env.step_dt)
summary = {"fixed_mass": {}, "drop": []}
sheet_w = writer(f"{out}/carry_vel_v1_masses_grid.mp4")
tiles = {}
for mass in MASSES:
  q, v, fell, lost = run(mass, steps, scheduled)
  summary["fixed_mass"][str(mass)] = dict(envs=N, fell=int((fell >= 0).sum()), dropped=int((lost >= 0).sum()), env0_fell_s=fell[0] * env.step_dt, env0_lost_s=lost[0] * env.step_dt)
  print("MASS", mass, summary["fixed_mass"][str(mass)], flush=True)
  tag = f"{mass:g} kg   of {N}: dropped {(lost >= 0).sum()}, fell {(fell >= 0).sum()}"
  w = writer(f"{out}/carry_vel_v1_{mass:g}kg.mp4")
  frames = []
  for t in range(0, steps, 2):
    vx, _, wz = scheduled(t * env.step_dt)
    state = " BOX LOST" if 0 <= lost[0] <= t else " FELL" if 0 <= fell[0] <= t else ""
    img = frame(q[t], 0, [f"carry_vel_v1  {tag}", f"t {t * env.step_dt:.1f} s  cmd {vx:.1f} m/s {wz:+.1f} rad/s  actual {v[t, 0, 0]:.2f} {v[t, 0, 1]:+.2f}{state}"])
    w.append_data(np.asarray(img))
    frames.append(img.resize((640, 360)))
  w.close()
  tiles[mass] = frames
for k in range(len(tiles[MASSES[0]])):
  sheet = Image.new("RGB", (1920, 720))
  for j, mass in enumerate(MASSES):
    sheet.paste(tiles[mass][k], (j % 3 * 640, j // 3 * 360))
  sheet_w.append_data(np.asarray(sheet))
sheet_w.close()
del tiles

n_drop = int(DROP_SECS / env.step_dt)
for vx, wz in DROP_CELLS:
  command = lambda t, vx=vx, wz=wz: (0.0, 0.0, 0.0) if t < 1.5 else (vx, 0.0, wz)  # noqa: E731
  heavy = run(5.0, n_drop, command)
  light = run(0.5, n_drop, command)
  hit = np.flatnonzero(heavy[3] >= 0)
  rec = dict(cmd_vx=vx, cmd_wz=wz, dropped_5kg=len(hit), fell_5kg=int((heavy[2] >= 0).sum()), dropped_0p5kg=int((light[3] >= 0).sum()), fell_0p5kg=int((light[2] >= 0).sum()), envs=N)
  summary["drop"].append(rec)
  print("DROP", rec, flush=True)
  if not len(hit):
    continue
  i5 = int(hit[np.argmin(heavy[3][hit])])
  t_drop = int(heavy[3][i5])
  end = min(n_drop, t_drop + int(2.5 / env.step_dt))
  w = writer(f"{out}/carry_vel_v1_drop_vx{vx:g}_wz{wz:g}.mp4")
  held = None
  for t in range(0, end, 2):
    if t >= t_drop and held is None:  # stop following, so the falling box stays in frame
      held = heavy[0][t, i5, FREE[1][:2]] - ORIGINS[i5][:2]
    a = frame(heavy[0][t], i5, [f"5 kg, picked: {len(hit)} of {N} dropped", f"t {t * env.step_dt:.1f} s  actual {heavy[1][t, i5, 0]:.2f} m/s" + ("  BOX LOST" if t >= t_drop else "")], held)
    b = frame(light[0][t], 0, [f"0.5 kg, env 0: {(light[3] >= 0).sum()} of {N} dropped", f"cmd {vx:g} m/s {wz:+.2f} rad/s, BEYOND trained range"])
    sheet = Image.new("RGB", (1280, 360))
    sheet.paste(a.resize((640, 360)), (0, 0))
    sheet.paste(b.resize((640, 360)), (640, 0))
    w.append_data(np.asarray(sheet))
  w.close()
json.dump(summary, open(f"{out}/carry_vel_v1_clips.json", "w"), indent=1)
print("CLIPS_DONE", flush=True)
