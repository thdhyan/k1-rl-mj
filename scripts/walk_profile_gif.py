"""Walking velocity profile as a GIF: one tile per (vx, wz) command, robots walking, border color = tracking status.
Usage: k1 python scripts/walk_profile_gif.py <task> <checkpoint.pt> <grid.json> <out.gif>

Status comes from the scripts/velocity_grid.py JSON of the same checkpoint (8 robots per cell, 10 s), not from the one
robot in the tile. Text in each tile: command, then this robot's speed averaged over the last 2 s so far."""
import json, math, sys
from dataclasses import asdict

import mujoco
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
import mjlab_k1  # noqa: F401
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls

task, ckpt, grid_path, out = sys.argv[1:5]
dev, SETTLE, STEPS, EVERY, TW, TH, HEAD = "cuda:0", 75, 500, 4, 216, 150, 34
VX = [0.0, 0.5, 1.0, 1.5, 2.0]
WZ = [-math.pi, -math.pi / 2, 0.0, math.pi / 2, math.pi]
STATUS = {0: ("both ok", "#1baf7a"), 1: ("vx ok", "#2a78d6"), 2: ("yaw ok", "#e87ba4"), 3: ("neither", "#c3c2b7"),
          4: ("fell", "#2b2b29"), 5: ("dropped", "#eda100")}
grid = json.load(open(grid_path))
prof = grid["profiles"]["none"]
status = [[prof["status"][grid["vx"].index(v)][min(range(len(grid["wz"])), key=lambda k: abs(grid["wz"][k] - w))] for w in WZ] for v in VX]

N = len(VX) * len(WZ)
cfg = load_env_cfg(task, play=True)
cfg.scene.num_envs = N
cfg.episode_length_s = 60.0
cfg.commands["twist"].resampling_time_range = (1e6, 1e6)
agent = load_rl_cfg(task)
env = RslRlVecEnvWrapper(ManagerBasedRlEnv(cfg=cfg, device=dev), clip_actions=agent.clip_actions)
runner = load_runner_cls(task)(env, asdict(agent), device=dev)
runner.load(ckpt, load_cfg={"actor": True}, strict=True, map_location=dev)
policy = runner.get_inference_policy(device=dev)
u = env.unwrapped
twist, robot = u.command_manager.get_term("twist"), u.scene["robot"]
vx_t = torch.tensor(VX, device=dev)[:, None].expand(-1, len(WZ)).flatten()
wz_t = torch.tensor(WZ, device=dev, dtype=torch.float)[None, :].expand(len(VX), -1).flatten()
ramp_v, ramp_w = 1.0 * u.step_dt, 2.0 * u.step_dt

m, d = u.sim.mj_model, mujoco.MjData(u.sim.mj_model)
r = mujoco.Renderer(m, TH, TW)
cam = mujoco.MjvCamera()
font = ImageFont.load_default(size=13)
legend = ImageFont.load_default(size=15)
W, H = TW * len(WZ), TH * len(VX) + HEAD
ema_v, ema_w = torch.zeros(N, device=dev), torch.zeros(N, device=dev)
frames = []
obs, _ = env.reset()
for t in range(STEPS):
  k = max(t - SETTLE, 0)
  twist.vel_command_b[:] = torch.stack([torch.minimum(vx_t, torch.full_like(vx_t, k * ramp_v)), torch.zeros(N, device=dev),
                                        wz_t.sign() * torch.minimum(wz_t.abs(), torch.full_like(wz_t, k * ramp_w))], -1)
  twist.is_standing_env[:] = (vx_t == 0) & (wz_t == 0)
  with torch.no_grad():
    a = policy(obs)
  obs, _, _, _ = env.step(a)
  ema_v += (robot.data.root_link_lin_vel_b[:, 0] - ema_v) * 0.01
  ema_w += (robot.data.root_link_ang_vel_b[:, 2] - ema_w) * 0.01
  if t % EVERY:
    continue
  canvas = Image.new("RGB", (W, H), "white")
  draw = ImageDraw.Draw(canvas)
  x = 8
  for s, (name, col) in STATUS.items():
    if s == 5:
      continue
    draw.rectangle((x, 10, x + 14, 24), fill=col)
    draw.text((x + 20, 9), name, font=legend, fill="#222222")
    x += 30 + int(draw.textlength(name, font=legend)) + 12
  draw.text((x + 6, 9), f"walk_still_v4   t = {t * u.step_dt:.1f} s", font=legend, fill="#222222")
  qpos = u.sim.data.qpos.cpu().numpy()
  for i in range(N):
    d.qpos[:] = qpos[i]
    mujoco.mj_forward(m, d)
    cam.lookat[:] = (d.qpos[0], d.qpos[1], 0.45)
    cam.azimuth, cam.elevation, cam.distance = 135.0, -18.0, 2.1
    r.update_scene(d, cam)
    tile = Image.fromarray(r.render())
    td = ImageDraw.Draw(tile)
    ro, co = i // len(WZ), i % len(WZ)
    td.rectangle((0, 0, TW - 1, TH - 1), outline=STATUS[status[ro][co]][1], width=5)
    td.text((9, 7), f"cmd {VX[ro]:.1f} m/s {WZ[co]:+.2f} rad/s", font=font, fill="white", stroke_width=2, stroke_fill="black")
    td.text((9, TH - 24), f"got {ema_v[i].item():.2f} m/s {ema_w[i].item():+.2f} rad/s", font=font, fill="white",
            stroke_width=2, stroke_fill="black")
    canvas.paste(tile, (co * TW, HEAD + ro * TH))
  frames.append(canvas.convert("P", palette=Image.ADAPTIVE, colors=96))
frames[0].save(out, save_all=True, append_images=frames[1:], duration=int(EVERY * u.step_dt * 1000), loop=0, optimize=True)
print(f"wrote {out}: {len(frames)} frames, {W}x{H}, status {status}")
