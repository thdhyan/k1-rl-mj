"""Render Mjlab-PushIK from MuJoCo Warp physics with box and goal corners and errors drawn on the frame.

  scripts/push_ik_render.py push <ckpt> <out.mp4>    one episode per box size in SIZES
  scripts/push_ik_render.py height <ckpt> <out.mp4>  scripted trunk height sweep, no policy: robots stand, dh 0 -> -0.12 -> 0
  scripts/push_ik_render.py grid <ckpt> <out.mp4>    16 parallel envs, random box sizes, one 30 s stage per curriculum value
Green = goal slab and corners, red = box corners, yellow = corner error. Prints EPISODE / HEIGHT lines.
"""
import sys

import imageio.v2 as imageio
import mujoco
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
import mjlab_k1  # noqa: F401
from mjlab.envs import ManagerBasedRlEnv
from mjlab.envs.mdp import dr
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab_k1 import tasks_push_ik as T
from k1_runtime.wbc import H_STAND, PolicyModule

SIZES = (("small", (0.2, 0.4, 0.15)), ("mid", (0.275, 0.575, 0.225)), ("wide", (0.2, 0.75, 0.2)),
         ("deep tall", (0.35, 0.4, 0.3)), ("large", (0.35, 0.75, 0.3)))
STAGES = (0.0, 0.33, 0.66, 1.0)
mode, ckpt, out = sys.argv[1:4]
GRID = mode == "grid"
N, TW, TH = (16, 480, 270) if GRID else (1, 1280, 720)
cfg = T.push_ik_env_cfg(play=not GRID)  # play pins the curriculum at 1
cfg.scene.num_envs = N
if not GRID:
  cfg.events["box_size"].mode = "startup"  # set_size picks the sizes
env = ManagerBasedRlEnv(cfg, device="cuda:0")
obs, _ = env.reset()
m, d = env.sim.mj_model, mujoco.MjData(env.sim.mj_model)
m.vis.global_.offwidth, m.vis.global_.offheight = 1280, 720
r = mujoco.Renderer(m, TH, TW)
cam = mujoco.MjvCamera()
st = T._state(env)
box = env.scene["box"]
gid = box.indexing.geom_ids[0].item()
adr = box.indexing.free_joint_q_adr
trunk = [env.scene[a] for a in T.AGENTS]
FONT = ImageFont.load_default(size=24)
SMALL = ImageFont.load_default(size=15)
w = imageio.get_writer(out, fps=25, codec="libx264", quality=8, macro_block_size=8)


def set_size(h) -> None:
  asset = SceneEntityCfg("box", geom_names=("box_geom",))
  asset.resolve(env.scene)
  dr.geom_size(env, torch.arange(1, device=env.device), asset_cfg=asset, operation="abs",
               ranges={i: (v, v) for i, v in enumerate(h)})
  m.geom_size[gid] = h


def add(kind, size, pos, rgba, mat=np.eye(3)) -> None:
  mujoco.mjv_initGeom(r.scene.geoms[r.scene.ngeom], kind, np.array(size, dtype=float), np.array(pos, dtype=float),
                      np.asarray(mat, dtype=float).flatten(), np.array(rgba, dtype=np.float32))
  r.scene.ngeom += 1


def draw_goal(half, i=0) -> None:
  x, y, yaw = st.goal[i, 0].item(), st.goal[i, 1].item(), st.goal_yaw[i].item()
  c, s = np.cos(yaw), np.sin(yaw)
  add(mujoco.mjtGeom.mjGEOM_BOX, (half[0], half[1], 0.004), (x, y, 0.005), (0.1, 0.9, 0.2, 0.45),
      ((c, -s, 0), (s, c, 0), (0, 0, 1)))
  zt = 2 * half[2]
  xy = env.scene["box"].data.root_link_pos_w[:, :2]
  b = env.scene["box"].data
  now = T.corners_xy(xy, T._yaw(b.root_link_quat_w), T._half(env))[i].cpu().numpy()
  goal = T.corners_xy(st.goal[:, :2], st.goal_yaw, T._half(env))[i].cpu().numpy()
  if np.linalg.norm(np.roll(now, 2, 0) - goal, axis=1).mean() < np.linalg.norm(now - goal, axis=1).mean():
    now = np.roll(now, 2, 0)
  for p, q in zip(now, goal):
    add(mujoco.mjtGeom.mjGEOM_SPHERE, (0.035, 0, 0), (*q, 0.02), (0.1, 0.9, 0.2, 1))
    add(mujoco.mjtGeom.mjGEOM_SPHERE, (0.035, 0, 0), (*p, zt + 0.01), (0.95, 0.1, 0.1, 1))
    g = r.scene.geoms[r.scene.ngeom]
    add(mujoco.mjtGeom.mjGEOM_LINE, (0, 0, 0), (0, 0, 0), (1, 0.85, 0.1, 1))
    mujoco.mjv_connector(g, mujoco.mjtGeom.mjGEOM_LINE, 3.0, np.array((*p, zt + 0.01)), np.array((*q, 0.02)))


def frame(lines, lookat, az, el, dist, goal_half=None):
  d.qpos[:] = env.sim.data.qpos[0].cpu().numpy()
  mujoco.mj_forward(m, d)
  cam.lookat[:], cam.azimuth, cam.elevation, cam.distance = lookat, az, el, dist
  r.update_scene(d, cam)
  if goal_half is not None:
    draw_goal(goal_half)
  img = Image.fromarray(r.render())
  draw = ImageDraw.Draw(img)
  for i, s in enumerate(lines):
    draw.text((16, 12 + 30 * i), s, font=FONT, fill="white", stroke_width=3, stroke_fill="black")
  w.append_data(np.asarray(img))


def z(k) -> float:
  return trunk[k].data.root_link_pos_w[0, 2].item()


def dh_now(k) -> float:
  return T.HEIGHT_RANGE * env.action_manager.get_term(f"base_{T.AGENTS[k]}").raw_action[0, 3].clamp(-1, 0).item()


def run_push() -> None:
  policy = PolicyModule(ckpt, env.device)
  tm = env.termination_manager
  all_dh = []
  for name, h in SIZES:
    set_size(h)
    obs, _ = env.reset()
    t, dhs = 0, []
    while True:
      e_d, e_y = T._errors(env)
      e_d, e_y, e_c = e_d[0].item(), np.degrees(e_y[0].item()), T.corner_error(env)[0].item()
      with torch.no_grad():
        a = policy(obs["actor"].reshape(2, -1)).reshape(1, -1)
      if t % 2 == 0:
        x, y = d.qpos[adr[0].item()], d.qpos[adr[1].item()]
        frame([f"box {name}: {2 * h[0]:.2f} x {2 * h[1]:.2f} x {2 * h[2]:.2f} m   t {t * env.step_dt:.1f} s",
               f"corner err {e_c:.2f} m   pos {e_d:.2f} m   yaw {e_y:.1f} deg",
               f"trunk z A {z(0):.3f} B {z(1):.3f} m   dh cmd A {dh_now(0):+.3f} B {dh_now(1):+.3f} m"],
              (x, y, 0.3), 120.0, -30.0, 6.0, h)
      obs, _, term, tout, _ = env.step(a)
      t += 1
      dhs.append((dh_now(0), dh_now(1), z(0), z(1)))
      if term[0] or tout[0]:
        break
    why = next((k for k in ("goal_reached", "fell_a", "fell_b", "box_oob", "box_tip") if tm.get_term(k)[0]), "timeout")
    v = np.array(dhs)
    all_dh.append(v)
    print(f"EPISODE {name} {h}: {why} t={t * env.step_dt:.1f}s corner {e_c:.2f} m pos {e_d:.2f} m yaw {e_y:.1f} deg | "
          f"dh mean {v[:, :2].mean():+.3f} min {v[:, :2].min():+.3f} max {v[:, :2].max():+.3f} std {v[:, :2].std():.3f} | "
          f"trunk z mean {v[:, 2:].mean():.3f}", flush=True)


def run_height() -> None:
  # away from the box: robots at their reset spots, zero velocity, hands at centre; only the height action moves
  T_SWEEP = 500
  a = torch.zeros(1, env.action_manager.total_action_dim, device=env.device)
  rows = []
  env.scene["box"].data  # noqa: B018
  for t in range(100 + T_SWEEP):
    u = 0.0 if t < 100 else float(-np.sin(np.pi * (t - 100) / T_SWEEP))  # 0 -> -1 -> 0
    a[:, 3] = a[:, 13] = u
    env.step(a)
    want = H_STAND + T.HEIGHT_RANGE * u
    rows.append((want, z(0), z(1)))
    if t % 2 == 0:
      p = trunk[0].data.root_link_pos_w[0].cpu().numpy()
      frame([f"scripted height sweep, no policy   t {t * env.step_dt:.1f} s",
             f"dh cmd {T.HEIGHT_RANGE * u:+.3f} m   target z {want:.3f} m",
             f"trunk z A {z(0):.3f} m  B {z(1):.3f} m"], (p[0], p[1], 0.35), 90.0, -5.0, 2.2)
  v = np.array(rows)[100:]
  print(f"HEIGHT target {v[:, 0].min():.3f}..{v[:, 0].max():.3f} A {v[:, 1].min():.3f}..{v[:, 1].max():.3f} "
        f"B {v[:, 2].min():.3f}..{v[:, 2].max():.3f} mean_abs_err_A {np.abs(v[:, 1] - v[:, 0]).mean():.3f}", flush=True)


def tile(i, half, lines, color):
  m.geom_size[gid] = half
  d.qpos[:] = env.sim.data.qpos[i].cpu().numpy()
  mujoco.mj_kinematics(m, d)
  box_xy, goal_xy = d.qpos[adr[:2].cpu().numpy()], st.goal[i, :2].cpu().numpy()
  gap = np.linalg.norm(box_xy - goal_xy)
  cam.lookat[:], cam.azimuth, cam.elevation, cam.distance = (*((box_xy + goal_xy) / 2), 0.2), 120.0, -40.0, max(4.0, 2.5 + 1.2 * gap)
  r.update_scene(d, cam)
  draw_goal(half, i)
  img = Image.fromarray(r.render())
  draw = ImageDraw.Draw(img)
  for k, s in enumerate(lines):
    draw.text((6, 4 + 18 * k), s, font=SMALL, fill=color if k == 2 else "white", stroke_width=2, stroke_fill="black")
  return img


def run_grid() -> None:
  policy = PolicyModule(ckpt, env.device)
  tm = env.termination_manager
  COLOR = {"success": (90, 255, 120), "running": "white"}
  for p in STAGES:
    st.prog = p
    obs, _ = env.reset()
    st.prog = p
    d0 = T._errors(env)[0].cpu().numpy().copy()
    last, first, count = ["running"] * N, [None] * N, {}
    for t in range(int(30.0 / env.step_dt)):
      with torch.no_grad():
        a = policy(obs["actor"].reshape(2 * N, -1)).reshape(N, -1)
      if t % 2 == 0:
        c, half = T.corner_error(env).cpu().numpy(), T._half(env).cpu().numpy()
        sheet = Image.new("RGB", (4 * TW, 4 * TH + 40), "black")
        for i in range(N):
          h = half[i]
          sheet.paste(tile(i, h, [f"#{i} box {2 * h[0]:.2f} x {2 * h[1]:.2f} x {2 * h[2]:.2f} m",
                                  f"start dist {d0[i]:.2f} m  corner err {c[i]:.2f} m",
                                  f"last ep: {last[i]}"], COLOR.get(last[i], (255, 110, 90))),
                      (i % 4 * TW, 40 + i // 4 * TH))
        ImageDraw.Draw(sheet).text((12, 6), f"16 parallel envs   curriculum {p:.2f}: goal dist 0.40 to {1.0 + 1.5 * p:.2f} m + "
                                   f"box/goal yaw, robot ring   t {t * env.step_dt:.1f} s", font=FONT, fill="white")
        w.append_data(np.asarray(sheet))
      obs, _, term, tout, _ = env.step(a)
      st.prog = p
      done = (term | tout).nonzero().flatten().tolist()
      if done:
        d_new = T._errors(env)[0].cpu().numpy()
        for i in done:
          why = next((k for k in ("goal_reached", "fell_a", "fell_b", "box_oob", "box_tip") if tm.get_term(k)[i]), "timeout")
          why = "success" if why == "goal_reached" else why
          last[i], d0[i] = why, d_new[i]
          first[i] = first[i] or why
          count[why] = count.get(why, 0) + 1
    print(f"STAGE p={p} episodes {count} first-episode success {first.count('success')}/{N} unfinished {first.count(None)}", flush=True)


{"push": run_push, "height": run_height, "grid": run_grid}[mode]()
w.close()
