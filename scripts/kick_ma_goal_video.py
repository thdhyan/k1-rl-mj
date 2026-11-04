"""One picked episode: robot_a's pass is received and robot_b's touch is the last before the goal. <ckpt> <out.mp4> [envs] [label]
Touches are ball velocity jumps credited by root distance, so the pick is the episode where that credit is clearest.
First episode of each env, deterministic policy. Prints how many episodes qualified, so the pick can be read against the rate."""
import sys

import imageio.v2 as imageio
import mujoco
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
import mjlab_k1  # noqa: F401
from mjlab.envs import ManagerBasedRlEnv
from mjlab_k1 import tasks_kick_ma as T
from k1_runtime.wbc import PolicyModule

ckpt, out = sys.argv[1:3]
n = int(sys.argv[3]) if len(sys.argv) > 3 else 512
LABEL = sys.argv[4] if len(sys.argv) > 4 else "kick_ma"
TOUCH_MPS = 0.4
REACH_M = 0.7  # robot_b root to ball at its touch
cfg = T.kick_ma_env_cfg(play=True)
cfg.scene.num_envs = n
env = ManagerBasedRlEnv(cfg, device="cuda:0")
policy = PolicyModule(ckpt, env.device)
obs, _ = env.reset()
tm, st = env.termination_manager, T._st(env)
ball, a, b = env.scene["ball"].data, env.scene["robot_a"].data, env.scene["robot_b"].data
done = torch.zeros(n, dtype=torch.bool, device=env.device)
goal = done.clone()
qs, recv, da, db, vs = [], [], [], [], []
with torch.no_grad():
  for t in range(int(env.max_episode_length) + 2):
    qs.append(env.sim.data.qpos.cpu().numpy().astype(np.float32))
    recv.append(st.received.cpu().numpy())
    p = ball.root_link_pos_w[:, :2]
    da.append((p - a.root_link_pos_w[:, :2]).norm(dim=-1).cpu().numpy())
    db.append((p - b.root_link_pos_w[:, :2]).norm(dim=-1).cpu().numpy())
    vs.append(ball.root_link_lin_vel_w[:, :2].cpu().numpy())
    act = policy(obs["actor"].reshape(2 * n, -1)).reshape(n, -1)
    live = ~done
    obs, _, term, tout, _ = env.step(act)
    new = live & (term | tout)
    goal |= new & tm.get_term("goal_scored")
    if t == 0:
      end = np.full(n, -1)
    end[new.cpu().numpy()] = t
    done |= new
    if done.all():
      break
qs, recv, da, db, vs = map(np.stack, (qs, recv, da, db, vs))
dv = np.linalg.norm(np.diff(vs, axis=0), axis=-1)
picked = []
goals = np.flatnonzero(goal.cpu().numpy())
for i in goals:
  e = end[i]
  touches = np.flatnonzero(dv[:e, i] > TOUCH_MPS)
  if not recv[:e + 1, i].any() or not len(touches):
    continue
  first_recv = int(np.argmax(recv[:e + 1, i]))
  last = touches[-1]
  passed = touches[touches < first_recv]  # robot_a's touch sent the ball there
  if len(passed) and da[passed[-1], i] < db[passed[-1], i] and last >= first_recv and db[last, i] < min(da[last, i], REACH_M):
    picked.append((i, e, first_recv, int(last)))
picked.sort(key=lambda c: db[c[3], c[0]] - da[c[3], c[0]])
gaps = [float(da[c[3], c[0]] - db[c[3], c[0]]) for c in picked]
print(f"QUALIFIED {len(picked)} of {n} episodes; goals {len(goals)}; clear by over 1 m: {sum(g > 1 for g in gaps)}", flush=True)
i, e, first_recv, last = picked[0]
print(f"PICK env {i} t_end {e * env.step_dt:.2f} s received {first_recv * env.step_dt:.2f} s robot_b last touch {last * env.step_dt:.2f} s "
      f"at that touch robot_b {db[last, i]:.2f} m, robot_a {da[last, i]:.2f} m from ball", flush=True)

m, d = env.sim.mj_model, mujoco.MjData(env.sim.mj_model)
m.vis.global_.offwidth, m.vis.global_.offheight = 1280, 720
r = mujoco.Renderer(m, 720, 1280)
cam = mujoco.MjvCamera()
FONT = ImageFont.load_default(size=26)
w = imageio.get_writer(out, fps=25, codec="libx264", quality=8, macro_block_size=8)



ORIGIN = env.scene.env_origins[i].cpu().numpy()
MARK = (np.array([0.2, 0.5, 1.0, 1.0], np.float32), np.array([1.0, 0.15, 0.15, 1.0], np.float32))
FREE = [env.scene[k].indexing.free_joint_q_adr.cpu().numpy() for k in ("robot_a", "robot_b", "ball")]


def frame(t):
  d.qpos[:] = qs[t, i]
  for adr in FREE:  # the renderer draws the pitch once, at the model origin
    d.qpos[adr[:3]] -= ORIGIN
  mujoco.mj_kinematics(m, d)
  cam.lookat[:], cam.azimuth, cam.elevation, cam.distance = (0.5, 0.0, 0.2), 90.0, -45.0, 10.0
  r.update_scene(d, cam)
  for adr, rgba in zip(FREE[:2], MARK):  # marker above each head: the robots are otherwise identical
    g = r.scene.geoms[r.scene.ngeom]
    mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_SPHERE, np.array([0.13, 0, 0]), d.qpos[adr[:3]] + [0, 0, 0.65], np.eye(3).ravel(), rgba)
    r.scene.ngeom += 1
  img = Image.fromarray(r.render())
  phase = "goal" if t >= e else "robot_b shot" if t >= last else "robot_b chases" if t >= first_recv else "robot_a passes"
  lines = [f"{LABEL}  picked goal, 1 of {len(picked)} in {n} episodes   blue robot_a, red robot_b", f"t {t * env.step_dt:.1f} s   {phase}"]
  draw = ImageDraw.Draw(img)
  for k, s in enumerate(lines):
    draw.text((16, 12 + 32 * k), s, font=FONT, fill="white", stroke_width=3, stroke_fill="black")
  return np.asarray(img)


for t in range(0, e + 1, 2):
  w.append_data(frame(t))
for _ in range(38):  # hold the goal frame 1.5 s
  w.append_data(frame(e))
w.close()
