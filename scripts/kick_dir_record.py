"""Record pass episodes from Mjlab-KickDir for the Isaac RTX replay, including the per-episode target.
Usage: k1 python scripts/kick_dir_record.py <checkpoint.pt> <out.npz> [episodes] [level]

Keeps the first episode of envs 0..episodes-1 whatever its outcome or length, so nothing is picked. Prints each outcome."""
import sys
from dataclasses import asdict

import numpy as np
import torch
import mjlab_k1  # noqa: F401
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab_k1 import tasks_kick as K
from mjlab_k1.tasks_kick_dir import _st

ckpt, out = sys.argv[1:3]
WANT = int(sys.argv[3]) if len(sys.argv) > 3 else 4
LEVEL = float(sys.argv[4]) if len(sys.argv) > 4 else 1.0
TASK, dev, N = "Mjlab-KickDir-Booster-K1", "cuda:0", 64
cfg = load_env_cfg(TASK, play=True)
cfg.scene.num_envs = N
agent = load_rl_cfg(TASK)
env = RslRlVecEnvWrapper(ManagerBasedRlEnv(cfg=cfg, device=dev), clip_actions=agent.clip_actions)
runner = load_runner_cls(TASK)(env, asdict(agent), device=dev)
runner.load(ckpt, load_cfg={"actor": True}, strict=True, map_location=dev)
policy = runner.get_inference_policy(device=dev)
u = env.unwrapped
st = _st(u)
st.level, st.freeze_level = LEVEL, True
obs, _ = env.reset()
robot, ball, tm = u.scene["robot"], u.scene["ball"], u.termination_manager
org = u.scene.env_origins
pos, quat, bpos, bquat, tgt = [], [], [], [], []
kept = {}
for t in range(int(cfg.episode_length_s / u.step_dt) + 1):
  pos.append((robot.data.body_link_pos_w - org[:, None]).cpu().numpy())
  quat.append(robot.data.body_link_quat_w.cpu().numpy())
  bpos.append((ball.data.root_link_pos_w - org).cpu().numpy())
  bquat.append(ball.data.root_link_quat_w.cpu().numpy())
  tgt.append((u._kick_target - org).cpu().numpy())
  with torch.no_grad():
    a = policy(obs)
  obs, _, done, _ = env.step(a)
  for i in torch.nonzero(done.bool()).flatten().tolist():
    if i < WANT and i not in kept:
      end = "fell" if tm.get_term("fell")[i] else "pass_done" if tm.get_term("pass_done")[i] else "time_out"
      gap = float(np.linalg.norm(bpos[t][i, :2] - tgt[t][i, :2]))
      print(f"env {i}: {t + 1} steps, end {end}, ball to target at end {gap:.2f} m", flush=True)
      kept[i] = {"pos": np.stack([p[i] for p in pos]), "quat": np.stack([q[i] for q in quat]),
                   "ball": np.stack([b[i] for b in bpos]), "ball_quat": np.stack([b[i] for b in bquat]),
                   "target": tgt[t][i, :2]}
  if len(kept) >= WANT:
    break
np.savez_compressed(out, names=np.array(list(robot.body_names)), dt=u.step_dt, goal=np.array(K.GOAL_POS), goal_w=K.GOAL_W, goal_h=K.GOAL_H,
                    ball_r=K.BALL_RADIUS, n=len(kept), **{f"{k}_{i}": v for i, e in kept.items() for k, v in e.items()})
