"""Record kick episodes from mjlab for the Isaac RTX replay: every body's world pose per step, plus ball and goal.
Usage: k1 python scripts/kick_record.py <checkpoint.pt> <out.npz> [episodes]

Runs 512 envs and keeps the first episodes that started with the ball over 2 m away (the robot has to walk up) and ended
in a goal, as they come, without picking by looks. Prints how many of the eligible episodes scored."""
import sys
from dataclasses import asdict

import numpy as np
import torch
import mjlab_k1  # noqa: F401
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab_k1 import tasks_kick as K

ckpt, out = sys.argv[1:3]
WANT = int(sys.argv[3]) if len(sys.argv) > 3 else 3
TASK, dev, N = "Mjlab-Kick-Booster-K1", "cuda:0", 512
cfg = load_env_cfg(TASK, play=True)
cfg.scene.num_envs = N
agent = load_rl_cfg(TASK)
env = RslRlVecEnvWrapper(ManagerBasedRlEnv(cfg=cfg, device=dev), clip_actions=agent.clip_actions)
runner = load_runner_cls(TASK)(env, asdict(agent), device=dev)
runner.load(ckpt, load_cfg={"actor": True}, strict=True, map_location=dev)
policy = runner.get_inference_policy(device=dev)
u = env.unwrapped
robot, ball, tm = u.scene["robot"], u.scene["ball"], u.termination_manager
org = u.scene.env_origins
names = list(robot.body_names)

obs, _ = env.reset()
pos, quat, bpos, bquat = [], [], [], []
start = torch.zeros(N, dtype=torch.long, device=dev)
eligible = (ball.data.root_link_pos_w[:, :2] - robot.data.root_link_pos_w[:, :2]).norm(dim=-1) > 2.0
kept, n_elig, n_goal = [], 0, 0
HORIZON = int(cfg.episode_length_s / u.step_dt) * 3
for t in range(HORIZON):
  pos.append((robot.data.body_link_pos_w - org[:, None]).cpu().numpy())
  quat.append(robot.data.body_link_quat_w.cpu().numpy())
  bpos.append((ball.data.root_link_pos_w - org).cpu().numpy())
  bquat.append(ball.data.root_link_quat_w.cpu().numpy())
  with torch.no_grad():
    a = policy(obs)
  obs, _, done, _ = env.step(a)
  scored = tm.get_term("goal_scored").bool()
  for i in torch.nonzero(done.bool()).flatten().tolist():
    if eligible[i]:
      n_elig += 1
      n_goal += int(scored[i])
      if scored[i] and len(kept) < WANT and t - start[i].item() > 100:
        s = start[i].item()
        kept.append({"pos": np.stack([p[i] for p in pos[s:t + 1]]), "quat": np.stack([q[i] for q in quat[s:t + 1]]),
                     "ball": np.stack([b[i] for b in bpos[s:t + 1]]), "ball_quat": np.stack([b[i] for b in bquat[s:t + 1]])})
    start[i] = t + 1
    eligible[i] = (ball.data.root_link_pos_w[i, :2] - robot.data.root_link_pos_w[i, :2]).norm() > 2.0
if len(kept) < WANT:
  print(f"only {len(kept)} of {WANT} episodes found")
print(f"walk-up episodes: {n_elig}, scored: {n_goal} ({n_goal / max(n_elig, 1):.1%}); kept lengths {[len(e['pos']) for e in kept]}")
np.savez_compressed(out, names=np.array(names), dt=u.step_dt, goal=np.array(K.GOAL_POS), goal_w=K.GOAL_W, goal_h=K.GOAL_H,
                    ball_r=K.BALL_RADIUS, n=len(kept), **{f"{k}_{i}": v for i, e in enumerate(kept) for k, v in e.items()})
