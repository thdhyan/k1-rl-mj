"""Where a kick policy sends the ball: heading error and landing spread against the commanded target, sim only.
Usage: k1 python scripts/kick_spread.py <task> <checkpoint.pt> <out.json> [envs] [level]
task is Mjlab-Kick-Booster-K1 (fixed goal, as v2 trained) or Mjlab-KickDir-Booster-K1 (targets at any bearing; level default 1.0, the hardest).
First episode of each env, deterministic actions, nothing ends the episode on the ball arriving, so the ball rolls on.
Kick moment: first step the ball moves over 0.3 m/s with the robot within 1 m. Heading is read at 1 m of ball travel.
Bearing is the target's direction seen from the robot facing at the kick moment.
Line miss is the ball's signed lateral offset when it first crosses the line through the target, square to kick-to-target."""
import json
import sys
from dataclasses import asdict

import numpy as np
import torch
import mjlab_k1  # noqa: F401
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab_k1 import tasks_kick as K

task, ckpt, out = sys.argv[1:4]
N, dev = int(sys.argv[4]) if len(sys.argv) > 4 else 1024, "cuda:0"
LEVEL = float(sys.argv[5]) if len(sys.argv) > 5 else 1.0
cfg = load_env_cfg(task, play=True)
cfg.scene.num_envs = N
cfg.episode_length_s = load_env_cfg(task).episode_length_s
for t in ("goal_scored", "pass_done", "strike_done"):
  cfg.terminations.pop(t, None)
agent = load_rl_cfg(task)
env = RslRlVecEnvWrapper(ManagerBasedRlEnv(cfg=cfg, device=dev), clip_actions=agent.clip_actions)
runner = load_runner_cls(task)(env, asdict(agent), device=dev)
runner.load(ckpt, load_cfg={"actor": True}, strict=True, map_location=dev)
policy = runner.get_inference_policy(device=dev)
u = env.unwrapped
if task in ("Mjlab-KickDir-Booster-K1", "Mjlab-KickStrike-Booster-K1"):
  from mjlab_k1.tasks_kick_dir import _st
  st = _st(u)
  st.level, st.freeze_level = LEVEL, True
  u._strike_level_fixed = True
obs, _ = env.reset()
robot = u.scene["robot"].data
ball = u.scene["ball"].data
z = lambda: torch.zeros(N, device=dev)
b = lambda: torch.zeros(N, dtype=torch.bool, device=dev)
over, kicked, read = b(), b(), b()
kick_xy, target_xy = torch.zeros(N, 2, device=dev), torch.zeros(N, 2, device=dev)
bearing, kick_t, err, speed1m, end_speed = z(), z(), z(), z(), z()
along_end, lat_end, dist = z(), z(), z()
fell, crossed = b(), b()
line_miss = z()
wrap = lambda a: torch.atan2(a.sin(), a.cos())
for t in range(int(cfg.episode_length_s / u.step_dt)):
  live = ~over
  bp, bv = ball.root_link_pos_w[:, :2], ball.root_link_lin_vel_w[:, :2]
  near = (bp - robot.root_link_pos_w[:, :2]).norm(dim=-1) < 1.0
  new = live & ~kicked & near & (bv.norm(dim=-1) > 0.3)
  if new.any():
    goal = K._goal_w(u)[:, :2]
    kick_xy[new], target_xy[new] = bp[new], goal[new]
    v = goal - robot.root_link_pos_w[:, :2]
    bearing[new] = wrap(torch.atan2(v[:, 1], v[:, 0]) - robot.heading_w)[new]
    kick_t[new] = t * u.step_dt
    kicked |= new
  d = bp - kick_xy
  arrive = live & kicked & ~read & (d.norm(dim=-1) >= 1.0)
  if arrive.any():
    c = target_xy - kick_xy
    err[arrive] = wrap(torch.atan2(d[:, 1], d[:, 0]) - torch.atan2(c[:, 1], c[:, 0]))[arrive]
    speed1m[arrive] = bv.norm(dim=-1)[arrive]
    read |= arrive
  # latest ball position in the kick-to-target frame; frozen once the episode ends
  c = target_xy - kick_xy
  dn = c.norm(dim=-1).clamp_min(1e-6)
  ax = c / dn[:, None]
  al, la = (d * ax).sum(-1), d[:, 0] * ax[:, 1] - d[:, 1] * ax[:, 0]
  cross = live & kicked & ~crossed & (al >= dn)
  line_miss[cross] = la[cross]
  crossed |= cross
  along_end = torch.where(live, al, along_end)
  lat_end = torch.where(live, la, lat_end)
  dist = torch.where(live, dn, dist)
  end_speed = torch.where(live, bv.norm(dim=-1), end_speed)
  with torch.no_grad():
    obs, _, done, _ = env.step(policy(obs))
  fell |= live & done.bool() & ~u.termination_manager.time_outs
  over |= done.bool()
over_all = over.all().item()
q = lambda v: {"n": len(v), "mean": round(float(np.mean(v)), 3), "std": round(float(np.std(v)), 3), **{f"p{p}": round(float(np.percentile(v, p)), 3) for p in (5, 50, 95)}} if len(v) else None
e, br = err[read].cpu().numpy(), bearing[read].cpu().numpy()
edges = np.radians([-180, -120, -60, -30, 0, 30, 60, 120, 180])
bins = {f"{round(np.degrees(a))}..{round(np.degrees(c))} deg": q(np.degrees(e[(br >= a) & (br < c)])) for a, c in zip(edges[:-1], edges[1:])}
res = {"task": task, "checkpoint": ckpt, "envs": N, "kicked": round(kicked.float().mean().item(), 4), "read_at_1m": round(read.float().mean().item(), 4),
       "fell": round(fell.float().mean().item(), 4), "kick_time_s": q(kick_t[kicked].cpu().numpy()),
       "heading_error_deg": q(np.degrees(e)), "heading_error_abs_deg": q(np.abs(np.degrees(e))), "heading_error_by_bearing": bins,
       "speed_at_1m_mps": q(speed1m[read].cpu().numpy()), "ball_end_speed_mps": q(end_speed[read].cpu().numpy()),
       "ball_end_along_m": q(along_end[read].cpu().numpy()), "ball_end_lateral_m": q(lat_end[read].cpu().numpy()), "target_dist_m": q(dist[read].cpu().numpy()),
       "crossed_line": round(crossed.float().mean().item(), 4), "line_miss_m": q(line_miss[crossed].cpu().numpy()),
       "samples": {"read": read.tolist(), "crossed": crossed.tolist(), "kicked": kicked.tolist(), "err_deg": [round(x, 2) for x in np.degrees(err.cpu().numpy()).tolist()],
                   "bearing_deg": [round(x, 2) for x in np.degrees(bearing.cpu().numpy()).tolist()], "line_miss_m": [round(x, 3) for x in line_miss.tolist()],
                   "dist_m": [round(x, 3) for x in dist.tolist()], "speed_1m": [round(x, 3) for x in speed1m.tolist()]}}
print("SPREAD", json.dumps({k: v for k, v in res.items() if k != "samples"}), flush=True)
json.dump(res, open(out, "w"), indent=1)
