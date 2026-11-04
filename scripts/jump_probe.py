"""Measure every landed jump of a checkpoint over one 20 s episode, and film env 0 without selection.
Usage: k1 python scripts/jump_probe.py <checkpoint.pt> <level> <out.json> <out.mp4> [envs]
level in [0, 1] sets the height target range; it is held fixed here. Every episode starts standing (play config)."""
import json
import sys
from dataclasses import asdict

import imageio.v2 as imageio
import numpy as np
import torch
import mjlab_k1  # noqa: F401
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls

ckpt, level, out, mp4 = sys.argv[1], float(sys.argv[2]), sys.argv[3], sys.argv[4]
N, dev, TASK = int(sys.argv[5]) if len(sys.argv) > 5 else 512, "cuda:0", "Mjlab-Jump-Booster-K1"
cfg = load_env_cfg(TASK, play=True)
cfg.scene.num_envs = N
cfg.episode_length_s = load_env_cfg(TASK).episode_length_s
agent = load_rl_cfg(TASK)
env = RslRlVecEnvWrapper(ManagerBasedRlEnv(cfg=cfg, device=dev, render_mode="rgb_array"), clip_actions=agent.clip_actions)
runner = load_runner_cls(TASK)(env, asdict(agent), device=dev)
runner.load(ckpt, load_cfg={"actor": True}, strict=True, map_location=dev)
policy = runner.get_inference_policy(device=dev)
u = env.unwrapped
j = u.command_manager.get_term("jump")
j.level, j.freeze_level = level, True
robot = u.scene["robot"]
obs, _ = env.reset()
steps = int(cfg.episode_length_s / u.step_dt)
apex, target, air, drift, frames = [], [], [], [], []
falls = 0
alive = torch.ones(N, dtype=torch.bool, device=dev)
windows = torch.zeros(N, device=dev)
for t in range(steps):
  was_win = j.in_window.clone()
  with torch.no_grad():
    obs, _, done, _ = env.step(policy(obs))
  opening = j.in_window & ~was_win
  windows += (opening & alive).float()
  m = j.just_landed & alive
  if m.any():
    apex += j.apex[m].tolist()
    target += j.target[m].tolist()
    air += (j.air_steps[m] * u.step_dt).tolist()
    drift += (robot.data.root_link_pos_w[m, :2] - j.xy0[m]).norm(dim=1).tolist()
  falls += int((alive & done.bool() & ~u.termination_manager.time_outs).sum())
  alive &= ~done.bool()
  if t % 2 == 0:
    frames.append(u.render())
imageio.mimsave(mp4, frames, fps=25, codec="libx264", macro_block_size=None)
q = lambda v: {"p10": round(float(np.percentile(v, 10)), 3), "p50": round(float(np.percentile(v, 50)), 3), "p90": round(float(np.percentile(v, 90)), 3)} if v else None
res = {"checkpoint": ckpt, "level": level, "envs": N, "falls": falls, "windows": int(windows.sum()),
       "landings": len(apex), "trunk_rise_m": q(apex), "target_m": q(target), "air_s": q(air), "drift_m": q(drift),
       "reached_target": round(float(np.mean(np.array(apex) >= np.array(target))), 4) if apex else None}
print("PROBE", json.dumps(res), flush=True)
json.dump(res, open(out, "w"), indent=1)
