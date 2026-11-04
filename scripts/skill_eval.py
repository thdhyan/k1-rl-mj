"""First episode of each env, deterministic actions: share of envs per termination, mean episode seconds, twist error if the task has one.
Usage: k1 python scripts/skill_eval.py <task> <checkpoint.pt> <out.json> [envs]"""
import json
import sys
from dataclasses import asdict

import torch
import mjlab_k1  # noqa: F401
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls

task, ckpt, out = sys.argv[1:4]
N, dev = int(sys.argv[4]) if len(sys.argv) > 4 else 512, "cuda:0"
cfg = load_env_cfg(task, play=True)
cfg.scene.num_envs = N
cfg.episode_length_s = load_env_cfg(task).episode_length_s  # play configs may set 1e9 s, which would never end the loop
agent = load_rl_cfg(task)
env = RslRlVecEnvWrapper(ManagerBasedRlEnv(cfg=cfg, device=dev), clip_actions=agent.clip_actions)
runner = load_runner_cls(task)(env, asdict(agent), device=dev)
runner.load(ckpt, load_cfg={"actor": True}, strict=True, map_location=dev)
policy = runner.get_inference_policy(device=dev)
u = env.unwrapped
tm = u.termination_manager
names = list(tm.active_terms)
counts = dict.fromkeys(names, 0)
seen = torch.zeros(N, dtype=torch.bool, device=dev)
length = torch.zeros(N, device=dev)
obs, _ = env.reset()
for t in range(int(cfg.episode_length_s / u.step_dt) + 5):
  with torch.no_grad():
    a = policy(obs)
  obs, _, done, _ = env.step(a)
  new = done.bool() & ~seen
  for n in names:
    counts[n] += int((tm.get_term(n).bool() & new).sum())
  length[new] = (t + 1) * u.step_dt
  seen |= new
  if seen.all():
    break
res = {"task": task, "checkpoint": ckpt, "envs": N, "finished": int(seen.sum()), "mean_s": round(length[seen].mean().item(), 2),
       **{n: round(c / max(int(seen.sum()), 1), 4) for n, c in counts.items()}}
print("EVAL", res, flush=True)
json.dump(res, open(out, "w"), indent=1)
