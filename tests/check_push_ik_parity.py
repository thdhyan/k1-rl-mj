"""Run deploy/push_ik_core.py inside the training env: its obs and decoded commands must equal the env's.
  python test/check_push_ik_parity.py <ckpt>   (mjlab venv, K1_PIK_BASE set, GPU)"""
import sys
from pathlib import Path

import numpy as np
import torch
import mjlab_k1  # noqa: F401
from mjlab.envs import ManagerBasedRlEnv
from mjlab_k1 import tasks_push_ik as T

from k1_runtime.push_ik_core import PushIKController  # noqa: E402

N, STEPS = 8, 300
ckpt = sys.argv[1]
cfg = T.push_ik_env_cfg(play=True)
cfg.scene.num_envs = N
env = ManagerBasedRlEnv(cfg, device="cuda:0")
obs, _ = env.reset()
half = T._half(env).cpu().numpy()
ctrls = [PushIKController(ckpt, half[i]) for i in range(N)]
worst = dict(obs=0.0, action=0.0, cmd=0.0, height=0.0, arms=0.0)
checked = 0
for step in range(STEPS):
  actions = []
  for i, c in enumerate(ctrls):
    a, b, box = (T._mocap(env, k)[i].tolist() for k in range(3))
    st = T._state(env)
    goal = (*st.goal[i, :2].tolist(), st.goal_yaw[i].item())
    o = c.obs(a, b, box, goal)
    worst["obs"] = max(worst["obs"], (o.flatten() - obs["actor"][i].cpu()).abs().max().item())
    raw = c.act(o)
    cmd, dh, arms = c.decode(raw)
    actions.append((raw.flatten(), cmd, dh, arms))
  a_t = torch.tensor(np.stack([x[0] for x in actions]), device=env.device, dtype=torch.float32)
  obs, _, term, tout, _ = env.step(a_t)
  am = env.action_manager
  for i, (_, cmd, dh, arms) in enumerate(actions):
    if term[i] or tout[i]:
      ctrls[i].reset()
      continue
    checked += 1
    for k, ag in enumerate(T.AGENTS):
      base, hands = am.get_term(f"base_{ag}"), am.get_term(f"hands_{ag}")
      worst["cmd"] = max(worst["cmd"], np.abs(base._cmd[i].cpu().numpy() - cmd[k]).max())
      worst["height"] = max(worst["height"], abs(base._h[i].item() - (0.52 + dh[k])))
      worst["arms"] = max(worst["arms"], np.abs(hands._q[i].cpu().numpy() - arms[k]).max())
print("worst differences over", checked, "env steps:", {k: float(v) for k, v in worst.items()})
assert all(v < 1e-3 for v in worst.values()), worst
print("PARITY OK")
