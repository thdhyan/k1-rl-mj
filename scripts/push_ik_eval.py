"""Success rate of a Mjlab-PushIK checkpoint at the full reset ranges: <ckpt> [envs] [out.json]. Success = box within
0.2 m and 10 deg (mod 180), slow; each env counts its first episode only."""
import json
import os
import sys

import torch
import mjlab_k1  # noqa: F401
from mjlab.envs import ManagerBasedRlEnv
from mjlab_k1 import tasks_push_ik as T
from k1_runtime.wbc import PolicyModule

ckpt, n = sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 512
PROG = os.environ.get("K1_EVAL_PROG")  # evaluate at this curriculum value instead of the full ranges
cfg = T.push_ik_env_cfg(play=PROG is None)
cfg.scene.num_envs = n
env = ManagerBasedRlEnv(cfg, device="cuda:0")
policy = PolicyModule(ckpt, env.device)
if PROG is not None:
  T._state(env).prog = float(PROG)
obs, _ = env.reset()
tm = env.termination_manager
done = torch.zeros(n, dtype=torch.bool, device=env.device)
out = {k: torch.zeros(n, dtype=torch.bool, device=env.device) for k in ("success", "fell", "box_oob", "box_tip", "timeout")}
t_end, d_end, e_end, c_end = (torch.zeros(n, device=env.device) for _ in range(4))
touch_b = touch_r = hands = both = left = right = steps = 0.0
with torch.no_grad():
  for t in range(int(env.max_episode_length) + 2):
    d, e, c = *T._errors(env), T.corner_error(env)
    a = policy(obs["actor"].reshape(2 * n, -1)).reshape(n, -1)
    live = ~done
    obs, _, term, tout, _ = env.step(a)
    touch_b += (T._body_contact(env)[live] > 0).float().sum().item()
    touch_r += T._robot_contact(env)[live].sum().item()
    hands += T._hand_frac(env)[live].sum().item()
    both += T._both_hands(env)[live].sum().item()
    per_hand = torch.stack([T._touch(env, f"{g}_hands_box") for g in T.AGENTS]).mean(0)[live].sum(0)
    left, right = left + per_hand[0].item(), right + per_hand[1].item()
    steps += live.sum().item()
    new = live & (term | tout)
    out["success"] |= new & tm.get_term("goal_reached")
    out["fell"] |= new & (tm.get_term("fell_a") | tm.get_term("fell_b"))
    out["box_oob"] |= new & tm.get_term("box_oob")
    out["box_tip"] |= new & tm.get_term("box_tip")
    out["timeout"] |= new & tout
    t_end[new], d_end[new], e_end[new], c_end[new] = (t + 1) * env.step_dt, d[new], e[new], c[new]
    done |= new
    if done.all():
      break
s = out["success"]
r = {k: v.float().mean().item() for k, v in out.items()}
r.update(n=n, finished=done.float().mean().item(), body_touch_frac=touch_b / steps, robots_touch_frac=touch_r / steps,
         hands_on_frac=hands / steps, both_hands_frac=both / steps, left_on_frac=left / steps, right_on_frac=right / steps, corner_err_end_m=c_end.mean().item(),
         t_success=t_end[s].mean().item() if s.any() else None,
         last_err_m_fail=d_end[~s].mean().item() if (~s).any() else None,
         last_err_deg_fail=torch.rad2deg(e_end[~s]).mean().item() if (~s).any() else None)
print(json.dumps(r, indent=1))
if len(sys.argv) > 3:
  json.dump(r, open(sys.argv[3], "w"), indent=1)
