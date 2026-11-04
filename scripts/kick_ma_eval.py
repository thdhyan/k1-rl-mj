"""Pass-and-shoot rates of a Mjlab-KickMA checkpoint: <ckpt> [envs] [out.json]. Each env counts its first episode only.
Touches are ball velocity jumps over TOUCH_MPS, credited to the nearer root (no foot contact check). pass_and_shot: robot_a
touched before the ball was received, then the last touch before the goal was robot_b's within REACH_M."""
import json
import sys

import torch
import mjlab_k1  # noqa: F401
from mjlab.envs import ManagerBasedRlEnv
from mjlab_k1 import tasks_kick_ma as T
from k1_runtime.wbc import PolicyModule

TOUCH_MPS, REACH_M = 0.4, 0.7
ckpt, n = sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 512
cfg = T.kick_ma_env_cfg(play=True)
cfg.scene.num_envs = n
env = ManagerBasedRlEnv(cfg, device="cuda:0")
policy = PolicyModule(ckpt, env.device)
obs, _ = env.reset()
tm, st = env.termination_manager, T._st(env)
b = lambda: torch.zeros(n, dtype=torch.bool, device=env.device)
done = b()
out = {k: b() for k in ("received", "switched", "goal", "fell_a", "fell_b", "ball_oob", "timeout")}
t_end = torch.zeros(n, device=env.device)
ball, ra, rb = env.scene["ball"].data, env.scene["robot_a"].data, env.scene["robot_b"].data
a_touch, last_b, moved = b(), b(), b()
d_a = torch.zeros(n, device=env.device)
b_touches, b_back, far_falls, fall_after_move = 0, 0, b(), b()
v_prev = ball.root_link_lin_vel_w[:, :2].clone()
with torch.no_grad():
  for t in range(int(env.max_episode_length) + 2):
    a = policy(obs["actor"].reshape(2 * n, -1)).reshape(n, -1)
    # latches are cleared by the auto-reset inside step, so read them before stepping
    live = ~done
    p, v = ball.root_link_pos_w, ball.root_link_lin_vel_w[:, :2]
    d_a = (p - ra.root_link_pos_w)[:, :2].norm(dim=-1)
    d_b = (p - rb.root_link_pos_w)[:, :2].norm(dim=-1)
    moved |= live & (v.norm(dim=-1) > 0.3)
    if t:
      touch = live & ((v - v_prev).norm(dim=-1) > TOUCH_MPS)
      by_b = touch & (d_b < d_a)
      a_touch |= touch & ~by_b & ~st.received
      last_b = (last_b & ~touch) | (by_b & (d_b < REACH_M))
      to_goal = (T._goal_w(env) - p)[:, :2]
      b_touches += by_b.sum().item()
      b_back += (by_b & ((v * to_goal).sum(-1) < 0)).sum().item()
    v_prev = v.clone()
    out["received"] |= live & st.received
    out["switched"] |= live & st.switched
    obs, _, term, tout, _ = env.step(a)
    new = live & (term | tout)
    out["goal"] |= new & tm.get_term("goal_scored")
    fa = new & tm.get_term("fell_a")
    out["fell_a"] |= fa
    far_falls |= fa & (d_a > 1.0)
    fall_after_move |= fa & moved
    out["fell_b"] |= new & tm.get_term("fell_b")
    out["ball_oob"] |= new & tm.get_term("ball_oob")
    out["timeout"] |= new & tout
    t_end[new] = (t + 1) * env.step_dt
    done |= new
    if done.all():
      break
r = {k: v.float().mean().item() for k, v in out.items()}
g = out["goal"]
r.update(pass_and_shot=(g & out["received"] & a_touch & last_b).float().mean().item(),
         fell_a_ball_far=far_falls.float().mean().item(), fell_a_after_ball_moved=fall_after_move.float().mean().item(),
         b_touches=b_touches, b_touches_ball_away_from_goal=b_back)
r.update(n=n, finished=done.float().mean().item(), goal_given_received=(g & out["received"]).sum().item() / max(out["received"].sum().item(), 1),
         goal_without_received=(g & ~out["received"]).float().mean().item(), t_goal=t_end[g].mean().item() if g.any() else None)
print(json.dumps(r, indent=1))
if len(sys.argv) > 3:
  json.dump(r, open(sys.argv[3], "w"), indent=1)
