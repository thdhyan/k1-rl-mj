"""Scripted checks of the KickStrike supervision terms. Prints PASS or FAIL per check."""
import math

import torch
import mjlab_k1  # noqa: F401
from mjlab.envs import ManagerBasedRlEnv
from mjlab_k1 import tasks_kick as K, tasks_kick_dir as D, tasks_kick_strike as S

cfg = S.kick_strike_env_cfg(play=False)
cfg.scene.num_envs = 256
env = ManagerBasedRlEnv(cfg, device="cuda:0")
obs, _ = env.reset()
dev = env.device


def check(name, ok, info=""):
  print("PASS" if ok else "FAIL", name, info)


check("actor obs is 45", obs["actor"].shape[-1] == 45, obs["actor"].shape)
check("action is 15", env.action_manager.total_action_dim == 15)

# side band share, bearing as seen from the robot at reset
ps, r = D._st(env), env.scene["robot"].data
v = (ps.target - r.root_link_pos_w)[:, :2]
bear = torch.atan2(v[:, 1], v[:, 0]) - r.heading_w
bear = torch.atan2(bear.sin(), bear.cos()).abs()
share = ((bear > math.radians(30)) & (bear < math.radians(95))).float().mean().item()
check("many targets in side bands", share > 0.5, f"share {share:.2f}")

st = S._st(env)
dirs = S._dir(env)


def set_ball(vel_xy):
  b = env.scene["ball"]
  s = b.data.root_link_pose_w.clone()
  full = torch.zeros(env.num_envs, 13, device=dev)
  full[:, :7] = s
  full[:, 7:9] = vel_xy
  b.write_root_state_to_sim(full)
  env.sim.forward()


# strike point is behind the ball, on the line to the target
b = K._ball(env).root_link_pos_w
sp = S._strike_point(env)
along = ((b - sp)[:, :2] * dirs).sum(-1)
check("strike point behind ball by the gap", (along - S.STRIKE_GAP).abs().max() < 1e-4)

# first touch along the target: small error, line re-anchored; next touch counts as second
st.prev_v[:] = 0
st.touched[:] = False
set_ball(dirs * 1.5)
S.strike_done(env)
check("first touch detected", st.first_now.all().item() and st.touched.all().item())
check("first touch along target has no error", st.first_err.max() < 0.01, f"max {st.first_err.max():.4f}")
check("first_impulse pays full on target", (S.first_impulse(env) > 0.98).all().item())
check("line re-anchored at the ball", (ps.ball0 - K._ball(env).root_link_pos_w)[:, :2].norm(dim=-1).max() < 1e-4)
rot = torch.stack([-dirs[:, 1], dirs[:, 0]], -1)
set_ball(rot * 3.0)
S.strike_done(env)
check("later touch is a second touch", st.again_now.all().item() and not st.first_now.any().item())

# 90 deg off first touch pays nothing
st.prev_v[:] = 0
st.touched[:] = False
set_ball(rot * 1.5)
S.strike_done(env)
check("first touch 90 deg off pays nothing", (S.first_impulse(env) < 1e-3).all().item(), f"err {st.first_err.mean():.3f}")

# contact model: foot behind ball moving along the normal; magnitude and direction of the closed form
v_pred, d = S.predicted_ball_velocity(env)
pos, vel = S._feet(env)
to = K._ball(env).root_link_pos_w[:, None, :2] - pos[:, :, :2]
i = to.norm(dim=-1).argmin(1)
idx = torch.arange(env.num_envs, device=dev)
n = to[idx, i] / to[idx, i].norm(dim=-1, keepdim=True)
gain = 1.5 * 2.0 / 2.43
want = gain * (vel[idx, i, :2] * n).sum(-1, keepdim=True).clamp_min(0) * n
check("predicted velocity matches the closed form", (v_pred - want).abs().max() < 1e-5)
check("prediction lies on the normal", (v_pred[:, 0] * n[:, 1] - v_pred[:, 1] * n[:, 0]).abs().max() < 1e-5)

# real physics: does a ball hit by a moving box-free impulse leave along the velocity? rollout with zero action runs clean
a = torch.zeros(env.num_envs, 15, device=dev)
for _ in range(50):
  o, rew, term, tout, _ = env.step(a)
check("50 steps run, rewards finite", torch.isfinite(rew).all().item() and torch.isfinite(o["actor"]).all().item())
