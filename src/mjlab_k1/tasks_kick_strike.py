"""Directed kick teacher (Mjlab-KickStrike-Booster-K1): tight line supervision plus a contact model of the strike.

Builds on Mjlab-KickDir. A frictionless sphere leaves along the contact normal, so with d = ball-to-target the strike
point is the back of the ball, ball - r * d, and for foot velocity v_f, normal n (foot to ball) the ball leaves at
  v_ball = (1 + e) * m_f / (m_f + m_b) * max(v_f . n, 0) * n
That closed form is the limit of sampling points on the ball, so no point cloud is needed in the teacher.

New supervision:
  strike_zone       the nearer foot sits behind the ball on the line to the target, before the first touch.
  momentum_preview  the model's predicted ball velocity from the nearer foot points at the target, before the first touch.
  first_impulse     one-shot on the measured ball velocity change of the first touch, std 0.1 rad.
  off_line          tighter. The line is re-anchored to the ball at the first touch, so approach nudges do not count.
  second_touch      cost for each further touch, so the result is one kick and not a dribble.

Half of the targets are sampled in the side bands (30-90 deg either side of the robot). The hit-rate level rule of
KickDir stayed at level 0, so the thresholds here are lower. Observations add feet and strike point to the 30 of KickDir;
scripts/kick_strike_init.py pads the kick_dir_v1 weights to match.
"""
from __future__ import annotations

import math
import os

import torch

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.metrics_manager import MetricsTermCfg
from mjlab.managers.observation_manager import ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.termination_manager import TerminationTermCfg

from . import tasks_kick as K
from .ball_gaze import attach_ball_gaze
from . import tasks_kick_dir as D

SIDE_FRAC = float(os.environ.get("K1_KS_SIDE_FRAC", "0.5"))
SIDE_BAND = (math.radians(30.0), math.radians(90.0))
LEVEL_UP, LEVEL_DOWN = float(os.environ.get("K1_KS_LEVEL_UP", "0.35")), 0.1
IMPULSE_MPS = 0.4  # ball velocity change in one step that counts as a touch
RESTITUTION, FOOT_MASS, REF_SPEED = 0.5, 2.0, 1.5  # contact model; effective foot mass in kg
STRIKE_GAP = K.BALL_RADIUS + 0.06  # foot site to ball centre at the strike, m


class StrikeState:
  def __init__(self, env):
    n, dev = env.num_envs, env.device
    self.prev_v = torch.zeros(n, 2, device=dev)
    self.touched = torch.zeros(n, dtype=torch.bool, device=dev)
    self.first_now = torch.zeros(n, dtype=torch.bool, device=dev)
    self.again_now = torch.zeros(n, dtype=torch.bool, device=dev)
    self.first_err = torch.zeros(n, device=dev)  # rad, heading error of the first touch
    self.attempts, self.hits = 0, 0
    r = env.scene["robot"]
    self.sites = torch.tensor(r.find_sites(("left_foot", "right_foot"), preserve_order=True)[0], device=dev)
    self.bodies = torch.tensor(r.find_bodies(("left_foot_link", "right_foot_link"), preserve_order=True)[0], device=dev)


def _st(env) -> StrikeState:
  st = getattr(env, "_strike_state", None)
  if st is None:
    st = env._strike_state = StrikeState(env)
  return st


def _dir(env) -> torch.Tensor:
  """Unit ball-to-target direction, xy."""
  v = (D._st(env).target - K._ball(env).root_link_pos_w)[:, :2]
  return v / v.norm(dim=-1, keepdim=True).clamp_min(1e-6)


def _feet(env):
  st, d = _st(env), env.scene["robot"].data
  return d.site_pos_w[:, st.sites], d.body_link_lin_vel_w[:, st.bodies]


def _strike_point(env) -> torch.Tensor:
  b = K._ball(env).root_link_pos_w
  return torch.cat([b[:, :2] - STRIKE_GAP * _dir(env), b[:, 2:]], -1)


def predicted_ball_velocity(env):
  """Contact model for the foot nearer the ball: predicted ball velocity (n, 2) and that foot's distance (n,)."""
  pos, vel = _feet(env)
  to = K._ball(env).root_link_pos_w[:, None, :2] - pos[:, :, :2]
  dist = to.norm(dim=-1)
  i = dist.argmin(1, keepdim=True)
  d = dist.gather(1, i)
  normal = to.gather(1, i[..., None].expand(-1, -1, 2))[:, 0] / d.clamp_min(1e-6)
  vf = vel[:, :, :2].gather(1, i[..., None].expand(-1, -1, 2))[:, 0]
  gain = (1.0 + RESTITUTION) * FOOT_MASS / (FOOT_MASS + K.BALL_MASS)
  return gain * (vf * normal).sum(-1, keepdim=True).clamp_min(0.0) * normal, d[:, 0]


def _angle(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
  return torch.atan2(a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0], (a * b).sum(-1))


##
# Termination that also keeps the touch state (terminations run before rewards each step)
##
def strike_done(env) -> torch.Tensor:
  st, ps = _st(env), D._st(env)
  v = K._ball(env).root_link_lin_vel_w[:, :2]
  dv = v - st.prev_v
  touch = dv.norm(dim=-1) > IMPULSE_MPS
  st.first_now = touch & ~st.touched
  st.again_now = touch & st.touched
  st.first_err[st.first_now] = _angle(dv, _dir(env))[st.first_now].abs()
  # the line now starts where the kick happened
  ps.ball0[st.first_now] = K._ball(env).root_link_pos_w[st.first_now]
  st.touched = st.touched | touch
  st.prev_v = v.clone()
  return D.pass_done(env)


##
# Observations (yaw frame of the trunk)
##
def obs_feet_pos(env) -> torch.Tensor:
  pos, _ = _feet(env)
  rel = pos - env.scene["robot"].data.root_link_pos_w[:, None]
  return torch.cat([K._yaw_rel(env, rel[:, i]) for i in range(2)], -1)


def obs_feet_vel(env) -> torch.Tensor:
  _, vel = _feet(env)
  return torch.cat([K._yaw_rel(env, vel[:, i]) for i in range(2)], -1)


def obs_strike_point(env) -> torch.Tensor:
  return K._yaw_rel(env, _strike_point(env) - env.scene["robot"].data.root_link_pos_w)


##
# Rewards
##
def strike_zone(env, std: float = 0.12) -> torch.Tensor:
  pos, _ = _feet(env)
  d = (pos[:, :, :2] - _strike_point(env)[:, None, :2]).norm(dim=-1).min(1).values
  return torch.exp(-(d / std) ** 2) * (~_st(env).touched).float()


def momentum_preview(env, std: float = 0.2, near: float = 0.3) -> torch.Tensor:
  v, d = predicted_ball_velocity(env)
  speed = v.norm(dim=-1)
  aim = torch.exp(-(_angle(v, _dir(env)) / std) ** 2)
  return aim * (speed / REF_SPEED).clamp_max(1.0) * (d < near).float() * (~_st(env).touched).float()


def first_impulse(env, std: float = 0.1) -> torch.Tensor:
  st = _st(env)
  return torch.exp(-(st.first_err / std) ** 2) * st.first_now.float()


def second_touch(env) -> torch.Tensor:
  return _st(env).again_now.float()


def off_line_tight(env, scale: float = 0.2) -> torch.Tensor:
  return D.off_line(env, scale=scale) * _st(env).touched.float()


##
# Reset: level rule, side-band targets, touch state. Runs after reset_pass_target.
##
def reset_strike(env, env_ids: torch.Tensor) -> None:
  st, ps, n, dev = _st(env), D._st(env), len(env_ids), env.device
  r = env.scene["robot"].data
  side = torch.rand(n, device=dev) < SIDE_FRAC
  if side.any():
    ids = env_ids[side]
    k = len(ids)
    sign = 2.0 * (torch.rand(k, device=dev) < 0.5).float() - 1.0
    world = r.heading_w[ids] + sign * (SIDE_BAND[0] + (SIDE_BAND[1] - SIDE_BAND[0]) * torch.rand(k, device=dev))
    dist = D.DIST_M[0] + (D.DIST_M[1] - D.DIST_M[0]) * torch.rand(k, device=dev)
    xy = r.root_link_pos_w[ids, :2] + dist[:, None] * torch.stack([world.cos(), world.sin()], 1)
    v = xy - ps.ball0[ids, :2]
    norm = v.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    ps.target[ids, :2] = ps.ball0[ids, :2] + v / norm * norm.clamp_min(D.MIN_BALL_TO_TARGET)
    K._state(env).g0[ids] = (ps.ball0[ids] - ps.target[ids]).norm(dim=-1)
  st.prev_v[env_ids] = 0.0
  st.touched[env_ids] = st.first_now[env_ids] = st.again_now[env_ids] = False
  st.first_err[env_ids] = 0.0


def reset_level(env, env_ids: torch.Tensor) -> None:
  """Before reset_pass_target clears the hit flags: the level rule with the lower thresholds."""
  st, ps = _st(env), D._st(env)
  ps.freeze_level = True  # KickDir's own rule stays off
  if ps.fresh:
    return
  st.attempts += len(env_ids)
  st.hits += int(ps.hit[env_ids].sum())
  if st.attempts >= D.LEVEL_BATCH:
    rate = st.hits / st.attempts
    if not getattr(env, "_strike_level_fixed", False):
      ps.level = min(ps.level + D.LEVEL_STEP, 1.0) if rate > LEVEL_UP else max(ps.level - D.LEVEL_STEP, 0.0) if rate < LEVEL_DOWN else ps.level
    st.attempts, st.hits = 0, 0


##
# Metrics
##
def touched_metric(env) -> torch.Tensor:
  return _st(env).touched.float()


def first_err_metric(env) -> torch.Tensor:
  return _st(env).first_err


def kick_strike_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  cfg = D.kick_dir_env_cfg(play=play)
  # settings of kick_dir_v1, fixed here so that launches cannot drift from it
  base = cfg.actions["base"]
  base.residual, base.residual_gate, base.cmd_alpha = True, K.residual_near_ball, 0.3
  attach_ball_gaze(cfg)

  extra = {
    "feet_pos": ObservationTermCfg(func=obs_feet_pos),
    "feet_vel": ObservationTermCfg(func=obs_feet_vel),
    "strike_point": ObservationTermCfg(func=obs_strike_point),
  }
  for group in ("actor", "critic"):
    cfg.observations[group].terms = {**cfg.observations[group].terms, **extra}

  cfg.events = {"reset_level": EventTermCfg(func=reset_level, mode="reset"), **cfg.events}
  cfg.events["reset_strike"] = EventTermCfg(func=reset_strike, mode="reset")  # after reset_pass_target

  cfg.terminations.pop("pass_done")
  cfg.terminations["strike_done"] = TerminationTermCfg(func=strike_done)

  cfg.rewards["pass_arrival"] = RewardTermCfg(func=D.pass_arrival, weight=60.0, params={"std": 0.15})
  cfg.rewards["ball_heading"] = RewardTermCfg(func=D.ball_heading, weight=3.0, params={"std": 0.1})
  cfg.rewards["off_line"] = RewardTermCfg(func=off_line_tight, weight=-2.0)
  cfg.rewards["strike_zone"] = RewardTermCfg(func=strike_zone, weight=1.0)
  cfg.rewards["momentum_preview"] = RewardTermCfg(func=momentum_preview, weight=3.0)
  cfg.rewards["first_impulse"] = RewardTermCfg(func=first_impulse, weight=40.0)
  cfg.rewards["second_touch"] = RewardTermCfg(func=second_touch, weight=-5.0)

  cfg.metrics = {**(cfg.metrics or {}), "strike_touched": MetricsTermCfg(func=touched_metric),
                 "strike_first_err": MetricsTermCfg(func=first_err_metric)}
  return cfg
