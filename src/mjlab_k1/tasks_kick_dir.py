"""Directed kick (Mjlab-KickDir-Booster-K1): pass the ball to a target given relative to the robot.

Same observations and actions as Mjlab-Kick, so a kick checkpoint loads and training warm-starts from it. The goal_pos
observation now carries a target sampled at reset in the robot's frame (bearing and distance) and then fixed in the world,
as a teammate standing still would be. Base gate and command filter come from the same K1_KICK_* variables.

An episode ends when the ball crosses the line through the target, square to ball-to-target, or strays too far sideways.
A level in [0, 1], moved by measured hit rate, widens the bearing range and narrows the hit tolerance.
Requires reset_ball to run first; it overrides the fixed goal through env._kick_target (see tasks_kick._goal_w).
"""
from __future__ import annotations

import math

import torch

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.metrics_manager import MetricsTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.termination_manager import TerminationTermCfg

from . import tasks_kick as K

DIST_M = (2.0, 5.0)  # target distance from the robot
MIN_BALL_TO_TARGET = 1.5
BEARING_EASY, BEARING_HARD = math.pi / 2, math.pi  # half range of the target bearing, robot frame
TOL_EASY, TOL_HARD = 0.6, 0.2  # lateral hit tolerance at the target line, m
LEVEL_BATCH, LEVEL_STEP, LEVEL_UP, LEVEL_DOWN = 2000, 0.05, 0.5, 0.2
LAUNCH_TRAVEL_M = 0.5  # the launch heading is read once the ball has travelled this far
MAX_LATERAL_M = 2.0
MOVING_MPS = 0.3
MIN_BALL_GAP_M = 0.35  # ball centre to trunk at reset


class PassState:
  def __init__(self, env):
    n, dev = env.num_envs, env.device
    self.target = torch.zeros(n, 3, device=dev)  # world
    self.ball0 = torch.zeros(n, 3, device=dev)
    self.launched = torch.zeros(n, dtype=torch.bool, device=dev)
    self.launch_now = torch.zeros(n, dtype=torch.bool, device=dev)
    self.arrived = torch.zeros(n, dtype=torch.bool, device=dev)
    self.hit = torch.zeros(n, dtype=torch.bool, device=dev)
    self.level, self.freeze_level, self.attempts, self.hits, self.fresh = 0.0, False, 0, 0, True

  def tolerance(self) -> float:
    return TOL_EASY + self.level * (TOL_HARD - TOL_EASY)


def _st(env) -> PassState:
  st = getattr(env, "_pass_state", None)
  if st is None:
    st = env._pass_state = PassState(env)
  return st


def _line(env):
  """Ball position along and across the line from its start to the target, plus the distance to cover and axis."""
  st = _st(env)
  ax = (st.target - st.ball0)[:, :2]
  dist = ax.norm(dim=-1).clamp_min(1e-6)
  ax = ax / dist[:, None]
  d = (K._ball(env).root_link_pos_w - st.ball0)[:, :2]
  along = (d * ax).sum(-1)
  lateral = d[:, 0] * ax[:, 1] - d[:, 1] * ax[:, 0]
  return along, lateral, dist, ax


def reset_pass_target(env, env_ids: torch.Tensor) -> None:
  st, dev, n = _st(env), env.device, len(env_ids)
  if not st.fresh:
    st.attempts += n
    st.hits += int(st.hit[env_ids].sum())
    if st.attempts >= LEVEL_BATCH and not st.freeze_level:
      rate = st.hits / st.attempts
      st.level = min(st.level + LEVEL_STEP, 1.0) if rate > LEVEL_UP else max(st.level - LEVEL_STEP, 0.0) if rate < LEVEL_DOWN else st.level
    if st.attempts >= LEVEL_BATCH:
      st.attempts, st.hits = 0, 0
  st.fresh = False
  env.sim.forward()
  r, ball = env.scene["robot"].data, K._ball(env)
  half = BEARING_EASY + st.level * (BEARING_HARD - BEARING_EASY)
  world = r.heading_w[env_ids] + half * (2.0 * torch.rand(n, device=dev) - 1.0)
  dist = DIST_M[0] + (DIST_M[1] - DIST_M[0]) * torch.rand(n, device=dev)
  xy = r.root_link_pos_w[env_ids, :2] + dist[:, None] * torch.stack([world.cos(), world.sin()], dim=1)
  # Plain reset_ball places the ball around the env origin, while the robot spawns up to 0.5 m from it. Clear overlaps.
  st13 = K._ball(env).default_root_state[env_ids].clone()
  rp = r.root_link_pos_w[env_ids, :2]
  away = ball.root_link_pos_w[env_ids, :2] - rp
  gap = away.norm(dim=-1, keepdim=True).clamp_min(1e-6)
  st13[:, :3] = ball.root_link_pos_w[env_ids]
  st13[:, :2] = rp + away / gap * gap.clamp_min(MIN_BALL_GAP_M)
  st13[:, 7:] = 0.0
  env.scene["ball"].write_root_state_to_sim(st13, env_ids=env_ids)
  b0 = st13[:, :3]
  v = xy - b0[:, :2]
  norm = v.norm(dim=-1, keepdim=True).clamp_min(1e-6)
  xy = b0[:, :2] + v / norm * norm.clamp_min(MIN_BALL_TO_TARGET)
  st.target[env_ids, :2] = xy
  st.target[env_ids, 2] = env.scene.env_origins[env_ids, 2] + K.GOAL_POS[2]
  st.ball0[env_ids] = b0
  for flag in (st.launched, st.launch_now, st.arrived, st.hit):
    flag[env_ids] = False
  env._kick_target = st.target
  K._state(env).g0[env_ids] = (b0 - st.target[env_ids]).norm(dim=-1)


def pass_done(env) -> torch.Tensor:
  """Ball reached the target line (hit or miss) or strayed too far sideways. Also updates the launch and hit flags."""
  st = _st(env)
  along, lateral, dist, ax = _line(env)
  st.arrived = along >= dist
  st.hit = st.hit | (st.arrived & (lateral.abs() < st.tolerance()))
  st.launch_now = (along.abs() + lateral.abs() >= LAUNCH_TRAVEL_M) & ~st.launched
  st.launched = st.launched | st.launch_now
  return st.arrived | (lateral.abs() > MAX_LATERAL_M)


def pass_arrival(env, std: float = 0.25) -> torch.Tensor:
  """One-shot when the ball reaches the target line: how close it passes the target."""
  _, lateral, _, _ = _line(env)
  return torch.exp(-(lateral / std) ** 2) * _st(env).arrived.float()


def launch_heading(env, std: float = 0.1) -> torch.Tensor:
  """One-shot when the ball has travelled 0.5 m: heading error of that travel against the direction to the target."""
  along, lateral, _, _ = _line(env)
  return torch.exp(-(torch.atan2(lateral, along) / std) ** 2) * _st(env).launch_now.float()


def _moving(env) -> torch.Tensor:
  return K._ball(env).root_link_lin_vel_w[:, :2].norm(dim=-1) > MOVING_MPS


def ball_heading(env, std: float = 0.2) -> torch.Tensor:
  """While the ball rolls: its velocity lines up with the direction to the target."""
  st, b = _st(env), K._ball(env)
  v, to = b.root_link_lin_vel_w[:, :2], (st.target - b.root_link_pos_w)[:, :2]
  err = torch.atan2(v[:, 0] * to[:, 1] - v[:, 1] * to[:, 0], (v * to).sum(-1))
  return torch.exp(-(err / std) ** 2) * _moving(env).float()


def off_line(env, scale: float = 0.5) -> torch.Tensor:
  """Cost while the ball rolls: distance from the start-to-target line."""
  _, lateral, _, _ = _line(env)
  return (lateral.abs() / scale).clamp_max(3.0) * _moving(env).float()


def level_metric(env) -> torch.Tensor:
  return torch.full((env.num_envs,), _st(env).level, device=env.device)


def hit_metric(env) -> torch.Tensor:
  return _st(env).hit.float()


def kick_dir_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  cfg = K.kick_env_cfg(play=play)
  cfg.events["reset_pass_target"] = EventTermCfg(func=reset_pass_target, mode="reset")  # after reset_ball
  cfg.terminations.pop("goal_scored")
  cfg.terminations["pass_done"] = TerminationTermCfg(func=pass_done)
  cfg.rewards.pop("goal_scored")
  cfg.rewards["pass_arrival"] = RewardTermCfg(func=pass_arrival, weight=50.0)
  cfg.rewards["launch_heading"] = RewardTermCfg(func=launch_heading, weight=30.0)
  cfg.rewards["ball_heading"] = RewardTermCfg(func=ball_heading, weight=3.0)
  cfg.rewards["off_line"] = RewardTermCfg(func=off_line, weight=-1.0)
  cfg.metrics = {**(cfg.metrics or {}), "pass_level": MetricsTermCfg(func=level_metric), "pass_hit": MetricsTermCfg(func=hit_metric)}
  return cfg
