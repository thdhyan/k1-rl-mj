"""Jump in place on the K1 (Mjlab-Jump-Booster-K1). Method follows Olsen 2025; see docs/k1_jump_strategies.md.

The frozen walk base stands the robot at zero command. A jump command opens a 1.5 s window, and inside it the policy
replaces the base: its 12 outputs are absolute leg targets for crouch, push and landing. The base returns after the
window and has to recover from whatever pose the policy left.

Episodes start in one of five states: standing (whole jump), squat with the window open, in flight, just before
touchdown, landed. Deep squats come first. A level in [0, 1], raised and lowered from measured success, widens the squat
range and raises the height target. The target is trunk rise above standing height, so tucking the legs earns nothing.
Before takeoff and in flight the reward uses the ballistic estimate z + vz^2 / 2g, with a cost above the target.

JumpAction caps leg torque on a linear torque-speed line from the robot URDF's effort and velocity limits. The sim has
no speed limit of its own. Whether the real motors follow this line is an assumption, not a measurement.

The actor sees only what the robot measures, with 5 frames of history. Contact, height and linear velocity go to the
critic. K1_JUMP_RESIDUAL scales the action: |a| = 1 is 25% of joint effort over kp around the home pose.
"""
from __future__ import annotations

import math
import os
import re
from dataclasses import dataclass, replace

import torch

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.command_manager import CommandTerm, CommandTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.tasks.velocity import mdp

from .env_cfg import SITES, k1_velocity_env_cfg
from .frozen_base import FrozenBaseAction, FrozenBaseActionCfg
from mjlab_k1.k1_constants import K1_ACTUATORS, LEG_REGEX
from .tasks_kick import BASE_CKPT, failure_terminated
from .upper_body import RandomUpperBodyActionCfg

JUMP_BASE_CKPT = os.environ.get("K1_JUMP_BASE_CKPT", BASE_CKPT)
JUMP_RESIDUAL = float(os.environ.get("K1_JUMP_RESIDUAL", "3.0"))
WINDOW_S, WAIT_S, RAMP_IN_S, RAMP_OUT_S, MIN_AIR_STEPS = 1.5, (0.5, 1.5), 0.04, 0.2, 3
STAND_Z, SITE_Z = 0.523, 0.0  # measured on the frozen base: trunk height standing, foot site height on the ground
TARGET_EASY, TARGET_HARD = (0.0, 0.03), (0.15, 0.30)  # trunk rise above standing, m
KNEE_DEEP, KNEE_EASY, KNEE_HARD = 1.8, 1.4, 0.8  # squat knee range at level 0 is [EASY, DEEP], at level 1 [HARD, DEEP]
LEVEL_BATCH, LEVEL_STEP, LEVEL_UP, LEVEL_DOWN = 2000, 0.05, 0.6, 0.3
START_PROBS = (0.2, 0.4, 0.15, 0.1, 0.15)  # stand, squat, flight, before touchdown, landed
LAND_FORCE_N = 600.0  # net vertical force on both feet that costs nothing; about 3 x body weight
VEL_LIMITS = {".*_Hip_Pitch": 14.66, ".*_Knee_Pitch": 12.57, ".*_Ankle_Pitch": 17.59}  # rad/s, real_robot URDF
AIR_MIN_RISE = -0.03  # air time counts only with the trunk this close to standing height, so a leg tuck earns nothing
G = 9.81


def _feet_down(env) -> torch.Tensor:
  return env.scene["feet_ground_contact"].data.found[:, :2] > 0


@dataclass(kw_only=True)
class JumpCommandCfg(CommandTermCfg):
  start_probs: tuple[float, ...] = START_PROBS

  def build(self, env) -> "JumpCommand":
    return JumpCommand(self, env)


class JumpCommand(CommandTerm):
  """Per cycle: wait, 1.5 s window, rest until the next resample. Command = (active, window phase, target)."""
  cfg: JumpCommandCfg

  def __init__(self, cfg: JumpCommandCfg, env):
    super().__init__(cfg, env)
    n, dev = self.num_envs, self.device
    z = lambda: torch.zeros(n, device=dev)
    b = lambda: torch.zeros(n, dtype=torch.bool, device=dev)
    self.t, self.wait, self.target, self.apex, self.air_steps, self.yaw0, self.age, self.high_air = z(), z(), z(), z(), z(), z(), z(), z()
    self.xy0 = torch.zeros(n, 2, device=dev)
    self.started, self.landed, self.just_landed, self.closed, self.genuine = b(), b(), b(), b(), b()
    self.start_phase = torch.zeros(n, dtype=torch.long, device=dev)
    self._cmd = torch.zeros(n, 3, device=dev)
    self._sites = env.scene["robot"].find_sites(SITES)[0]
    self.level, self.freeze_level = 0.0, False
    self._attempts, self._hits = 0, 0
    for m in ("jumps", "rise_m", "air_s", "hits", "level"):
      self.metrics[m] = z()

  @property
  def command(self) -> torch.Tensor:
    return self._cmd

  @property
  def foot_z(self) -> torch.Tensor:
    return self._env.scene["robot"].data.site_pos_w[:, self._sites, 2].min(dim=1).values

  @property
  def rise(self) -> torch.Tensor:
    return self._env.scene["robot"].data.root_link_pos_w[:, 2] - STAND_Z

  @property
  def ballistic(self) -> torch.Tensor:
    vz = self._env.scene["robot"].data.root_link_lin_vel_w[:, 2].clamp_min(0.0)
    return self.rise + vz.square() / (2.0 * G)

  @property
  def in_window(self) -> torch.Tensor:
    return (self.t >= self.wait) & (self.t < self.wait + WINDOW_S)

  @property
  def airborne(self) -> torch.Tensor:
    return ~_feet_down(self._env).any(dim=1)

  @property
  def gate(self) -> torch.Tensor:
    return (((self.t - self.wait) / RAMP_IN_S).clamp(0.0, 1.0) * ((self.wait + WINDOW_S - self.t) / RAMP_OUT_S).clamp(0.0, 1.0))

  def knee_low(self) -> float:
    return KNEE_EASY + self.level * (KNEE_HARD - KNEE_EASY)

  def _tally(self, attempts: int, hits: int) -> None:
    self._attempts += attempts
    self._hits += hits
    if self._attempts < LEVEL_BATCH or self.freeze_level:
      return
    rate = self._hits / self._attempts
    if rate > LEVEL_UP:
      self.level = min(self.level + LEVEL_STEP, 1.0)
    elif rate < LEVEL_DOWN:
      self.level = max(self.level - LEVEL_STEP, 0.0)
    self._attempts, self._hits = 0, 0

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    n, dev = len(env_ids), self.device
    self._tally(int((self.genuine[env_ids] & self.started[env_ids] & ~self.closed[env_ids]).sum()), 0)  # fell mid-window
    lo, hi = (a + self.level * (b - a) for a, b in zip(TARGET_EASY, TARGET_HARD))
    phase = self.start_phase[env_ids]
    t0 = torch.tensor([0.0, 0.0, 0.2, 0.4, 0.4], device=dev)[phase]
    t1 = torch.tensor([0.0, 0.0, 0.5, 0.8, 1.0], device=dev)[phase]
    self.t[env_ids] = t0 + (t1 - t0) * torch.rand(n, device=dev)
    wait = WAIT_S[0] + (WAIT_S[1] - WAIT_S[0]) * torch.rand(n, device=dev)
    self.wait[env_ids] = torch.where(phase == 0, wait, 0.0)
    self.target[env_ids] = lo + (hi - lo) * torch.rand(n, device=dev)
    self.apex[env_ids] = 0.0
    self.high_air[env_ids] = 0.0
    self.age[env_ids] = 0.0
    self.air_steps[env_ids] = torch.where((phase == 2) | (phase == 3), float(MIN_AIR_STEPS), 0.0)
    self.started[env_ids] = False
    self.closed[env_ids] = False
    self.just_landed[env_ids] = False
    self.landed[env_ids] = phase == 4
    self.genuine[env_ids] = phase <= 1  # only a jump pushed off the ground earns the landing payout or counts as success
    self._cmd[env_ids] = 0.0
    self.start_phase[env_ids] = 0

  def _update_command(self, env_ids: torch.Tensor | None = None) -> None:
    self.t += self._env.step_dt
    self.age += 1.0
    win = self.in_window
    start = win & ~self.started
    if start.any():
      d = self._env.scene["robot"].data
      self.xy0[start] = d.root_link_pos_w[start, :2]
      self.yaw0[start] = d.heading_w[start]
      self.started |= start
    valid = self.age > 1.0  # contact data is stale on the step right after a reset
    down = _feet_down(self._env).any(dim=1) & valid
    air = win & ~_feet_down(self._env).any(dim=1) & valid & ~self.landed
    self.air_steps = torch.where(air, self.air_steps + 1, self.air_steps)
    self.apex = torch.where(air, torch.maximum(self.apex, self.rise), self.apex)
    self.high_air = self.high_air + (air & (self.rise > AIR_MIN_RISE)).float()
    self.just_landed = (self.air_steps >= MIN_AIR_STEPS) & down & ~self.landed
    self.landed |= self.just_landed
    close = self.started & ~win & ~self.closed
    if close.any():
      self.closed |= close
      real = close & self.genuine
      self._tally(int(real.sum()), int((real & self.landed & (self.apex >= self.target)).sum()))
    self._cmd[:, 0] = win.float()
    self._cmd[:, 1] = torch.where(win, (self.t - self.wait) / WINDOW_S, 0.0)
    self._cmd[:, 2] = torch.where(win, self.target, 0.0)

  def _update_metrics(self) -> None:
    real = self.just_landed & self.genuine
    self.metrics["jumps"] += real.float()
    self.metrics["rise_m"] += self.apex * real.float()
    self.metrics["air_s"] += self.high_air * self._env.step_dt * real.float()
    self.metrics["hits"] += (real & (self.apex >= self.target)).float()
    self.metrics["level"][:] = self.level


def _jump(env) -> JumpCommand:
  return env.command_manager.get_term("jump")


def jump_gate(env) -> torch.Tensor:
  return _jump(env).gate


@dataclass(kw_only=True)
class JumpActionCfg(FrozenBaseActionCfg):
  def build(self, env) -> "JumpAction":
    return JumpAction(self, env)


class JumpAction(FrozenBaseAction):
  """Frozen base outside the window, policy inside it, leg torque capped on the torque-speed line at every physics step."""

  def __init__(self, cfg: JumpActionCfg, env):
    super().__init__(cfg, env)
    names = self._entity.find_joints(LEG_REGEX)[1]
    def lookup(n, attr):
      return next(getattr(a, attr) for a in K1_ACTUATORS[:6] if any(re.fullmatch(e, n) for e in a.target_names_expr))
    t = lambda v: torch.tensor(v, device=self.device, dtype=torch.float)
    self._kp = t([lookup(n, "stiffness") for n in names])
    self._kd = t([lookup(n, "damping") for n in names])
    self._effort = t([lookup(n, "effort_limit") for n in names])
    self._vmax = t([next((v for k, v in VEL_LIMITS.items() if re.fullmatch(k, n)), math.inf) for n in names])
    self.sym_sign = t([-1.0 if re.search("Roll|Yaw", n) else 1.0 for n in names[:6]])

  def apply_actions(self) -> None:
    d = self._entity.data
    q, qd = d.joint_pos[:, self._ids], d.joint_vel[:, self._ids]
    hi = self._effort * (1.0 - (qd / self._vmax).clamp(0.0, 1.0))
    lo = -self._effort * (1.0 - (-qd / self._vmax).clamp(0.0, 1.0))
    tau = (self._kp * (self._target - q) - self._kd * qd).clamp(lo, hi)
    self._entity.set_joint_position_target(q + (tau + self._kd * qd) / self._kp, joint_ids=self._ids)


def reset_jump_state(env, env_ids: torch.Tensor) -> None:
  """Picks each new episode's start state; JumpCommand reads the phase when the command manager resets right after."""
  j, r, dev = _jump(env), env.scene["robot"], env.device
  n = len(env_ids)
  phase = torch.multinomial(torch.tensor(j.cfg.start_probs, device=dev), n, replacement=True)
  j.start_phase[env_ids] = phase
  m = phase > 0
  if not m.any():
    return
  ids, phase, k = env_ids[m], phase[m], int(m.sum())
  u = lambda a, b: a + (b - a) * torch.rand(k, device=dev)
  in_air = (phase == 2) | (phase == 3)
  knee = torch.where(phase == 1, u(j.knee_low(), KNEE_DEEP), torch.where(in_air, u(0.2, 1.2), u(0.8, 1.6)))
  lean = u(-0.25, 0.0)
  q = r.data.default_joint_pos[ids].clone()
  j_hip, j_knee, j_ank = (r.find_joints((f".*_{x}",))[0] for x in ("Hip_Pitch", "Knee_Pitch", "Ankle_Pitch"))
  hip = -knee / 2 + lean
  q[:, j_hip] = hip[:, None]
  q[:, j_knee] = knee[:, None]
  q[:, j_ank] = (-(hip + knee)).clamp_min(-0.8)[:, None]
  q[:, j_hip + j_knee + j_ank] += 0.05 * (2.0 * torch.rand(k, 6, device=dev) - 1.0)
  q = q.clamp(*r.data.soft_joint_pos_limits[ids].unbind(-1))
  yaw = u(-math.pi, math.pi)
  pitch = torch.where(in_air, u(-0.1, 0.1), 0.0)
  root = torch.zeros(k, 13, device=dev)
  root[:, :2] = env.scene.env_origins[ids, :2]
  root[:, 2] = env.scene.env_origins[ids, 2] + 1.0
  cy, sy, cp, sp = (yaw / 2).cos(), (yaw / 2).sin(), (pitch / 2).cos(), (pitch / 2).sin()
  root[:, 3:7] = torch.stack([cy * cp, -sy * sp, cy * sp, sy * cp], dim=1)
  r.write_root_state_to_sim(root, env_ids=ids)
  r.write_joint_state_to_sim(q, torch.zeros_like(q), env_ids=ids)
  env.sim.forward()
  env.scene.update(dt=0.0)
  clear = torch.where(phase == 2, u(0.03, 0.15), torch.where(phase == 3, u(0.005, 0.04), 0.0))
  root[:, 2] -= j.foot_z[ids] - env.scene.env_origins[ids, 2] - SITE_Z - clear
  vz = torch.where(phase == 2, u(-1.0, 1.0), torch.where(phase == 3, u(-1.5, -0.3), torch.where(phase == 4, u(-0.6, 0.0), 0.0)))
  root[:, 9] = vz
  moving = (phase >= 2).float()[:, None]
  root[:, 7:9] = 0.2 * (2.0 * torch.rand(k, 2, device=dev) - 1.0) * moving
  root[:, 10:13] = 0.5 * (2.0 * torch.rand(k, 3, device=dev) - 1.0) * moving
  r.write_root_state_to_sim(root, env_ids=ids)


def takeoff_estimate(env) -> torch.Tensor:
  """Before the first landing: ballistic height as a share of the target. Shaping for the push."""
  j = _jump(env)
  return (j.ballistic.clamp_min(0.0) / j.target.clamp_min(0.03)).clamp(0.0, 1.0) * (j.in_window & ~j.landed).float()


def flight_height(env) -> torch.Tensor:
  """Both feet up before the first landing: ballistic height as a share of the target."""
  j = _jump(env)
  return takeoff_estimate(env) * j.airborne.float()


def air_time(env, full_s: float = 0.1) -> torch.Tensor:
  """Each step off the ground with the trunk up, before the first landing. Grows the longer the flight lasts."""
  j = _jump(env)
  high = j.in_window & j.airborne & ~j.landed & (j.rise > AIR_MIN_RISE)
  return (1.0 + j.high_air * env.step_dt / full_s) * high.float()


def landing_air(env, unit_s: float = 0.1, cap: float = 8.0) -> torch.Tensor:
  """One-shot on landing a jump that left the ground: pays for the whole flight, in units of 0.1 s."""
  j = _jump(env)
  return (j.high_air * env.step_dt / unit_s).clamp_max(cap) * (j.just_landed & j.genuine).float()


def overshoot(env, margin: float = 0.10, scale: float = 0.05) -> torch.Tensor:
  """Cost once the ballistic height passes the target: the command is a height, not 'as high as possible'."""
  j = _jump(env)
  return ((j.ballistic - j.target - margin) / scale).clamp(0.0, 3.0) * (j.in_window & ~j.landed).float()


def landing_apex(env, short_std: float = 0.03, over_std: float = 0.06) -> torch.Tensor:
  """One-shot on landing a jump that left the ground: how close the trunk peak came to the target."""
  j = _jump(env)
  e = j.apex - j.target
  return torch.exp(-(e / torch.where(e < 0, short_std, over_std)) ** 2) * (j.just_landed & j.genuine).float()


def leg_symmetry(env) -> torch.Tensor:
  """Inside the window: left and right legs mirror each other (pitch equal, roll and yaw opposite)."""
  j, a = _jump(env), env.action_manager.get_term("base")
  q = env.scene["robot"].data.joint_pos[:, a._ids] - a._default
  return (q[:, :6] - a.sym_sign * q[:, 6:]).square().sum(dim=1) * j.in_window.float()


def air_and_land_pose(env, std: float = 0.7) -> torch.Tensor:
  """In flight and after landing, none before or during the push: stay near home so the base can take over."""
  j, a = _jump(env), env.action_manager.get_term("base")
  e = (env.scene["robot"].data.joint_pos[:, a._ids] - a._default).square().sum(dim=1)
  return torch.exp(-e / std**2) * (j.in_window & (j.airborne | j.landed)).float()


def stay_in_place(env, std: float = 0.15) -> torch.Tensor:
  j = _jump(env)
  d = (env.scene["robot"].data.root_link_pos_w[:, :2] - j.xy0).norm(dim=-1)
  return torch.exp(-(d / std) ** 2) * (j.in_window & j.started).float()


def hold_heading(env, std: float = 0.2) -> torch.Tensor:
  j = _jump(env)
  e = env.scene["robot"].data.heading_w - j.yaw0
  e = torch.atan2(e.sin(), e.cos())
  return torch.exp(-(e / std) ** 2) * (j.in_window & j.started).float()


def tumble_rate(env) -> torch.Tensor:
  j = _jump(env)
  return env.scene["robot"].data.root_link_ang_vel_b[:, :2].square().sum(dim=1) * j.in_window.float()


def landing_force(env, free_n: float = LAND_FORCE_N) -> torch.Tensor:
  f = env.scene["feet_ground_contact"].data.force[..., 2].abs().sum(dim=-1)
  return ((f - free_n) / free_n).clamp(0.0, 3.0)


def windowed_action_rate(env) -> torch.Tensor:
  """Action rate only while the policy is in control: outside the window its output is ignored."""
  return mdp.action_rate_l2(env) * _jump(env).gate


def trunk_too_low(env, min_z: float = 0.30) -> torch.Tensor:
  robot = env.scene["robot"]
  return robot.data.body_link_pos_w[:, robot.find_bodies(("Trunk",))[0][0], 2] < min_z


def feet_down_obs(env) -> torch.Tensor:
  return _feet_down(env).float()


def rise_obs(env) -> torch.Tensor:
  return _jump(env).rise[:, None]


def jump_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  cfg = k1_velocity_env_cfg(rough=False, play=play)
  cfg.episode_length_s = 8.0
  twist = cfg.commands["twist"]
  twist.rel_standing_envs, twist.rel_heading_envs, twist.heading_command = 1.0, 0.0, False
  twist.ranges = replace(twist.ranges, lin_vel_x=(0.0, 0.0), lin_vel_y=(0.0, 0.0), ang_vel_z=(0.0, 0.0), heading=None)
  cfg.commands["jump"] = JumpCommandCfg(resampling_time_range=(3.5, 5.0), start_probs=(1.0, 0.0, 0.0, 0.0, 0.0) if play else START_PROBS)
  cfg.curriculum = {}
  cfg.actions = {
    "base": JumpActionCfg(entity_name="robot", checkpoint=JUMP_BASE_CKPT, residual=True, residual_replaces_base=True,
                          residual_scale=JUMP_RESIDUAL, residual_gate=jump_gate, command_name="twist"),
    "upper_body": RandomUpperBodyActionCfg(entity_name="robot"),
  }
  for group in ("actor", "critic"):
    cfg.observations[group].terms["command"].params["command_name"] = "jump"
  cfg.observations["critic"].terms["feet_down"] = ObservationTermCfg(func=feet_down_obs)
  cfg.observations["critic"].terms["rise"] = ObservationTermCfg(func=rise_obs)
  cfg.events = {k: v for k, v in cfg.events.items() if k in ("reset_base", "reset_robot_joints", "foot_friction", "base_com")}
  cfg.events["reset_jump_state"] = EventTermCfg(func=reset_jump_state, mode="reset")  # after the plain resets above
  trunk = SceneEntityCfg("robot", body_names=("Trunk",))
  cfg.rewards = {
    "alive": RewardTermCfg(func=mdp.is_alive, weight=0.5),
    "termination": RewardTermCfg(func=failure_terminated, weight=-200.0, params={"term_names": ("fell", "trunk_low")}),
    "upright": RewardTermCfg(func=mdp.flat_orientation_l2, weight=-4.0, params={"asset_cfg": trunk}),
    "tumble": RewardTermCfg(func=tumble_rate, weight=-0.05),
    "takeoff": RewardTermCfg(func=takeoff_estimate, weight=3.0),
    "flight": RewardTermCfg(func=flight_height, weight=10.0),
    "air_time": RewardTermCfg(func=air_time, weight=5.0),
    "landing_air": RewardTermCfg(func=landing_air, weight=30.0),
    "overshoot": RewardTermCfg(func=overshoot, weight=-2.0),
    "landing_apex": RewardTermCfg(func=landing_apex, weight=100.0),
    "symmetry": RewardTermCfg(func=leg_symmetry, weight=-0.5),
    "pose": RewardTermCfg(func=air_and_land_pose, weight=2.0),
    "in_place": RewardTermCfg(func=stay_in_place, weight=1.0),
    "heading": RewardTermCfg(func=hold_heading, weight=0.5),
    "landing_force": RewardTermCfg(func=landing_force, weight=-1.0),
    "action_rate": RewardTermCfg(func=windowed_action_rate, weight=-0.01),
  }
  cfg.terminations = {
    "time_out": TerminationTermCfg(func=mdp.time_out, time_out=True),
    "fell": TerminationTermCfg(func=mdp.bad_orientation, params={"limit_angle": 0.8}),
    "trunk_low": TerminationTermCfg(func=trunk_too_low),
  }
  return cfg
