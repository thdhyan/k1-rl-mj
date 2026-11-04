"""Ports of the remaining Isaac Lab K1 tasks to mjlab: stand (P1 Basic), squat (H* command), steps (residual).

* ``Mjlab-Stand-Booster-K1``  (Isaac-Basic-*): the velocity env with every command = 0 (standing envs only),
  pushes kept. Same 12-leg actor, so a stand policy is a drop-in walk-base replacement.
* ``Mjlab-Squat-Booster-K1``  (Isaac-Velocity-Squat-*): velocity env + a commanded trunk height H* (relative to
  the feet), as a 4th command channel in the actor obs. Squat depth ramps 0.45-0.50 -> 0.35-0.43 m.
* ``Mjlab-Steps-Booster-K1``  (Isaac-Steps-Residual-*): stairs-only rough terrain, frozen walk base under the
  twist command, task policy = 12 leg residuals seeing the height scan.

Isaac-Velocity-PartialCtrl needs no port: the mjlab velocity task already is legs-only with random arms.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, replace

import torch

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.command_manager import CommandTerm, CommandTermCfg
from mjlab.managers.observation_manager import ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.tasks.velocity import mdp
from mjlab.terrains.config import (flat, hf_pyramid_slope, hf_pyramid_slope_inv, pyramid_stairs, pyramid_stairs_inv,
                                   random_rough, wave_terrain)

from .env_cfg import SITES, k1_velocity_env_cfg
from .frozen_base import FrozenBaseActionCfg
from .tasks_kick import BASE_CKPT
from .upper_body import RandomUpperBodyActionCfg

# ---------------------------------------------------------------- stand


def stand_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  cfg = k1_velocity_env_cfg(rough=True, play=play)
  twist = cfg.commands["twist"]
  twist.rel_standing_envs = 1.0
  twist.rel_heading_envs = 0.0
  twist.heading_command = False
  twist.ranges = replace(twist.ranges, lin_vel_x=(0.0, 0.0), lin_vel_y=(0.0, 0.0), ang_vel_z=(0.0, 0.0), heading=None)
  cfg.curriculum.pop("command_vel", None)
  cfg.rewards["air_time"].weight = 0.0
  return cfg


# ---------------------------------------------------------------- squat

#: trunk height above the mean foot-site height while the walk base stands (0.516 measured in mjlab; Isaac's
#: world-frame 0.57). Squat ranges = Isaac's shifted by the same -0.05.
H_STAND = float(os.environ.get("K1_H_STAND", "0.52"))
SQUAT_START, SQUAT_END, SQUAT_IT = (0.45, 0.50), (0.35, 0.43), (500, 3500)
STEPS_PER_IT = 24


def trunk_height(env) -> torch.Tensor:
  """Trunk z minus mean foot-site z: terrain-free trunk height while the feet are down."""
  robot = env.scene["robot"]
  bid = robot.find_bodies(("Trunk",))[0][0]
  sids = robot.find_sites(SITES)[0]
  return robot.data.body_link_pos_w[:, bid, 2] - robot.data.site_pos_w[:, sids, 2].mean(dim=1)


@dataclass(kw_only=True)
class TrunkHeightCommandCfg(CommandTermCfg):
  stand_height: float = H_STAND
  rel_squat_envs: float = 0.5

  def build(self, env) -> "TrunkHeightCommand":
    return TrunkHeightCommand(self, env)


class TrunkHeightCommand(CommandTerm):
  """H* per env: stand height, or (rel_squat_envs) U(lo, hi) with the range deepening over SQUAT_IT."""
  cfg: TrunkHeightCommandCfg

  def __init__(self, cfg: TrunkHeightCommandCfg, env):
    super().__init__(cfg, env)
    self.h = torch.full((self.num_envs, 1), cfg.stand_height, device=self.device)
    self.metrics["error_height"] = torch.zeros(self.num_envs, device=self.device)

  @property
  def command(self) -> torch.Tensor:
    return self.h

  def _range(self) -> tuple[float, float]:
    it = self._env.common_step_counter / STEPS_PER_IT
    t = min(max((it - SQUAT_IT[0]) / (SQUAT_IT[1] - SQUAT_IT[0]), 0.0), 1.0)
    return tuple(a + t * (b - a) for a, b in zip(SQUAT_START, SQUAT_END))

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    lo, hi = self._range()
    n = len(env_ids)
    squat = torch.rand(n, device=self.device) < self.cfg.rel_squat_envs
    h = lo + (hi - lo) * torch.rand(n, device=self.device)
    self.h[env_ids, 0] = torch.where(squat, h, self.cfg.stand_height)

  def _update_command(self, env_ids: torch.Tensor | None) -> None:
    pass

  def _update_metrics(self) -> None:
    steps = self.cfg.resampling_time_range[1] / self._env.step_dt
    self.metrics["error_height"] += (trunk_height(self._env) - self.h[:, 0]).abs() / steps


def track_trunk_height(env, command_name: str = "height", std: float = 0.05) -> torch.Tensor:
  h = env.command_manager.get_command(command_name)[:, 0]
  return torch.exp(-(trunk_height(env) - h).abs() / std)


def squat_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  cfg = k1_velocity_env_cfg(rough=True, play=play)
  twist = cfg.commands["twist"]
  # VR teleop BASE_LIMITS, as the Isaac squat task: fixed envelope, no widening curriculum
  twist.ranges = replace(twist.ranges, lin_vel_x=(-0.5, 0.5), lin_vel_y=(-0.3, 0.3), ang_vel_z=(-0.8, 0.8))
  cfg.curriculum.pop("command_vel", None)
  return _add_height_command(cfg, play)


def height_walk_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Deployable walk base with the squat height channel: flat + blind (225 + 5 obs), flat_swing10's command
  ranges and curriculum. Run with K1_STAND_W / K1_STAND_FRAC for a quiet stand at zero twist."""
  return _add_height_command(k1_velocity_env_cfg(rough=False, play=play), play)


def rough_height_walk_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Rough-terrain walk base, deployable like walk_still_v1: same blind 225 + 5 actor obs (height_scan only in
  the critic as privileged info), terrain-level curriculum over stairs up/down, slopes, rough and waves, and
  stronger random pushes (K1_PUSH_V m/s). Run with the walk_still env vars (K1_STAND_W ... K1_SYM_W).
  K1_TERRAIN_FOCUS=1: mostly stairs + slopes (up-heavy); K1_STAIR_MAX m / K1_SLOPE_MAX (rise/run) = hardest row."""
  cfg = _add_height_command(k1_velocity_env_cfg(rough=True, play=play), play)
  stair_max, slope_max = float(os.environ.get("K1_STAIR_MAX", "0.12")), float(os.environ.get("K1_SLOPE_MAX", "0.45"))
  w = ((0.05, 0.3, 0.15, 0.25, 0.15, 0.1, 0.0) if os.environ.get("K1_TERRAIN_FOCUS") == "1"
       else (0.15, 0.2, 0.15, 0.125, 0.125, 0.15, 0.1))
  stair = dict(step_height_range=(0.02, stair_max), step_width=0.30, platform_width=2.0)
  cfg.scene.terrain.terrain_generator = replace(cfg.scene.terrain.terrain_generator, num_rows=10, num_cols=20,
                                                curriculum=not play, sub_terrains={k: v for k, v in {
    "flat": flat(proportion=w[0]),
    "stairs_up": pyramid_stairs_inv(proportion=w[1], **stair),
    "stairs_down": pyramid_stairs(proportion=w[2], **stair),
    "slope_up": hf_pyramid_slope_inv(proportion=w[3], slope_range=(0.0, slope_max)),
    "slope_down": hf_pyramid_slope(proportion=w[4], slope_range=(0.0, slope_max)),
    "rough": random_rough(proportion=w[5], noise_range=(0.01, 0.06), noise_step=0.01),
    "waves": wave_terrain(proportion=w[6], amplitude_range=(0.0, 0.08)),
  }.items() if v.proportion > 0})
  cfg.observations["actor"].terms.pop("height_scan", None)  # blind actor: no scanner on the real K1
  v = float(os.environ.get("K1_PUSH_V", "0.7"))
  if "push_robot" in cfg.events:
    cfg.events["push_robot"].params["velocity_range"] = {
      "x": (-v, v), "y": (-v, v), "z": (-0.2, 0.2), "roll": (-0.4, 0.4), "pitch": (-0.4, 0.4), "yaw": (-0.6, 0.6)}
  if play:
    cfg.scene.terrain.terrain_generator = replace(cfg.scene.terrain.terrain_generator, num_rows=5, num_cols=7)
  return cfg


def _add_height_command(cfg: ManagerBasedRlEnvCfg, play: bool) -> ManagerBasedRlEnvCfg:
  cfg.commands["height"] = TrunkHeightCommandCfg(resampling_time_range=(4.0, 8.0))
  for g in ("actor", "critic"):
    cfg.observations[g].terms["height_cmd"] = ObservationTermCfg(
      func=mdp.generated_commands, params={"command_name": "height"})
  cfg.rewards["trunk_height"] = RewardTermCfg(func=track_trunk_height, weight=2.0, params={"std": 0.05})
  # a commanded squat leaves HOME: standing posture gets the walking tolerance
  cfg.rewards["pose"].params["std_standing"] = dict(cfg.rewards["pose"].params["std_walking"])
  if play:
    cfg.commands["height"].rel_squat_envs = 1.0
  return cfg


# ---------------------------------------------------------------- steps

STEPS_BASE_CKPT = os.environ.get("K1_STEPS_BASE_CKPT", BASE_CKPT)
STEPS_RESIDUAL = 0.3


def residual_l2(env, action_name: str = "base") -> torch.Tensor:
  return env.action_manager.get_term(action_name).raw_action.square().sum(dim=1)


def feet_stumble(env, sensor_name: str = "feet_ground_contact") -> torch.Tensor:
  """1 per foot whose horizontal contact force exceeds 2x its vertical one (toe hits a riser)."""
  f = env.scene[sensor_name].data.force
  return (f[..., :2].norm(dim=-1) > 2.0 * f[..., 2].abs()).float().sum(dim=1)


def steps_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  cfg = k1_velocity_env_cfg(rough=True, play=play)
  stair = dict(step_height_range=(0.04, 0.14), step_width=0.30, platform_width=2.0)
  gen = cfg.scene.terrain.terrain_generator
  cfg.scene.terrain.terrain_generator = replace(gen, num_rows=10, sub_terrains={
    "stairs_up": pyramid_stairs_inv(proportion=0.5, **stair),
    "stairs_down": pyramid_stairs(proportion=0.4, **stair),
    "flat": flat(proportion=0.1),
  })
  twist = cfg.commands["twist"]
  twist.ranges = replace(twist.ranges, lin_vel_x=(-0.3, 0.5), lin_vel_y=(-0.2, 0.2), ang_vel_z=(-0.5, 0.5))
  cfg.curriculum.pop("command_vel", None)
  cfg.actions = {
    "base": FrozenBaseActionCfg(entity_name="robot", checkpoint=STEPS_BASE_CKPT, residual=True,
                                residual_scale=STEPS_RESIDUAL, command_name="twist"),
    "upper_body": RandomUpperBodyActionCfg(entity_name="robot"),
  }
  # the base keeps its own 5-step history; the residual sees one frame + the height scan
  cfg.observations["actor"].history_length = 1
  # the frozen base owns the gait: task terms + stairs-specific ones only
  keep = ("track_linear_velocity", "track_angular_velocity", "upright", "dof_pos_limits", "action_rate_l2",
          "foot_clearance")
  cfg.rewards = {k: v for k, v in cfg.rewards.items() if k in keep}
  cfg.rewards["action_rate_l2"].weight = -0.01
  cfg.rewards["feet_stumble"] = RewardTermCfg(func=feet_stumble, weight=-1.0)
  cfg.rewards["residual_l2"] = RewardTermCfg(func=residual_l2, weight=-0.05)
  cfg.rewards["termination"] = RewardTermCfg(func=mdp.is_terminated, weight=-200.0)
  if play:
    cfg.scene.terrain.terrain_generator = replace(cfg.scene.terrain.terrain_generator, num_rows=5, num_cols=4,
                                                  curriculum=False)
  return cfg
