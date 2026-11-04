"""Carry a free box of varying mass under a velocity command: residual layer on the frozen walk base.

Same no-weld hold as carry v2 (forearm stubs squeeze the box, 1 s weld hand-off). No goal: the frozen base gets the
twist command directly, and the policy outputs 12 leg residuals + 8 arm offsets to walk at that command with the load.
Box mass is resampled per env at each reset (K1_CV_MASS_MIN..MAX kg) and given to the actor: on the robot, the operator
enters it. Used for the velocity profile vs weight (scripts/velocity_grid.py).
"""
from __future__ import annotations

import os

import torch

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.event_manager import EventTermCfg, RecomputeLevel, requires_model_fields
from mjlab.managers.observation_manager import ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.tasks.velocity import mdp

from .env_cfg import k1_velocity_env_cfg
from .frozen_base import FrozenBaseActionCfg
from k1_common.k1_constants import ARM_REGEX, LEG_REGEX
from .tasks_carry2 import carry2_env_cfg
from .tasks_kick import BASE_CKPT
from .tasks_skills import residual_l2

CV_BASE = os.environ.get("K1_CV_BASE", BASE_CKPT)
MASS_MIN = float(os.environ.get("K1_CV_MASS_MIN", "0.5"))
MASS_MAX = float(os.environ.get("K1_CV_MASS_MAX", "5.0"))
RESIDUAL = float(os.environ.get("K1_CV_RESIDUAL", "0.3"))
VX_MAX = float(os.environ.get("K1_CV_VX_MAX", "1.3"))
WZ_MAX = float(os.environ.get("K1_CV_WZ_MAX", "1.5"))


@requires_model_fields("body_mass", "body_inertia", recompute=RecomputeLevel.set_const)
def reset_box_mass(env, env_ids: torch.Tensor) -> None:
  """New mass per env, inertia scaled with it. env.box_mass_fixed (kg, set by the eval) overrides the sampling."""
  bid = env.scene["box"].indexing.body_ids[0]
  model = env.sim.model
  if not hasattr(env, "_box_mass"):
    env._box_mass = torch.full((env.num_envs,), MASS_MIN, device=env.device)
    env._box_unit_inertia = model.body_inertia[0, bid].clone() / model.body_mass[0, bid].clone()
  fixed = getattr(env, "box_mass_fixed", None)
  m = torch.rand(len(env_ids), device=env.device) * (MASS_MAX - MASS_MIN) + MASS_MIN if fixed is None \
    else torch.full((len(env_ids),), float(fixed), device=env.device)
  env._box_mass[env_ids] = m
  model.body_mass[env_ids, bid] = m
  model.body_inertia[env_ids, bid] = m[:, None] * env._box_unit_inertia


def box_mass_obs(env) -> torch.Tensor:
  m = getattr(env, "_box_mass", None)
  return torch.zeros(env.num_envs, 1, device=env.device) if m is None else (m / MASS_MAX).unsqueeze(-1)


def carry_vel_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  walk = k1_velocity_env_cfg(rough=False, play=play)
  cfg = carry2_env_cfg(play=play)
  twist = walk.commands["twist"]
  twist.ranges.lin_vel_x = (0.0, VX_MAX)
  twist.ranges.lin_vel_y = (-0.2, 0.2)
  twist.ranges.ang_vel_z = (-WZ_MAX, WZ_MAX)
  cfg.commands = {"twist": twist}

  cfg.actions["base"] = FrozenBaseActionCfg(entity_name="robot", checkpoint=CV_BASE, residual=True,
                                            residual_scale=RESIDUAL, command_name="twist")
  cfg.actions["arms"] = JointPositionActionCfg(entity_name="robot", actuator_names=ARM_REGEX, scale=0.25,
                                               use_default_offset=True)

  legs = SceneEntityCfg("robot", joint_names=LEG_REGEX)
  obs = {k: v for k, v in cfg.observations["actor"].terms.items() if k != "goal_pos"}
  obs.update({
    "command": ObservationTermCfg(func=mdp.generated_commands, params={"command_name": "twist"}),
    "leg_pos": ObservationTermCfg(func=mdp.joint_pos_rel, params={"asset_cfg": legs}),
    "leg_vel": ObservationTermCfg(func=mdp.joint_vel_rel, params={"asset_cfg": legs}),
    "box_mass": ObservationTermCfg(func=box_mass_obs),
  })
  for g in ("actor", "critic"):
    cfg.observations[g].terms = dict(obs)

  cfg.events["box_mass"] = EventTermCfg(func=reset_box_mass, mode="reset")

  cfg.rewards = {k: v for k, v in cfg.rewards.items() if k not in ("goal_progress", "goal_near", "success")}
  for k in ("track_linear_velocity", "track_angular_velocity"):
    cfg.rewards[k] = walk.rewards[k]
  cfg.rewards["residual_l2"] = RewardTermCfg(func=residual_l2, weight=-0.05)
  cfg.terminations = {k: v for k, v in cfg.terminations.items() if k != "goal_reached"}
  return cfg
