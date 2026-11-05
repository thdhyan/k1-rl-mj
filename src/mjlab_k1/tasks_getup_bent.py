"""Get-up with a bent-elbow goal: ``Mjlab-GetUpBent-Booster-K1``.

Joint motion is capped (targets slew at most K1_GETUP_VCAP=1.5 rad/s; penalties on joint speed over K1_GETUP_VLIM=4 rad/s, trunk motion, joint vel/acc) and every
episode starts fallen. Same task as :mod:`tasks_getup` (random fallen orientation -> stand, full 22-joint position targets, no residual,
no base policy), but the goal pose is HOME with the forearms bent forward. ``default_joint_pos`` is the goal, so the
action offset, the fallen-reset joint noise and ``stand_pose_r`` all target it with no other change.

Elbow_Yaw is the forward bend with the arm hanging at the side (forward kinematics: hand 0.1 m ahead of and level
with the trunk at Elbow_Yaw -1.5 / +1.5); Shoulder_Roll stays at HOME. Left/right mirror: roll and yaw flip sign,
pitch does not.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, replace

import torch

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.envs.mdp.actions.actions import JointPositionAction
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.tasks.velocity import mdp

from mjlab_k1.k1_constants import HOME, get_k1_getup_robot_cfg
from .tasks_getup import ACTION_SCALE, getup_env_cfg, reset_fallen

SHOULDER_PITCH = -0.3   # rad, upper arm slightly forward
ELBOW_BEND = 1.5        # rad, |Elbow_Yaw|; left negative, right positive

GOAL_JOINT_POS = {
  **{k: v for k, v in HOME.joint_pos.items() if k != ".*Shoulder_Pitch"},
  ".*Shoulder_Pitch": SHOULDER_PITCH,
  "Left_Elbow_Yaw": -ELBOW_BEND,
  "Right_Elbow_Yaw": ELBOW_BEND,
}

# Joint motion cap: position targets move at most VEL_CAP rad/s, so the PD loop cannot snap the robot up.
VEL_CAP = float(os.environ.get("K1_GETUP_VCAP", "1.5"))
VEL_LIMIT = float(os.environ.get("K1_GETUP_VLIM", "4.0"))   # rad/s, soft limit on measured joint speed
SETTLE_S = 1.0                                              # skip the 0.55 m drop / first impact


def _settled(env) -> torch.Tensor:
  return (env.episode_length_buf * env.step_dt > SETTLE_S).float()


def joint_speed_excess(env) -> torch.Tensor:
  """Sum of squared joint speed above VEL_LIMIT (excess clipped at 10 rad/s), after the settle time."""
  v = env.scene["robot"].data.joint_vel.abs()
  return _settled(env) * (v - VEL_LIMIT).clamp(0.0, 10.0).square().sum(-1)


def base_motion_l2(env) -> torch.Tensor:
  """Trunk linear + angular speed squared (no whipping up), after the settle time."""
  r = env.scene["robot"].data
  return _settled(env) * (r.root_link_lin_vel_b.square().sum(-1) + r.root_link_ang_vel_b.square().sum(-1))


class SlewJointPositionAction(JointPositionAction):
  """Position target = raw target slew-limited to +-vel_cap * step_dt per control step."""

  def __init__(self, cfg, env):
    super().__init__(cfg, env)
    self._max_delta = cfg.vel_cap * env.step_dt
    self._cmd = torch.zeros_like(self._raw_actions)
    self._fresh = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)

  def process_actions(self, actions: torch.Tensor):
    super().process_actions(actions)
    if self._fresh.any():  # new episode: start from where the joints actually are (after the reset event)
      q = self._entity.data.joint_pos[:, self._target_ids]
      self._cmd = torch.where(self._fresh[:, None], q, self._cmd)
      self._fresh[:] = False
    self._cmd = self._cmd + (self._processed_actions - self._cmd).clamp(-self._max_delta, self._max_delta)
    self._processed_actions = self._cmd

  def reset(self, env_ids=None):
    super().reset(env_ids)
    self._fresh[slice(None) if env_ids is None else env_ids] = True


@dataclass(kw_only=True)
class SlewJointPositionActionCfg(JointPositionActionCfg):
  vel_cap: float = VEL_CAP

  def build(self, env):
    return SlewJointPositionAction(self, env)


def getup_bent_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  cfg = getup_env_cfg(play=play)
  robot = get_k1_getup_robot_cfg()
  cfg.scene.entities = {"robot": replace(robot, init_state=replace(HOME, joint_pos=GOAL_JOINT_POS))}
  cfg.actions = {"joint_pos": SlewJointPositionActionCfg(
    entity_name="robot", actuator_names=(".*",), scale=ACTION_SCALE, use_default_offset=True, preserve_order=True)}
  # every episode starts fallen (no standing resets); slower motion needs a longer episode
  cfg.events["reset_fallen"] = EventTermCfg(func=reset_fallen, mode="reset", params={"stand_frac": 0.0})
  cfg.episode_length_s = 12.0 if not play else 1e9
  cfg.rewards["joint_vel"] = RewardTermCfg(func=mdp.joint_vel_l2, weight=-5e-4)
  cfg.rewards["joint_speed_excess"] = RewardTermCfg(func=joint_speed_excess, weight=-5e-3)
  cfg.rewards["base_motion"] = RewardTermCfg(func=base_motion_l2, weight=-0.02)
  cfg.rewards["joint_acc"] = RewardTermCfg(func=mdp.joint_acc_l2, weight=-2.5e-7)
  return cfg
