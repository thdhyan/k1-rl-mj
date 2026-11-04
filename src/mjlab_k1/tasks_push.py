"""Box push (port of Isaac Lab `Isaac-Push-Simple-K1-v0`) on mjlab: an HRL task layer with arms.

Action (11) = (vx, vy, wz) in [-1, 1] -> frozen walk base | 8 arm joint offsets (scale 0.5 rad) about home.
The Isaac task drove the wrists through DifferentialIK; here the arms are joint-space so the HRL contract
stays 'task layer -> base command + arm targets'. Arms use the Isaac push/carry gains (kp 40, kd 2) and
collide (k1_constants.MANIP). Head holds home. Rewards follow the Isaac SIMPLE set: hands to box, box
progress toward the goal, box near goal, success bonus, failure penalty, light regularisation.
"""
from __future__ import annotations

import math
import os

import mujoco
import torch

from mjlab.entity import EntityCfg
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.tasks.velocity import mdp
from mjlab.utils.lab_api.math import quat_apply, quat_apply_inverse

from .env_cfg import k1_velocity_env_cfg
from .frozen_base import FrozenBaseActionCfg
from mjlab_k1.k1_constants import ARM_REGEX, HAND_BODIES, get_k1_manip_robot_cfg
from .tasks_kick import BASE_CKPT, _yaw_rel, failure_terminated
from .upper_body import RandomUpperBodyActionCfg

BOX_HALF, BOX_MASS = 0.35, 4.0
BOX_X, BOX_Y, BOX_YAW = (0.8, 1.1), (-0.2, 0.2), (-0.3, 0.3)   # spawn, env frame (robot faces +x)
GOAL_D, GOAL_ANGLE = (0.5, 1.5), (-0.5, 0.5)                   # goal = box + d * (cos a, sin a)
SUCCESS_R, OOB_R = 0.15, 3.0
FAILURES = ("fell", "box_oob", "box_tip")


def box_spec() -> mujoco.MjSpec:
  spec = mujoco.MjSpec()
  body = spec.worldbody.add_body(name="box")
  body.add_freejoint(name="box_joint")
  body.add_geom(
    name="box_geom", type=mujoco.mjtGeom.mjGEOM_BOX, size=(BOX_HALF,) * 3, mass=BOX_MASS,
    rgba=(0.8, 0.5, 0.2, 1.0), friction=(0.6, 0.005, 0.0001), condim=3, contype=1, conaffinity=1)
  return spec


##
# State + MDP terms
##
def _state(env):
  st = getattr(env, "_push_state", None)
  if st is None:
    n, dev = env.num_envs, env.device
    st = env._push_state = type("S", (), {"goal": torch.zeros(n, 3, device=dev), "d_prev": torch.zeros(n, device=dev)})()
  return st


def _box(env):
  return env.scene["box"].data


def _hands_w(env) -> torch.Tensor:
  robot = env.scene["robot"]
  ids = getattr(env, "_hand_ids", None)
  if ids is None:
    ids = env._hand_ids = robot.find_bodies(HAND_BODIES, preserve_order=True)[0]
  return robot.data.body_link_pos_w[:, ids]  # (n, 2, 3)


def _box_goal_dist(env) -> torch.Tensor:
  return (_box(env).root_link_pos_w[:, :2] - _state(env).goal[:, :2]).norm(dim=-1)


def box_pos_b(env) -> torch.Tensor:
  return _yaw_rel(env, _box(env).root_link_pos_w - env.scene["robot"].data.root_link_pos_w)


def box_vel_b(env) -> torch.Tensor:
  return _yaw_rel(env, _box(env).root_link_lin_vel_w)


def box_heading_b(env) -> torch.Tensor:
  """Box x-axis in the robot yaw frame (sin/cos of relative yaw; a cube is 90-deg symmetric, keep both)."""
  ax = quat_apply(_box(env).root_link_quat_w, torch.tensor([1.0, 0.0, 0.0], device=env.device).expand(env.num_envs, 3))
  return _yaw_rel(env, ax)[:, :2]


def goal_pos_b(env) -> torch.Tensor:
  return _yaw_rel(env, _state(env).goal - env.scene["robot"].data.root_link_pos_w)


def hands_pos_b(env) -> torch.Tensor:
  rel = _hands_w(env) - env.scene["robot"].data.root_link_pos_w[:, None]
  return torch.cat([_yaw_rel(env, rel[:, 0]), _yaw_rel(env, rel[:, 1])], -1)


def hands_to_box(env, std: float = 0.2) -> torch.Tensor:
  """exp(-d / std) per hand, averaged; d = hand distance to the box surface."""
  b = _box(env)
  rel = _hands_w(env) - b.root_link_pos_w[:, None]
  q = b.root_link_quat_w[:, None].expand(-1, 2, -1)
  local = quat_apply_inverse(q.reshape(-1, 4), rel.reshape(-1, 3)).reshape(-1, 2, 3)
  d = (local.abs() - BOX_HALF).clamp_min(0.0).norm(dim=-1)
  return torch.exp(-d / std).mean(-1)


def box_progress_rate(env) -> torch.Tensor:
  """Metres per second the box closed on the goal this step (clamped)."""
  st = _state(env)
  d = _box_goal_dist(env)
  r = ((st.d_prev - d) / env.step_dt).clamp(-1.0, 1.0)
  st.d_prev = d
  return r


def box_near_goal(env, std: float = 0.5) -> torch.Tensor:
  return torch.exp(-(_box_goal_dist(env) ** 2) / std**2)


def goal_reached(env) -> torch.Tensor:
  return _box_goal_dist(env) < SUCCESS_R


def box_oob(env) -> torch.Tensor:
  return (_box(env).root_link_pos_w[:, :2] - env.scene.env_origins[:, :2]).norm(dim=-1) > OOB_R


def box_tip(env, limit_deg: float = 45.0) -> torch.Tensor:
  # cube: any face may point up -> tipped when no box axis is within limit of vertical
  q, eye = _box(env).root_link_quat_w, torch.eye(3, device=env.device)
  up_z = torch.stack([quat_apply(q, eye[i].expand(env.num_envs, 3))[:, 2] for i in range(3)], -1)
  return up_z.abs().max(-1).values < math.cos(math.radians(limit_deg))


def reset_box(env, env_ids: torch.Tensor) -> None:
  n, dev = len(env_ids), env.device
  u = lambda lo_hi: lo_hi[0] + (lo_hi[1] - lo_hi[0]) * torch.rand(n, device=dev)
  pos = torch.stack([u(BOX_X), u(BOX_Y), torch.full((n,), BOX_HALF + 0.005, device=dev)], -1)
  yaw = u(BOX_YAW)
  box = env.scene["box"]
  st = box.data.default_root_state[env_ids].clone()
  st[:, :3] = pos + env.scene.env_origins[env_ids]
  st[:, 3:7] = torch.stack([torch.cos(yaw / 2), torch.zeros_like(yaw), torch.zeros_like(yaw), torch.sin(yaw / 2)], -1)
  st[:, 7:] = 0.0
  box.write_root_state_to_sim(st, env_ids=env_ids)
  d, a = u(GOAL_D), u(GOAL_ANGLE)
  s = _state(env)
  s.goal[env_ids] = st[:, :3] + torch.stack([d * torch.cos(a), d * torch.sin(a), torch.zeros_like(d)], -1)
  s.d_prev[env_ids] = d


def push_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  cfg = k1_velocity_env_cfg(rough=False, play=play)
  cfg.scene.entities = {"robot": get_k1_manip_robot_cfg(), "box": EntityCfg(
    spec_fn=box_spec, init_state=EntityCfg.InitialStateCfg(pos=(1.0, 0.0, BOX_HALF + 0.005)))}
  cfg.sim.mujoco.ls_iterations = int(os.environ.get("K1_LS_ITERS", "50"))
  cfg.sim.nconmax = 80
  cfg.sim.njmax = 600
  cfg.commands = {}
  cfg.curriculum = {}
  cfg.episode_length_s = 15.0

  cfg.actions = {
    "base": FrozenBaseActionCfg(entity_name="robot", checkpoint=BASE_CKPT),
    "arms": JointPositionActionCfg(entity_name="robot", actuator_names=ARM_REGEX, scale=0.5, use_default_offset=True),
    "head": RandomUpperBodyActionCfg(entity_name="robot", joint_names=(".*Head.*",), frozen_prob=1.0),
  }

  arms = SceneEntityCfg("robot", joint_names=ARM_REGEX)
  terms = {
    "gravity": ObservationTermCfg(func=mdp.projected_gravity),
    "ang_vel": ObservationTermCfg(func=mdp.builtin_sensor, params={"sensor_name": "robot/imu_ang_vel"}),
    "box_pos": ObservationTermCfg(func=box_pos_b),
    "box_heading": ObservationTermCfg(func=box_heading_b),
    "box_vel": ObservationTermCfg(func=box_vel_b),
    "goal_pos": ObservationTermCfg(func=goal_pos_b),
    "hands_pos": ObservationTermCfg(func=hands_pos_b),
    "arm_pos": ObservationTermCfg(func=mdp.joint_pos_rel, params={"asset_cfg": arms}),
    "arm_vel": ObservationTermCfg(func=mdp.joint_vel_rel, params={"asset_cfg": arms}),
    "last_action": ObservationTermCfg(func=mdp.last_action),
  }
  cfg.observations = {
    "actor": ObservationGroupCfg(dict(terms), enable_corruption=False),
    "critic": ObservationGroupCfg(dict(terms), enable_corruption=False),
  }

  cfg.events = {k: v for k, v in cfg.events.items() if k in ("reset_base", "reset_robot_joints", "foot_friction")}
  cfg.events["reset_base"].params["pose_range"] = {"x": (0.0, 0.0), "y": (0.0, 0.0), "z": (0.01, 0.03), "yaw": (-0.2, 0.2)}
  cfg.events["reset_box"] = EventTermCfg(func=reset_box, mode="reset")

  cfg.rewards = {
    "alive": RewardTermCfg(func=mdp.is_alive, weight=0.2),
    "hands_to_box": RewardTermCfg(func=hands_to_box, weight=1.0),
    "box_progress": RewardTermCfg(func=box_progress_rate, weight=5.0),
    "box_near_goal": RewardTermCfg(func=box_near_goal, weight=2.0),
    "success": RewardTermCfg(func=goal_reached, weight=500.0),   # x dt = +10 once
    "failure": RewardTermCfg(func=failure_terminated, weight=-500.0, params={"term_names": FAILURES}),
    "flat_orientation": RewardTermCfg(
      func=mdp.flat_orientation_l2, weight=-1.0, params={"asset_cfg": SceneEntityCfg("robot", body_names=("Trunk",))}),
    "action_rate": RewardTermCfg(func=mdp.action_rate_l2, weight=-0.005),
    "arm_limits": RewardTermCfg(func=mdp.joint_pos_limits, weight=-1.0, params={"asset_cfg": arms}),
  }
  cfg.terminations = {
    "time_out": TerminationTermCfg(func=mdp.time_out, time_out=True),
    "fell": TerminationTermCfg(func=mdp.bad_orientation, params={"limit_angle": float(os.environ.get("K1_FELL_ANGLE", "0.8"))}),
    "goal_reached": TerminationTermCfg(func=goal_reached),
    "box_oob": TerminationTermCfg(func=box_oob),
    "box_tip": TerminationTermCfg(func=box_tip),
  }
  return cfg
