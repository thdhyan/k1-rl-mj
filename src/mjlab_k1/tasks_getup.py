"""Get-up: from lying (prone, supine, side) or a random tumble back to the HOME stand, all 22 joints.

* ``Mjlab-GetUp-Booster-K1``: flat ground, whole-body ground contact (no self-collision), arms at the manip
  gains (kp 40) so the hands can push. Actor = what the robot measures (gyro, gravity, q, dq, last action), 5-step
  history, no command. Episodes never end early: the policy must get up and then stand still at HOME, which is the
  hand-over pose for the walk base (deploy FSM: fallen -> damp -> get-up -> ramp to HOME -> walk).
* Resets: 15% standing at HOME (stay up), the rest lying with random joints, a share of them dropped in a random
  orientation from 0.55 m (the "disoriented fall").
"""
from __future__ import annotations

import math
import os

import torch

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.tasks.velocity import mdp
from mjlab.utils.lab_api.math import quat_from_euler_xyz

from .env_cfg import k1_velocity_env_cfg
from mjlab_k1.k1_constants import K1_ACTION_SCALE, get_k1_getup_robot_cfg

STAND_FRAC = float(os.environ.get("K1_GETUP_STAND_FRAC", "0.15"))
JOINT_NOISE = 0.6          # rad, uniform around HOME for fallen resets
H_LO, H_UP = 0.15, 0.48    # trunk height: lying ~0.1, HOME 0.53
ACTION_SCALE = {**K1_ACTION_SCALE, ".*Shoulder.*": 1.0, ".*Elbow.*": 1.0, ".*Head.*": 0.2}


def reset_fallen(env, env_ids: torch.Tensor | None, stand_frac: float = STAND_FRAC, joint_noise: float = JOINT_NOISE):
  """Per env: standing (stand_frac), or prone / supine / left / right side / random tumble (equal shares)."""
  robot = env.scene["robot"]
  if env_ids is None:
    env_ids = torch.arange(env.num_envs, device=env.device)
  n, dev = len(env_ids), env.device
  u = lambda lo, hi: lo + (hi - lo) * torch.rand(n, device=dev)  # noqa: E731
  cat = torch.randint(0, 5, (n,), device=dev)
  roll = torch.zeros(n, device=dev)
  pitch = torch.zeros(n, device=dev)
  pitch = torch.where(cat == 0, math.pi / 2, pitch)    # prone (face down)
  pitch = torch.where(cat == 1, -math.pi / 2, pitch)   # supine
  roll = torch.where(cat == 2, math.pi / 2, roll)
  roll = torch.where(cat == 3, -math.pi / 2, roll)
  roll = torch.where(cat == 4, u(-math.pi, math.pi), roll + u(-0.3, 0.3))
  pitch = torch.where(cat == 4, u(-math.pi / 2, math.pi / 2), pitch + u(-0.3, 0.3))
  z = torch.where(cat == 4, 0.55, 0.32)
  stand = torch.rand(n, device=dev) < stand_frac
  roll, pitch = torch.where(stand, 0.0, roll), torch.where(stand, 0.0, pitch)

  root = robot.data.default_root_state[env_ids].clone()
  root[:, 2] = torch.where(stand, root[:, 2] + 0.01, z)
  root[:, :3] += env.scene.env_origins[env_ids]
  root[:, 3:7] = quat_from_euler_xyz(roll, pitch, u(-math.pi, math.pi))
  root[:, 7:] = 0.0
  robot.write_root_link_pose_to_sim(root[:, :7], env_ids=env_ids)
  robot.write_root_link_velocity_to_sim(root[:, 7:], env_ids=env_ids)

  q = robot.data.default_joint_pos[env_ids].clone()
  noise = torch.where(stand, 0.05, joint_noise)[:, None]
  q += (2 * torch.rand_like(q) - 1) * noise
  lim = robot.data.soft_joint_pos_limits[env_ids]
  q = q.clamp(lim[..., 0], lim[..., 1])
  robot.write_joint_state_to_sim(q, torch.zeros_like(q), env_ids=env_ids)


def _up(env) -> torch.Tensor:
  return -env.scene["robot"].data.projected_gravity_b[:, 2]   # 1 upright, -1 upside down


def _height(env) -> torch.Tensor:
  return env.scene["robot"].data.root_link_pos_w[:, 2] - env.scene.env_origins[:, 2]


def _stood(env) -> torch.Tensor:
  return (_height(env) > 0.45) & (_up(env) > 0.9)


def base_height(env) -> torch.Tensor:
  return _height(env)[:, None]


def trunk_height_r(env) -> torch.Tensor:
  return ((_height(env) - H_LO) / (H_UP - H_LO)).clamp(0.0, 1.0)


def upright_r(env) -> torch.Tensor:
  return ((_up(env) + 1.0) / 2.0).square()


def stand_pose_r(env, std: float = 1.0) -> torch.Tensor:
  """Once up: joints at HOME (the walk base's hand-over pose)."""
  r = env.scene["robot"].data
  err = (r.joint_pos - r.default_joint_pos).square().sum(-1)
  stood = _stood(env)
  env.extras["log"]["Metrics/stood"] = stood.float().mean()
  return stood.float() * torch.exp(-err / std**2)


def stand_still_r(env) -> torch.Tensor:
  r = env.scene["robot"].data
  return _stood(env).float() * torch.exp(-r.root_link_ang_vel_b.square().sum(-1) - r.joint_vel.square().sum(-1) / 20.0)


def getup_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  cfg = k1_velocity_env_cfg(rough=False, play=play)
  cfg.scene.entities = {"robot": get_k1_getup_robot_cfg()}
  cfg.sim.nconmax = 120
  cfg.sim.njmax = 800
  cfg.commands = {}
  cfg.curriculum = {}
  cfg.episode_length_s = 8.0 if not play else 1e9

  cfg.actions = {"joint_pos": JointPositionActionCfg(
    entity_name="robot", actuator_names=(".*",), scale=ACTION_SCALE, use_default_offset=True, preserve_order=True)}

  actor = {
    "ang_vel": ObservationTermCfg(func=mdp.builtin_sensor, params={"sensor_name": "robot/imu_ang_vel"}),
    "gravity": ObservationTermCfg(func=mdp.projected_gravity),
    "joint_pos": ObservationTermCfg(func=mdp.joint_pos_rel),
    "joint_vel": ObservationTermCfg(func=mdp.joint_vel_rel),
    "actions": ObservationTermCfg(func=mdp.last_action),
  }
  critic = {**actor, "base_height": ObservationTermCfg(func=base_height),
            "lin_vel": ObservationTermCfg(func=mdp.builtin_sensor, params={"sensor_name": "robot/imu_lin_vel"})}
  cfg.observations = {
    "actor": ObservationGroupCfg(actor, enable_corruption=not play, history_length=5),
    "critic": ObservationGroupCfg(critic, enable_corruption=False),
  }

  cfg.events = {k: v for k, v in cfg.events.items() if k in ("foot_friction", "encoder_bias", "base_com")}
  cfg.events["reset_fallen"] = EventTermCfg(func=reset_fallen, mode="reset")

  cfg.rewards = {
    "trunk_height": RewardTermCfg(func=trunk_height_r, weight=2.0),
    "upright": RewardTermCfg(func=upright_r, weight=1.0),
    "stand_pose": RewardTermCfg(func=stand_pose_r, weight=2.0),
    "stand_still": RewardTermCfg(func=stand_still_r, weight=1.0),
    "action_rate": RewardTermCfg(func=mdp.action_rate_l2, weight=-float(os.environ.get("K1_GETUP_AR_W", "0.01"))),
    "joint_vel": RewardTermCfg(func=mdp.joint_vel_l2, weight=-1e-4),
    "dof_pos_limits": RewardTermCfg(func=mdp.joint_pos_limits, weight=-1.0),
  }
  cfg.terminations = {"time_out": TerminationTermCfg(func=mdp.time_out, time_out=True)}
  return cfg
