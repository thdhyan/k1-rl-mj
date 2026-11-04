"""Ball tracking v2: keep the ball centred in the head camera with the head AND the body (walk base command).

v1 (tasks_head) owned only the head and saw an image-plane detection while the legs walked a random command.
v2: the policy sees the ball POSITION in the camera frame (x forward, y left, z up; what detection + ZED depth
gives on the robot), with range-dependent noise, dropouts and last-seen hold + age. Action = head (yaw, pitch)
offsets | (vx, vy, wz) to the frozen walk base: it may turn and walk to follow a ball that wanders, speeds up
(curriculum) and leaves the field of view. Reward: ball centred in the image, in frame, kept at FOLLOW_D,
head near straight (turn the body, not just the neck), light regularisation.
"""
from __future__ import annotations

import math
import os

import torch

from mjlab.entity import EntityCfg
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.tasks.velocity import mdp
from mjlab.utils.lab_api.math import quat_apply, quat_apply_inverse

from .env_cfg import k1_velocity_env_cfg
from .frozen_base import FrozenBaseActionCfg
from mjlab_k1.k1_constants import ARM_REGEX
from .tasks_head import CAM_OFFSET, HEAD, HEAD_JOINTS, HFOV, VFOV
from .tasks_kick import BALL_RADIUS, BASE_CKPT, ball_spec
from .upper_body import RandomUpperBodyActionCfg

#: ball speed curriculum 0.2 -> V_MAX over iterations IT0 -> IT1 (24 env steps per iteration)
V_MIN, V_MAX, IT0, IT1, STEPS_PER_IT = 0.2, 1.0, 200, 2000, 24
FOLLOW_D, FAR_D = 1.5, 4.0       # m: wanted ball distance; beyond FAR_D the ball turns back toward the robot
TURN_PROB = 0.3                  # per 1 s interval: new random roll direction
DEPTH_NOISE = 0.02               # m per m of range (ZED-like), plus 1 cm
DROPOUT = 0.1


def _state(env):
  st = getattr(env, "_track_state", None)
  if st is None:
    n, dev = env.num_envs, env.device
    st = env._track_state = type("S", (), {
      "dir": torch.zeros(n, 2, device=dev), "rel": torch.zeros(n, 3, device=dev),
      "obs": torch.zeros(n, 5, device=dev), "step": -1})()
  return st


def _ball_speed(env) -> float:
  it = env.common_step_counter / STEPS_PER_IT
  return V_MIN + (V_MAX - V_MIN) * min(max((it - IT0) / (IT1 - IT0), 0.0), 1.0)


def _cam_rel(env) -> torch.Tensor:
  """Ball centre in the camera frame (Head_2 + CAM_OFFSET, x forward)."""
  robot = env.scene["robot"]
  bid = robot.find_bodies(("Head_2",))[0][0]
  q = robot.data.body_link_quat_w[:, bid]
  cam = robot.data.body_link_pos_w[:, bid] + quat_apply(q, torch.tensor(CAM_OFFSET, device=env.device).expand_as(q[:, :3]))
  return quat_apply_inverse(q, env.scene["ball"].data.root_link_pos_w - cam)


def _image(rel: torch.Tensor) -> torch.Tensor:
  """(visible, du, dv): du/dv in [-1, 1] at the image edges, + = left / up."""
  du = torch.atan2(rel[:, 1], rel[:, 0]) / (HFOV / 2)
  dv = torch.atan2(rel[:, 2], rel[:, 0].abs().clamp_min(1e-3)) / (VFOV / 2)
  vis = (rel[:, 0] > 0.05) & (du.abs() < 1.0) & (dv.abs() < 1.0)
  return torch.stack([vis.float(), du, dv], -1)


def _update(env):
  st = _state(env)
  if st.step != env.common_step_counter:  # once per env step, shared by obs and rewards
    st.step = env.common_step_counter
    st.rel = _cam_rel(env)
    seen = _image(st.rel)[:, 0] * (torch.rand(env.num_envs, device=env.device) > DROPOUT).float()
    noisy = st.rel + (0.01 + DEPTH_NOISE * st.rel.norm(dim=-1, keepdim=True)) * torch.randn_like(st.rel)
    # obs = (seen now, x, y, z, age s): last detection held while unseen, age counts up (capped 2 s)
    age = torch.where(seen > 0, 0.0, (st.obs[:, 4] + env.step_dt).clamp_max(2.0))
    xyz = torch.where(seen[:, None] > 0, noisy, st.obs[:, 1:4])
    st.obs = torch.cat([seen[:, None], xyz, age[:, None]], -1)
  return st


def ball_cam(env) -> torch.Tensor:
  return _update(env).obs


def ball_centered(env, std: float = 0.35) -> torch.Tensor:
  t = _image(_update(env).rel)
  return t[:, 0] * torch.exp(-(t[:, 1] ** 2 + t[:, 2] ** 2) / std**2)


def ball_in_frame(env) -> torch.Tensor:
  return _image(_update(env).rel)[:, 0]


def follow_distance(env, std: float = 0.5) -> torch.Tensor:
  rel = env.scene["ball"].data.root_link_pos_w[:, :2] - env.scene["robot"].data.root_link_pos_w[:, :2]
  return torch.exp(-((rel.norm(dim=-1) - FOLLOW_D) ** 2) / std**2)


def head_straight(env) -> torch.Tensor:
  """Squared head yaw: turn the body toward the ball rather than holding the neck at its limit."""
  robot = env.scene["robot"]
  return robot.data.joint_pos[:, robot.find_joints(HEAD_JOINTS[:1])[0]].square().sum(-1)


def _set_ball(env, env_ids, pos=None):
  ball = env.scene["ball"]
  rs = ball.data.default_root_state[env_ids].clone()
  rs[:, :3] = ball.data.root_link_pos_w[env_ids] if pos is None else pos
  rs[:, 7:] = 0.0
  rs[:, 7:9] = _state(env).dir[env_ids] * _ball_speed(env)
  ball.write_root_state_to_sim(rs, env_ids=env_ids)


def reset_ball(env, env_ids: torch.Tensor) -> None:
  """Ball 1-3 m away within +-0.9 rad of the robot heading (part of it outside the 69 deg FOV), random roll."""
  n, dev = len(env_ids), env.device
  robot = env.scene["robot"].data
  root = robot.data.qpos[env_ids][:, robot.indexing.free_joint_q_adr]  # pose already reset (events run pre-forward)
  w, z = root[:, 3], root[:, 6]
  yaw = torch.atan2(2 * w * z, 1 - 2 * z * z)
  d = 1.0 + 2.0 * torch.rand(n, device=dev)
  b = yaw + 1.8 * (torch.rand(n, device=dev) - 0.5)
  pos = torch.stack([root[:, 0] + d * torch.cos(b), root[:, 1] + d * torch.sin(b), torch.full_like(d, BALL_RADIUS + 0.01)], -1)
  th = 2 * math.pi * torch.rand(n, device=dev)
  _state(env).dir[env_ids] = torch.stack([torch.cos(th), torch.sin(th)], -1)
  _set_ball(env, env_ids, pos)


def wander_ball(env, env_ids: torch.Tensor) -> None:
  """Interval: sometimes a new roll direction; past FAR_D roll back toward the robot; re-assert the speed."""
  n, dev = len(env_ids), env.device
  st = _state(env)
  th = 2 * math.pi * torch.rand(n, device=dev)
  new = torch.stack([torch.cos(th), torch.sin(th)], -1)
  to_robot = env.scene["robot"].data.root_link_pos_w[env_ids, :2] - env.scene["ball"].data.root_link_pos_w[env_ids, :2]
  far = to_robot.norm(dim=-1, keepdim=True) > FAR_D
  turn = torch.rand(n, 1, device=dev) < TURN_PROB
  d = torch.where(turn, new, st.dir[env_ids])
  st.dir[env_ids] = torch.where(far, to_robot / to_robot.norm(dim=-1, keepdim=True).clamp_min(1e-6), d)
  _set_ball(env, env_ids)


def track_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  cfg = k1_velocity_env_cfg(rough=False, play=play)
  cfg.scene.entities = {**cfg.scene.entities, "ball": EntityCfg(
    spec_fn=ball_spec, init_state=EntityCfg.InitialStateCfg(pos=(1.5, 0.0, BALL_RADIUS + 0.01)))}
  cfg.commands = {}
  cfg.curriculum = {}
  cfg.episode_length_s = 20.0
  cfg.actions = {
    "head": JointPositionActionCfg(entity_name="robot", actuator_names=HEAD_JOINTS, scale=0.5, preserve_order=True),
    "base": FrozenBaseActionCfg(entity_name="robot", checkpoint=BASE_CKPT),
    "arms": RandomUpperBodyActionCfg(entity_name="robot", joint_names=ARM_REGEX),
  }
  terms = {
    "ball_cam": ObservationTermCfg(func=ball_cam),
    "head_pos": ObservationTermCfg(func=mdp.joint_pos_rel, params={"asset_cfg": HEAD}),
    "head_vel": ObservationTermCfg(func=mdp.joint_vel_rel, params={"asset_cfg": HEAD}),
    "ang_vel": ObservationTermCfg(func=mdp.builtin_sensor, params={"sensor_name": "robot/imu_ang_vel"}),
    "gravity": ObservationTermCfg(func=mdp.projected_gravity),
    "last_action": ObservationTermCfg(func=mdp.last_action),
  }
  cfg.observations = {
    "actor": ObservationGroupCfg(dict(terms), enable_corruption=False, history_length=3),
    "critic": ObservationGroupCfg(dict(terms), enable_corruption=False, history_length=3),
  }
  cfg.events = {k: v for k, v in cfg.events.items() if k in ("reset_base", "reset_robot_joints", "foot_friction")}
  cfg.events["reset_ball"] = EventTermCfg(func=reset_ball, mode="reset")
  cfg.events["wander_ball"] = EventTermCfg(func=wander_ball, mode="interval", interval_range_s=(1.0, 1.0))
  cfg.rewards = {
    "ball_centered": RewardTermCfg(func=ball_centered, weight=2.0),
    "ball_in_frame": RewardTermCfg(func=ball_in_frame, weight=0.5),
    "follow_distance": RewardTermCfg(func=follow_distance, weight=1.0),
    "head_straight": RewardTermCfg(func=head_straight, weight=-0.2),
    "action_rate": RewardTermCfg(func=mdp.action_rate_l2, weight=-0.05),
    "joint_pos_limits": RewardTermCfg(func=mdp.joint_pos_limits, weight=-1.0, params={"asset_cfg": HEAD}),
    "flat_orientation": RewardTermCfg(func=mdp.flat_orientation_l2, weight=-1.0),
  }
  cfg.terminations = {
    "time_out": TerminationTermCfg(func=mdp.time_out, time_out=True),
    "fell": TerminationTermCfg(func=mdp.bad_orientation, params={"limit_angle": float(os.environ.get("K1_FELL_ANGLE", "0.8"))}),
  }
  return cfg
