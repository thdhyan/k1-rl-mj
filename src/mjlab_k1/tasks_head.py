"""Head tracking (port of Isaac Lab `Isaac-HeadTrack-K1-v0`) on mjlab: the head layer of the HRL stack.

Policy owns the 2 head joints only. Legs = frozen walk base with a per-episode random velocity command
(the head must track while the body walks/turns; Isaac held the legs still), arms = random upper-body
driver. Obs carry no ground truth: the detection is a geometric camera model (the Isaac YOLO fallback),
ball projected into a pinhole camera on Head_2 with ZED-like FOV, noise and 5 % dropouts.
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
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.tasks.velocity import mdp
from mjlab.utils.lab_api.math import quat_apply, quat_apply_inverse

from .env_cfg import k1_velocity_env_cfg
from .frozen_base import FrozenBaseActionCfg
from mjlab_k1.k1_constants import ARM_REGEX
from .tasks_kick import BALL_RADIUS, BASE_CKPT, ball_spec
from .upper_body import RandomUpperBodyActionCfg

HEAD_JOINTS = ("AAHead_yaw", "Head_pitch")
HEAD = SceneEntityCfg("robot", joint_names=HEAD_JOINTS, preserve_order=True)
CAM_OFFSET = (0.06, 0.0, 0.05)  # camera in the Head_2 frame (x forward)
HFOV = math.radians(69.4)       # ZED 2c, 320x240 crop
VFOV = 2.0 * math.atan(math.tan(HFOV / 2.0) * 240.0 / 320.0)
#: ball roll speed curriculum: 0 -> V_MAX over iterations IT0 -> IT1 (24 env steps per iteration)
V_MAX, IT0, IT1, STEPS_PER_IT = 0.8, 200, 1200, 24
#: walking command ranges (vx, vy, wz) for the frozen base, held per episode
WALK_CMD = ((-0.2, 0.4), (-0.2, 0.2), (-0.5, 0.5))


def _state(env):
  st = getattr(env, "_head_state", None)
  if st is None:
    n, dev = env.num_envs, env.device
    st = env._head_state = type("S", (), {
      "dir": torch.zeros(n, 2, device=dev), "det": torch.zeros(n, 3, device=dev),
      "true": torch.zeros(n, 3, device=dev), "step": -1})()
  return st


def _ball_speed(env) -> float:
  it = env.common_step_counter / STEPS_PER_IT
  return V_MAX * min(max((it - IT0) / (IT1 - IT0), 0.0), 1.0)


def _project(env) -> torch.Tensor:
  """(n, 3) true (visible, du, dv), du/dv in [-1, 1] at the image edges, + = left / up."""
  robot = env.scene["robot"]
  bid = robot.find_bodies(("Head_2",))[0][0]
  q = robot.data.body_link_quat_w[:, bid]
  cam = robot.data.body_link_pos_w[:, bid] + quat_apply(q, torch.tensor(CAM_OFFSET, device=env.device).expand_as(q[:, :3]))
  rel = quat_apply_inverse(q, env.scene["ball"].data.root_link_pos_w - cam)
  du = torch.atan2(rel[:, 1], rel[:, 0]) / (HFOV / 2)
  dv = torch.atan2(rel[:, 2], rel[:, 0].abs().clamp_min(1e-3)) / (VFOV / 2)
  vis = (rel[:, 0] > 0.05) & (du.abs() < 1.0) & (dv.abs() < 1.0)
  return torch.stack([vis.float(), du.clamp(-1.5, 1.5), dv.clamp(-1.5, 1.5)], dim=-1)


def _update(env):
  st = _state(env)
  if st.step != env.common_step_counter:  # once per env step, shared by obs and rewards
    st.step = env.common_step_counter
    st.true = _project(env)
    det = st.true.clone()
    det[:, 1:] += 0.02 * torch.randn_like(det[:, 1:])
    det[:, 0] *= (torch.rand_like(det[:, 0]) > 0.05).float()  # detector dropouts
    det[:, 1:] *= det[:, :1]                                   # nothing seen -> (0, 0, 0)
    st.det = det
  return st


def ball_detection(env) -> torch.Tensor:
  return _update(env).det


def ball_centered(env, std: float = 0.35) -> torch.Tensor:
  t = _update(env).true
  return t[:, 0] * torch.exp(-(t[:, 1] ** 2 + t[:, 2] ** 2) / std**2)


def ball_in_frame(env) -> torch.Tensor:
  return _update(env).true[:, 0]


def reset_ball(env, env_ids: torch.Tensor) -> None:
  """Ball 1.0-2.5 m ahead of the robot (in its yaw frame), rolling mostly sideways at the curriculum speed."""
  n, dev = len(env_ids), env.device
  robot = env.scene["robot"].data
  root = robot.data.qpos[env_ids][:, robot.indexing.free_joint_q_adr]  # pose already reset (events run pre-forward)
  w, z = root[:, 3], root[:, 6]
  yaw = torch.atan2(2 * w * z, 1 - 2 * z * z)  # base reset randomises yaw only
  d = 1.0 + 1.5 * torch.rand(n, device=dev)
  lat = 0.5 * (2 * torch.rand(n, device=dev) - 1)
  c, s = torch.cos(yaw), torch.sin(yaw)
  pos = torch.stack([root[:, 0] + c * d - s * lat, root[:, 1] + s * d + c * lat,
                     torch.full_like(d, BALL_RADIUS + 0.01)], dim=-1)
  th = yaw + (torch.rand(n, device=dev) - 0.5) * 1.2 + torch.where(torch.rand(n, device=dev) < 0.5, 1.0, -1.0) * math.pi / 2
  st = _state(env)
  st.dir[env_ids] = torch.stack([torch.cos(th), torch.sin(th)], dim=-1)
  ball = env.scene["ball"]
  rs = ball.data.default_root_state[env_ids].clone()
  rs[:, :3] = pos
  rs[:, 7:] = 0.0
  rs[:, 7:9] = st.dir[env_ids] * _ball_speed(env)
  ball.write_root_state_to_sim(rs, env_ids=env_ids)


def keep_ball_rolling(env, env_ids: torch.Tensor) -> None:
  """Interval: re-assert the roll speed friction eats."""
  ball = env.scene["ball"]
  rs = ball.data.default_root_state[env_ids].clone()
  rs[:, :3] = ball.data.root_link_pos_w[env_ids]
  rs[:, 7:] = 0.0
  rs[:, 7:9] = _state(env).dir[env_ids] * _ball_speed(env)
  ball.write_root_state_to_sim(rs, env_ids=env_ids)


def head_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  cfg = k1_velocity_env_cfg(rough=False, play=play)
  cfg.scene.entities = {**cfg.scene.entities, "ball": EntityCfg(
    spec_fn=ball_spec, init_state=EntityCfg.InitialStateCfg(pos=(1.5, 0.0, BALL_RADIUS + 0.01)))}
  cfg.commands = {}
  cfg.curriculum = {}
  cfg.episode_length_s = 15.0
  cfg.actions = {
    "head": JointPositionActionCfg(entity_name="robot", actuator_names=HEAD_JOINTS, scale=0.5, preserve_order=True),
    "base": FrozenBaseActionCfg(entity_name="robot", checkpoint=BASE_CKPT, auto_cmd=WALK_CMD),
    "arms": RandomUpperBodyActionCfg(entity_name="robot", joint_names=ARM_REGEX),
  }
  terms = {
    "detection": ObservationTermCfg(func=ball_detection),
    "head_pos": ObservationTermCfg(func=mdp.joint_pos_rel, params={"asset_cfg": HEAD}),
    "head_vel": ObservationTermCfg(func=mdp.joint_vel_rel, params={"asset_cfg": HEAD}),
    "ang_vel": ObservationTermCfg(func=mdp.builtin_sensor, params={"sensor_name": "robot/imu_ang_vel"}),
    "last_action": ObservationTermCfg(func=mdp.last_action),
  }
  cfg.observations = {
    "actor": ObservationGroupCfg(dict(terms), enable_corruption=False),
    "critic": ObservationGroupCfg(dict(terms), enable_corruption=False),
  }
  cfg.events = {k: v for k, v in cfg.events.items() if k in ("reset_base", "reset_robot_joints", "foot_friction")}
  cfg.events["reset_ball"] = EventTermCfg(func=reset_ball, mode="reset")
  cfg.events["keep_ball_rolling"] = EventTermCfg(func=keep_ball_rolling, mode="interval", interval_range_s=(0.5, 0.5))
  cfg.rewards = {
    "ball_centered": RewardTermCfg(func=ball_centered, weight=2.0),
    "ball_in_frame": RewardTermCfg(func=ball_in_frame, weight=0.5),
    "action_rate": RewardTermCfg(func=mdp.action_rate_l2, weight=-0.1),
    "joint_pos_limits": RewardTermCfg(func=mdp.joint_pos_limits, weight=-1.0, params={"asset_cfg": HEAD}),
  }
  cfg.terminations = {
    "time_out": TerminationTermCfg(func=mdp.time_out, time_out=True),
    "fell": TerminationTermCfg(func=mdp.bad_orientation, params={"limit_angle": float(os.environ.get("K1_FELL_ANGLE", "0.8"))}),
  }
  return cfg
