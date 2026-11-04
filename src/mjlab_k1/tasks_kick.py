"""Kick task (port of Isaac Lab `Isaac-Kick-*`) on mjlab: the first HRL task layer.

Policy = task layer. Action = (vx, vy, wz) in [-1, 1] -> frozen walk base (frozen_base.py), optionally
+ 12 leg deltas (K1_KICK_RESIDUAL=1, the Isaac `p9_kick_residual` form). Ball is a free sphere; the goal
is analytic (a box volume at (FIELD_L/2, 0)), as in the Isaac reward code. Head/arms follow the random
upper-body driver for now (head tracking is a separate layer).
"""
from __future__ import annotations

import os

import mujoco
import torch

from mjlab.entity import EntityCfg
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp import dr  # noqa: F401
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.tasks.velocity import mdp
from mjlab.utils.lab_api.math import quat_apply_inverse, yaw_quat

from k1_common.paths import model

from .env_cfg import k1_velocity_env_cfg
from .frozen_base import FrozenBaseActionCfg
from .upper_body import RandomUpperBodyActionCfg

FIELD_L, GOAL_W, GOAL_H = 8.0, 2.0, 0.9
BALL_RADIUS, BALL_MASS = 0.11, 0.43
GOAL_POS = (FIELD_L / 2, 0.0, GOAL_H / 2)
BASE_CKPT = os.environ.get(
  "K1_BASE_CKPT", str(model("mjlab_k1/flat_swing10/model_4999.pt")))


def ball_spec() -> mujoco.MjSpec:
  spec = mujoco.MjSpec()
  body = spec.worldbody.add_body(name="ball")
  body.add_freejoint(name="ball_joint")
  body.add_geom(
    name="ball_geom", type=mujoco.mjtGeom.mjGEOM_SPHERE, size=(BALL_RADIUS, 0, 0), mass=BALL_MASS,
    rgba=(1.0, 0.85, 0.1, 1.0), friction=(0.6, 0.005, 0.0001), condim=3, contype=1, conaffinity=1)
  return spec


##
# MDP terms (all take env first; ball/robot read through the scene)
##
def _goal_w(env) -> torch.Tensor:
  target = getattr(env, "_kick_target", None)  # per-env pass target (tasks_kick_dir), else the fixed goal
  if target is not None:
    return target
  return torch.tensor(GOAL_POS, device=env.device) + env.scene.env_origins


def _ball(env):
  return env.scene["ball"].data


def _yaw_rel(env, vec_w: torch.Tensor) -> torch.Tensor:
  q = yaw_quat(env.scene["robot"].data.root_link_quat_w)
  return quat_apply_inverse(q, vec_w)


def ball_pos_b(env) -> torch.Tensor:
  return _yaw_rel(env, _ball(env).root_link_pos_w - env.scene["robot"].data.root_link_pos_w)


def ball_vel_b(env) -> torch.Tensor:
  return _yaw_rel(env, _ball(env).root_link_lin_vel_w)


def goal_pos_b(env) -> torch.Tensor:
  return _yaw_rel(env, _goal_w(env) - env.scene["robot"].data.root_link_pos_w)


def goal_scored(env) -> torch.Tensor:
  rel = _ball(env).root_link_pos_w - _goal_w(env)
  return (rel[:, 0] > -0.1) & (rel[:, 0] < 0.2) & (rel[:, 1].abs() < GOAL_W / 2) & (rel[:, 2].abs() < GOAL_H / 2)


def _state(env):
  st = getattr(env, "_kick_state", None)
  if st is None:
    z = lambda: torch.zeros(env.num_envs, device=env.device)
    st = env._kick_state = type("S", (), {"g0": z(), "r0": z()})()
  return st


def ball_to_goal_progress(env) -> torch.Tensor:
  st = _state(env)
  d = (_ball(env).root_link_pos_w - _goal_w(env)).norm(dim=-1)
  st.g0 = torch.where(st.g0 < 1e-6, d, st.g0)
  return ((st.g0 - d) / st.g0.clamp_min(1e-6)).clamp(0.0, 1.0)


def approach_ball(env, min_dist: float = 0.5) -> torch.Tensor:
  st = _state(env)
  d = (_ball(env).root_link_pos_w[:, :2] - env.scene["robot"].data.root_link_pos_w[:, :2]).norm(dim=-1)
  st.r0 = torch.where(st.r0 < 1e-6, d, st.r0)
  return ((st.r0 - d) / st.r0.clamp_min(1e-6)).clamp(0.0, 1.0) * (d > min_dist).float()


def kick_toward_goal(env, speed_scale: float = 0.5) -> torch.Tensor:
  b = _ball(env)
  v = b.root_link_lin_vel_w
  ax = _goal_w(env) - b.root_link_pos_w
  ax = ax / ax.norm(dim=-1, keepdim=True).clamp_min(1e-6)
  along = (v * ax).sum(-1)
  return torch.exp((along / speed_scale).clamp(0.0, 2.0)) * (v.norm(dim=-1) > 0.05).float()


def align_stance(env, max_dist: float = 0.75) -> torch.Tensor:
  bp = _ball(env).root_link_pos_w[:, :2]
  rp = env.scene["robot"].data.root_link_pos_w[:, :2]
  r2b, b2g = bp - rp, _goal_w(env)[:, :2] - bp
  n = lambda x: x / x.norm(dim=-1, keepdim=True).clamp_min(1e-6)
  cos = (n(r2b) * n(b2g)).sum(-1).clamp(-1, 1)
  return 0.5 * (cos + 1.0) * (r2b.norm(dim=-1) < max_dist).float()


def face_ball(env, std: float = 0.5) -> torch.Tensor:
  p = ball_pos_b(env)
  bearing = torch.atan2(p[:, 1], p[:, 0])
  return torch.exp(-(bearing**2) / std**2)


def failure_terminated(env, term_names: tuple[str, ...] = ("fell",)) -> torch.Tensor:
  """Failure terminations only: mdp.is_terminated would also charge the success (goal) termination."""
  tm = env.termination_manager
  return torch.stack([tm.get_term(n) for n in term_names]).any(0).float()


def residual_near_ball(env, near: float = 0.8, far: float = 1.2) -> torch.Tensor:
  """1 within near metres of the ball, 0 beyond far: the frozen base alone walks up to the ball."""
  d = (_ball(env).root_link_pos_w[:, :2] - env.scene["robot"].data.root_link_pos_w[:, :2]).norm(dim=-1)
  return ((far - d) / (far - near)).clamp(0.0, 1.0)


def reset_ball(env, env_ids: torch.Tensor) -> None:
  """OmniReset families: at_ball_shoot 50 %, stand_ready 30 %, walk_up 20 % (ball relative to the robot origin)."""
  n, dev = len(env_ids), env.device
  r = torch.rand(n, device=dev)
  pos = torch.zeros(n, 3, device=dev)
  pos[:, 2] = BALL_RADIUS + 0.01
  shoot, walk = r < 0.5, r >= 0.8
  k = int(shoot.sum())
  pos[shoot, 0] = 0.3 + 0.3 * torch.rand(k, device=dev)
  pos[shoot, 1] = 0.10 * (2 * torch.rand(k, device=dev) - 1)
  k = int(walk.sum())
  pos[walk, 0] = 2.0 + 1.5 * torch.rand(k, device=dev)
  pos[walk, 1] = 2.0 * (torch.rand(k, device=dev) - 0.5)
  ball = env.scene["ball"]
  st = ball.data.default_root_state[env_ids].clone()
  st[:, :3] = pos + env.scene.env_origins[env_ids]
  st[:, 7:] = 0.0
  ball.write_root_state_to_sim(st, env_ids=env_ids)
  s = _state(env)
  s.g0[env_ids] = (pos - torch.tensor(GOAL_POS, device=dev)).norm(dim=-1)
  s.r0[env_ids] = pos[:, :2].norm(dim=-1)


def kick_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  cfg = k1_velocity_env_cfg(rough=False, play=play)
  cfg.scene.entities = {**cfg.scene.entities, "ball": EntityCfg(
    spec_fn=ball_spec, init_state=EntityCfg.InitialStateCfg(pos=(0.5, 0.0, BALL_RADIUS + 0.01)))}
  cfg.sim.mujoco.ls_iterations = int(os.environ.get("K1_LS_ITERS", "50"))
  cfg.commands = {}
  cfg.curriculum = {}
  cfg.episode_length_s = 15.0

  residual = os.environ.get("K1_KICK_RESIDUAL", "0") == "1"
  cfg.actions = {
    "base": FrozenBaseActionCfg(
      entity_name="robot", checkpoint=BASE_CKPT, residual=residual,
      residual_gate=residual_near_ball if os.environ.get("K1_KICK_GATE", "0") == "1" else None,
      cmd_alpha=float(os.environ.get("K1_KICK_CMD_ALPHA", "1.0"))),
    "upper_body": RandomUpperBodyActionCfg(entity_name="robot"),
  }

  terms = {
    "gravity": ObservationTermCfg(func=mdp.projected_gravity),
    "ang_vel": ObservationTermCfg(func=mdp.builtin_sensor, params={"sensor_name": "robot/imu_ang_vel"}),
    "ball_pos": ObservationTermCfg(func=ball_pos_b),
    "ball_vel": ObservationTermCfg(func=ball_vel_b),
    "goal_pos": ObservationTermCfg(func=goal_pos_b),
    "last_action": ObservationTermCfg(func=mdp.last_action),
  }
  cfg.observations = {
    "actor": ObservationGroupCfg(dict(terms), enable_corruption=False),
    "critic": ObservationGroupCfg(dict(terms), enable_corruption=False),
  }

  cfg.events = {k: v for k, v in cfg.events.items() if k in ("reset_base", "reset_robot_joints", "foot_friction")}
  cfg.events["reset_ball"] = EventTermCfg(func=reset_ball, mode="reset")

  cfg.rewards = {
    "alive": RewardTermCfg(func=mdp.is_alive, weight=0.5),
    "termination": RewardTermCfg(func=failure_terminated, weight=-50.0),
    "flat_orientation": RewardTermCfg(
      func=mdp.flat_orientation_l2, weight=-1.0, params={"asset_cfg": SceneEntityCfg("robot", body_names=("Trunk",))}),
    "action_rate": RewardTermCfg(func=mdp.action_rate_l2, weight=-0.01),
    "goal_scored": RewardTermCfg(func=goal_scored, weight=50.0),
    "progress": RewardTermCfg(func=ball_to_goal_progress, weight=1.0),
    "approach": RewardTermCfg(func=approach_ball, weight=1.0),
    "kick_toward_goal": RewardTermCfg(func=kick_toward_goal, weight=2.0),
    "align_stance": RewardTermCfg(func=align_stance, weight=0.5),
    "face_ball": RewardTermCfg(func=face_ball, weight=0.3),
  }
  cfg.terminations = {
    "time_out": TerminationTermCfg(func=mdp.time_out, time_out=True),
    "fell": TerminationTermCfg(func=mdp.bad_orientation, params={"limit_angle": float(os.environ.get("K1_FELL_ANGLE", "0.8"))}),
    "goal_scored": TerminationTermCfg(func=goal_scored),
  }
  return cfg
