"""Two-robot box push (MAPPO, parameter sharing): two K1s push one long box to a goal with their HANDS.

Per agent (11) = (vx, vy, wz) -> its own frozen walk base | 8 arm offsets (0.5 rad), exactly the single-robot
push layer; one shared actor (marl.SharedActorModel) runs on each robot's egocentric obs, the critic sees both.
Box 1.2 m wide x 0.5 m deep x 0.4 m, BOX_MASS kg: heavier than one robot pushes comfortably, wide enough for two.
Collisions: every robot geom is solid against the ground, the box and the other robot, never against itself
(contype/conaffinity bits per robot). Hands only: any non-hand body touching the box is penalised per step
(HAND_ONLY_W) and counted in Metrics/body_touch; hand contact on the box is rewarded.
Team reward (shared): hands near/on the box, box progress to the goal, box near goal, success, failure.
"""
from __future__ import annotations

import math
import os
from dataclasses import replace

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
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.tasks.velocity import mdp
from mjlab.utils.lab_api.math import quat_apply, quat_apply_inverse, yaw_quat
from mjlab.utils.spec_config import CollisionCfg

from .env_cfg import k1_velocity_env_cfg
from .frozen_base import FrozenBaseActionCfg
from k1_common.k1_constants import _FOOT, ARM_REGEX, HAND_BODIES, get_getup_spec, get_k1_manip_robot_cfg
from .tasks_kick import BASE_CKPT, failure_terminated
from .upper_body import RandomUpperBodyActionCfg

AGENTS = ("robot_a", "robot_b")
SPAWN_Y = 0.4                                         # robots at (0, +-SPAWN_Y) facing +x
BOX_HALF = (0.25, 0.6, 0.2)                           # depth (x), width (y), height (z) half sizes
BOX_MASS = float(os.environ.get("K1_MA_BOX_MASS", "8.0"))
BOX_X, BOX_Y, BOX_YAW = (0.75, 0.95), (-0.1, 0.1), (-0.2, 0.2)
GOAL_D, GOAL_ANGLE = (0.5, 1.5), (-0.5, 0.5)
SUCCESS_R, OOB_R = 0.2, 3.5
HAND_ONLY_W = float(os.environ.get("K1_MA_HAND_ONLY_W", "1.0"))
FAILURES = ("fell_a", "fell_b", "box_oob", "box_tip")


def box_spec() -> mujoco.MjSpec:
  spec = mujoco.MjSpec()
  body = spec.worldbody.add_body(name="box")
  body.add_freejoint(name="box_joint")
  body.add_geom(
    name="box_geom", type=mujoco.mjtGeom.mjGEOM_BOX, size=BOX_HALF, mass=BOX_MASS,
    rgba=(0.8, 0.5, 0.2, 1.0), friction=(0.6, 0.005, 0.0001), condim=3, contype=1, conaffinity=1)
  return spec


def robot_cfg(k: int) -> EntityCfg:
  """Manip robot (arms kp 40) with every body geom solid; bit 2<<k so a robot never collides with itself."""
  base = get_k1_manip_robot_cfg()
  bit = 2 << k                                   # a: 2, b: 4
  other = sum(2 << j for j in range(len(AGENTS)) if j < k)  # b collides with a's bit
  col = CollisionCfg(
    geom_names_expr=(_FOOT, r".*_collision\d+"), contype={_FOOT: 0, ".*": bit}, conaffinity={_FOOT: 1, ".*": 1 | other},
    condim=3, priority={_FOOT: 1, ".*": 0}, friction=(0.6,))
  init = replace(base.init_state, pos=(0.0, SPAWN_Y * (1 - 2 * k), base.init_state.pos[2]))
  return replace(base, init_state=init, collisions=(col,), spec_fn=get_getup_spec)


##
# State + helpers
##
def _state(env):
  st = getattr(env, "_ma_state", None)
  if st is None:
    n, dev = env.num_envs, env.device
    st = env._ma_state = type("S", (), {"goal": torch.zeros(n, 3, device=dev), "d_prev": torch.zeros(n, device=dev)})()
  return st


def _box(env):
  return env.scene["box"].data


def _yaw_rel(env, agent: str, v: torch.Tensor) -> torch.Tensor:
  return quat_apply_inverse(yaw_quat(env.scene[agent].data.root_link_quat_w), v)


def _hands_w(env, agent: str) -> torch.Tensor:
  robot = env.scene[agent]
  ids = robot.find_bodies(HAND_BODIES, preserve_order=True)[0]
  return robot.data.body_link_pos_w[:, ids]


def _box_goal_dist(env) -> torch.Tensor:
  return (_box(env).root_link_pos_w[:, :2] - _state(env).goal[:, :2]).norm(dim=-1)


##
# Observations (per agent, egocentric)
##
def box_pos_b(env, agent: str) -> torch.Tensor:
  return _yaw_rel(env, agent, _box(env).root_link_pos_w - env.scene[agent].data.root_link_pos_w)


def box_vel_b(env, agent: str) -> torch.Tensor:
  return _yaw_rel(env, agent, _box(env).root_link_lin_vel_w)


def box_heading_b(env, agent: str) -> torch.Tensor:
  """Box y-axis (its long side) in the agent's yaw frame; sign-ambiguous (180-deg symmetric) -> doubled angle."""
  ax = quat_apply(_box(env).root_link_quat_w, torch.tensor([0.0, 1.0, 0.0], device=env.device).expand(env.num_envs, 3))
  v = _yaw_rel(env, agent, ax)
  a = 2.0 * torch.atan2(v[:, 1], v[:, 0])
  return torch.stack([torch.sin(a), torch.cos(a)], -1)


def goal_pos_b(env, agent: str) -> torch.Tensor:
  return _yaw_rel(env, agent, _state(env).goal - env.scene[agent].data.root_link_pos_w)


def mate_b(env, agent: str) -> torch.Tensor:
  """Teammate trunk position and heading (sin, cos of relative yaw) in the agent's yaw frame."""
  mate = env.scene[AGENTS[1 - AGENTS.index(agent)]].data
  me = env.scene[agent].data
  fwd = quat_apply(mate.root_link_quat_w, torch.tensor([1.0, 0.0, 0.0], device=env.device).expand(env.num_envs, 3))
  f = _yaw_rel(env, agent, fwd)
  return torch.cat([_yaw_rel(env, agent, mate.root_link_pos_w - me.root_link_pos_w), f[:, :2]], -1)


def hands_pos_b(env, agent: str) -> torch.Tensor:
  rel = _hands_w(env, agent) - env.scene[agent].data.root_link_pos_w[:, None]
  return torch.cat([_yaw_rel(env, agent, rel[:, 0]), _yaw_rel(env, agent, rel[:, 1])], -1)


def own_last_action(env, agent: str) -> torch.Tensor:
  am = env.action_manager
  return torch.cat([am.get_term(f"base_{agent}").raw_action, am.get_term(f"arms_{agent}").raw_action], -1)


##
# Rewards / terminations (team)
##
def hands_to_box(env, std: float = 0.2) -> torch.Tensor:
  """exp(-d / std) averaged over all 4 hands; d = hand distance to the box surface."""
  b = _box(env)
  hands = torch.cat([_hands_w(env, a) for a in AGENTS], 1)                     # (n, 4, 3)
  rel = hands - b.root_link_pos_w[:, None]
  q = b.root_link_quat_w[:, None].expand(-1, hands.shape[1], -1)
  local = quat_apply_inverse(q.reshape(-1, 4), rel.reshape(-1, 3)).reshape(hands.shape)
  d = (local.abs() - torch.tensor(BOX_HALF, device=env.device)).clamp_min(0.0).norm(dim=-1)
  return torch.exp(-d / std).mean(-1)


def _touching(env, sensor: str) -> torch.Tensor:
  return (env.scene[sensor].data.found > 0).float()


def hands_on_box(env) -> torch.Tensor:
  """Fraction of the 4 hands in contact with the box."""
  return torch.cat([_touching(env, f"{a}_hands_box") for a in AGENTS], -1).mean(-1)


def body_touch(env) -> torch.Tensor:
  """Number of robots touching the box with anything but a hand (0, 1, 2)."""
  n = sum(_touching(env, f"{a}_body_box").amax(-1) for a in AGENTS)
  env.extras["log"]["Metrics/body_touch"] = n.mean()
  return n


def box_progress_rate(env) -> torch.Tensor:
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


def box_tip(env, limit_deg: float = 30.0) -> torch.Tensor:
  up = quat_apply(_box(env).root_link_quat_w, torch.tensor([0.0, 0.0, 1.0], device=env.device).expand(env.num_envs, 3))
  return up[:, 2] < math.cos(math.radians(limit_deg))


def reset_box(env, env_ids: torch.Tensor) -> None:
  n, dev = len(env_ids), env.device
  u = lambda lo_hi: lo_hi[0] + (lo_hi[1] - lo_hi[0]) * torch.rand(n, device=dev)  # noqa: E731
  pos = torch.stack([u(BOX_X), u(BOX_Y), torch.full((n,), BOX_HALF[2] + 0.005, device=dev)], -1)
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


def push_ma_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  cfg = k1_velocity_env_cfg(rough=False, play=play)
  cfg.scene.entities = {a: robot_cfg(k) for k, a in enumerate(AGENTS)}
  cfg.scene.entities["box"] = EntityCfg(spec_fn=box_spec, init_state=EntityCfg.InitialStateCfg(pos=(0.85, 0.0, BOX_HALF[2] + 0.005)))
  cfg.scene.sensors = tuple(
    ContactSensorCfg(
      name=f"{a}_{part}_box", fields=("found",), num_slots=1,
      primary=ContactMatch(mode="body", pattern=pat, entity=a, exclude=excl),
      secondary=ContactMatch(mode="body", pattern="box", entity="box"))
    for a in AGENTS for part, pat, excl in (("hands", HAND_BODIES, ()), ("body", ".*", HAND_BODIES)))
  cfg.sim.mujoco.ls_iterations = int(os.environ.get("K1_LS_ITERS", "50"))
  cfg.sim.nconmax = 200
  cfg.sim.njmax = 1500
  cfg.commands = {}
  cfg.curriculum = {}
  cfg.episode_length_s = 20.0
  cfg.viewer.entity_name = "robot_a"

  cfg.actions = {}
  for a in AGENTS:  # dict order = action layout: [a: base 3, arms 8 | b: base 3, arms 8]
    cfg.actions[f"base_{a}"] = FrozenBaseActionCfg(entity_name=a, checkpoint=BASE_CKPT)
    cfg.actions[f"arms_{a}"] = JointPositionActionCfg(entity_name=a, actuator_names=ARM_REGEX, scale=0.5, use_default_offset=True)
  for a in AGENTS:
    cfg.actions[f"head_{a}"] = RandomUpperBodyActionCfg(entity_name=a, joint_names=(".*Head.*",), frozen_prob=1.0)

  def agent_terms(a: str) -> dict:
    arms = SceneEntityCfg(a, joint_names=ARM_REGEX)
    p = {"agent": a}
    return {
      f"{a}_gravity": ObservationTermCfg(func=mdp.projected_gravity, params={"asset_cfg": SceneEntityCfg(a)}),
      f"{a}_ang_vel": ObservationTermCfg(func=mdp.builtin_sensor, params={"sensor_name": f"{a}/imu_ang_vel"}),
      f"{a}_box_pos": ObservationTermCfg(func=box_pos_b, params=p),
      f"{a}_box_heading": ObservationTermCfg(func=box_heading_b, params=p),
      f"{a}_box_vel": ObservationTermCfg(func=box_vel_b, params=p),
      f"{a}_goal_pos": ObservationTermCfg(func=goal_pos_b, params=p),
      f"{a}_mate": ObservationTermCfg(func=mate_b, params=p),
      f"{a}_hands_pos": ObservationTermCfg(func=hands_pos_b, params=p),
      f"{a}_arm_pos": ObservationTermCfg(func=mdp.joint_pos_rel, params={"asset_cfg": arms}),
      f"{a}_arm_vel": ObservationTermCfg(func=mdp.joint_vel_rel, params={"asset_cfg": arms}),
      f"{a}_last_action": ObservationTermCfg(func=own_last_action, params=p),
    }

  terms = {k: v for a in AGENTS for k, v in agent_terms(a).items()}  # [agent a block | agent b block]
  cfg.observations = {
    "actor": ObservationGroupCfg(dict(terms), enable_corruption=False),
    "critic": ObservationGroupCfg(dict(terms), enable_corruption=False),
  }

  reset_base, reset_joints = cfg.events["reset_base"], cfg.events["reset_robot_joints"]
  cfg.events = {}
  for a in AGENTS:
    cfg.events[f"reset_base_{a}"] = replace(reset_base, params={
      **reset_base.params, "asset_cfg": SceneEntityCfg(a),
      "pose_range": {"x": (-0.05, 0.05), "y": (-0.05, 0.05), "z": (0.01, 0.03), "yaw": (-0.2, 0.2)}})
    cfg.events[f"reset_joints_{a}"] = replace(reset_joints, params={**reset_joints.params, "asset_cfg": SceneEntityCfg(a)})
  cfg.events["reset_box"] = EventTermCfg(func=reset_box, mode="reset")

  cfg.rewards = {
    "alive": RewardTermCfg(func=mdp.is_alive, weight=0.2),
    "hands_to_box": RewardTermCfg(func=hands_to_box, weight=1.0),
    "hands_on_box": RewardTermCfg(func=hands_on_box, weight=1.0),
    "body_touch": RewardTermCfg(func=body_touch, weight=-HAND_ONLY_W),
    "box_progress": RewardTermCfg(func=box_progress_rate, weight=5.0),
    "box_near_goal": RewardTermCfg(func=box_near_goal, weight=2.0),
    "success": RewardTermCfg(func=goal_reached, weight=500.0),
    "failure": RewardTermCfg(func=failure_terminated, weight=-500.0, params={"term_names": FAILURES}),
    "action_rate": RewardTermCfg(func=mdp.action_rate_l2, weight=-0.005),
  }
  for a in AGENTS:
    cfg.rewards[f"flat_{a}"] = RewardTermCfg(
      func=mdp.flat_orientation_l2, weight=-1.0, params={"asset_cfg": SceneEntityCfg(a, body_names=("Trunk",))})
    cfg.rewards[f"arm_limits_{a}"] = RewardTermCfg(
      func=mdp.joint_pos_limits, weight=-1.0, params={"asset_cfg": SceneEntityCfg(a, joint_names=ARM_REGEX)})

  fell = float(os.environ.get("K1_FELL_ANGLE", "0.8"))
  cfg.terminations = {
    "time_out": TerminationTermCfg(func=mdp.time_out, time_out=True),
    "fell_a": TerminationTermCfg(func=mdp.bad_orientation, params={"limit_angle": fell, "asset_cfg": SceneEntityCfg("robot_a")}),
    "fell_b": TerminationTermCfg(func=mdp.bad_orientation, params={"limit_angle": fell, "asset_cfg": SceneEntityCfg("robot_b")}),
    "goal_reached": TerminationTermCfg(func=goal_reached),
    "box_oob": TerminationTermCfg(func=box_oob),
    "box_tip": TerminationTermCfg(func=box_tip),
  }
  return cfg
