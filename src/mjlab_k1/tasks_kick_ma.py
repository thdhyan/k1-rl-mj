"""Two-robot soccer kick (Mjlab-KickMA-Booster-K1): robot A passes to robot B, B chases and shoots at the goal centre.

Multi-agent (MAPPO, parameter sharing, marl.SharedActorModel) on a soccer pitch: green ground, white lines, goal frame
at +x (posts and bar are solid, the goal itself is the analytic volume of tasks_kick.goal_scored).

Per agent, egocentric, 30 dims = exactly the Mjlab-KickDir observation (gravity, ang vel, ball pos, ball vel, goal
pos, last action), so kick_dir_v1 loads into the shared actor unchanged. Only the goal point differs by role:
  robot_a: goal = pass point (where robot_b stands at reset, +-PASS_NOISE)
  robot_b: goal = goal centre
Action per agent 15 = (vx, vy, wz) + 12 leg residuals -> frozen walk base (walk_still_v4), as kick_dir_v1.
Upper body: random arms; head is the ball bbox servo (ball_gaze.py) on both robots.

Phases (latched per env, see _update_phase):
  pass  (switched = False): robot_b is HELD in place (its command is gated to 0, residual off). Team reward for the pass:
        ball progress to the pass point, ball heading on the pass point, one-shot when the ball first reaches it.
  shoot (switched = True): robot_b switches to chase and kick: command and residual on, its goal is the goal centre.
        Team reward for the shot: ball progress to goal, robot_b closing on the ball, kick toward goal, goal bonus.
  Switch when the ball leaves towards robot_b (speed > SWITCH_SPEED, heading within SWITCH_TOWARD), or passes within
  SWITCH_NEAR of it, or after SWITCH_T seconds.

robot_a release (latched): once it has kicked (ball moving with it within KICK_REACH) and the ball is RELEASE_DIST away, its
command and residual fade to zero over RELEASE_FADE_S, then the frozen walk base alone balances it for the rest of the episode.

Known limits: feet-only collisions (as kick_dir_v1), so the robots pass through each other; robot_b's held actions still
enter the PPO log-probs (zero-mean noise, no reward effect).
"""
from __future__ import annotations

import math
import os
from dataclasses import replace
from pathlib import Path

import mujoco
import torch

from mjlab.entity import EntityCfg
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.metrics_manager import MetricsTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.tasks.velocity import mdp
from mjlab.utils.lab_api.math import quat_apply_inverse, yaw_quat

from .ball_gaze import attach_ball_gaze
from .env_cfg import k1_velocity_env_cfg
from .frozen_base import FrozenBaseActionCfg
from mjlab_k1.k1_constants import get_k1_robot_cfg
from .tasks_kick import BALL_RADIUS, GOAL_H, GOAL_POS, GOAL_W, ball_spec, failure_terminated
from .upper_body import RandomUpperBodyActionCfg

AGENTS = ("robot_a", "robot_b")
BASE_CKPT = os.environ.get(
  "K1_KMA_BASE", str(Path(__file__).resolve().parents[3] / "models/mjlab_k1/walk_still_v4/model_18996.pt"))
CMD_ALPHA = float(os.environ.get("K1_KMA_CMD_ALPHA", "0.3"))  # kick_dir_v1 was trained with 0.3

PITCH_X, PITCH_Y = 4.0, 3.0                  # touch lines and goal line (+x), m, half sizes
OOB_X = 4.3                                  # ball past this without a goal is out
LINE_W, LINE_Z = 0.05, 0.002
A_SPOT, B_SPOT = (-1.5, 0.0), (0.8, 0.5)     # robot start points, m (world, env origin added at reset)
PASS_NOISE = 0.3                             # pass point jitter, m
BALL_AHEAD = (0.35, 1.5)                     # ball distance from robot_a at reset, m
A_X, A_Y = (-3.3, 0.5), 2.3                  # robot_a spawn range, m
B_X, B_Y = (0.3, 3.5), 2.7                   # robot_b spawn range, goal half of the pitch, m
PASS_RANGE = (1.5, 4.5)                      # robot_b distance from robot_a, m
B_BALL_MIN = 1.5                             # robot_b distance from the ball at reset, m (past SWITCH_NEAR: a pass is needed)
B_YAW_STD = math.radians(30.0)               # robot_b facing about the direction to the goal centre, rad
KICK_REACH, RELEASE_DIST = 0.7, 1.0          # robot_a release, m
RELEASE_FADE_S = 1.0                         # robot_a command and residual fade after release, s
RECEIVE_R = 1.0                              # ball within this of the pass point = received
SWITCH_T, SWITCH_NEAR, SWITCH_SPEED, SWITCH_TOWARD = 6.0, 1.0, 0.3, 0.3
MOVING_MPS = 0.3


def _pitch_spec() -> mujoco.MjSpec:
  """Visual pitch: green slab, white touch/half-way/centre-circle/penalty/goal-area lines, goal frames (solid)."""
  spec = mujoco.MjSpec()
  body = spec.worldbody.add_body(name="pitch")
  white, green = (1.0, 1.0, 1.0, 1.0), (0.16, 0.5, 0.18, 1.0)

  def box(name, pos, half, rgba=white, solid=False):
    body.add_geom(name=name, type=mujoco.mjtGeom.mjGEOM_BOX, pos=pos, size=half, rgba=rgba,
                  contype=1 if solid else 0, conaffinity=1 if solid else 0)

  def rect(tag, x0, x1, y0, y1):  # outline of an axis-aligned rectangle
    w = LINE_W / 2
    box(f"{tag}_n", ((x0 + x1) / 2, y1, LINE_Z), ((x1 - x0) / 2 + w, w, LINE_Z))
    box(f"{tag}_s", ((x0 + x1) / 2, y0, LINE_Z), ((x1 - x0) / 2 + w, w, LINE_Z))
    box(f"{tag}_e", (x1, (y0 + y1) / 2, LINE_Z), (w, (y1 - y0) / 2 + w, LINE_Z))
    box(f"{tag}_w", (x0, (y0 + y1) / 2, LINE_Z), (w, (y1 - y0) / 2 + w, LINE_Z))

  box("grass", (0.0, 0.0, 0.0005), (PITCH_X + 0.6, PITCH_Y + 0.6, 0.0005), green)
  rect("touch", -PITCH_X, PITCH_X, -PITCH_Y, PITCH_Y)
  box("halfway", (0.0, 0.0, LINE_Z), (LINE_W / 2, PITCH_Y, LINE_Z))
  seg, r, n = 2 * math.pi * 0.75 / 24, 0.75, 24          # centre circle, 24 tangent dashes
  for i in range(n):
    th = 2 * math.pi * i / n
    body.add_geom(name=f"circle_{i}", type=mujoco.mjtGeom.mjGEOM_BOX, pos=(r * math.cos(th), r * math.sin(th), LINE_Z),
                  quat=(math.cos(th / 2), 0.0, 0.0, math.sin(th / 2)), size=(LINE_W / 2, seg / 2, LINE_Z),
                  rgba=white, contype=0, conaffinity=0)
  for sx in (-1.0, 1.0):  # penalty and goal areas, both ends
    x_far = sx * PITCH_X
    rect(f"pen_{sx:+.0f}", min(x_far, sx * 3.0), max(x_far, sx * 3.0), -1.6, 1.6)
    rect(f"goal_area_{sx:+.0f}", min(x_far, sx * 3.5), max(x_far, sx * 3.5), -1.0, 1.0)
  # goal frame at +x: solid posts and bar (ball and robots bounce off), the goal volume itself is analytic
  box("post_l", (PITCH_X, GOAL_W / 2, GOAL_H / 2), (0.03, 0.03, GOAL_H / 2), (0.95, 0.95, 0.95, 1.0), solid=True)
  box("post_r", (PITCH_X, -GOAL_W / 2, GOAL_H / 2), (0.03, 0.03, GOAL_H / 2), (0.95, 0.95, 0.95, 1.0), solid=True)
  box("bar", (PITCH_X, 0.0, GOAL_H), (0.03, GOAL_W / 2, 0.03), (0.95, 0.95, 0.95, 1.0), solid=True)
  return spec


##
# State (latched phase, pass point, reward memory)
##
class _S:
  def __init__(self, env):
    n, dev = env.num_envs, env.device
    z = lambda: torch.zeros(n, device=dev)
    b = lambda: torch.zeros(n, dtype=torch.bool, device=dev)
    self.pass_w = torch.zeros(n, 3, device=dev)   # world point robot_a passes to
    self.switched, self.received, self.prev_received = b(), b(), b()
    self.a_kicked, self.a_released = b(), b()
    self.a_release_step = z()
    self.d_pass, self.d_goal, self.d_chase, self.d_b = z(), z(), z(), z()


def _st(env) -> _S:
  st = getattr(env, "_kma_state", None)
  if st is None:
    st = env._kma_state = _S(env)
  return st


def _ball(env):
  return env.scene["ball"].data


def _pos(env, agent: str) -> torch.Tensor:
  return env.scene[agent].data.root_link_pos_w


def _yaw_rel(env, agent: str, vec_w: torch.Tensor) -> torch.Tensor:
  return quat_apply_inverse(yaw_quat(env.scene[agent].data.root_link_quat_w), vec_w)


def _goal_w(env) -> torch.Tensor:
  return torch.tensor(GOAL_POS, device=env.device) + env.scene.env_origins


def _update_phase(env) -> None:
  """Latch the switch and the receive flags. Idempotent within a step, so gates and rewards may both call it."""
  st, b = _st(env), _ball(env)
  to = (_pos(env, "robot_b") - b.root_link_pos_w)[:, :2]
  dist = to.norm(dim=-1).clamp_min(1e-6)
  v = b.root_link_lin_vel_w[:, :2]
  toward = (v * (to / dist[:, None])).sum(-1)
  passed = (v.norm(dim=-1) > SWITCH_SPEED) & (toward > SWITCH_TOWARD)
  timeout = env.episode_length_buf.float() * env.step_dt >= SWITCH_T
  st.switched = st.switched | passed | (dist < SWITCH_NEAR) | timeout
  st.received = st.received | ((b.root_link_pos_w[:, :2] - st.pass_w[:, :2]).norm(dim=-1) < RECEIVE_R)
  d_a = (b.root_link_pos_w - _pos(env, "robot_a"))[:, :2].norm(dim=-1)
  st.a_kicked = st.a_kicked | ((v.norm(dim=-1) > MOVING_MPS) & (d_a < KICK_REACH))
  new = st.a_kicked & (d_a > RELEASE_DIST) & ~st.a_released
  st.a_release_step[new] = env.episode_length_buf[new].float()
  st.a_released = st.a_released | new


def _gate_b(env) -> torch.Tensor:
  _update_phase(env)
  return _st(env).switched.float()


def _near_ball(env, agent: str, near: float = 0.8, far: float = 1.2) -> torch.Tensor:
  """kick_dir's residual gate, per agent: 1 within near m of the ball, 0 beyond far."""
  d = (_ball(env).root_link_pos_w[:, :2] - _pos(env, agent)[:, :2]).norm(dim=-1)
  return ((far - d) / (far - near)).clamp(0.0, 1.0)


def _gate_a(env) -> torch.Tensor:
  """1 until robot_a's release, then a linear fade to 0 over RELEASE_FADE_S."""
  _update_phase(env)
  st = _st(env)
  fade = 1.0 - (env.episode_length_buf.float() - st.a_release_step) * env.step_dt / RELEASE_FADE_S
  return torch.where(st.a_released, fade.clamp(0.0, 1.0), torch.ones_like(fade))


def _res_gate_a(env) -> torch.Tensor:
  return _gate_a(env) * _near_ball(env, "robot_a")


def _res_gate_b(env) -> torch.Tensor:
  return _gate_b(env) * _near_ball(env, "robot_b")


##
# Observations (per agent, egocentric; same order and width as Mjlab-KickDir)
##
def obs_gravity(env, agent: str) -> torch.Tensor:
  return mdp.projected_gravity(env, SceneEntityCfg(agent))


def obs_ball_pos(env, agent: str) -> torch.Tensor:
  return _yaw_rel(env, agent, _ball(env).root_link_pos_w - _pos(env, agent))


def obs_ball_vel(env, agent: str) -> torch.Tensor:
  return _yaw_rel(env, agent, _ball(env).root_link_lin_vel_w)


def obs_goal(env, agent: str) -> torch.Tensor:
  st = _st(env)
  target = st.pass_w if agent == "robot_a" else _goal_w(env)
  return _yaw_rel(env, agent, target - _pos(env, agent))


def obs_last(env, agent: str) -> torch.Tensor:
  return env.action_manager.get_term(f"base_{agent}").raw_action


def obs_phase(env) -> torch.Tensor:
  """Critic only: switched, received."""
  _update_phase(env)
  st = _st(env)
  return torch.stack([st.switched.float(), st.received.float()], -1)


def obs_ball_abs(env) -> torch.Tensor:
  return _ball(env).root_link_pos_w - env.scene.env_origins


##
# Rewards, metrics, terminations (team)
##
def _rate(prev: torch.Tensor, now: torch.Tensor, dt: float) -> torch.Tensor:
  return ((prev - now) / dt).clamp(-1.0, 1.0)


def pass_progress(env) -> torch.Tensor:
  """Pre-switch: ball distance to the pass point shrinking (per second, clipped)."""
  st, b = _st(env), _ball(env)
  d = (st.pass_w[:, :2] - b.root_link_pos_w[:, :2]).norm(dim=-1)
  r = _rate(st.d_pass, d, env.step_dt) * (~st.switched).float()
  st.d_pass = d
  return r


def pass_heading(env, std: float = 0.2) -> torch.Tensor:
  """Pre-switch, ball rolling: velocity lined up with the pass point."""
  st, b = _st(env), _ball(env)
  v, to = b.root_link_lin_vel_w[:, :2], (st.pass_w - b.root_link_pos_w)[:, :2]
  err = torch.atan2(v[:, 0] * to[:, 1] - v[:, 1] * to[:, 0], (v * to).sum(-1))
  moving = (v.norm(dim=-1) > MOVING_MPS).float()
  return torch.exp(-(err / std) ** 2) * moving * (~st.switched).float()


def pass_received(env) -> torch.Tensor:
  """One-shot on the step the ball first comes within RECEIVE_R of the pass point."""
  st = _st(env)
  r = (st.received & ~st.prev_received).float()
  st.prev_received = st.received.clone()
  return r


def robot_a_approach(env) -> torch.Tensor:
  """Pre-switch: robot_a closing on the ball."""
  d = (_ball(env).root_link_pos_w[:, :2] - _pos(env, "robot_a")[:, :2]).norm(dim=-1)
  st = _st(env)
  r = _rate(st.d_chase, d, env.step_dt) * (~st.switched & ~st.a_released).float()
  st.d_chase = d
  return r


def chase_approach(env) -> torch.Tensor:
  """Post-switch: robot_b closing on the ball."""
  st = _st(env)
  d = (_ball(env).root_link_pos_w[:, :2] - _pos(env, "robot_b")[:, :2]).norm(dim=-1)
  r = _rate(st.d_b, d, env.step_dt) * st.switched.float()
  st.d_b = d
  return r


def goal_progress(env) -> torch.Tensor:
  """Post-switch: ball distance to the goal centre shrinking."""
  st, b = _st(env), _ball(env)
  d = (_goal_w(env)[:, :2] - b.root_link_pos_w[:, :2]).norm(dim=-1)
  r = _rate(st.d_goal, d, env.step_dt) * st.switched.float()
  st.d_goal = d
  return r


def kick_toward_goal(env, speed_scale: float = 0.5) -> torch.Tensor:
  """Post-switch, ball rolling: velocity towards the goal centre (kick_dir-style exp reward)."""
  b = _ball(env)
  v = b.root_link_lin_vel_w
  ax = _goal_w(env) - b.root_link_pos_w
  ax = ax / ax.norm(dim=-1, keepdim=True).clamp_min(1e-6)
  along = (v * ax).sum(-1)
  return torch.exp((along / speed_scale).clamp(0.0, 2.0)) * (v.norm(dim=-1) > 0.05).float() * _st(env).switched.float()


def goal_scored(env) -> torch.Tensor:
  rel = _ball(env).root_link_pos_w - _goal_w(env)
  return (rel[:, 0] > -0.1) & (rel[:, 0] < 0.2) & (rel[:, 1].abs() < GOAL_W / 2) & (rel[:, 2].abs() < GOAL_H / 2)


def ball_oob(env) -> torch.Tensor:
  p = _ball(env).root_link_pos_w - env.scene.env_origins
  out = (p[:, 0].abs() > OOB_X) | (p[:, 1].abs() > PITCH_Y)
  return out & ~goal_scored(env)


def _metric_pass(env) -> torch.Tensor:
  return _st(env).received.float()


def _metric_switch(env) -> torch.Tensor:
  _update_phase(env)
  return _st(env).switched.float()


def _metric_goal(env) -> torch.Tensor:
  return goal_scored(env).float()


##
# Resets
##
def _place_robot(env, agent: str, env_ids: torch.Tensor, xy: torch.Tensor, yaw: torch.Tensor) -> None:
  e = env.scene[agent]
  st13 = e.data.default_root_state[env_ids].clone()
  st13[:, :2] = xy
  st13[:, 2] += env.scene.env_origins[env_ids, 2]
  st13[:, 3:7] = torch.stack([(yaw / 2).cos(), 0 * yaw, 0 * yaw, (yaw / 2).sin()], -1)
  st13[:, 7:] = 0.0
  e.write_root_state_to_sim(st13, env_ids=env_ids)


def reset_kick_ma(env, env_ids: torch.Tensor) -> None:
  """Random layout on the pitch: robot_a in the back part with any facing, the ball near it at any bearing, robot_b in the
  goal half PASS_RANGE from robot_a and clear of the ball, facing the goal centre +- a B_YAW_STD gaussian, pass point at
  robot_b's spot (+-PASS_NOISE). Goal stays fixed."""
  n, dev = len(env_ids), env.device
  st, origin = _st(env), env.scene.env_origins[env_ids]
  u = lambda lo, hi, k=n: lo + (hi - lo) * torch.rand(k, device=dev)
  a0 = torch.stack([u(*A_X), u(-A_Y, A_Y)], -1)
  # ball around robot_a, any bearing
  th, dist = u(-math.pi, math.pi), u(*BALL_AHEAD)
  xy = a0 + dist[:, None] * torch.stack([th.cos(), th.sin()], -1)
  ball0 = torch.stack([xy[:, 0].clamp(-PITCH_X + 0.3, PITCH_X - 0.5), xy[:, 1].clamp(-PITCH_Y + 0.3, PITCH_Y - 0.3)], -1)
  # robot_b: uniform in the goal half, resampled until it is in pass range and clear of the ball
  b0 = torch.stack([u(*B_X), u(-B_Y, B_Y)], -1)
  for _ in range(32):
    d = (b0 - a0).norm(dim=-1)
    bad = (d < PASS_RANGE[0]) | (d > PASS_RANGE[1]) | ((b0 - ball0).norm(dim=-1) < B_BALL_MIN)
    if not bad.any():
      break
    k = int(bad.sum())
    b0[bad] = torch.stack([u(*B_X, k), u(-B_Y, B_Y, k)], -1)
  to_goal = torch.tensor(GOAL_POS[:2], device=dev) - b0
  yaw_b = torch.atan2(to_goal[:, 1], to_goal[:, 0]) + B_YAW_STD * torch.randn(n, device=dev)
  _place_robot(env, "robot_a", env_ids, a0 + origin[:, :2], u(-math.pi, math.pi))
  _place_robot(env, "robot_b", env_ids, b0 + origin[:, :2], yaw_b)
  ptx = b0 + (2 * torch.rand(n, 2, device=dev) - 1) * PASS_NOISE
  st.pass_w[env_ids, :2] = ptx + origin[:, :2]
  st.pass_w[env_ids, 2] = origin[:, 2] + GOAL_POS[2]
  xy = ball0 + origin[:, :2]
  st13 = env.scene["ball"].data.default_root_state[env_ids].clone()
  st13[:, :2] = xy
  st13[:, 2] = BALL_RADIUS + 0.01 + origin[:, 2]
  st13[:, 3:7] = torch.tensor((1.0, 0.0, 0.0, 0.0), device=dev)
  st13[:, 7:] = 0.0
  env.scene["ball"].write_root_state_to_sim(st13, env_ids=env_ids)
  for flag in (st.switched, st.received, st.prev_received, st.a_kicked, st.a_released):
    flag[env_ids] = False
  b = st13[:, :2]
  st.d_pass[env_ids] = (st.pass_w[env_ids, :2] - b).norm(dim=-1)
  st.d_goal[env_ids] = (_goal_w(env)[env_ids, :2] - b).norm(dim=-1)
  st.d_chase[env_ids] = (b - (a0 + origin[:, :2])).norm(dim=-1)
  st.d_b[env_ids] = (b - (b0 + origin[:, :2])).norm(dim=-1)


##
# Env cfg
##
def kick_ma_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  cfg = k1_velocity_env_cfg(rough=False, play=play)
  base = get_k1_robot_cfg()
  robots = {}
  for agent, spot in zip(AGENTS, (A_SPOT, B_SPOT)):
    init = replace(base.init_state, pos=(spot[0], spot[1], base.init_state.pos[2]))
    robots[agent] = replace(base, init_state=init)
  cfg.scene.entities = {**robots, "ball": EntityCfg(spec_fn=ball_spec, init_state=EntityCfg.InitialStateCfg(
    pos=(A_SPOT[0] + 0.5, A_SPOT[1], BALL_RADIUS + 0.01))), "pitch": EntityCfg(spec_fn=_pitch_spec)}
  cfg.scene.sensors = ()
  cfg.sim.mujoco.ls_iterations = int(os.environ.get("K1_LS_ITERS", "50"))
  cfg.sim.nconmax = 200
  cfg.sim.njmax = 1500
  cfg.commands = {}
  cfg.curriculum = {}
  cfg.episode_length_s = 15.0
  cfg.viewer.entity_name = "robot_b"

  cfg.actions = {}
  for agent in AGENTS:  # dict order = action layout: [a: base 15 | b: base 15]
    gate_cmd, gate_res = (_gate_a, _res_gate_a) if agent == "robot_a" else (_gate_b, _res_gate_b)
    cfg.actions[f"base_{agent}"] = FrozenBaseActionCfg(
      entity_name=agent, checkpoint=BASE_CKPT, residual=True, residual_gate=gate_res, cmd_gate=gate_cmd,
      cmd_alpha=CMD_ALPHA)
    cfg.actions[f"upper_{agent}"] = RandomUpperBodyActionCfg(entity_name=agent)
    attach_ball_gaze(cfg, entity_name=agent, upper=f"upper_{agent}", term=f"gaze_{agent}")

  def agent_terms(agent: str) -> dict:
    return {
      f"{agent}_gravity": ObservationTermCfg(func=obs_gravity, params={"agent": agent}),
      f"{agent}_ang_vel": ObservationTermCfg(func=mdp.builtin_sensor, params={"sensor_name": f"{agent}/imu_ang_vel"}),
      f"{agent}_ball_pos": ObservationTermCfg(func=obs_ball_pos, params={"agent": agent}),
      f"{agent}_ball_vel": ObservationTermCfg(func=obs_ball_vel, params={"agent": agent}),
      f"{agent}_goal_pos": ObservationTermCfg(func=obs_goal, params={"agent": agent}),
      f"{agent}_last_action": ObservationTermCfg(func=obs_last, params={"agent": agent}),
    }

  actor = {k: v for a in AGENTS for k, v in agent_terms(a).items()}  # [a block | b block], kick_dir order
  critic = {**actor, "phase": ObservationTermCfg(func=obs_phase), "ball_abs": ObservationTermCfg(func=obs_ball_abs)}
  cfg.observations = {
    "actor": ObservationGroupCfg(actor, enable_corruption=False),
    "critic": ObservationGroupCfg(critic, enable_corruption=False),
  }

  reset_base, reset_joints = cfg.events["reset_base"], cfg.events["reset_robot_joints"]
  cfg.events = {}
  for agent in AGENTS:
    cfg.events[f"reset_base_{agent}"] = replace(reset_base, params={
      **reset_base.params, "asset_cfg": SceneEntityCfg(agent),
      "pose_range": {"x": (-0.05, 0.05), "y": (-0.05, 0.05), "z": (0.01, 0.03), "yaw": (-0.1, 0.1)}})
    cfg.events[f"reset_joints_{agent}"] = replace(reset_joints, params={**reset_joints.params, "asset_cfg": SceneEntityCfg(agent)})
  cfg.events["reset_kick_ma"] = EventTermCfg(func=reset_kick_ma, mode="reset")  # after the robots: fixes ball and phases

  cfg.rewards = {
    "alive": RewardTermCfg(func=mdp.is_alive, weight=0.5),
    "pass_progress": RewardTermCfg(func=pass_progress, weight=2.0),
    "pass_heading": RewardTermCfg(func=pass_heading, weight=1.0),
    "pass_received": RewardTermCfg(func=pass_received, weight=30.0),
    "robot_a_approach": RewardTermCfg(func=robot_a_approach, weight=1.0),
    "chase_approach": RewardTermCfg(func=chase_approach, weight=1.0),
    "goal_progress": RewardTermCfg(func=goal_progress, weight=2.0),
    "kick_toward_goal": RewardTermCfg(func=kick_toward_goal, weight=1.0),
    "goal": RewardTermCfg(func=goal_scored, weight=100.0),
    "failure": RewardTermCfg(func=failure_terminated, weight=-50.0, params={"term_names": ("fell_a", "fell_b", "ball_oob")}),
    "action_rate": RewardTermCfg(func=mdp.action_rate_l2, weight=-0.01),
  }
  for agent in AGENTS:
    cfg.rewards[f"flat_{agent}"] = RewardTermCfg(
      func=mdp.flat_orientation_l2, weight=-1.0, params={"asset_cfg": SceneEntityCfg(agent, body_names=("Trunk",))})

  cfg.terminations = {
    "time_out": TerminationTermCfg(func=mdp.time_out, time_out=True),
    "fell_a": TerminationTermCfg(func=mdp.bad_orientation, params={"limit_angle": float(os.environ.get("K1_FELL_ANGLE", "0.8")), "asset_cfg": SceneEntityCfg("robot_a")}),
    "fell_b": TerminationTermCfg(func=mdp.bad_orientation, params={"limit_angle": float(os.environ.get("K1_FELL_ANGLE", "0.8")), "asset_cfg": SceneEntityCfg("robot_b")}),
    "goal_scored": TerminationTermCfg(func=goal_scored),
    "ball_oob": TerminationTermCfg(func=ball_oob),
  }
  cfg.metrics = {
    "Metrics/received": MetricsTermCfg(func=_metric_pass),
    "Metrics/switched": MetricsTermCfg(func=_metric_switch),
    "Metrics/goal": MetricsTermCfg(func=_metric_goal),
  }
  return cfg
