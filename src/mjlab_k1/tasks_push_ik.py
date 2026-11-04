"""Two K1s push a box to a goal POSE, with only what the PhaseSpace caps give (MAPPO, shared actor).

Per agent, egocentric, 25 dims x 5 history steps, delayed + noisy like the mocap:
  box xy, box long axis (sin, cos of the doubled angle: 180-deg symmetric) | goal xy, goal axis | mate xy, mate heading |
  box half sizes | own last action. No gravity, joints or hands: the frozen walk base owns balance.
Per agent action (10): vx, vy, wz, trunk height -> frozen walk base | left + right hand xyz goal, trunk frame -> wbc.ArmIK.
Critic: same + true hand positions and box velocity. Reset ranges widen while the success rate stays above PROG_GATE: behind a face, close
goal, small turn -> anywhere around the box, facing anywhere, goal up to 2.5 m and any box yaw. Sizes, mass, friction are
per world. Goal shaping is on the box's top-view corners (mean corner distance to the goal corners). Success: box within
SUCCESS_R and SUCCESS_YAW, slow. Non-hand contact with the box or the other robot is
penalised; box progress pays only while a hand is on the box and nothing else is.
"""
from __future__ import annotations

import math
import os
from dataclasses import dataclass, replace

import torch

from mjlab.entity import EntityCfg
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp import dr
from mjlab.managers.event_manager import RecomputeLevel, requires_model_fields
from mjlab.managers.action_manager import ActionTerm, ActionTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.tasks.velocity import mdp
from mjlab.utils.lab_api.math import quat_apply, quat_apply_inverse, yaw_quat

from k1_common.paths import model

from .env_cfg import k1_velocity_env_cfg
from .frozen_base import FrozenBaseActionCfg
from mjlab_k1.k1_constants import ARM_REGEX, HAND_BODIES
from .tasks_kick import failure_terminated
from .tasks_push_ma import AGENTS, box_spec, robot_cfg
from .upper_body import RandomUpperBodyActionCfg
from k1_runtime.wbc import ArmIK, JOINTS

BASE_CKPT = os.environ.get(
  "K1_PIK_BASE", str(model("mjlab_k1/walk_still_v2/model_8998.pt")))
# half depth, half width, half height (m); K1_PIK_HALF_Z="lo,hi" overrides the half height range
SIZE_RANGE = ((0.2, 0.35), (0.4, 0.75), tuple(float(v) for v in os.environ.get("K1_PIK_HALF_Z", "0.15,0.3").split(",")))
# mass = 8 kg * exp(2 alpha) = 2 .. 5 kg, so drag <= 29 N: scripted hand pushing stalls above ~35 N (scripts/push_ik_probe.py)
BOX_ALPHA, FRICTION_RANGE = (-0.69, -0.235), (0.3, 0.6)
INERTIA_SPREAD = (0.5, 1.6)
TALL_BOX = "K1_PIK_HALF_Z" in os.environ                     # also resamples box size, mass and inertia every episode
CMD_SCALE = (0.5, 0.3, 0.8)                                  # the deploy cap
HEIGHT_RANGE = 0.12                                          # H_STAND - 0.12 .. H_STAND
HAND_CENTER, HAND_SCALE = (0.27, 0.2, -0.1), (0.13, 0.12, 0.2)  # left hand, trunk frame; the right mirrors y
SUCCESS_R, SUCCESS_YAW, SUCCESS_V = 0.2, math.radians(10.0), 0.15
OOB_R = 5.0
PROG_GATE, PROG_STEP = 0.4, 1e-4                             # widen resets while the success rate is above the gate
BODY_W = float(os.environ.get("K1_PIK_BODY_W", "6.0"))
ONE_HAND = float(os.environ.get("K1_PIK_ONE_HAND", "1.0"))    # share of progress and goal credit a robot with one hand on the box earns
BOTH_W = float(os.environ.get("K1_PIK_BOTH_W", "0.0"))        # per second, per share of robots with both hands on the box
NEAR_FRAC = float(os.environ.get("K1_PIK_NEAR_FRAC", "0.0"))  # resets with the box near the goal, robots squared up at its face
FACE_FRAC = float(os.environ.get("K1_PIK_FACE_FRAC", "0.0"))  # resets with the robots squared up at the face, usual goal distance
MOCAP_DELAY, MOCAP_XY, MOCAP_YAW = (1, 3), 0.005, math.radians(1.0)  # steps, m, rad (1 sigma)
FAILURES = ("fell_a", "fell_b", "box_oob", "box_tip")


##
# Hand IK action
##
@dataclass(kw_only=True)
class HandIKActionCfg(ActionTermCfg):
  alpha: float = 0.2
  """First-order filter on the hand goals each step."""

  def build(self, env) -> "HandIKAction":
    return HandIKAction(self, env)


class HandIKAction(ActionTerm):
  def __init__(self, cfg: HandIKActionCfg, env):
    super().__init__(cfg=cfg, env=env)
    ids, names = self._entity.find_joints(ARM_REGEX)
    assert tuple(names) == JOINTS[2:10], names
    self._ids = torch.tensor(ids, device=self.device, dtype=torch.long)
    self._ik = ArmIK(self.num_envs, self.device, dt=env.step_dt)
    c, s = torch.tensor(HAND_CENTER, device=self.device), torch.tensor(HAND_SCALE, device=self.device)
    flip = torch.tensor((1.0, -1.0, 1.0), device=self.device)
    self._center, self._scale = torch.cat([c, c * flip]), torch.cat([s, s * flip])
    self._raw = torch.zeros(self.num_envs, 6, device=self.device)
    self._goal = self._center.repeat(self.num_envs, 1)
    self._q = self._ik.q.clone()

  @property
  def action_dim(self) -> int:
    return 6

  @property
  def raw_action(self) -> torch.Tensor:
    return self._raw

  def process_actions(self, actions: torch.Tensor) -> None:
    self._raw[:] = actions
    goal = self._center + self._scale * actions.clamp(-1.0, 1.0)
    self._goal = self._goal + self.cfg.alpha * (goal - self._goal)
    self._q = self._ik.step(self._goal)

  def apply_actions(self) -> None:
    self._entity.set_joint_position_target(self._q, joint_ids=self._ids)

  def reset(self, env_ids=None) -> None:
    if env_ids is None:
      env_ids = slice(None)
    self._raw[env_ids] = 0.0
    self._goal[env_ids] = self._center
    self._ik.reset(env_ids)


##
# State: goal, per-world box, mocap ring buffer
##
def _state(env):
  st = getattr(env, "_pik_state", None)
  if st is None:
    n, dev, D = env.num_envs, env.device, MOCAP_DELAY[1] + 1
    st = env._pik_state = type("S", (), dict(
      goal=torch.zeros(n, 3, device=dev), goal_yaw=torch.zeros(n, device=dev), pot=torch.zeros(n, device=dev),
      ring=torch.zeros(D, n, 3, 3, device=dev), delay=torch.randint(MOCAP_DELAY[0], MOCAP_DELAY[1] + 1, (n,), device=dev),
      hand_n=torch.zeros(n, device=dev), body_n=torch.zeros(n, device=dev), near=torch.zeros(n, dtype=torch.bool, device=dev),
      prog=0.0, rate=0.0, stamp=-1, fresh=torch.ones(n, dtype=torch.bool, device=dev)))()
  return st


def _half(env) -> torch.Tensor:
  return env.sim.model.geom_size[:, env.scene["box"].indexing.geom_ids[0]]


def _yaw(q: torch.Tensor) -> torch.Tensor:
  return torch.atan2(2.0 * (q[:, 0] * q[:, 3] + q[:, 1] * q[:, 2]), 1.0 - 2.0 * (q[:, 2] ** 2 + q[:, 3] ** 2))


def _mocap(env, k: int) -> torch.Tensor:
  """(n, 3) x, y, yaw of object k (0 robot_a, 1 robot_b, 2 box) as the caps report it; updated once per env step."""
  st, step = _state(env), env.common_step_counter
  if st.stamp != step or st.fresh.any():
    now = torch.stack([torch.cat([d.root_link_pos_w[:, :2], _yaw(d.root_link_quat_w)[:, None]], -1)
                       for d in (env.scene[a].data for a in (*AGENTS, "box"))], 1)
    now = now + torch.randn_like(now) * torch.tensor((MOCAP_XY, MOCAP_XY, MOCAP_YAW), device=env.device)
    if st.stamp != step:
      st.ring, st.stamp = torch.cat([st.ring[1:], now[None]]), step
    st.ring[:, st.fresh] = now[st.fresh]
    st.fresh[:] = False
  return st.ring[-1 - st.delay, torch.arange(env.num_envs, device=env.device), k]


def _rel(env, agent: str, target: torch.Tensor) -> torch.Tensor:
  """World (n, 2) point -> this agent's mocap heading frame."""
  me = _mocap(env, AGENTS.index(agent))
  d, c, s = target - me[:, :2], me[:, 2].cos(), me[:, 2].sin()
  return torch.stack([c * d[:, 0] + s * d[:, 1], -s * d[:, 0] + c * d[:, 1]], -1)


def _axis(yaw: torch.Tensor) -> torch.Tensor:
  return torch.stack([(2.0 * yaw).sin(), (2.0 * yaw).cos()], -1)


##
# Observations
##
def obs_box(env, agent: str) -> torch.Tensor:
  b = _mocap(env, 2)
  return torch.cat([_rel(env, agent, b[:, :2]), _axis(b[:, 2] - _mocap(env, AGENTS.index(agent))[:, 2])], -1)


def obs_goal(env, agent: str) -> torch.Tensor:
  st = _state(env)
  return torch.cat([_rel(env, agent, st.goal[:, :2]), _axis(st.goal_yaw - _mocap(env, AGENTS.index(agent))[:, 2])], -1)


def obs_mate(env, agent: str) -> torch.Tensor:
  k = AGENTS.index(agent)
  m, dy = _mocap(env, 1 - k), _mocap(env, 1 - k)[:, 2] - _mocap(env, k)[:, 2]
  return torch.cat([_rel(env, agent, m[:, :2]), dy.sin()[:, None], dy.cos()[:, None]], -1)


def obs_size(env, agent: str) -> torch.Tensor:
  return _half(env)


def obs_last(env, agent: str) -> torch.Tensor:
  am = env.action_manager
  return torch.cat([am.get_term(f"base_{agent}").raw_action, am.get_term(f"hands_{agent}").raw_action], -1)


def hands_true(env, agent: str) -> torch.Tensor:
  r = env.scene[agent]
  rel = r.data.body_link_pos_w[:, r.find_bodies(HAND_BODIES, preserve_order=True)[0]] - r.data.root_link_pos_w[:, None]
  q = yaw_quat(r.data.root_link_quat_w)
  return torch.cat([quat_apply_inverse(q, rel[:, 0]), quat_apply_inverse(q, rel[:, 1])], -1)


def box_vel_true(env, agent: str) -> torch.Tensor:
  return quat_apply_inverse(yaw_quat(env.scene[agent].data.root_link_quat_w), env.scene["box"].data.root_link_lin_vel_w)


##
# Task geometry
##
def _errors(env) -> tuple[torch.Tensor, torch.Tensor]:
  """True box position error (m) and yaw error (rad, mod 180 deg, in [0, pi/2])."""
  st, b = _state(env), env.scene["box"].data
  d = (b.root_link_pos_w[:, :2] - st.goal[:, :2]).norm(dim=-1)
  e = _yaw(b.root_link_quat_w) - st.goal_yaw
  return d, torch.atan2((2.0 * e).sin(), (2.0 * e).cos()).abs() / 2.0


def corners_xy(xy: torch.Tensor, yaw: torch.Tensor, half: torch.Tensor) -> torch.Tensor:
  """(n, 4, 2) top-view corners of boxes at xy (n, 2), yaw (n,), half sizes (n, >=2), in order around the box."""
  sx, sy = torch.tensor((1.0, -1.0, -1.0, 1.0), device=xy.device), torch.tensor((1.0, 1.0, -1.0, -1.0), device=xy.device)
  lx, ly = sx * half[:, :1], sy * half[:, 1:2]
  c, s = yaw.cos()[:, None], yaw.sin()[:, None]
  return xy[:, None] + torch.stack([c * lx - s * ly, s * lx + c * ly], -1)


def corner_error(env) -> torch.Tensor:
  """Mean corner distance (m) box -> goal. The box is 180-deg symmetric: the closer of the two corner pairings."""
  st, b, half = _state(env), env.scene["box"].data, _half(env)
  now = corners_xy(b.root_link_pos_w[:, :2], _yaw(b.root_link_quat_w), half)
  goal = corners_xy(st.goal[:, :2], st.goal_yaw, half)
  return torch.minimum((now - goal).norm(dim=-1).mean(-1), (now.roll(2, 1) - goal).norm(dim=-1).mean(-1))


def _potential(env) -> torch.Tensor:
  return corner_error(env)


def _touch(env, sensor: str) -> torch.Tensor:
  return (env.scene[sensor].data.found > 0).float()


def _robot_contact(env) -> torch.Tensor:
  return _touch(env, "robots_touch").amax(-1)


def _body_contact(env) -> torch.Tensor:
  return sum(_touch(env, f"{a}_body_box").amax(-1) for a in AGENTS)


def _hand_frac(env) -> torch.Tensor:
  return torch.cat([_touch(env, f"{a}_hands_box") for a in AGENTS], -1).mean(-1)


def _both_hands(env) -> torch.Tensor:
  """Share of the two robots that have both hands on the box: 0, 0.5 or 1."""
  return torch.stack([_touch(env, f"{a}_hands_box").prod(-1) for a in AGENTS], -1).mean(-1)


def _credit(env) -> torch.Tensor:
  """1 with both hands of both robots on the box, down to ONE_HAND. ONE_HAND = 1 is the old one-hand-is-enough rule."""
  return ONE_HAND + (1.0 - ONE_HAND) * _both_hands(env)


##
# Rewards / terminations
##
def _hand_dist(env) -> torch.Tensor:
  """(n, 4) distance of each hand to the box surface: robot a left, right, robot b left, right."""
  b = env.scene["box"].data
  hands = torch.cat([env.scene[a].data.body_link_pos_w[:, env.scene[a].find_bodies(HAND_BODIES, preserve_order=True)[0]]
                     for a in AGENTS], 1)
  q = b.root_link_quat_w[:, None].expand(-1, 4, -1).reshape(-1, 4)
  local = quat_apply_inverse(q, (hands - b.root_link_pos_w[:, None]).reshape(-1, 3)).reshape(hands.shape)
  return (local.abs() - _half(env)[:, None]).clamp_min(0.0).norm(dim=-1)


def hands_near(env, std: float = 0.25) -> torch.Tensor:
  return torch.exp(-_hand_dist(env) / std).mean(-1)


def hands_on_box(env) -> torch.Tensor:
  return _hand_frac(env)


def body_touch(env) -> torch.Tensor:
  n = _body_contact(env)
  env.extras["log"]["Metrics/body_touch"] = n.mean()
  return n


def robots_touch(env) -> torch.Tensor:
  c = _robot_contact(env)
  env.extras["log"]["Metrics/robots_touch"] = c.mean()
  return c


def both_hands(env, std: float = 0.1) -> torch.Tensor:
  """Per robot, how close its farther hand is to the box. Contact alone almost never fires from a one-handed policy."""
  return torch.exp(-_hand_dist(env).reshape(-1, 2, 2).amax(-1) / std).mean(-1)


def hand_push_progress(env) -> torch.Tensor:
  """Potential drop rate; counts only while at least one hand and no other body is on the box. Log true errors.
  Gains scale with _credit, losses do not: pushing the box the wrong way one-handed must not get cheaper."""
  st = _state(env)
  pot, d_e = _potential(env), _errors(env)
  rate = ((st.pot - pot) / env.step_dt).clamp(-1.0, 1.0)
  body = _body_contact(env) > 0
  st.body_n += body.float()
  credit = _credit(env)
  st.hand_n += ((_hand_frac(env) > 0) & ~body).float() * credit
  env.extras["log"]["Metrics/hand_ratio"] = _hand_ratio(env).mean()
  st.pot = pot
  env.extras["log"]["Metrics/box_corner_err"] = pot.mean()
  env.extras["log"]["Metrics/box_pos_err"] = d_e[0].mean()
  env.extras["log"]["Metrics/box_yaw_err_deg"] = torch.rad2deg(d_e[1]).mean()
  env.extras["log"]["Metrics/hands_on_box"] = _hand_frac(env).mean()
  env.extras["log"]["Metrics/both_hands"] = _both_hands(env).mean()
  per_hand = torch.stack([_touch(env, f"{a}_hands_box") for a in AGENTS]).mean((0, 1))  # left, right
  env.extras["log"]["Metrics/left_on_box"], env.extras["log"]["Metrics/right_on_box"] = per_hand[0], per_hand[1]
  return torch.where(rate > 0, rate * credit, rate) * ((_hand_frac(env) > 0) & ~body).float()


def _hand_ratio(env) -> torch.Tensor:
  """Share of this episode's box contact that was hands only: goal rewards scale with it, so shoving earns nothing."""
  st = _state(env)
  return st.hand_n / (st.hand_n + st.body_n + 1.0)


def near_goal(env, std: float = 0.5) -> torch.Tensor:
  return _hand_ratio(env) * torch.exp(-(corner_error(env) / std) ** 2)


def action_bound(env) -> torch.Tensor:
  """Actions are clamped to +-1 downstream: without this, the policy can drift outside at no cost."""
  return (env.action_manager.action.abs() - 1.0).clamp_min(0.0).square().sum(-1)


def success(env) -> torch.Tensor:
  d, e = _errors(env)
  v = env.scene["box"].data.root_link_lin_vel_w[:, :2].norm(dim=-1)
  return (d < SUCCESS_R) & (e < SUCCESS_YAW) & (v < SUCCESS_V)


def success_bonus(env) -> torch.Tensor:
  """1 + the fraction of the episode left: success ends the episode, so it must outpay staying and collecting."""
  left = 1.0 - env.episode_length_buf.float() / env.max_episode_length
  return success(env).float() * (1.0 + left) * _hand_ratio(env)


def box_oob(env) -> torch.Tensor:
  return (env.scene["box"].data.root_link_pos_w[:, :2] - env.scene.env_origins[:, :2]).norm(dim=-1) > OOB_R


def box_tip(env, limit_deg: float = 30.0) -> torch.Tensor:
  b = env.scene["box"].data
  up = quat_apply(b.root_link_quat_w, torch.tensor([0.0, 0.0, 1.0], device=env.device).expand(env.num_envs, 3))
  return up[:, 2] < math.cos(math.radians(limit_deg))


##
# Reset
##
def progress(env, env_ids: torch.Tensor) -> float:
  """Curriculum 0..1: follows the success rate of the episodes that end (not timeouts cut short at the run's start)."""
  if getattr(env, "_pik_play", False):
    return 1.0
  st, tm = _state(env), env.termination_manager
  if env.common_step_counter > 0:
    far = env_ids[~st.near[env_ids]]  # near-goal starts are easy: they must not widen the ranges
    if len(far):
      st.rate += min(1.0, len(far) / env.num_envs) * (tm.get_term("goal_reached")[far].float().mean().item() - st.rate)
    if st.rate > PROG_GATE:
      st.prog = min(1.0, st.prog + PROG_STEP)
  env.extras["log"]["Metrics/curriculum"] = st.prog
  env.extras["log"]["Metrics/success_rate"] = st.rate
  return st.prog


@requires_model_fields("body_inertia", recompute=RecomputeLevel.set_const)
def box_inertia(env, env_ids: torch.Tensor | None = None) -> None:
  """Inertia of this env's size and mass, as a solid box with each axis's mass spread scaled by INERTIA_SPREAD
  (loaded or hollow boxes). Any positive spread keeps the inertia physically valid."""
  ids = torch.arange(env.num_envs, device=env.device) if env_ids is None else env_ids
  bid, sq = env.scene["box"].indexing.body_ids[0], (2.0 * _half(env)[ids]) ** 2
  sq = sq * (INERTIA_SPREAD[0] + (INERTIA_SPREAD[1] - INERTIA_SPREAD[0]) * torch.rand_like(sq))
  x, y, z = sq[:, 0], sq[:, 1], sq[:, 2]
  env.sim.model.body_inertia[ids, bid] = env.sim.model.body_mass[ids, bid][:, None] / 12.0 * torch.stack([y + z, x + z, x + y], -1)


def _quat_z(yaw: torch.Tensor) -> torch.Tensor:
  return torch.stack([(yaw / 2).cos(), torch.zeros_like(yaw), torch.zeros_like(yaw), (yaw / 2).sin()], -1)


def reset_scene(env, env_ids: torch.Tensor, play: bool = False) -> None:
  env._pik_play = play
  n, dev, p = len(env_ids), env.device, progress(env, env_ids)
  st, half = _state(env), _half(env)[env_ids]
  r = lambda lo, hi: lo + (hi - lo) * torch.rand(n, device=dev)      # noqa: E731
  sym = lambda a: a * (2.0 * torch.rand(n, device=dev) - 1.0)        # noqa: E731
  org = env.scene.env_origins[env_ids]

  # box and goal: push direction th; at p = 0 the box faces it, at p = 1 any box yaw and any goal yaw
  near = torch.rand(n, device=dev) < (0.0 if play else NEAR_FRAC)
  close = near | (torch.rand(n, device=dev) < (0.0 if play else FACE_FRAC))
  st.near[env_ids] = near
  th = sym(math.pi)
  box_xy = torch.stack([sym(0.5), sym(0.5)], -1)
  box_yaw = th + sym(0.2 + p * (math.pi - 0.2))
  dist = torch.where(near, r(0.25, 0.7), r(0.4, 1.0 + 1.5 * p))  # over SUCCESS_R, so no near start is won at once
  st.goal[env_ids, :2] = org[:, :2] + box_xy + dist[:, None] * torch.stack([th.cos(), th.sin()], -1)
  st.goal[env_ids, 2] = org[:, 2]
  st.goal_yaw[env_ids] = box_yaw + torch.where(near, sym(0.35), sym(0.1 + p * (math.pi / 2 - 0.1)))

  box = env.scene["box"]
  s = box.data.default_root_state[env_ids].clone()
  s[:, :2], s[:, 2] = org[:, :2] + box_xy, org[:, 2] + half[:, 2] + 0.005
  s[:, 3:7], s[:, 7:] = _quat_z(box_yaw), 0.0
  box.write_root_state_to_sim(s, env_ids=env_ids)

  # robots: behind the box's push face (prob 1 - p), else anywhere on a ring around it, always clear of box and mate
  c, sn = box_yaw.cos(), box_yaw.sin()
  back = -torch.sign(th.cos() * c + th.sin() * sn).clamp_min(0.0) * 2.0 + 1.0  # -1: robots on the -x face if it faces th
  ring = (torch.rand(n, device=dev) < p) & ~close
  rad = half[:, :2].norm(dim=-1) + r(0.45, 0.45 + 1.0 * p)
  ang = sym(math.pi)
  gap = (0.75 / rad).clamp_max(math.pi) + r(0.0, 0.5 * p)
  bx = back * (half[:, 0] + torch.where(close, r(0.32, 0.45), r(0.35, 0.6)))  # HAND_CENTER x is 0.27: hands start clear
  lat = (half[:, 1] * torch.where(close, torch.full((n,), 0.5, device=dev), r(0.3, 0.7))).clamp_min(0.33)  # shoulders ~0.45 m wide
  square = box_yaw + (back > 0).float() * math.pi  # facing the box face head on: both hands can reach it
  yaw_noise = 0.2 + p * (math.pi - 0.2)
  for k, a in enumerate(AGENTS):
    sgn = 1.0 - 2.0 * k
    face = torch.stack([bx, sgn * lat + sym(0.05)], -1)
    al = ang + k * gap
    circ = torch.stack([rad * al.cos(), rad * al.sin()], -1)
    loc = torch.where(ring[:, None], circ, face)
    pos = box_xy + torch.stack([c * loc[:, 0] - sn * loc[:, 1], sn * loc[:, 0] + c * loc[:, 1]], -1)
    to_box = torch.atan2(box_xy[:, 1] - pos[:, 1], box_xy[:, 0] - pos[:, 0])
    robot = env.scene[a]
    t = robot.data.default_root_state[env_ids].clone()
    t[:, :2] = org[:, :2] + pos
    t[:, 2] = org[:, 2] + t[:, 2] + 0.01
    t[:, 3:7], t[:, 7:] = _quat_z(torch.where(close, square + sym(0.15), to_box + sym(yaw_noise))), 0.0
    robot.write_root_state_to_sim(t, env_ids=env_ids)

  st.hand_n[env_ids], st.body_n[env_ids] = 0.0, 0.0
  st.fresh[env_ids] = True
  st.delay[env_ids] = torch.randint(MOCAP_DELAY[0], MOCAP_DELAY[1] + 1, (n,), device=dev)
  now, goal = corners_xy(s[:, :2], box_yaw, half), corners_xy(st.goal[env_ids, :2], st.goal_yaw[env_ids], half)
  st.pot[env_ids] = torch.minimum((now - goal).norm(dim=-1).mean(-1), (now.roll(2, 1) - goal).norm(dim=-1).mean(-1))


def push_ik_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  cfg = k1_velocity_env_cfg(rough=False, play=play)
  cfg.scene.entities = {a: robot_cfg(k) for k, a in enumerate(AGENTS)}
  cfg.scene.entities["box"] = EntityCfg(spec_fn=box_spec, init_state=EntityCfg.InitialStateCfg(pos=(0.85, 0.0, 0.3)))
  cfg.scene.sensors = tuple(
    ContactSensorCfg(name=f"{a}_{part}_box", fields=("found",), num_slots=1,
                     primary=ContactMatch(mode="body", pattern=pat, entity=a, exclude=excl),
                     secondary=ContactMatch(mode="body", pattern="box", entity="box"))
    for a in AGENTS for part, pat, excl in (("hands", HAND_BODIES, ()), ("body", ".*", HAND_BODIES))
  ) + (ContactSensorCfg(name="robots_touch", fields=("found",), num_slots=1,
                        primary=ContactMatch(mode="body", pattern=".*", entity=AGENTS[0]),
                        secondary=ContactMatch(mode="body", pattern=".*", entity=AGENTS[1])),)
  cfg.sim.mujoco.ls_iterations = int(os.environ.get("K1_LS_ITERS", "50"))
  cfg.sim.nconmax, cfg.sim.njmax = 200, 1500
  cfg.commands, cfg.curriculum = {}, {}
  cfg.episode_length_s = 30.0
  cfg.viewer.entity_name = "robot_a"

  cfg.actions = {}
  for a in AGENTS:  # dict order = action layout: [a: base 4, hands 6 | b: base 4, hands 6]
    cfg.actions[f"base_{a}"] = FrozenBaseActionCfg(entity_name=a, checkpoint=BASE_CKPT, cmd_scale=CMD_SCALE,
                                                   height_range=HEIGHT_RANGE, cmd_alpha=0.2)
    cfg.actions[f"hands_{a}"] = HandIKActionCfg(entity_name=a)
  for a in AGENTS:
    cfg.actions[f"head_{a}"] = RandomUpperBodyActionCfg(entity_name=a, joint_names=(".*Head.*",), frozen_prob=1.0)

  def terms(a: str, fn=None) -> dict:
    p = {"agent": a}
    return {f"{a}_{name}": ObservationTermCfg(func=f, params=p) for name, f in
            (("box", obs_box), ("goal", obs_goal), ("mate", obs_mate), ("size", obs_size), ("last", obs_last))}

  actor = {k: v for a in AGENTS for k, v in terms(a).items()}  # [agent a block | agent b block]
  critic = dict(actor)
  for a in AGENTS:
    critic[f"{a}_hands"] = ObservationTermCfg(func=hands_true, params={"agent": a})
    critic[f"{a}_box_vel"] = ObservationTermCfg(func=box_vel_true, params={"agent": a})
  cfg.observations = {
    "actor": ObservationGroupCfg(actor, enable_corruption=False, history_length=5),
    "critic": ObservationGroupCfg(critic, enable_corruption=False),
  }

  reset_joints = cfg.events["reset_robot_joints"]
  box_geom = SceneEntityCfg("box", geom_names=("box_geom",))
  box_body = SceneEntityCfg("box", body_names=("box",))
  box_mode = "reset" if TALL_BOX else "startup"
  cfg.events = {
    "box_size": EventTermCfg(func=dr.geom_size, mode=box_mode, params={
      "asset_cfg": box_geom, "operation": "abs", "ranges": {i: r for i, r in enumerate(SIZE_RANGE)}}),
    "box_mass": EventTermCfg(func=dr.pseudo_inertia, mode=box_mode, params={
      "asset_cfg": box_body, "alpha_range": BOX_ALPHA}),
    "box_friction": EventTermCfg(func=dr.geom_friction, mode="startup", params={
      "asset_cfg": box_geom, "operation": "abs", "axes": [0], "ranges": FRICTION_RANGE}),
  }
  if TALL_BOX:  # after box_size and box_mass, before reset_scene, which reads the new size
    cfg.events["box_inertia"] = EventTermCfg(func=box_inertia, mode="reset")
  for a in AGENTS:
    cfg.events[f"reset_joints_{a}"] = replace(reset_joints, params={**reset_joints.params, "asset_cfg": SceneEntityCfg(a)})
  cfg.events["reset_scene"] = EventTermCfg(func=reset_scene, mode="reset", params={"play": play})

  cfg.rewards = {
    "hands_near": RewardTermCfg(func=hands_near, weight=0.5),
    "hands_on_box": RewardTermCfg(func=hands_on_box, weight=1.0),
    "body_touch": RewardTermCfg(func=body_touch, weight=-BODY_W),
    "robots_touch": RewardTermCfg(func=robots_touch, weight=-BODY_W),
    "push_progress": RewardTermCfg(func=hand_push_progress, weight=40.0),
    "near_goal": RewardTermCfg(func=near_goal, weight=2.0),
    "success": RewardTermCfg(func=success_bonus, weight=5000.0),  # x dt: 100 + up to 100; staying pays at most 72
    "failure": RewardTermCfg(func=failure_terminated, weight=-500.0, params={"term_names": FAILURES}),
    "action_rate": RewardTermCfg(func=mdp.action_rate_l2, weight=-0.01),
    "action_bound": RewardTermCfg(func=action_bound, weight=-0.1),
  }
  if BOTH_W > 0.0:
    cfg.rewards["both_hands"] = RewardTermCfg(func=both_hands, weight=BOTH_W)
  for a in AGENTS:
    cfg.rewards[f"flat_{a}"] = RewardTermCfg(
      func=mdp.flat_orientation_l2, weight=-1.0, params={"asset_cfg": SceneEntityCfg(a, body_names=("Trunk",))})

  fell = float(os.environ.get("K1_FELL_ANGLE", "0.8"))
  cfg.terminations = {
    "time_out": TerminationTermCfg(func=mdp.time_out, time_out=True),
    "fell_a": TerminationTermCfg(func=mdp.bad_orientation, params={"limit_angle": fell, "asset_cfg": SceneEntityCfg("robot_a")}),
    "fell_b": TerminationTermCfg(func=mdp.bad_orientation, params={"limit_angle": fell, "asset_cfg": SceneEntityCfg("robot_b")}),
    "goal_reached": TerminationTermCfg(func=success),
    "box_oob": TerminationTermCfg(func=box_oob),
    "box_tip": TerminationTermCfg(func=box_tip),
  }
  return cfg
