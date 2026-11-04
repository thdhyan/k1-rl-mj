"""Box carry v2: no weld. The box is a free, colliding body held only by the forearm stubs squeezing its sides.

Same hold pose, FK geometry, base/arm action split and goal as carry v1 (tasks_carry), but:
  * box collides with the robot (and ground), high friction, torsional condim, lighter (K1_CARRY2_MASS, 1 kg);
  * box is SQUEEZE wider per side than the stub gap at the hold pose, so the default arms already press it and
    the policy must keep pressing (arm offsets) while walking or it slips out;
  * hand-off: the v1 welds hold the box for the first ASSIST_S of each episode (arms at kp 40 sag ~0.2 rad in
    Elbow_Yaw within 0.1 s, so the box drops before an untrained policy can react), then release; the policy
    sees the assist flag and must have its squeeze in place at release;
  * rewards: hold quality (box at its nominal spot, upright trunk), both stubs in contact, goal progress gated by
    both; terminate when the box drops / slides away (box_lost) or the robot falls.
"""
from __future__ import annotations

import os

import mujoco
import torch

from mjlab.entity import EntityCfg
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.tasks.velocity import mdp

from mjlab_k1.k1_constants import HAND_BODIES
from .tasks_carry import (BOX_DEPTH, BOX_HEIGHT, GEO, SUCCESS_R, VIS_INSET, _goal_dist, _state, add_welds,
                          box_err_b, carry_env_cfg, hold_quality)
from .tasks_kick import failure_terminated

BOX_MASS = float(os.environ.get("K1_CARRY2_MASS", "1.0"))
SQUEEZE = float(os.environ.get("K1_CARRY2_SQUEEZE", "0.015"))  # m per side beyond the stub faces
FRICTION = (float(os.environ.get("K1_CARRY2_MU", "1.2")), 0.02, 0.001)
LOST = 0.08                                                    # m: box this far from its hold spot = dropped
FAILURES = ("fell", "box_lost")
ASSIST_S = float(os.environ.get("K1_CARRY2_ASSIST_S", "1.0"))


def box_spec() -> mujoco.MjSpec:
  spec = mujoco.MjSpec()
  body = spec.worldbody.add_body(name="box")
  body.add_freejoint(name="box_joint")
  body.add_geom(
    name="box_geom", type=mujoco.mjtGeom.mjGEOM_BOX,
    size=(BOX_DEPTH / 2, GEO["width"] / 2 - VIS_INSET + SQUEEZE, BOX_HEIGHT / 2),
    rgba=(0.8, 0.5, 0.2, 1.0), contype=1, conaffinity=1, condim=4, friction=FRICTION, mass=BOX_MASS)
  return spec


def _assisted(env) -> torch.Tensor:
  return env.episode_length_buf * env.step_dt < ASSIST_S


def assist_weld(env, env_ids: torch.Tensor | None = None) -> None:
  """Every step (and on reset): welds on while the episode is younger than ASSIST_S."""
  env.sim.data.eq_active[:] = _assisted(env)[:, None]


def assist_obs(env) -> torch.Tensor:
  return _assisted(env).float().unsqueeze(-1)


def hands_on_box(env) -> torch.Tensor:
  """Fraction of the 2 stubs touching the box."""
  return (env.scene["hands_box"].data.found > 0).float().mean(-1)


def hands_obs(env) -> torch.Tensor:
  return (env.scene["hands_box"].data.found > 0).float()


def _grip(env) -> torch.Tensor:
  return hold_quality(env) * (hands_on_box(env) == 1.0).float()


def grip_progress_rate(env) -> torch.Tensor:
  """Metres per second closed on the goal (clamped), only while both stubs grip the box at its spot."""
  st = _state(env)
  d = _goal_dist(env)
  r = ((st.d_prev - d) / env.step_dt).clamp(-1.0, 1.0)
  st.d_prev = d
  return r * _grip(env)


def goal_near(env, std: float = 0.5) -> torch.Tensor:
  return torch.exp(-(_goal_dist(env) ** 2) / std**2) * _grip(env)


def goal_reached(env) -> torch.Tensor:
  return (_goal_dist(env) < SUCCESS_R) & (box_err_b(env).norm(dim=-1) < 0.05) & (hands_on_box(env) == 1.0)


def box_lost(env, threshold: float = LOST) -> torch.Tensor:
  return (box_err_b(env).norm(dim=-1) > threshold) & ~_assisted(env)


def box_lin_vel_rel(env) -> torch.Tensor:
  """Box slip: its velocity relative to the trunk (world frame, squared)."""
  return (env.scene["box"].data.root_link_lin_vel_w - env.scene["robot"].data.root_link_lin_vel_w).square().sum(-1)


def carry2_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  cfg = carry_env_cfg(play=play)
  cfg.scene.entities["box"] = EntityCfg(spec_fn=box_spec, init_state=EntityCfg.InitialStateCfg(pos=(0.3, 0.0, 0.7)))
  cfg.scene.spec_fn = add_welds                                # hand-off assist only (assist_weld)
  cfg.scene.sensors = (ContactSensorCfg(
    name="hands_box", fields=("found",), num_slots=1,
    primary=ContactMatch(mode="body", pattern=HAND_BODIES, entity="robot"),
    secondary=ContactMatch(mode="body", pattern="box", entity="box")),)
  cfg.sim.nconmax = 120
  # grasp by friction: elliptic cone + high impratio, else the pyramidal/impratio-1 default lets the box creep out
  cfg.sim.mujoco.cone = os.environ.get("K1_CARRY2_CONE", "elliptic")
  cfg.sim.mujoco.impratio = float(os.environ.get("K1_CARRY2_IMPRATIO", "10"))
  cfg.sim.njmax = 800

  obs = {k: v for k, v in cfg.observations["actor"].terms.items() if k != "weld"}
  obs["hands_contact"] = ObservationTermCfg(func=hands_obs)
  obs["assist"] = ObservationTermCfg(func=assist_obs)
  for g in ("actor", "critic"):
    cfg.observations[g].terms = dict(obs)

  cfg.events["assist_weld"] = EventTermCfg(func=assist_weld, mode="step")
  cfg.events["assist_weld_reset"] = EventTermCfg(func=assist_weld, mode="reset")

  cfg.rewards = {k: v for k, v in cfg.rewards.items() if k not in ("weld_error", "goal_progress", "goal_near", "success")}
  cfg.rewards.update({
    "hands_on_box": RewardTermCfg(func=hands_on_box, weight=1.0),
    "goal_progress": RewardTermCfg(func=grip_progress_rate, weight=5.0),
    "goal_near": RewardTermCfg(func=goal_near, weight=1.0),
    "success": RewardTermCfg(func=goal_reached, weight=500.0),
    "box_slip": RewardTermCfg(func=box_lin_vel_rel, weight=-0.5),
  })
  cfg.rewards["failure"] = RewardTermCfg(func=failure_terminated, weight=-200.0, params={"term_names": FAILURES})
  cfg.terminations = {k: v for k, v in cfg.terminations.items() if k != "weld_broken"}
  cfg.terminations["goal_reached"] = TerminationTermCfg(func=goal_reached)
  cfg.terminations["box_lost"] = TerminationTermCfg(func=box_lost)
  return cfg
