"""Box carry (port of Isaac Lab `Isaac-Carry-K1-v0`, carry v1) on mjlab: an HRL task layer with arms.

The box is welded between the two forearm stubs, as in the Isaac task, here with two MuJoCo weld equalities
(scene spec_fn) instead of USD FixedJoints. Box has no collider. Action (11) = (vx, vy, wz) -> frozen walk
base | 8 arm offsets about the hold pose. Arms at the Isaac carry gains (kp 40, kd 2, k1_constants.MANIP).
Rewards follow the simplified Isaac v1 set: hold quality always pays, goal progress is gated by it,
weld error / box tilt / fall are penalised.

Hold pose: the Isaac offline-IK table entry for a 0.325 m box (carry_mdp.HOLD_Q_LEFT), mirrored for the
right arm; box centre and weld frames come from MuJoCo FK of that pose on the mjlab model itself.
"""
from __future__ import annotations

import math
import os

import mujoco
import numpy as np
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
from mjlab_k1.k1_constants import ARM_REGEX, HAND_BODIES, LEG_REGEX, get_k1_manip_robot_cfg, get_spec
from .tasks_kick import BASE_CKPT, _yaw_rel, failure_terminated
from .upper_body import RandomUpperBodyActionCfg

# Isaac carry_mdp: HOLD_Q_LEFT[3] (box width 0.325), right arm mirrored through the sagittal plane
HOLD_Q_LEFT = (-0.532, -1.458, 0.066, -1.081)
MIRROR = (1.0, -1.0, 1.0, -1.0)
ARM_ORDER = ("Shoulder_Pitch", "Shoulder_Roll", "Elbow_Pitch", "Elbow_Yaw")
STUB_TIP = 0.20                       # m along the forearm's +-y
BOX_DEPTH, BOX_HEIGHT, BOX_MASS = 0.18, 0.18, float(os.environ.get("K1_CARRY_MASS", "3.0"))
VIS_INSET = 0.03                      # stub half-thickness + gap: drawn box clears the forearms
GOAL_D = (1.0, 3.0)
SUCCESS_R, WELD_BREAK = 0.3, 0.10
FAILURES = ("fell", "weld_broken")


def _hold_joint_pos() -> dict[str, float]:
  q = {}
  for side, sign in (("Left", 1.0), ("Right", -1.0)):
    for name, v, m in zip(ARM_ORDER, HOLD_Q_LEFT, MIRROR):
      q[f".*{side}_{name}"] = v * (m if sign < 0 else 1.0)
  return q


def _carry_geometry() -> dict:
  """Hand poses at the hold pose in the Trunk frame (MuJoCo FK), box centre and weld frames."""
  model = get_spec().compile()
  data = mujoco.MjData(model)
  for pat, v in _hold_joint_pos().items():
    side, name = pat[2:].split("_", 1)
    jid = next(j for j in range(model.njnt) if model.joint(j).name.endswith(f"{side}_{name}"))
    data.qpos[model.jnt_qposadr[jid]] = v
  mujoco.mj_kinematics(model, data)
  t = model.body("Trunk").id
  R_t, p_t = data.xmat[t].reshape(3, 3), data.xpos[t]
  out, tips = {}, []
  for body, sign in zip(HAND_BODIES, (1.0, -1.0)):
    b = model.body(body).id
    R = R_t.T @ data.xmat[b].reshape(3, 3)
    p = R_t.T @ (data.xpos[b] - p_t)
    out[body] = (p, R)
    tips.append(p + R @ np.array([0.0, sign * STUB_TIP, 0.0]))
  tl, tr = tips
  center = np.array([0.5 * (tl[0] + tr[0]) + 0.02 - 0.5 * BOX_DEPTH, 0.0, 0.5 * (tl[2] + tr[2])])
  width = float(tl[1] - tr[1])
  welds = {}
  for (body, (p, R)), sign in zip(out.items(), (1.0, -1.0)):
    quat = np.zeros(4)
    mujoco.mju_mat2Quat(quat, R.flatten())
    welds[body] = (p - center, quat, sign)                    # hand pose in the box frame
  return {"center": center, "width": width, "welds": welds,
          "anchors": np.stack([tl - center, tr - center])}


GEO = _carry_geometry()


def box_spec() -> mujoco.MjSpec:
  spec = mujoco.MjSpec()
  body = spec.worldbody.add_body(name="box")
  body.add_freejoint(name="box_joint")
  w = GEO["width"]
  body.add_geom(
    name="box_geom", type=mujoco.mjtGeom.mjGEOM_BOX, size=(BOX_DEPTH / 2, w / 2 - VIS_INSET, BOX_HEIGHT / 2),
    rgba=(0.8, 0.5, 0.2, 1.0), contype=0, conaffinity=0, mass=BOX_MASS)
  return spec


def add_welds(spec: mujoco.MjSpec) -> None:
  """Scene spec_fn: weld each forearm stub to its face of the box (after entities are attached)."""
  for k, (body, (_, quat, sign)) in enumerate(GEO["welds"].items()):
    eq = spec.add_equality(name=f"weld_{body}", type=mujoco.mjtEq.mjEQ_WELD, objtype=mujoco.mjtObj.mjOBJ_BODY,
                           name1="box/box", name2=f"robot/{body}")
    # anchor = stub tip (hand frame) | that point in the box frame + hand orientation in the box frame |
    # torquescale (checked on CPU: zero efc violation at the hold pose). Assign whole: eq.data is a copy.
    eq.data = np.concatenate([[0.0, sign * STUB_TIP, 0.0], GEO["anchors"][k], quat, [1.0]])


##
# State + MDP terms
##
def _state(env):
  st = getattr(env, "_carry_state", None)
  if st is None:
    n, dev = env.num_envs, env.device
    st = env._carry_state = type("S", (), {"goal": torch.zeros(n, 3, device=dev), "d_prev": torch.zeros(n, device=dev),
                                           "center": torch.tensor(GEO["center"], device=dev, dtype=torch.float32),
                                           "anchors": torch.tensor(GEO["anchors"], device=dev, dtype=torch.float32)})()
  return st


def _robot(env):
  return env.scene["robot"].data


def _box(env):
  return env.scene["box"].data


def _goal_dist(env) -> torch.Tensor:
  return (_robot(env).root_link_pos_w[:, :2] - _state(env).goal[:, :2]).norm(dim=-1)


def box_err_b(env) -> torch.Tensor:
  """Box position in the trunk frame minus its nominal hold spot."""
  r = _robot(env)
  return quat_apply_inverse(r.root_link_quat_w, _box(env).root_link_pos_w - r.root_link_pos_w) - _state(env).center


def box_gravity(env) -> torch.Tensor:
  g = torch.tensor([0.0, 0.0, -1.0], device=env.device).expand(env.num_envs, 3)
  return quat_apply_inverse(_box(env).root_link_quat_w, g)[:, :2]


def weld_error(env) -> torch.Tensor:
  """Worst distance (m) between a stub tip and its weld point on the box."""
  robot, b, st = env.scene["robot"], _box(env), _state(env)
  ids = getattr(env, "_hand_ids", None)
  if ids is None:
    ids = env._hand_ids = robot.find_bodies(HAND_BODIES, preserve_order=True)[0]
  err = []
  for k, sign in enumerate((1.0, -1.0)):
    tip_local = torch.tensor([0.0, sign * STUB_TIP, 0.0], device=env.device).expand(env.num_envs, 3)
    tip = robot.data.body_link_pos_w[:, ids[k]] + quat_apply(robot.data.body_link_quat_w[:, ids[k]], tip_local)
    anchor = b.root_link_pos_w + quat_apply(b.root_link_quat_w, st.anchors[k].expand(env.num_envs, 3))
    err.append((tip - anchor).norm(dim=-1))
  return torch.stack(err, -1).max(-1).values


def weld_obs(env) -> torch.Tensor:
  return weld_error(env).unsqueeze(-1)


def goal_pos_b(env) -> torch.Tensor:
  return _yaw_rel(env, _state(env).goal - _robot(env).root_link_pos_w)


def hold_quality(env, pos_std: float = 0.08, tilt_std: float = 0.3) -> torch.Tensor:
  """[0, 1]: box at its nominal spot in front of the trunk and the trunk upright."""
  pos = torch.exp(-box_err_b(env).square().sum(-1) / pos_std**2)
  tilt = torch.exp(-_robot(env).projected_gravity_b[:, :2].square().sum(-1) / tilt_std**2)
  return pos * tilt


def goal_progress_rate(env) -> torch.Tensor:
  """Metres per second closed on the goal (clamped), gated by hold quality."""
  st = _state(env)
  d = _goal_dist(env)
  r = ((st.d_prev - d) / env.step_dt).clamp(-1.0, 1.0)
  st.d_prev = d
  return r * hold_quality(env)


def goal_near(env, std: float = 0.5) -> torch.Tensor:
  return torch.exp(-(_goal_dist(env) ** 2) / std**2) * hold_quality(env)


def goal_reached(env) -> torch.Tensor:
  return (_goal_dist(env) < SUCCESS_R) & (weld_error(env) < 0.05)


def weld_broken(env, threshold: float = WELD_BREAK) -> torch.Tensor:
  return weld_error(env) > threshold


def box_tilt_l2(env) -> torch.Tensor:
  return box_gravity(env).square().sum(-1)


def arm_deviation_l1(env, asset_cfg: SceneEntityCfg) -> torch.Tensor:
  d = env.scene["robot"].data
  return (d.joint_pos[:, asset_cfg.joint_ids] - d.default_joint_pos[:, asset_cfg.joint_ids]).abs().sum(-1)


def reset_box(env, env_ids: torch.Tensor) -> None:
  """Box at its hold spot in front of the (already reset) trunk; goal 1-3 m away in any direction."""
  n, dev = len(env_ids), env.device
  robot = env.scene["robot"]
  arm_ids = getattr(env, "_arm_ids", None)
  if arm_ids is None:
    arm_ids = env._arm_ids = torch.tensor(robot.find_joints(ARM_REGEX)[0], device=dev)
  hold = robot.data.default_joint_pos[env_ids][:, arm_ids]
  robot.write_joint_state_to_sim(hold, torch.zeros_like(hold), env_ids=env_ids, joint_ids=arm_ids)
  root = robot.data.data.qpos[env_ids][:, robot.data.indexing.free_joint_q_adr]
  pos, quat = root[:, :3], root[:, 3:7]
  st = _state(env)
  box = env.scene["box"]
  s = torch.zeros(n, 13, device=dev)
  s[:, :3] = pos + quat_apply(quat, st.center.expand(n, 3))
  s[:, 3:7] = quat
  box.write_root_state_to_sim(s, env_ids=env_ids)
  d = GOAL_D[0] + (GOAL_D[1] - GOAL_D[0]) * torch.rand(n, device=dev)
  a = math.pi * (2 * torch.rand(n, device=dev) - 1)
  st.goal[env_ids] = pos + torch.stack([d * torch.cos(a), d * torch.sin(a), torch.zeros_like(d)], -1)
  st.d_prev[env_ids] = d


def carry_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  cfg = k1_velocity_env_cfg(rough=False, play=play)
  robot = get_k1_manip_robot_cfg()
  robot.init_state = EntityCfg.InitialStateCfg(
    pos=robot.init_state.pos,
    joint_pos={**{k: v for k, v in robot.init_state.joint_pos.items() if "Shoulder" not in k}, **_hold_joint_pos()},
    joint_vel=robot.init_state.joint_vel)
  cfg.scene.entities = {"robot": robot, "box": EntityCfg(
    spec_fn=box_spec, init_state=EntityCfg.InitialStateCfg(pos=(0.3, 0.0, 0.7)))}
  cfg.scene.spec_fn = add_welds
  cfg.sim.mujoco.ls_iterations = int(os.environ.get("K1_LS_ITERS", "50"))
  cfg.sim.njmax = 600
  cfg.commands = {}
  cfg.curriculum = {}
  cfg.episode_length_s = 20.0

  cfg.actions = {
    "base": FrozenBaseActionCfg(entity_name="robot", checkpoint=BASE_CKPT),
    "arms": JointPositionActionCfg(entity_name="robot", actuator_names=ARM_REGEX, scale=0.25, use_default_offset=True),
    "head": RandomUpperBodyActionCfg(entity_name="robot", joint_names=(".*Head.*",), frozen_prob=1.0),
  }

  arms = SceneEntityCfg("robot", joint_names=ARM_REGEX)
  terms = {
    "gravity": ObservationTermCfg(func=mdp.projected_gravity),
    "ang_vel": ObservationTermCfg(func=mdp.builtin_sensor, params={"sensor_name": "robot/imu_ang_vel"}),
    "goal_pos": ObservationTermCfg(func=goal_pos_b),
    "box_err": ObservationTermCfg(func=box_err_b),
    "box_gravity": ObservationTermCfg(func=box_gravity),
    "weld": ObservationTermCfg(func=weld_obs),
    "arm_pos": ObservationTermCfg(func=mdp.joint_pos_rel, params={"asset_cfg": arms}),
    "arm_vel": ObservationTermCfg(func=mdp.joint_vel_rel, params={"asset_cfg": arms}),
    "last_action": ObservationTermCfg(func=mdp.last_action),
  }
  cfg.observations = {
    "actor": ObservationGroupCfg(dict(terms), enable_corruption=False),
    "critic": ObservationGroupCfg(dict(terms), enable_corruption=False),
  }

  cfg.events = {k: v for k, v in cfg.events.items() if k in ("reset_base", "reset_robot_joints", "foot_friction")}
  cfg.events["reset_base"].params["pose_range"] = {"x": (0.0, 0.0), "y": (0.0, 0.0), "z": (0.01, 0.03), "yaw": (-3.14, 3.14)}
  # arms start exactly at the hold pose (welds satisfied); only the legs are jittered
  cfg.events["reset_robot_joints"].params["asset_cfg"] = SceneEntityCfg("robot", joint_names=LEG_REGEX)
  cfg.events["reset_box"] = EventTermCfg(func=reset_box, mode="reset")

  cfg.rewards = {
    "stand_hold": RewardTermCfg(func=hold_quality, weight=2.0),
    "goal_progress": RewardTermCfg(func=goal_progress_rate, weight=5.0),
    "goal_near": RewardTermCfg(func=goal_near, weight=1.0),
    "success": RewardTermCfg(func=goal_reached, weight=500.0),   # x dt = +10 once
    "failure": RewardTermCfg(func=failure_terminated, weight=-200.0, params={"term_names": FAILURES}),
    "weld_error": RewardTermCfg(func=weld_error, weight=-10.0),
    "box_tilt": RewardTermCfg(func=box_tilt_l2, weight=-2.0),
    "flat_orientation": RewardTermCfg(
      func=mdp.flat_orientation_l2, weight=-2.0, params={"asset_cfg": SceneEntityCfg("robot", body_names=("Trunk",))}),
    "action_rate": RewardTermCfg(func=mdp.action_rate_l2, weight=-0.01),
    "arm_deviation": RewardTermCfg(func=arm_deviation_l1, weight=-0.5, params={"asset_cfg": arms}),
  }
  cfg.terminations = {
    "time_out": TerminationTermCfg(func=mdp.time_out, time_out=True),
    "fell": TerminationTermCfg(func=mdp.bad_orientation, params={"limit_angle": float(os.environ.get("K1_FELL_ANGLE", "0.8"))}),
    "goal_reached": TerminationTermCfg(func=goal_reached),
    "weld_broken": TerminationTermCfg(func=weld_broken),
  }
  return cfg
