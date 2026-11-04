"""Booster K1 (22 DoF) for mjlab.

MJCF is Booster's own ``K1_22dof.xml``, resolved by :mod:`k1_description` (override: ``$K1_XML``). Actuator numbers
(effort, velocity, armature) and the PD recipe come from Booster's reference
``booster_train/assets/robots/booster.py``: kp = armature * (2 pi f)^2, kd = 2 zeta armature (2 pi f)
with f = 4 Hz. Do NOT stiffen these to make learning easier; fix the policy side instead
(AGILE LESSONS_LEARNED: if hardware differs from sim, fix the sim).
"""
import math
import os
import re
from pathlib import Path

import k1_description
import mujoco

from mjlab.actuator import BuiltinPositionActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.utils.spec_config import CollisionCfg

K1_XML = Path(os.environ.get("K1_XML") or k1_description.robot_file(k1_description.MJCF))

FOOT_GEOMS = ("left_foot1_collision", "right_foot1_collision")
LEG_JOINTS = (
  "Left_Hip_Pitch", "Left_Hip_Roll", "Left_Hip_Yaw", "Left_Knee_Pitch", "Left_Ankle_Pitch", "Left_Ankle_Roll",
  "Right_Hip_Pitch", "Right_Hip_Roll", "Right_Hip_Yaw", "Right_Knee_Pitch", "Right_Ankle_Pitch", "Right_Ankle_Roll",
)
LEG_REGEX = (".*_Hip_Pitch", ".*_Hip_Roll", ".*_Hip_Yaw", ".*_Knee_Pitch", ".*_Ankle_Pitch", ".*_Ankle_Roll")
UPPER_REGEX = (".*Head.*", ".*Shoulder.*", ".*Elbow.*")
ARM_REGEX = (".*Shoulder.*", ".*Elbow.*")
ARM_BODY_REGEX = r"(Left|Right)_Arm_\d|(left|right)_hand_link"
HAND_BODIES = ("left_hand_link", "right_hand_link")


def get_spec() -> mujoco.MjSpec:
  spec = mujoco.MjSpec.from_file(str(K1_XML))
  # floor comes from the mjlab terrain entity
  spec.delete(spec.geom("ground"))
  for a in list(spec.actuators):
    spec.delete(a)
  # feet are the only colliding geoms (FEET_ONLY recipe); falls end the episode on orientation
  for body_name, side in (("left_foot_link", "left"), ("right_foot_link", "right")):
    body = spec.body(body_name)
    for g in body.geoms:
      if g.type == mujoco.mjtGeom.mjGEOM_BOX:
        g.name = f"{side}_foot1_collision"
    body.add_site(name=f"{side}_foot", pos=(0.026, 0.0, -0.038), size=(0.01, 0.01, 0.01))
  # name Booster's arm/hand collision geoms (group 0; group 1 = visual copies) so MANIP can enable them
  for b in spec.bodies:
    if re.fullmatch(ARM_BODY_REGEX, b.name):
      for i, g in enumerate(x for x in b.geoms if x.group != 1):
        g.name = f"{b.name}_collision{i}"
  for g in spec.geoms:
    if g.name not in FOOT_GEOMS:
      g.contype = 0
      g.conaffinity = 0
  trunk = spec.body("Trunk")
  # sensors expected by the mjlab velocity task
  spec.add_sensor(name="imu_ang_vel", type=mujoco.mjtSensor.mjSENS_GYRO, objtype=mujoco.mjtObj.mjOBJ_SITE, objname="imu")
  spec.add_sensor(name="imu_lin_vel", type=mujoco.mjtSensor.mjSENS_VELOCIMETER, objtype=mujoco.mjtObj.mjOBJ_SITE, objname="imu")
  spec.add_sensor(name="imu_lin_acc", type=mujoco.mjtSensor.mjSENS_ACCELEROMETER, objtype=mujoco.mjtObj.mjOBJ_SITE, objname="imu")
  spec.add_sensor(name="imu_upvector", type=mujoco.mjtSensor.mjSENS_FRAMEZAXIS, objtype=mujoco.mjtObj.mjOBJ_BODY,
                  objname="world", reftype=mujoco.mjtObj.mjOBJ_SITE, refname="imu")
  spec.add_sensor(name="root_angmom", type=mujoco.mjtSensor.mjSENS_SUBTREEANGMOM, objtype=mujoco.mjtObj.mjOBJ_BODY,
                  objname=trunk.name)
  return spec


##
# Actuators (Booster joint models; 4 Hz natural frequency as in Booster's own config).
##
NATURAL_FREQ = 4.0 * 2.0 * math.pi


def _pd(armature: float, zeta: float = 1.5, ratio: float = 1.0):
  a = armature * ratio
  return a * NATURAL_FREQ**2, 2.0 * zeta * a * NATURAL_FREQ, a


def _act(names, effort, armature, zeta=1.5, ratio=1.0, delay=True):
  kp, kd, arm = _pd(armature, zeta, ratio)
  return BuiltinPositionActuatorCfg(
    target_names_expr=names, stiffness=kp, damping=kd, effort_limit=effort, armature=arm,
    # Booster's training config delays commands by 2-8 physics steps (5 ms) = 10-40 ms
    delay_min_lag=2 if delay else 0, delay_max_lag=8 if delay else 0,
  )


K1_ACTUATORS = (
  _act((".*_Hip_Pitch",), 68.0, 0.0478125),
  _act((".*_Hip_Roll",), 76.0, 0.0339552),
  _act((".*_Hip_Yaw",), 38.3, 0.0282528),
  _act((".*_Knee_Pitch",), 112.0, 0.095625, zeta=1.0),
  # K1 ankle is a 4-bar linkage: pitch/roll armature ratio (1.4, 0.4) of an E4310
  _act((".*_Ankle_Pitch",), 38.3, 0.0282528, ratio=1.4),
  _act((".*_Ankle_Roll",), 38.3, 0.0282528, ratio=0.4),
  _act((".*Shoulder.*", ".*Elbow.*"), 14.0, 0.001, zeta=1.0),
  _act((".*Head.*",), 6.0, 0.001, zeta=1.0),
)

# knees bent, flat feet (checked with forward kinematics: sole level, trunk 0.533 m above sole)
HOME = EntityCfg.InitialStateCfg(
  pos=(0.0, 0.0, 0.56),
  joint_pos={
    ".*_Hip_Pitch": -0.3, ".*_Knee_Pitch": 0.6, ".*_Ankle_Pitch": -0.3,
    "Left_Shoulder_Roll": -1.3, "Right_Shoulder_Roll": 1.3,
    ".*Shoulder_Pitch": 0.0,
  },
  joint_vel={".*": 0.0},
)

FEET_ONLY = CollisionCfg(
  geom_names_expr=FOOT_GEOMS, contype=0, conaffinity=1, condim=3, priority=1, friction=(0.6,))

K1_ARTICULATION = EntityArticulationInfoCfg(actuators=K1_ACTUATORS, soft_joint_pos_limit_factor=0.9)


def get_k1_robot_cfg() -> EntityCfg:
  return EntityCfg(init_state=HOME, collisions=(FEET_ONLY,), spec_fn=get_spec, articulation=K1_ARTICULATION)


# Manipulation tasks (push/carry): feet + arm/hand collisions, arms at the Isaac push/carry gains (kp 40, kd 2).
# One CollisionCfg: each cfg zeroes the geoms it does not match.
_FOOT = r".*_foot1_collision"
MANIP = CollisionCfg(
  geom_names_expr=(_FOOT, r".*_collision\d+"), contype={_FOOT: 0, ".*": 1}, conaffinity=1, condim=3,
  priority={_FOOT: 1, ".*": 0}, friction=(0.6,))
ARM_KP, ARM_KD = 40.0, 2.0


def get_k1_manip_robot_cfg() -> EntityCfg:
  arms = BuiltinPositionActuatorCfg(
    target_names_expr=ARM_REGEX, stiffness=ARM_KP, damping=ARM_KD, effort_limit=14.0, armature=0.001,
    delay_min_lag=2, delay_max_lag=8)
  acts = tuple(a for a in K1_ACTUATORS if a.target_names_expr != ARM_REGEX) + (arms,)
  return EntityCfg(init_state=HOME, collisions=(MANIP,), spec_fn=get_spec,
                   articulation=EntityArticulationInfoCfg(actuators=acts, soft_joint_pos_limit_factor=0.9))


# action = offset + scale * a; scale so |a|=1 is 0.25 * effort / kp (mjlab convention)
K1_ACTION_SCALE: dict[str, float] = {}
for _a in K1_ACTUATORS[:6]:
  for _n in _a.target_names_expr:
    K1_ACTION_SCALE[_n] = 0.25 * _a.effort_limit / _a.stiffness


# Get-up: every Booster collision geom touches the ground (no self-collision), arms at the manip gains so the
# hands can push the trunk up. Legs and head keep the training gains.
GETUP = CollisionCfg(
  geom_names_expr=(_FOOT, r".*_collision\d+"), contype=0, conaffinity=1, condim=3,
  priority={_FOOT: 1, ".*": 0}, friction=(0.6,))


def get_getup_spec() -> mujoco.MjSpec:
  spec = get_spec()
  for b in spec.bodies:
    for i, g in enumerate(x for x in b.geoms if x.group != 1 and not x.name):
      g.name = f"{b.name}_body_collision{i}"
  return spec


def get_k1_getup_robot_cfg() -> EntityCfg:
  manip = get_k1_manip_robot_cfg()
  return EntityCfg(init_state=HOME, collisions=(GETUP,), spec_fn=get_getup_spec, articulation=manip.articulation)
