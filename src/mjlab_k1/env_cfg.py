"""K1 decoupled lower-body velocity tracking on mjlab: AGILE-style, small reward set.

Policy = 12 leg joints only. Head/arms follow random targets (upper_body.py). Terrain curriculum
(mjlab terrain_levels_vel) on rough procedural terrain. Reward set is mjlab's G1 velocity task
(task + a handful of regularisers), no custom gait shaping, no phase clock.
"""
import os
import math

import torch

from mjlab.envs import ManagerBasedRlEnvCfg, mdp
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.utils.lab_api.math import quat_apply_inverse
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import (
  ContactMatch, ContactSensorCfg, ObjRef, RayCastSensorCfg, RingPatternCfg, TerrainHeightSensorCfg)
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg
from mjlab.tasks.velocity.velocity_env_cfg import make_velocity_env_cfg

from k1_common.k1_constants import K1_ACTION_SCALE, LEG_REGEX, get_k1_robot_cfg
from .upper_body import RandomUpperBodyActionCfg

SITES = ("left_foot", "right_foot")


def stand_still(env, command_name: str, sensor_name: str, asset_cfg: SceneEntityCfg,
                command_threshold: float = 0.1, vel_coef: float = 0.05):
  """Zero-command envs only: feet in the air + leg joint speed. flat_swing10 marched in place at
  1.8 steps/s under a zero command (only 10% standing envs, nothing penalised foot lifts)."""
  cmd = env.command_manager.get_command(command_name)
  standing = (torch.norm(cmd[:, :2], dim=1) + torch.abs(cmd[:, 2])) < command_threshold
  in_air = (env.scene[sensor_name].data.current_air_time > 0).float().sum(1)
  dq = env.scene[asset_cfg.name].data.joint_vel[:, asset_cfg.joint_ids]
  pen = in_air + vel_coef * torch.sum(dq**2, dim=1)
  env.extras["log"]["Metrics/stand_still_air"] = (in_air * standing).sum() / standing.sum().clamp(min=1)
  return pen * standing.float()


def foot_flat(env, sensor_name: str, asset_cfg: SceneEntityCfg) -> torch.Tensor:
  """Feet in contact: squared xy of gravity in the foot frame (0 = sole flat; the frame is flat at HOME).
  flat_swing10 toe-walks on the left foot."""
  asset = env.scene[asset_cfg.name]
  quat = asset.data.body_link_quat_w[:, asset_cfg.body_ids]
  g = quat_apply_inverse(quat, asset.data.gravity_vec_w.reshape(-1, 1, 3).expand(*quat.shape[:2], 3))
  contact = (env.scene[sensor_name].data.current_contact_time > 0).float()
  return torch.sum(contact * torch.sum(g[..., :2] ** 2, dim=-1), dim=1)


def feet_too_close(env, asset_cfg: SceneEntityCfg, min_dist: float = 0.13) -> torch.Tensor:
  """Hinge (m) on the left-right foot spacing in the trunk heading frame below min_dist. walk_still_v1 on
  A1 brought its feet to 3 cm (p5 8 cm under vy commands; 19 cm at HOME) and tripped on itself."""
  asset = env.scene[asset_cfg.name]
  p = asset.data.body_link_pos_w[:, asset_cfg.body_ids]
  rel = quat_apply_inverse(asset.data.root_link_quat_w, p[:, 0] - p[:, 1])  # left - right, trunk frame
  return torch.clamp(min_dist - rel[:, 1], min=0.0)


def track_lin_vel_y(env, command_name: str, std: float = 0.2) -> torch.Tensor:
  """Extra lateral-velocity tracking (walk_still_v1 tracked vy poorly on the robot)."""
  vy = env.scene["robot"].data.root_link_lin_vel_b[:, 1]
  return torch.exp(-(env.command_manager.get_command(command_name)[:, 1] - vy) ** 2 / std**2)


def gait_symmetry(env, sensor_name: str, command_name: str, command_threshold: float = 0.1) -> torch.Tensor:
  """Moving envs: left/right difference of the last swing and last stance durations (s)."""
  d = env.scene[sensor_name].data
  cmd = env.command_manager.get_command(command_name)
  moving = (torch.norm(cmd[:, :2], dim=1) + torch.abs(cmd[:, 2])) > command_threshold
  diff = (d.last_air_time[:, 0] - d.last_air_time[:, 1]).abs() + (d.last_contact_time[:, 0] - d.last_contact_time[:, 1]).abs()
  return diff * moving.float()


class side_shove:
  """Step event: a sideways force on the trunk for a short time, like a person shoving the robot. Respects the robot's
  mass, unlike the velocity kicks of push_robot. Force is along the trunk's y axis +- 30 deg, applied above the trunk
  centre so it also tips. Its top size grows from 30% to full over ramp_steps env steps (counted from this run's start)."""

  def __init__(self, cfg, env):
    self.robot, self.ids = env.scene[cfg.params["asset_cfg"].name], cfg.params["asset_cfg"].body_ids
    self.cooldown = cfg.params["cooldown_s"]
    self.left = torch.zeros(env.num_envs, device=env.device)
    self.wait = self._sample(self.cooldown, env.num_envs, env.device)

  @staticmethod
  def _sample(lo_hi, n, device):
    return lo_hi[0] + (lo_hi[1] - lo_hi[0]) * torch.rand(n, device=device)

  def __call__(self, env, env_ids, force, ramp_steps, duration_s, cooldown_s, asset_cfg):
    del env_ids, cooldown_s, asset_cfg
    dev, on = env.device, self.left > 0
    self.left[on] -= env.step_dt
    done = (on & (self.left <= 0)).nonzero().squeeze(-1)
    if len(done):
      zero = torch.zeros(len(done), 1, 3, device=dev)
      self.robot.write_external_wrench_to_sim(zero, zero, env_ids=done, body_ids=self.ids)
      self.wait[done] = self._sample(self.cooldown, len(done), dev)
    self.wait[~on] -= env.step_dt
    fire = (~on & (self.wait <= 0)).nonzero().squeeze(-1)
    if not len(fire):
      return
    n = len(fire)
    top = force * (0.3 + 0.7 * min(1.0, env.common_step_counter / ramp_steps))
    size = top * (0.3 + 0.7 * torch.rand(n, device=dev)) * torch.sign(torch.rand(n, device=dev) - 0.5)
    q = self.robot.data.body_link_quat_w[fire][:, self.ids[0]]
    yaw = torch.atan2(2.0 * (q[:, 0] * q[:, 3] + q[:, 1] * q[:, 2]), 1.0 - 2.0 * (q[:, 2] ** 2 + q[:, 3] ** 2))
    ang = yaw + math.pi / 2 + math.radians(30.0) * (2.0 * torch.rand(n, device=dev) - 1.0)
    forces = torch.stack([size * ang.cos(), size * ang.sin(), torch.zeros(n, device=dev)], -1)[:, None]
    above = torch.tensor((0.0, 0.0, 0.15), device=dev).expand_as(forces)
    self.robot.write_external_wrench_to_sim(forces, torch.cross(above, forces, dim=-1), env_ids=fire, body_ids=self.ids)
    self.left[fire] = self._sample(duration_s, n, dev)
    self.wait[fire] = 0.0

  def reset(self, env_ids=None):
    ids = torch.arange(len(self.left), device=self.left.device) if env_ids is None or isinstance(env_ids, slice) else env_ids
    active = ids[self.left[ids] > 0]
    if len(active):
      zero = torch.zeros(len(active), 1, 3, device=self.left.device)
      self.robot.write_external_wrench_to_sim(zero, zero, env_ids=active, body_ids=self.ids)
    self.left[ids] = 0.0
    self.wait[ids] = self._sample(self.cooldown, len(ids), self.left.device)


def k1_velocity_env_cfg(rough: bool = True, play: bool = False) -> ManagerBasedRlEnvCfg:
  cfg = make_velocity_env_cfg()
  cfg.sim.mujoco.ccd_iterations = 50
  cfg.sim.njmax = 300
  cfg.sim.nconmax = 40
  cfg.scene.entities = {"robot": get_k1_robot_cfg()}

  legs = SceneEntityCfg("robot", joint_names=LEG_REGEX)

  # sensors
  for s in cfg.scene.sensors or ():
    if s.name == "terrain_scan":
      assert isinstance(s, RayCastSensorCfg)
      s.frame.name = "Trunk"
    if s.name == "foot_height_scan":
      assert isinstance(s, TerrainHeightSensorCfg)
      s.frame = tuple(ObjRef(type="site", name=n, entity="robot") for n in SITES)
      s.pattern = RingPatternCfg.single_ring(radius=0.03, num_samples=6)
  feet = ContactSensorCfg(
    name="feet_ground_contact",
    primary=ContactMatch(mode="subtree", pattern=r"^(left_foot_link|right_foot_link)$", entity="robot"),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found", "force"), reduce="netforce", num_slots=1, track_air_time=True)
  cfg.scene.sensors = (cfg.scene.sensors or ()) + (feet,)

  # actions: legs from the policy, upper body from the random driver
  cfg.actions = {
    "joint_pos": JointPositionActionCfg(
      entity_name="robot", actuator_names=LEG_REGEX, scale=K1_ACTION_SCALE, use_default_offset=True,
      preserve_order=True),
    "upper_body": RandomUpperBodyActionCfg(entity_name="robot"),
  }

  # observations: actor sees only what the real robot measures (no base_lin_vel), legs only
  actor = cfg.observations["actor"].terms
  actor.pop("base_lin_vel")
  actor["joint_pos"].params["asset_cfg"] = legs
  actor["joint_vel"].params["asset_cfg"] = legs
  cfg.observations["actor"].history_length = 5  # TODO: confirm field semantics (GEAR uses 6)
  cfg.observations["critic"].terms["joint_pos"].params = {"asset_cfg": SceneEntityCfg("robot")}

  # command: AGILE-like adaptive bias toward low speeds is not available; keep mjlab's 10% standing
  twist = cfg.commands["twist"]
  assert isinstance(twist, UniformVelocityCommandCfg)
  twist.ranges.lin_vel_x = (-0.6, 0.8)   # K1 is 0.95 m tall: G1 numbers are too fast
  twist.ranges.lin_vel_y = (-0.4, 0.4)
  twist.ranges.ang_vel_z = (-0.8, 0.8)
  twist.viz.z_offset = 0.8
  twist.rel_standing_envs = float(os.environ.get("K1_STAND_FRAC", "0.1"))
  cfg.curriculum["command_vel"].params["velocity_stages"] = [
    {"step": 0, "lin_vel_x": (-0.4, 0.5), "lin_vel_y": (-0.2, 0.2), "ang_vel_z": (-0.5, 0.5)},
    {"step": 3000 * 24, "lin_vel_x": (-0.6, 0.8), "lin_vel_y": (-0.4, 0.4), "ang_vel_z": (-0.8, 0.8)},
    {"step": 8000 * 24, "lin_vel_x": (-0.8, 1.0), "lin_vel_y": (-0.5, 0.5), "ang_vel_z": (-1.0, 1.0)},
  ]

  # events
  cfg.events["foot_friction"].params["asset_cfg"].geom_names = ("left_foot1_collision", "right_foot1_collision")
  cfg.events["base_com"].params["asset_cfg"].body_names = ("Trunk",)
  cfg.events["push_robot"].params["velocity_range"] = {
    "x": (-0.4, 0.4), "y": (-0.4, 0.4), "z": (-0.2, 0.2), "roll": (-0.3, 0.3), "pitch": (-0.3, 0.3), "yaw": (-0.5, 0.5)}
  shove_n = float(os.environ.get("K1_SHOVE_N", "0"))  # top sideways force, N; 0 = no shoves
  if shove_n > 0 and not play:
    cfg.events["side_shove"] = EventTermCfg(func=side_shove, mode="step", params={
      "force": shove_n, "ramp_steps": int(os.environ.get("K1_SHOVE_RAMP_ITERS", "2000")) * 24,
      "duration_s": (0.1, 0.3), "cooldown_s": (2.0, 5.0), "asset_cfg": SceneEntityCfg("robot", body_names=("Trunk",))})

  # rewards: mjlab's set, pose reward restricted to legs
  r = cfg.rewards
  r["pose"].params["asset_cfg"] = legs
  r["pose"].params["std_standing"] = {".*": 0.05}
  walk = {r".*Hip_Pitch": 0.3, r".*Hip_Roll": 0.15, r".*Hip_Yaw": 0.15, r".*Knee.*": 0.35,
          r".*Ankle_Pitch": 0.25, r".*Ankle_Roll": 0.1}
  r["pose"].params["std_walking"] = walk
  r["pose"].params["std_running"] = {k: 1.5 * v for k, v in walk.items()}
  r["upright"].params["asset_cfg"].body_names = ("Trunk",)
  r["body_ang_vel"].params["asset_cfg"].body_names = ("Trunk",)
  r["body_ang_vel"].weight = -0.05
  r["angular_momentum"].weight = 0.0
  for name in ("foot_clearance", "foot_slip"):
    r[name].params["asset_cfg"].site_names = SITES
  # force K1-scale swing (~0.2-0.45 s, ~2.5-3 steps/s); baseline without it shuffled at 5.8 steps/s
  r["air_time"].weight = float(os.environ.get("K1_AIR_W", "0.8"))
  r["air_time"].params.update(threshold_min=0.2, threshold_max=0.45, command_threshold=0.15)
  # K1 is ~0.55 m trunk height: scale foot targets (G1 values are for 0.78 m)
  r["foot_clearance"].params["target_height"] = 0.06
  # swing height is scored once per landing, so it needs a big weight; baseline peaked at ~3 cm (p95)
  r["foot_swing_height"].params["target_height"] = float(os.environ.get("K1_SWING_H", "0.07"))
  r["foot_swing_height"].weight = -float(os.environ.get("K1_SWING_W", "10.0"))
  r["foot_clearance"].weight = -1.0
  # opt-in quiet stand at zero command (walk_still runs); 0 = off so the task envs keep their recipe
  stand_w = float(os.environ.get("K1_STAND_W", "0"))
  if stand_w:
    r["stand_still"] = RewardTermCfg(func=stand_still, weight=-stand_w, params={
      "command_name": "twist", "sensor_name": "feet_ground_contact", "asset_cfg": legs})
  # opt-in full-sole, even left/right steps
  flat_w, sym_w = float(os.environ.get("K1_FLAT_W", "0")), float(os.environ.get("K1_SYM_W", "0"))
  if flat_w:
    r["foot_flat"] = RewardTermCfg(func=foot_flat, weight=-flat_w, params={
      "sensor_name": "feet_ground_contact",
      "asset_cfg": SceneEntityCfg("robot", body_names=("left_foot_link", "right_foot_link"), preserve_order=True)})
  # opt-in foot spacing hinge and extra vy tracking (walk_still_v2 fine-tune)
  feet_w, vy_w = float(os.environ.get("K1_FEET_W", "0")), float(os.environ.get("K1_VY_W", "0"))
  if feet_w:
    r["feet_too_close"] = RewardTermCfg(func=feet_too_close, weight=-feet_w, params={
      "min_dist": float(os.environ.get("K1_FEET_MIN", "0.13")),
      "asset_cfg": SceneEntityCfg("robot", body_names=("left_foot_link", "right_foot_link"), preserve_order=True)})
  if vy_w:
    r["track_lin_vel_y"] = RewardTermCfg(func=track_lin_vel_y, weight=vy_w, params={"command_name": "twist"})
  if sym_w:
    r["gait_symmetry"] = RewardTermCfg(func=gait_symmetry, weight=-sym_w, params={
      "sensor_name": "feet_ground_contact", "command_name": "twist"})

  # opt-in smooth + fast fine-tune (walk v3): action rate, joint accel, trunk rate, landing, tracking weight, top speed
  r["action_rate_l2"].weight = -float(os.environ.get("K1_ACT_RATE_W", "0.1"))
  r["body_ang_vel"].weight = -float(os.environ.get("K1_ANG_VEL_W", "0.05"))
  r["soft_landing"].weight = -float(os.environ.get("K1_LAND_W", "1e-5"))
  acc_w = float(os.environ.get("K1_JOINT_ACC_W", "0"))
  if acc_w:
    r["joint_acc"] = RewardTermCfg(func=mdp.joint_acc_l2, weight=-acc_w, params={"asset_cfg": legs})
  track_w = float(os.environ.get("K1_TRACK_W", "2"))
  r["track_linear_velocity"].weight = track_w
  vx_max = float(os.environ.get("K1_VX_MAX", "0"))
  if vx_max:
    stages = cfg.curriculum["command_vel"].params["velocity_stages"]
    stages[-1] = {**stages[-1], "lin_vel_x": (-0.8, vx_max), "ang_vel_z": (-1.2, 1.2)}

  # termination: 70 deg bad-orientation is mjlab default; keep. Viewer.
  cfg.viewer.body_name = "Trunk"

  if not rough:
    cfg.scene.terrain.terrain_type = "plane"
    cfg.scene.terrain.terrain_generator = None
    cfg.scene.sensors = tuple(s for s in cfg.scene.sensors if s.name != "terrain_scan")
    for g in ("actor", "critic"):
      cfg.observations[g].terms.pop("height_scan", None)
    cfg.terminations.pop("out_of_terrain_bounds", None)
    cfg.curriculum.pop("terrain_levels", None)
  else:
    cfg.scene.terrain.terrain_generator.curriculum = True

  if play:
    cfg.episode_length_s = int(1e9)
    cfg.observations["actor"].enable_corruption = False
    cfg.events.pop("push_robot", None)
  return cfg
