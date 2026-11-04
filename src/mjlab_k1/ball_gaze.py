"""Head servo that keeps the ball's bounding box centred in the head camera. No learning: bbox centre error -> head joints.

On the robot the bbox comes from YOLO on the head RGB. In training it comes from detect(), a pinhole projection with the
detector's noise, dropouts and latency; scripts/kick_yolo_check.py compares it with real yolov8n on rendered frames.
Lost ball: hold for HOLD_S, then a slow scan.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

import torch

from mjlab.managers.action_manager import ActionTerm, ActionTermCfg
from mjlab.utils.lab_api.math import quat_apply, quat_apply_inverse

from .tasks_head import CAM_OFFSET, HEAD_JOINTS, HFOV, VFOV
from .tasks_kick import BALL_RADIUS

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

TAN_H, TAN_V = math.tan(HFOV / 2), math.tan(VFOV / 2)
HOLD_S, SCAN_YAW, SCAN_HZ, SCAN_PITCH = 1.0, 0.8, 0.25, 0.6
# The velocity tasks' head gain (kp 0.63 from armature) cannot hold the head up: it sags to the pitch limit whatever the
# target. kp is from Booster's official K1 configs (4.0); their kd 1.0 lags 0.25 s and makes this servo limit-cycle, so kd is lower.
# Check head lag on the robot before reusing the gain: with kd 1.0 it needs gain near 0.06.
HEAD_KP, HEAD_KD = 4.0, 0.2


def detect(env, entity_name: str = "robot", noise: float = 0.0, dropout: float = 0.0):
  """Bbox of the ball in the head camera: (n, 4) centre u, v (right, up, -1..1 at the frame edge), width, height (fraction
  of the frame), and seen (n,)."""
  r = env.scene[entity_name]
  bid = r.find_bodies(("Head_2",))[0][0]
  q = r.data.body_link_quat_w[:, bid]
  cam = r.data.body_link_pos_w[:, bid] + quat_apply(q, torch.tensor(CAM_OFFSET, device=env.device).expand(env.num_envs, 3))
  rel = quat_apply_inverse(q, env.scene["ball"].data.root_link_pos_w - cam)  # x forward, y left, z up
  x = rel[:, 0].clamp_min(1e-3)
  u, v = -rel[:, 1] / x / TAN_H, rel[:, 2] / x / TAN_V
  ang = BALL_RADIUS / x
  box = torch.stack([u, v, ang / TAN_H, ang / TAN_V], -1)
  seen = (rel[:, 0] > 0.05) & (u.abs() < 1.0) & (v.abs() < 1.0)
  if noise:
    box[:, :2] += noise * torch.randn_like(box[:, :2])
  if dropout:
    seen = seen & (torch.rand(env.num_envs, device=env.device) > dropout)
  return box, seen


@dataclass(kw_only=True)
class BallGazeActionCfg(ActionTermCfg):
  gain: float = 0.1  # fraction of the angular error corrected each 50 Hz step
  max_speed: float = 3.0  # rad/s
  noise: float = 0.015  # bbox centre, fraction of half frame
  dropout: float = 0.1
  latency_steps: int = 2  # about 40 ms at 50 Hz

  def build(self, env: "ManagerBasedRlEnv") -> "BallGazeAction":
    return BallGazeAction(self, env)


class BallGazeAction(ActionTerm):
  cfg: BallGazeActionCfg

  def __init__(self, cfg: BallGazeActionCfg, env: "ManagerBasedRlEnv"):
    super().__init__(cfg=cfg, env=env)
    ids, names = self._entity.find_joints(HEAD_JOINTS, preserve_order=True)
    self._ids = torch.tensor(ids, device=self.device, dtype=torch.long)
    assert "yaw" in names[0].lower() and "pitch" in names[1].lower(), names
    lim = self._entity.data.soft_joint_pos_limits[:, self._ids]
    self._lo, self._hi = lim[..., 0], lim[..., 1]
    n = self.num_envs
    self._target = torch.zeros(n, 2, device=self.device)
    self._target[:, 1] = SCAN_PITCH
    self._raw = torch.zeros(n, 0, device=self.device)
    self._lost = torch.zeros(n, device=self.device)
    self._queue: list[tuple[torch.Tensor, torch.Tensor]] = []
    self.box = torch.zeros(n, 4, device=self.device)  # last detection the servo acted on, for the student's observation
    self.seen = torch.zeros(n, dtype=torch.bool, device=self.device)

  @property
  def action_dim(self) -> int:
    return 0

  @property
  def raw_action(self) -> torch.Tensor:
    return self._raw

  def process_actions(self, actions: torch.Tensor) -> None:
    self._queue.append(detect(self._env, self.cfg.entity_name, self.cfg.noise, self.cfg.dropout))
    if len(self._queue) <= self.cfg.latency_steps:
      return
    box, seen = self._queue.pop(0)
    self.box, self.seen = box, seen
    dt = self._env.step_dt
    self._lost = torch.where(seen, torch.zeros_like(self._lost), self._lost + dt)
    # image error -> angle; yaw is positive left, pitch positive down, so both signs are negative of u and v
    err = torch.stack([-torch.atan(box[:, 0] * TAN_H), -torch.atan(box[:, 1] * TAN_V)], -1)
    step = (self.cfg.gain * err).clamp(-self.cfg.max_speed * dt, self.cfg.max_speed * dt)
    tgt = torch.where(seen[:, None], self._target + step, self._target)
    t = self._env.episode_length_buf.float() * dt
    scan = torch.stack([SCAN_YAW * torch.sin(2 * math.pi * SCAN_HZ * t), torch.full_like(t, SCAN_PITCH)], -1)
    tgt = torch.where((self._lost > HOLD_S)[:, None], scan, tgt)
    self._target = torch.max(torch.min(tgt, self._hi), self._lo)

  def apply_actions(self) -> None:
    self._entity.set_joint_position_target(self._target, joint_ids=self._ids)

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    if env_ids is None:
      env_ids = slice(None)
    self._target[env_ids, 0] = 0.0
    self._target[env_ids, 1] = SCAN_PITCH
    self._lost[env_ids] = 0.0
    self.seen[env_ids] = False
    self.box[env_ids] = 0.0
    for box, seen in self._queue:  # stale detections from the last episode
      seen[env_ids] = False


def attach_ball_gaze(cfg, entity_name: str = "robot", upper: str = "upper_body", term: str = "gaze", **kwargs) -> None:
  """Give one robot the servo: head gains that hold the head, and head joints taken off that robot's random upper body."""
  cfg.actions[upper].joint_names = (".*Shoulder.*", ".*Elbow.*")
  art = cfg.scene.entities[entity_name].articulation
  art.actuators = tuple(
    replace(a, stiffness=HEAD_KP, damping=HEAD_KD) if any("Head" in n for n in a.target_names_expr) else a
    for a in art.actuators)
  cfg.actions[term] = BallGazeActionCfg(entity_name=entity_name, **kwargs)
