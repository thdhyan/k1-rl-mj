"""Policy-less upper-body driver: the decoupled-WBC 'other half'.

The leg policy never commands head/arms. During training they are driven by random joint targets
(AGILE: 'upper-body joints receive random targets through trapezoidal velocity profiles'), so the
leg policy learns to balance under any upper-body motion, and a separate controller (IK/teleop/
residual skill) can own these joints at deploy time. action_dim is 0.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from mjlab.managers.action_manager import ActionTerm, ActionTermCfg

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


@dataclass(kw_only=True)
class RandomUpperBodyActionCfg(ActionTermCfg):
  joint_names: tuple[str, ...] = (".*Head.*", ".*Shoulder.*", ".*Elbow.*")
  amplitude: float = 0.6
  """Fraction of the soft joint range the random goals may span (around the default pose)."""
  max_speed: float = 2.0  # rad/s, per-joint velocity cap of the trapezoid ramp
  hold_time_range_s: tuple[float, float] = (0.5, 3.0)
  frozen_prob: float = 0.2  # envs that keep the default arm pose (deploy-time 'arms idle')

  def build(self, env: "ManagerBasedRlEnv") -> "RandomUpperBodyAction":
    return RandomUpperBodyAction(self, env)


class RandomUpperBodyAction(ActionTerm):
  cfg: RandomUpperBodyActionCfg

  def __init__(self, cfg: RandomUpperBodyActionCfg, env: "ManagerBasedRlEnv"):
    super().__init__(cfg=cfg, env=env)
    ids, _ = self._entity.find_joints(cfg.joint_names)
    self._ids = torch.tensor(ids, device=self.device, dtype=torch.long)
    n, k = self.num_envs, len(ids)
    self._default = self._entity.data.default_joint_pos[:, self._ids].clone()
    lim = self._entity.data.soft_joint_pos_limits[:, self._ids]  # (n, k, 2)
    self._lo = self._default + cfg.amplitude * (lim[..., 0] - self._default)
    self._hi = self._default + cfg.amplitude * (lim[..., 1] - self._default)
    self._cur = self._default.clone()
    self._goal = self._default.clone()
    self._timer = torch.zeros(n, device=self.device)
    self._frozen = torch.zeros(n, dtype=torch.bool, device=self.device)
    self._raw = torch.zeros(n, 0, device=self.device)

  @property
  def action_dim(self) -> int:
    return 0

  @property
  def raw_action(self) -> torch.Tensor:
    return self._raw

  def process_actions(self, actions: torch.Tensor) -> None:
    pass

  def _resample(self, env_ids: torch.Tensor) -> None:
    u = torch.rand(len(env_ids), len(self._ids), device=self.device)
    goal = self._lo[env_ids] + u * (self._hi[env_ids] - self._lo[env_ids])
    self._goal[env_ids] = torch.where(self._frozen[env_ids, None], self._default[env_ids], goal)
    a, b = self.cfg.hold_time_range_s
    self._timer[env_ids] = a + (b - a) * torch.rand(len(env_ids), device=self.device)

  def apply_actions(self) -> None:
    dt = self._env.step_dt
    self._timer -= dt
    due = (self._timer <= 0).nonzero(as_tuple=False).flatten()
    if len(due):
      self._resample(due)
    step = self.cfg.max_speed * dt
    self._cur += torch.clamp(self._goal - self._cur, -step, step)
    self._entity.set_joint_position_target(self._cur, joint_ids=self._ids)

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    if env_ids is None:
      env_ids = slice(None)
    ids = torch.arange(self.num_envs, device=self.device)[env_ids]
    self._cur[ids] = self._default[ids]
    self._frozen[ids] = torch.rand(len(ids), device=self.device) < self.cfg.frozen_prob
    self._resample(ids)
