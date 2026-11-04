"""Frozen lower-body policy as an action term: the HRL 'actuator' for every layer above it.

The task policy outputs a velocity command (vx, vy, wz) [+ optional 12-dim leg residual]; this term
rebuilds the walk policy's 225-dim actor observation (230 for Mjlab-Velocity-Height-* bases such as
walk_still_v1, fed the stand height; detected from the checkpoint) (5-step term-major history, same as
scripts/export_policy.py / policy_meta.json), runs the frozen MLP and writes leg PD targets.
Nothing here is trained. Upper body (head/arms) stays with another action term.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from mjlab.managers.action_manager import ActionTerm, ActionTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg

from mjlab.tasks.velocity import mdp

from k1_common.k1_constants import K1_ACTION_SCALE, LEG_REGEX
from k1_runtime.wbc import H_STAND, load_frozen_mlp

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

HISTORY = 5
CMD_DIM = 3


@dataclass(kw_only=True)
class FrozenBaseActionCfg(ActionTermCfg):
  checkpoint: str = ""
  """rsl-rl .pt of the walk policy (Mjlab-Velocity-*-Booster-K1)."""
  residual: bool = False
  """If True the task action is (vx, vy, wz, 12 leg deltas); residual_scale scales the deltas."""
  residual_scale: float = 0.3
  residual_gate: Callable | None = None
  """gate(env) -> (n,) weight in [0, 1] on the residual: 0 leaves the frozen base alone."""
  cmd_gate: Callable | None = None
  """cmd_gate(env) -> (n,) weight in [0, 1] on the velocity command (0 holds the robot in place). Default off."""
  residual_replaces_base: bool = False
  """If True the gate blends from the base output to the task action instead of adding to it (gate 1 = base off)."""
  cmd_scale: tuple[float, float, float] = (1.0, 0.5, 1.0)
  """Task action in [-1, 1] -> command; scale = max |vx|, |vy|, |wz| the base was trained for."""
  auto_cmd: tuple[tuple[float, float], tuple[float, float], tuple[float, float]] | None = None
  """If set, the task does not drive the base (action_dim 0): each reset samples a (vx, vy, wz) command
  uniformly from these ranges, held for the episode (layers that ride on a walking robot, e.g. head)."""
  auto_stand_prob: float = 0.3
  command_name: str | None = None
  """If set, the base command comes from this command term (e.g. "twist") and the task action is only the
  12 leg residuals (needs residual=True): a residual skill trained under the velocity command, e.g. steps."""

  height_range: float = 0.0
  """> 0: a 4th task action sets the trunk height: a in [-1, 0] -> H_STAND + a * height_range, a > 0 -> H_STAND.
  Height bases only (230-dim obs)."""
  cmd_alpha: float = 1.0
  """First-order filter on the command each step (1 = off), so a noisy task layer does not shake the walk base."""

  def build(self, env: "ManagerBasedRlEnv") -> "FrozenBaseAction":
    return FrozenBaseAction(self, env)


class FrozenBaseAction(ActionTerm):
  cfg: FrozenBaseActionCfg

  def __init__(self, cfg: FrozenBaseActionCfg, env: "ManagerBasedRlEnv"):
    super().__init__(cfg=cfg, env=env)
    # the trained base's action term resolves to model joint order (L leg 6, R leg 6), same as its
    # joint obs terms; find_joints(preserve_order=True) would give regex-group order and permute actions.
    ids, names = self._entity.find_joints(LEG_REGEX)
    self._ids = torch.tensor(ids, device=self.device, dtype=torch.long)
    self._default = self._entity.data.default_joint_pos[:, self._ids].clone()
    self._oids, self._odefault = self._ids, self._default
    import re
    self._scale = torch.tensor(
      [next(v for k, v in K1_ACTION_SCALE.items() if re.fullmatch(k, n)) for n in names], device=self.device)
    self._mlp, self._mean, self._std = load_frozen_mlp(cfg.checkpoint, self.device)
    n = self.num_envs
    self._off = 0 if cfg.command_name else CMD_DIM
    self._dim = 0 if cfg.auto_cmd else self._off + (12 if cfg.residual else 0) + (cfg.height_range > 0)
    assert not (cfg.height_range and self._mean.numel() != 230), "height_range needs a 230-dim height base"
    self._raw = torch.zeros(n, self._dim, device=self.device)
    self._cmd = torch.zeros(n, CMD_DIM, device=self.device)
    self._last = torch.zeros(n, 12, device=self.device)
    self._sizes = (3, 3, 12, 12, 12, 3) + (1,) * (self._mean.numel() == 230)  # + height cmd channel
    self._hist = torch.zeros(n, HISTORY, sum(self._sizes), device=self.device)
    self._h = torch.full((n, 1), H_STAND, device=self.device)
    self._fresh = torch.ones(n, dtype=torch.bool, device=self.device)
    self._cmd_scale = torch.tensor(cfg.cmd_scale, device=self.device)
    self._target = self._default.clone()

  @property
  def action_dim(self) -> int:
    return self._dim

  @property
  def raw_action(self) -> torch.Tensor:
    return self._raw

  @property
  def command(self) -> torch.Tensor:
    """Velocity command currently fed to the base (n, 3)."""
    return self._cmd

  def _frame(self) -> torch.Tensor:
    d = self._entity.data
    return torch.cat([
      mdp.builtin_sensor(self._env, f"{self.cfg.entity_name}/imu_ang_vel"),
      mdp.projected_gravity(self._env, SceneEntityCfg(self.cfg.entity_name)),
      d.joint_pos[:, self._oids] - self._odefault, d.joint_vel[:, self._oids],
      self._last, self._cmd] + [self._h] * (len(self._sizes) == 7), dim=-1)

  def process_actions(self, actions: torch.Tensor) -> None:
    self._raw[:] = actions
    if self.cfg.command_name:
      self._cmd = self._env.command_manager.get_command(self.cfg.command_name)[:, :CMD_DIM].clone()
    elif not self.cfg.auto_cmd:
      cmd = torch.clamp(actions[:, :CMD_DIM], -1.0, 1.0) * self._cmd_scale
      if self.cfg.cmd_gate is not None:
        cmd = cmd * self.cfg.cmd_gate(self._env)[:, None]
      self._cmd = self._cmd + self.cfg.cmd_alpha * (cmd - self._cmd)
    if self.cfg.height_range:
      self._h = H_STAND + self.cfg.height_range * torch.clamp(actions[:, CMD_DIM:CMD_DIM + 1], -1.0, 0.0)
    frame = self._frame()
    fresh = self._fresh
    self._hist = torch.where(
      fresh[:, None, None], frame[:, None, :].expand_as(self._hist),
      torch.cat([self._hist[:, 1:], frame[:, None, :]], dim=1))
    self._fresh = torch.zeros_like(fresh)
    # term-major stacking: per term oldest->newest, then next term
    obs = torch.cat([c.flatten(1) for c in self._hist.split(self._sizes, dim=-1)], dim=-1)
    a = self._mlp((obs - self._mean) / self._std)
    self._last = a.clone()
    if self.cfg.residual:
      res = self.cfg.residual_scale * torch.clamp(actions[:, self._off:], -1.0, 1.0)
      g = self.cfg.residual_gate(self._env)[:, None] if self.cfg.residual_gate is not None else 1.0
      a = (1.0 - g) * a + g * res if self.cfg.residual_replaces_base else a + g * res
    self._target = self._default + self._scale * a

  def apply_actions(self) -> None:
    self._entity.set_joint_position_target(self._target, joint_ids=self._ids)

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    if env_ids is None:
      env_ids = slice(None)
    self._raw[env_ids] = 0.0
    self._cmd[env_ids] = 0.0
    if self.cfg.auto_cmd:
      ids = torch.arange(self.num_envs, device=self.device)[env_ids]
      lo, hi = torch.tensor(self.cfg.auto_cmd, device=self.device).T
      cmd = lo + (hi - lo) * torch.rand(len(ids), CMD_DIM, device=self.device)
      stand = torch.rand(len(ids), 1, device=self.device) < self.cfg.auto_stand_prob
      self._cmd[ids] = torch.where(stand, 0.0, cmd)
    self._h[env_ids] = H_STAND
    self._last[env_ids] = 0.0
    self._fresh[env_ids] = True
    self._target[env_ids] = self._default[env_ids]
