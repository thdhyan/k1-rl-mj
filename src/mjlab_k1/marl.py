"""Multi-agent PPO with parameter sharing (MAPPO-style CTDE) on the single-agent rsl-rl runner.

The env packs N agents into one env row: actor obs = [agent 0 | agent 1 | ...] (each egocentric, same layout),
action = [agent 0 | agent 1 | ...]. SharedActorModel runs ONE MLP on every agent's slice (decentralised
execution: each robot only needs its own obs at deploy) and concatenates the actions; the team log-prob is
the sum over agents, which is the MAPPO objective with a shared team reward. The critic is a plain MLPModel
on the full state (centralised training). Checkpoints are plain MLPModel state dicts with per-agent shapes, so
wbc.load_frozen_mlp loads the actor for a single robot.
"""
from __future__ import annotations

import torch
from rsl_rl.models import MLPModel
from rsl_rl.modules import MLP, EmpiricalNormalization


class SharedActorModel(MLPModel):
  def __init__(self, obs, obs_groups, obs_set, output_dim, num_agents: int = 2, **kw):
    super().__init__(obs, obs_groups, obs_set, output_dim, **kw)
    assert self.obs_dim % num_agents == 0 and output_dim % num_agents == 0
    self.n_agents, self.agent_obs = num_agents, self.obs_dim // num_agents
    if self.obs_normalization:
      self.obs_normalizer = EmpiricalNormalization(self.agent_obs)
    out = self.distribution.input_dim if self.distribution is not None else output_dim
    self.mlp = MLP(self.agent_obs, out // num_agents, kw.get("hidden_dims", (256, 256, 256)), kw.get("activation", "elu"))
    if self.distribution is not None:
      self.distribution.init_mlp_weights(self.mlp)

  def get_latent(self, obs, masks=None, hidden_state=None) -> torch.Tensor:
    latent = torch.cat([obs[g] for g in self.obs_groups], dim=-1)
    return self.obs_normalizer(latent.reshape(-1, self.agent_obs))  # (B * agents, agent_obs)

  def forward(self, obs, masks=None, hidden_state=None, stochastic_output=False) -> torch.Tensor:
    latent = self.get_latent(obs, masks, hidden_state)
    out = self.mlp(latent).reshape(latent.shape[0] // self.n_agents, -1)  # agents side by side
    if self.distribution is not None:
      if stochastic_output:
        self.distribution.update(out)
        return self.distribution.sample()
      return self.distribution.deterministic_output(out)
    return out

  def update_normalization(self, obs) -> None:
    if self.obs_normalization:
      self.obs_normalizer.update(torch.cat([obs[g] for g in self.obs_groups], dim=-1).reshape(-1, self.agent_obs))
