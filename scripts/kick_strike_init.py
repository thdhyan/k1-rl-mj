"""Warm-start Mjlab-KickStrike-Booster-K1 from kick_dir_v1: <kick_dir_ckpt.pt> <out.pt>.

Actor and critic keep their weights. New obs inputs get zero first-layer weights, normaliser mean 0, var 1. So on any obs
the new net's mean action equals kick_dir's on the first 30 dims; the script checks that.
"""
import sys
from dataclasses import asdict

import torch
from tensordict import TensorDict

import mjlab_k1  # noqa: F401
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls

TASK, dev = "Mjlab-KickStrike-Booster-K1", "cuda:0"
src, out = sys.argv[1:3]
ck = torch.load(src, map_location=dev, weights_only=False)
cfg = load_env_cfg(TASK)
cfg.scene.num_envs = 2
agent = load_rl_cfg(TASK)
env = RslRlVecEnvWrapper(ManagerBasedRlEnv(cfg=cfg, device=dev), clip_actions=agent.clip_actions)
runner = load_runner_cls(TASK)(env, asdict(agent), device=dev)


def pad(net, old):
  sd, new = net.state_dict(), {}
  for k, v in sd.items():
    o = old[k]
    if o.shape == v.shape:
      new[k] = o.clone()
    elif "normalizer" in k:  # mean, var, std: 1-row (1, n) or (n,)
      fill = 0.0 if k.endswith("_mean") else 1.0
      new[k] = torch.full_like(v, fill)
      new[k][..., :o.shape[-1]] = o
    else:  # first linear layer (hidden, n)
      new[k] = torch.zeros_like(v)
      new[k][:, :o.shape[1]] = o
    print("INIT", k, tuple(o.shape), "->", tuple(v.shape)) if o.shape != v.shape else None
  net.load_state_dict(new, strict=True)


pad(runner.alg.actor, ck["actor_state_dict"])
pad(runner.alg.critic, ck["critic_state_dict"])

torch.manual_seed(0)
n_obs = runner.alg.actor.state_dict()["obs_normalizer._mean"].shape[-1]
x = torch.randn(64, n_obs, device=dev) * 0.5
a0 = runner.alg.actor(TensorDict({"actor": x}, batch_size=[64], device=dev))
x2 = x.clone()
x2[:, 30:] += 1.0
a1 = runner.alg.actor(TensorDict({"actor": x2}, batch_size=[64], device=dev))
err = (a0 - a1).abs().max().item()
print(f"INIT obs dim {n_obs}; effect of new inputs on the mean: {err:.2e}")
assert err < 1e-5
runner.save(out)
print("INIT saved", out)
