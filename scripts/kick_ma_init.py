"""Warm-start Mjlab-KickMA-Booster-K1 from a Mjlab-KickDir checkpoint (kick_dir_v1), and check it numerically.

usage: python scripts/kick_ma_init.py <kick_dir_ckpt.pt> <out.pt>

The shared actor takes the kick_dir actor weights unchanged (obs normaliser, MLP, and the distribution std copied
into both agent slots). Check: with the same observation fed to both agent slots, each slot's mean action equals the
kick_dir actor's mean on that observation. The critic starts fresh.
"""
import sys
from dataclasses import asdict

import torch
from rsl_rl.modules import MLP, EmpiricalNormalization
from tensordict import TensorDict

import mjlab_k1  # noqa: F401
from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.rl import RslRlVecEnvWrapper

TASK, dev = "Mjlab-KickMA-Booster-K1", "cuda:0"
src, out = sys.argv[1:3]
kd = torch.load(src, map_location=dev, weights_only=False)["actor_state_dict"]

cfg = load_env_cfg(TASK)
cfg.scene.num_envs = 2
agent = load_rl_cfg(TASK)
env = RslRlVecEnvWrapper(ManagerBasedRlEnv(cfg=cfg, device=dev), clip_actions=agent.clip_actions)
runner = load_runner_cls(TASK)(env, asdict(agent), device=dev)
actor = runner.alg.actor
sd = actor.state_dict()
new = {}
for k, v in sd.items():
  if k == "distribution.std_param":
    new[k] = torch.cat([kd["distribution.std_param"], kd["distribution.std_param"]]).to(v)
  else:
    assert k in kd and kd[k].shape == v.shape, (k, v.shape, kd.get(k, torch.empty(0)).shape)
    new[k] = kd[k].clone()
actor.load_state_dict(new, strict=True)
print("INIT copied", len(new), "tensors; std", tuple(new["distribution.std_param"].shape))

# reference: kick_dir actor (same architecture, one agent)
ref_mlp = MLP(30, 15, (256, 128), "elu").to(dev)
ref_mlp.load_state_dict({k[4:]: v for k, v in kd.items() if k.startswith("mlp.")})
ref_norm = EmpiricalNormalization(30).to(dev)
ref_norm.load_state_dict({k[len("obs_normalizer."):]: v for k, v in kd.items() if k.startswith("obs_normalizer.")})
ref_norm.eval()
ref_mlp.eval()

torch.manual_seed(0)
x = torch.randn(64, 30, device=dev) * 0.5
with torch.no_grad():
  want = ref_mlp(ref_norm(x))
  obs = TensorDict({"actor": torch.cat([x, x], -1)}, batch_size=[64], device=dev)
  got = actor(obs)  # deterministic: distribution mean
err_a = (got[:, :15] - want).abs().max().item()
err_b = (got[:, 15:] - want).abs().max().item()
print(f"INIT max |mean - kick_dir mean|: robot_a {err_a:.2e}, robot_b {err_b:.2e}")
assert max(err_a, err_b) < 1e-4, "warm start does not match kick_dir"

runner.save(out)
print("INIT saved", out)
