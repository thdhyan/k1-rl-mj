import sys, torch
import mjlab_k1  # noqa: F401
from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg
from mjlab_k1.frozen_base import load_frozen_mlp

cfg = load_env_cfg("Mjlab-Velocity-Flat-Booster-K1", play=True)
cfg.scene.num_envs = 32
env = ManagerBasedRlEnv(cfg=cfg, device="cuda:0")
mlp, mean, std = load_frozen_mlp(sys.argv[1], "cuda:0")
obs, _ = env.reset()
falls = 0
vs = []
for i in range(300):
  a = mlp((obs["actor"] - mean) / std)
  obs, _, term, _, _ = env.step(a)
  falls += int(term.sum())
  if i > 100: vs.append(env.scene["robot"].data.root_link_lin_vel_b[:, 0].mean().item())
cmd = env.command_manager.get_command("twist")
print(f"MLPCHECK falls={falls} mean vx={sum(vs)/len(vs):.3f} cmd_vx_mean={cmd[:,0].mean().item():.3f}")
