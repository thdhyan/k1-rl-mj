"""Joint speed stats of a get-up policy: scripts/getup_speed_check.py <ckpt> [envs] [seconds] [task]"""
import sys

import torch

import mjlab_k1  # noqa: F401
from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg
try:
  from k1_runtime.wbc import PolicyModule
except ImportError:  # older zz-bw snapshots
  from mjlab_k1.wbc import PolicyModule

ckpt = sys.argv[1]
n = int(sys.argv[2]) if len(sys.argv) > 2 else 256
secs = float(sys.argv[3]) if len(sys.argv) > 3 else 12.0
cfg = load_env_cfg(sys.argv[4] if len(sys.argv) > 4 else "Mjlab-GetUpBent-Booster-K1", play=True)
cfg.scene.num_envs = n
env = ManagerBasedRlEnv(cfg, device="cuda:0")
obs, _ = env.reset()
pol = PolicyModule(ckpt, env.device)
r = env.scene["robot"].data
arg, up_l = [], []
vmax, vall, stood_t = [], [], torch.full((n,), float("nan"), device=env.device)
with torch.no_grad():
  for t in range(int(secs / env.step_dt)):
    obs, *_ = env.step(pol(obs["actor"]))
    v = r.joint_vel.abs()
    vmax.append(v.max(-1).values)
    arg.append(v.argmax(-1))
    up_l.append(-r.projected_gravity_b[:, 2])
    vall.append(v.flatten())
    up = -r.projected_gravity_b[:, 2]
    h = r.root_link_pos_w[:, 2] - env.scene.env_origins[:, 2]
    ok = (h > 0.45) & (up > 0.9) & stood_t.isnan()
    stood_t[ok] = (t + 1) * env.step_dt
SKIP = int(1.0 / env.step_dt)  # first second = the 0.55 m drop / impact
vm, va = torch.stack(vmax)[SKIP:], torch.cat(vall[SKIP:])
print(f"SPEED (after first 1.0 s) max over joints+time {vm.max():.2f} rad/s | p99 of per-step max {vm.flatten().quantile(0.99):.2f} | "
      f"p99 all joint-steps {va.quantile(0.99):.2f} | mean {va.mean():.3f}")
big = vm > 10
names = env.scene["robot"].joint_names
if big.any():
  ids = torch.stack(arg)[SKIP:][big]
  top = torch.bincount(ids, minlength=len(names)).topk(3)
  upb = torch.stack(up_l)[SKIP:][big]
  print(f"SPIKES >10 rad/s: {int(big.sum())} of {big.numel()} env-steps ({big.float().mean():.4%}); envs affected {int(big.any(0).sum())}/{n}; "
        f"joints {[(names[i], int(c)) for c, i in zip(top.values, top.indices)]}; mean upright at spike {upb.mean():.2f}")
print(f"STOOD {int((~stood_t.isnan()).sum())}/{n}; time to first stood: median {stood_t.nan_to_num(99).median():.1f} s")
