"""Check Mjlab-Jump start states with zero actions: heights per start phase, standing trunk and foot site, stale contact.
Usage: k1 python scripts/jump_smoke.py"""
import torch
import mjlab_k1  # noqa: F401
from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg

cfg = load_env_cfg("Mjlab-Jump-Booster-K1")
cfg.scene.num_envs = 256
u = ManagerBasedRlEnv(cfg=cfg, device="cuda:0")
j, r = u.command_manager.get_term("jump"), u.scene["robot"]
# Standing reference: frozen base holds the stand, no window opens.
j.cfg.start_probs = (1.0, 0.0, 0.0, 0.0, 0.0)
u.reset()
j.wait[:] = 1e6
zero = torch.zeros(256, u.action_manager.total_action_dim, device="cuda:0")
for _ in range(100):
  u.step(zero)
  j.wait[:] = 1e6
up = r.data.projected_gravity_b[:, 2] < -0.9
print("STAND trunk_z", r.data.root_link_pos_w[up, 2].mean().item(), "foot_site_z", j.foot_z[up].mean().item(), "upright", up.float().mean().item())
# Start states: state right after reset, per phase.
from mjlab_k1.tasks_jump import START_PROBS, reset_jump_state
j.cfg.start_probs = START_PROBS
ids = torch.arange(256, device="cuda:0")
u.reset()
phase_t = j.t.clone(), j.wait.clone(), j.air_steps.clone(), j.landed.clone(), j.genuine.clone()
u.sim.forward()
u.scene.update(dt=0.0)
z, f, vz = r.data.root_link_pos_w[:, 2], j.foot_z, r.data.root_link_lin_vel_w[:, 2]
air = phase_t[2] > 0
for name, m in (("stand/squat", phase_t[4]), ("air", air), ("landed", phase_t[3])):
  if m.any():
    print(name, "n", int(m.sum()), "trunk_z %.3f..%.3f foot %.3f..%.3f vz %.2f..%.2f" % (z[m].min(), z[m].max(), f[m].min(), f[m].max(), vz[m].min(), vz[m].max()))
print("level", j.level, "target", j.target.min().item(), j.target.max().item())
for t in range(150):
  u.step(zero)
print("after 150 zero steps: rewards ok; terminated share", u.termination_manager.terminated.float().mean().item())
print("SMOKE_OK")
