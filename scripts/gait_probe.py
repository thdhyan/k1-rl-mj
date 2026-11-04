"""Roll out a trained K1 mjlab policy at fixed commands; report cadence and tracking.
Usage: uv run python scripts/gait_probe.py <task> <checkpoint.pt> [vx] [wz]"""
import sys, torch
from dataclasses import asdict
import mjlab_k1  # noqa: F401
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls

task, ckpt = sys.argv[1], sys.argv[2]
vx = float(sys.argv[3]) if len(sys.argv) > 3 else 0.4
wz = float(sys.argv[4]) if len(sys.argv) > 4 else 0.0
dev = "cuda:0"
cfg = load_env_cfg(task, play=True)
cfg.scene.num_envs = 64
agent = load_rl_cfg(task)
env = RslRlVecEnvWrapper(ManagerBasedRlEnv(cfg=cfg, device=dev), clip_actions=agent.clip_actions)
runner = load_runner_cls(task)(env, asdict(agent), device=dev)
runner.load(ckpt, load_cfg={"actor": True}, strict=True, map_location=dev)
policy = runner.get_inference_policy(device=dev)
u = env.unwrapped
term = u.command_manager.get_term("twist")
robot = u.scene["robot"]
sens = u.scene["feet_ground_contact"]
obs = env.get_observations()
T = 500
cmd = torch.tensor([vx, 0.0, wz], device=dev)
contacts, vels, zs, fh = [], [], [], []
hs = u.scene['foot_height_scan']
for t in range(T):
  term.vel_command_b[:] = cmd
  term.is_standing_env[:] = False
  with torch.no_grad():
    act = policy(obs)
  obs, _, done, _ = env.step(act)
  found = sens.data.found.reshape(64, -1)[:, :2] > 0
  contacts.append(found.clone())
  vels.append(robot.data.root_link_lin_vel_b[:, :2].clone())
  zs.append(robot.data.root_link_pos_w[:, 2].clone())
  fh.append(hs.data.heights.clone())
C = torch.stack(contacts).float()  # T,N,2
dt = u.step_dt
td = ((C[1:] > 0.5) & (C[:-1] < 0.5)).float().sum(0)  # touchdowns per env per foot
secs = (T - 1) * dt
print(f"cmd vx={vx} wz={wz}  step_dt={dt}")
print(f"touchdown rate per foot (Hz): {td.mean().item()/secs:.2f}  (gait cadence = 2x this steps/s: {2*td.mean().item()/secs:.2f})")
print(f"double-support frac: {(C.sum(-1)==2).float().mean().item():.2f}  flight frac: {(C.sum(-1)==0).float().mean().item():.2f}  single: {(C.sum(-1)==1).float().mean().item():.2f}")
V = torch.stack(vels)
print(f"mean vx: {V[100:,:,0].mean().item():.3f}  mean |vy|: {V[100:,:,1].abs().mean().item():.3f}")
print(f"trunk z: {torch.stack(zs)[100:].mean().item():.3f}")
H = torch.stack(fh)[100:]  # T,N,2 foot site height above terrain
print(f"foot height: p50 {H.median().item():.3f}  p95 {H.flatten().quantile(0.95).item():.3f}  max {H.max().item():.3f}")
