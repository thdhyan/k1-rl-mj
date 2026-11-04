"""Walk-base smoothness + top-speed table at fixed commands (flat, no pushes); same numbers for every checkpoint.
Usage: k1 python scripts/smooth_probe.py <task> <checkpoint.pt> [out.json]"""
import json, sys, torch
from dataclasses import asdict
import mjlab_k1  # noqa: F401
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls

task, ckpt = sys.argv[1], sys.argv[2]
dev, N, T, SKIP = "cuda:0", 64, 400, 100
cfg = load_env_cfg(task, play=True)
cfg.scene.num_envs = N
agent = load_rl_cfg(task)
env = RslRlVecEnvWrapper(ManagerBasedRlEnv(cfg=cfg, device=dev), clip_actions=agent.clip_actions)
runner = load_runner_cls(task)(env, asdict(agent), device=dev)
runner.load(ckpt, load_cfg={"actor": True}, strict=True, map_location=dev)
policy = runner.get_inference_policy(device=dev)
u = env.unwrapped
twist, robot, feet = u.command_manager.get_term("twist"), u.scene["robot"], u.scene["feet_ground_contact"]
legs = robot.find_joints(mjlab_k1.env_cfg.LEG_REGEX)[0]
out = {}
for vx in (0.0, 0.3, 0.5, 0.75, 1.0, 1.2):
  obs, _ = env.reset()
  rows = {k: [] for k in ("act", "vx", "gyro", "dq", "c")}
  falls = 0
  for t in range(T):
    twist.vel_command_b[:] = torch.tensor([vx, 0.0, 0.0], device=dev)
    twist.is_standing_env[:] = vx == 0.0
    with torch.no_grad():
      a = policy(obs)
    obs, _, done, _ = env.step(a)
    falls += int(done.sum())
    rows["act"].append(a.clone())
    rows["vx"].append(robot.data.root_link_lin_vel_b[:, 0].clone())
    rows["gyro"].append(robot.data.root_link_ang_vel_b[:, :2].clone())
    rows["dq"].append(robot.data.joint_vel[:, legs].clone())
    rows["c"].append(feet.data.found.reshape(N, -1)[:, :2] > 0)
  S = {k: torch.stack(v)[SKIP:] for k, v in rows.items()}
  da = S["act"].diff(dim=0)
  ddq = S["dq"].diff(dim=0) / u.step_dt
  C = S["c"].float()
  td = ((C[1:] > .5) & (C[:-1] < .5)).float().sum(0).mean().item() / ((T - SKIP - 1) * u.step_dt)
  out[vx] = dict(vx_real=round(S["vx"].mean().item(), 3), act_rate=round(da.pow(2).sum(-1).mean().sqrt().item(), 3),
                 act_acc=round(da.diff(dim=0).pow(2).sum(-1).mean().sqrt().item(), 3),
                 joint_acc=round(ddq.pow(2).mean().sqrt().item(), 1), gyro_rms=round(S["gyro"].pow(2).mean().sqrt().item(), 3),
                 steps_s=round(2 * td, 2), falls=falls)
  print(vx, out[vx], flush=True)
if len(sys.argv) > 3:
  json.dump(out, open(sys.argv[3], "w"), indent=1)
