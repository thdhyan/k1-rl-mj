"""Fall rate of a walk checkpoint under sideways shoves on the trunk, same forces for every checkpoint.
Usage: k1 python scripts/shove_eval.py <task> <checkpoint.pt> [out.json]
One shove per robot, at a random time, along the trunk's +-y, 0.15 m above the trunk centre. A robot counts as fallen if it
terminates within 3 s of the shove. Robot mass about 19.5 kg."""
import json, sys, torch
from dataclasses import asdict
import mjlab_k1  # noqa: F401
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls

task, ckpt = sys.argv[1], sys.argv[2]
dev, N, SETTLE, WINDOW, AFTER, HOLD = "cuda:0", 512, 100, 50, 150, 10  # steps of 0.02 s; shove lasts 0.2 s
cfg = load_env_cfg(task, play=True)
cfg.scene.num_envs = N
agent = load_rl_cfg(task)
env = RslRlVecEnvWrapper(ManagerBasedRlEnv(cfg=cfg, device=dev), clip_actions=agent.clip_actions)
runner = load_runner_cls(task)(env, asdict(agent), device=dev)
runner.load(ckpt, load_cfg={"actor": True}, strict=True, map_location=dev)
policy = runner.get_inference_policy(device=dev)
u = env.unwrapped
twist, robot = u.command_manager.get_term("twist"), u.scene["robot"]
trunk = robot.find_bodies(("Trunk",))[0]
out = {}
for vx in (0.0, 0.5):
  for force in (0, 50, 100, 150, 200, 300):
    obs, _ = env.reset()
    start = SETTLE + torch.randint(0, WINDOW, (N,), device=dev)
    sign = torch.where(torch.arange(N, device=dev) % 2 == 0, 1.0, -1.0)
    fell = torch.zeros(N, dtype=torch.bool, device=dev)
    for t in range(SETTLE + WINDOW + AFTER):
      twist.vel_command_b[:] = torch.tensor([vx, 0.0, 0.0], device=dev)
      twist.is_standing_env[:] = vx == 0.0
      on = (t >= start) & (t < start + HOLD)
      q = robot.data.body_link_quat_w[:, trunk[0]]
      yaw = torch.atan2(2.0 * (q[:, 0] * q[:, 3] + q[:, 1] * q[:, 2]), 1.0 - 2.0 * (q[:, 2] ** 2 + q[:, 3] ** 2))
      f = torch.stack([-yaw.sin(), yaw.cos(), torch.zeros(N, device=dev)], -1) * (force * sign * on)[:, None]
      arm = torch.tensor((0.0, 0.0, 0.15), device=dev).expand_as(f)
      robot.write_external_wrench_to_sim(f[:, None], torch.cross(arm, f, dim=-1)[:, None], body_ids=trunk)
      with torch.no_grad():
        a = policy(obs)
      obs, _, done, _ = env.step(a)
      fell |= done.bool() & (t >= start)
    out[f"vx{vx}_F{force}"] = round(fell.float().mean().item(), 4)
    print(f"vx {vx} force {force} N ({force * 0.2:.0f} N s): fell {out[f'vx{vx}_F{force}']:.1%}", flush=True)
if len(sys.argv) > 3:
  json.dump(out, open(sys.argv[3], "w"), indent=1)
