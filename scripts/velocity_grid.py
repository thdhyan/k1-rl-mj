"""Velocity profile: every (vx, wz) cell of a grid, per box mass, classified as tracking / fell / dropped the box.
Usage: k1 python scripts/velocity_grid.py <task> <checkpoint.pt> <out.json> [masses kg, comma list; "none" = no box]

vx 0..3.0 m/s step 0.1, wz -pi..pi step pi/16. Each cell: REPS robots. 1.5 s standing, then the command ramps at
1 m/s^2 and 2 rad/s^2 (at most 3 s) and holds; speed is measured over the last 4.5 s of 10 s.
Cell status from the robots still up at the end: fell or dropped if at least half ended that way, otherwise tracking,
tolerance |error| <= max(0.1 m/s, 20 %) for vx and max(0.2 rad/s, 20 %) for wz, on the cell's mean speed."""
import json, math, sys, time, torch
from dataclasses import asdict
import mjlab_k1  # noqa: F401
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls

task, ckpt, out_path = sys.argv[1:4]
masses = sys.argv[4].split(",") if len(sys.argv) > 4 else ["none"]
dev, REPS, STEPS, SETTLE, MEASURE = "cuda:0", 8, 500, 75, 275
VX = [round(0.1 * i, 1) for i in range(31)]
WZ = [round(math.pi * i / 16, 4) for i in range(-16, 17)]
n_cells = len(VX) * len(WZ)
N = n_cells * REPS
cfg = load_env_cfg(task, play=True)
cfg.scene.num_envs = N
cfg.episode_length_s = 60.0
cfg.commands["twist"].resampling_time_range = (1e6, 1e6)
agent = load_rl_cfg(task)
env = RslRlVecEnvWrapper(ManagerBasedRlEnv(cfg=cfg, device=dev), clip_actions=agent.clip_actions)
runner = load_runner_cls(task)(env, asdict(agent), device=dev)
runner.load(ckpt, load_cfg={"actor": True}, strict=True, map_location=dev)
policy = runner.get_inference_policy(device=dev)
u = env.unwrapped
twist, robot, tm = u.command_manager.get_term("twist"), u.scene["robot"], u.termination_manager
fall_term = "fell" if "fell" in tm.active_terms else "fell_over"
has_box = "box_lost" in tm.active_terms
vx_t = torch.tensor(VX, device=dev)[:, None].expand(-1, len(WZ)).flatten().repeat_interleave(REPS)
wz_t = torch.tensor(WZ, device=dev)[None, :].expand(len(VX), -1).flatten().repeat_interleave(REPS)
ramp_v, ramp_w = 1.0 * u.step_dt, 2.0 * u.step_dt
result = {"task": task, "checkpoint": ckpt, "vx": VX, "wz": WZ, "reps": REPS, "profiles": {}}
for m in masses:
  t0 = time.time()
  if m != "none":
    u.box_mass_fixed = float(m)
  obs, _ = env.reset()
  fell = torch.zeros(N, dtype=torch.bool, device=dev)
  drop = torch.zeros(N, dtype=torch.bool, device=dev)
  sum_v, sum_w = torch.zeros(N, device=dev), torch.zeros(N, device=dev)
  for t in range(STEPS):
    k = max(t - SETTLE, 0)
    cmd = torch.stack([torch.minimum(vx_t, torch.full_like(vx_t, k * ramp_v)), torch.zeros(N, device=dev),
                       wz_t.sign() * torch.minimum(wz_t.abs(), torch.full_like(wz_t, k * ramp_w))], -1)
    twist.vel_command_b[:] = cmd
    twist.is_standing_env[:] = (vx_t == 0) & (wz_t == 0)
    with torch.no_grad():
      a = policy(obs)
    obs, _, _, _ = env.step(a)
    ended = fell | drop
    fell |= tm.get_term(fall_term).bool() & ~ended
    if has_box:
      drop |= tm.get_term("box_lost").bool() & ~ended & ~fell
    if t >= STEPS - MEASURE:
      sum_v += robot.data.root_link_lin_vel_b[:, 0]
      sum_w += robot.data.root_link_ang_vel_b[:, 2]
  up = ~(fell | drop)
  c = lambda x: x.float().view(n_cells, REPS)                     # noqa: E731
  n_up = c(up).sum(1).clamp(min=1)
  mean_v = (c(sum_v / MEASURE) * c(up)).sum(1) / n_up
  mean_w = (c(sum_w / MEASURE) * c(up)).sum(1) / n_up
  cv, cw = vx_t[::REPS], wz_t[::REPS]
  v_ok = (mean_v - cv).abs() <= (0.2 * cv.abs()).clamp(min=0.1)
  w_ok = (mean_w - cw).abs() <= (0.2 * cw.abs()).clamp(min=0.2)
  fell_f, drop_f = c(fell).mean(1), c(drop).mean(1)
  status = torch.where(v_ok & w_ok, 0, torch.where(v_ok, 1, torch.where(w_ok, 2, 3)))     # both, vx, yaw, neither
  status = torch.where(fell_f + drop_f >= 0.5, torch.where(fell_f >= drop_f, 4, 5), status)  # fell, dropped
  shape = (len(VX), len(WZ))
  result["profiles"][m] = {"status": status.view(shape).tolist(), "fell": fell_f.view(shape).round(decimals=3).tolist(),
                           "dropped": drop_f.view(shape).round(decimals=3).tolist(),
                           "vx": mean_v.view(shape).round(decimals=3).tolist(),
                           "wz": mean_w.view(shape).round(decimals=3).tolist()}
  counts = [(status == i).sum().item() for i in range(6)]
  print(f"mass {m}: both/vx/yaw/neither/fell/dropped {counts} ({time.time() - t0:.0f} s)", flush=True)
  json.dump(result, open(out_path, "w"))
