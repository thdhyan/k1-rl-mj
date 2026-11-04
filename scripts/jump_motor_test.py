"""Open-loop motor test: settle in a squat, drive hips, knees and ankles toward full extension, then hold a landing pose.
No policy. Sweeps squat depth, torque requested (share of limit) and target ramp time. Torque is capped at each limit set.
Usage: k1 python scripts/jump_motor_test.py <out.json>"""
import itertools
import json
import sys

import torch
import mjlab_k1  # noqa: F401
from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg

out, dev, TASK = sys.argv[1], "cuda:0", "Mjlab-Jump-Booster-K1"
KNEES, SHARES, RAMPS = (1.0, 1.4, 1.8), (0.5, 0.75, 1.0), (-0.3, -0.15, 0.0, 0.15, 0.3)  # RAMPS slot now holds hip lean, rad
LIMITS = {"train_cfg": (68.0, 112.0, 38.3), "mjcf": (30.0, 40.0, 20.0)}  # hip pitch, knee, ankle pitch
REP = 8
grid = list(itertools.product(LIMITS, KNEES, SHARES, RAMPS))
N = len(grid) * REP
cfg = load_env_cfg(TASK)
cfg.scene.num_envs = N
cfg.terminations = {}
cfg.events = {k: v for k, v in cfg.events.items() if k == "reset_base"}
u = ManagerBasedRlEnv(cfg=cfg, device=dev)
u.reset()
r = u.scene["robot"]
ids = {k: r.find_joints((f".*_{k}",))[0] for k in ("Hip_Pitch", "Knee_Pitch", "Ankle_Pitch")}
trunk, sites = r.find_bodies(("Trunk",))[0][0], u.command_manager.get_term("jump")._sites
cell = lambda f: torch.tensor([f(g) for g in grid for _ in range(REP)], device=dev, dtype=torch.float)
knee, share, lean = cell(lambda g: g[1]), cell(lambda g: g[2]), cell(lambda g: g[3])
lim = torch.tensor([LIMITS[g[0]] for g in grid for _ in range(REP)], device=dev)
dt = u.physics_dt
q0 = r.data.default_joint_pos.clone()
squat = q0.clone()
squat[:, ids["Hip_Pitch"]] = -knee[:, None] / 2 + lean[:, None]
squat[:, ids["Knee_Pitch"]] = knee[:, None]
squat[:, ids["Ankle_Pitch"]] = (-knee[:, None] / 2).clamp_min(-0.8)
kp = r.data.joint_stiffness if hasattr(r.data, "joint_stiffness") else None
KP = {"Hip_Pitch": 0.0478125, "Knee_Pitch": 0.095625, "Ankle_Pitch": 0.0282528 * 1.4}
KP = {k: v * (4 * 2 * torch.pi) ** 2 for k, v in KP.items()}
KD = {"Hip_Pitch": 3.0, "Knee_Pitch": 2.0, "Ankle_Pitch": 3.0}
KD = {k: KD[k] * v / (4 * 2 * torch.pi) for k, v in KP.items()}  # 2 zeta a w = 2 zeta kp / w
SIGN = {"Hip_Pitch": 1.0, "Knee_Pitch": -1.0, "Ankle_Pitch": 1.0}

def step(target):
  r.set_joint_position_target(target)
  u.scene.write_data_to_sim()
  u.sim.step()
  u.scene.update(dt=dt)

# settle: write the squat, lower root until soles touch, hold 1 s
root = r.data.default_root_state.clone()
root[:, :3] += u.scene.env_origins
r.write_root_state_to_sim(root)
r.write_joint_state_to_sim(squat, torch.zeros_like(squat))
u.sim.forward()
u.scene.update(dt=0.0)
foot = r.data.site_pos_w[:, sites, 2].min(dim=1).values - u.scene.env_origins[:, 2]
root[:, 2] -= (foot - 0.03)
r.write_root_state_to_sim(root)
for _ in range(int(1.0 / dt)):
  step(squat)
z0 = r.data.body_link_pos_w[:, trunk, 2].clone()
foot0 = r.data.site_pos_w[:, sites, 2].min(dim=1).values.clone()
tilt0 = r.data.projected_gravity_b[:, 2].clone()
peak_z, peak_foot = z0.clone(), torch.zeros(N, device=dev)
air_t, peak_v, peak_tau = torch.zeros(N, device=dev), {k: torch.zeros(N, device=dev) for k in ids}, {k: torch.zeros(N, device=dev) for k in ids}
left = torch.zeros(N, dtype=torch.bool, device=dev)
feet = u.scene["feet_ground_contact"]
HORIZON = 0.6
vz_off, tilt_off, knee_off, z_off, both = torch.zeros(N, device=dev), torch.ones(N, device=dev) * 9, torch.zeros(N, device=dev), torch.zeros(N, device=dev), torch.zeros(N, dtype=torch.bool, device=dev)
knee_start = r.data.joint_pos[:, ids["Knee_Pitch"]].mean(dim=1).clone()
for s in range(int(HORIZON / dt)):
  t = s * dt
  a = 1.0
  down = feet.data.found[:, :2].sum(dim=1) > 0
  new = ~down & ~left & (t > 0.02)
  tilt = torch.acos((-r.data.projected_gravity_b[:, 2]).clamp(-1, 1))
  vz_off = torch.where(new, r.data.root_link_lin_vel_w[:, 2], vz_off)
  tilt_off = torch.where(new, tilt, tilt_off)
  z_off = torch.where(new, r.data.body_link_pos_w[:, trunk, 2], z_off)
  knee_off = torch.where(new, r.data.joint_pos[:, ids["Knee_Pitch"]].mean(dim=1), knee_off)
  left |= new
  push = ~left
  tgt = q0.clone()  # landing pose once airborne
  for i, (k, jid) in enumerate(ids.items()):
    d = SIGN[k] * share[:, None] * lim[:, i:i + 1] / KP[k] * a
    # request is measured from the current joint angle so the PD asks for the same torque through the stroke
    tgt[:, jid] = torch.where(push[:, None], r.data.joint_pos[:, jid] + d, q0[:, jid])
    v, e = r.data.joint_vel[:, jid], tgt[:, jid] - r.data.joint_pos[:, jid]
    tau = (KP[k] * e - KD[k] * v).clamp(-lim[:, i:i + 1], lim[:, i:i + 1])
    peak_v[k] = torch.maximum(peak_v[k], v.abs().max(dim=1).values)
    peak_tau[k] = torch.maximum(peak_tau[k], tau.abs().max(dim=1).values)
  step(tgt)
  z = r.data.body_link_pos_w[:, trunk, 2]
  peak_z = torch.maximum(peak_z, z)
  peak_foot = torch.maximum(peak_foot, r.data.site_pos_w[:, sites, 2].min(dim=1).values - foot0)
  air_t += (~down).float() * dt
upright = r.data.projected_gravity_b[:, 2] < -0.7
rows = []
for c, g in enumerate(grid):
  m = torch.arange(c * REP, (c + 1) * REP, device=dev)
  rows.append({"limits": g[0], "knee": g[1], "share": g[2], "lean": g[3], "knee_settled": round(knee_start[m].mean().item(), 2), "liftoff_share": round(left[m].float().mean().item(), 2),
               "vz_liftoff": round(vz_off[m].mean().item(), 2), "ballistic_h_m": round((vz_off[m].clamp_min(0) ** 2 / 19.62).mean().item(), 3),
               "tilt_liftoff_rad": round(tilt_off[m].mean().item(), 2), "z_liftoff": round(z_off[m].mean().item(), 3),
               "tilt_settled": round(torch.acos((-tilt0[m]).clamp(-1, 1)).mean().item(), 2),
               "knee_v": round(peak_v["Knee_Pitch"][m].mean().item(), 1), "hip_v": round(peak_v["Hip_Pitch"][m].mean().item(), 1), "ankle_v": round(peak_v["Ankle_Pitch"][m].mean().item(), 1)})
print("PD actuator note: test caps torque in the request only; simulator effort limit stays at train_cfg.", flush=True)
for row in rows:
  print(row, flush=True)
json.dump(rows, open(out, "w"), indent=1)
