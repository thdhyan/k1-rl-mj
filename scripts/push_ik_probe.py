"""Can the frozen walk base + hand IK move the box at all? Scripted: walk at the box, hands forward at the given reach.
Prints box travel and contact by mass. Args: [envs]."""
import sys

import torch
import mjlab_k1  # noqa: F401
from mjlab.envs import ManagerBasedRlEnv
from mjlab_k1 import tasks_push_ik as T

n = int(sys.argv[1]) if len(sys.argv) > 1 else 256
cfg = T.push_ik_env_cfg(play=False)
cfg.scene.num_envs = n
env = ManagerBasedRlEnv(cfg, device="cuda:0")
box = env.scene["box"]
import mujoco
mj = env.sim.mj_model
bid = next(i for i in range(mj.nbody) if mujoco.mj_id2name(mj, mujoco.mjtObj.mjOBJ_BODY, i).endswith("box"))
gid = next(i for i in range(mj.ngeom) if (mujoco.mj_id2name(mj, mujoco.mjtObj.mjOBJ_GEOM, i) or "").endswith("box_geom"))
drag = 9.81 * env.sim.model.body_mass[:, bid] * env.sim.model.geom_friction[:, gid, 0]
print(f"drag N: min {drag.min():.0f} median {drag.median():.0f} max {drag.max():.0f}", flush=True)

for vx in (1.0,):
  for reach, hz in ((0.0, -0.5), (-1.0, -0.5)):
    env.reset()
    st, origin = T._state(env), box.data.root_link_pos_w[:, :2].clone()
    mid = (env.scene["robot_a"].data.root_link_pos_w[:, :2] + env.scene["robot_b"].data.root_link_pos_w[:, :2]) / 2
    th = origin - mid
    th = th / th.norm(dim=-1, keepdim=True)
    st.goal[:, :2] = origin + 4.5 * th  # no early success
    alive = torch.ones(n, dtype=torch.bool, device=env.device)
    hand = body = 0.0
    for t in range(500):
      acts = []
      for a in T.AGENTS:
        r = env.scene[a].data
        to = box.data.root_link_pos_w[:, :2] - r.root_link_pos_w[:, :2]
        e = torch.atan2(to[:, 1], to[:, 0]) - T._yaw(r.root_link_quat_w)
        e = torch.atan2(e.sin(), e.cos())
        acts.append(torch.stack([torch.full_like(e, vx), 0 * e, (2.0 * e).clamp(-1, 1), 0 * e,
                                 torch.full_like(e, reach), 0 * e, torch.full_like(e, hz),
                                 torch.full_like(e, reach), 0 * e, torch.full_like(e, hz)], -1))
      _, _, term, tout, _ = env.step(torch.cat(acts, -1))
      alive &= ~(term | tout)
      hand += (T._hand_frac(env) > 0)[alive].float().sum().item()
      body += (T._body_contact(env) > 0)[alive].float().sum().item()
      if t == 498:
        d = ((box.data.root_link_pos_w[:, :2] - origin) * th).sum(-1)
    print(f"vx {vx} reach {reach}: alive {alive.float().mean():.2f} hand {hand / (500 * max(alive.sum().item(), 1)):.2f} "
          f"body {body / (500 * max(alive.sum().item(), 1)):.2f} travel mean {d[alive].mean():.2f} "
          f"p10 {d[alive].quantile(.1):.2f} p90 {d[alive].quantile(.9):.2f}", flush=True)
    for lo, hi in ((0, 15), (15, 25), (25, 35), (35, 50), (50, 100)):
      m = alive & (drag >= lo) & (drag < hi)
      if m.any():
        print(f"  drag {lo}-{hi} N: n {m.sum().item()} travel mean {d[m].mean():.2f} p10 {d[m].quantile(.1):.2f}", flush=True)
