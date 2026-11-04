"""Empirical limits of the one-robot box push (Mjlab-Push): success and failure rate per box footprint, height and mass.
Usage: k1 python scripts/push_limits.py <checkpoint.pt> <out.json> [reps per cell]

Every env runs one scene with its own box, same start and goal ranges as training, except the box centre moves so the
near face stays at the trained distance. First outcome of each env is latched: success (box centre within 0.15 m of
goal), robot fell, box tipped (no axis within 45 deg of vertical), box out of bounds (3 m), or timeout at 30 s
(twice the 15 s training episode). Solid uniform box, friction as in training. The policy sees box position, heading
and velocity, never size or mass, and was trained only on a 0.7 m cube of 4 kg."""
import json, sys, time, torch
from dataclasses import asdict
import mjlab_k1  # noqa: F401
from mjlab.envs import ManagerBasedRlEnv
from mjlab.envs.mdp import dr
from mjlab.managers.event_manager import EventTermCfg, RecomputeLevel
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab_k1 import tasks_push as T

ckpt, out_path = sys.argv[1:3]
REPS = int(sys.argv[3]) if len(sys.argv) > 3 else 32
TASK, dev, EPISODE_S = "Mjlab-Push-Booster-K1", "cuda:0", 30.0
EDGE = [0.3, 0.4, 0.5, 0.7, 0.9, 1.1]        # footprint edge, m (square)
HEIGHT = [0.2, 0.4, 0.7, 1.0, 1.4]           # m
MASS = [1, 2, 4, 8, 12, 16, 24, 32, 48]      # kg
OUTCOMES = ("success", "fell", "box_tip", "box_oob", "timeout")
n_cells = len(EDGE) * len(HEIGHT) * len(MASS)
N = n_cells * REPS
idx = torch.arange(N, device=dev) // REPS
e_i, h_i, m_i = idx // (len(HEIGHT) * len(MASS)), idx // len(MASS) % len(HEIGHT), idx % len(MASS)
half_xy = torch.tensor(EDGE, device=dev)[e_i] / 2
half_z = torch.tensor(HEIGHT, device=dev)[h_i] / 2
mass = torch.tensor(MASS, device=dev, dtype=torch.float)[m_i]


def reset_box_sized(env, env_ids):
  """tasks_push.reset_box with this env's size: near face at the trained distance, box resting on the floor."""
  n, d = len(env_ids), env.device
  u = lambda r: r[0] + (r[1] - r[0]) * torch.rand(n, device=d)  # noqa: E731
  x = u(T.BOX_X) + half_xy[env_ids] - T.BOX_HALF
  pos = torch.stack([x, u(T.BOX_Y), half_z[env_ids] + 0.005], -1)
  yaw = u(T.BOX_YAW)
  box = env.scene["box"]
  st = box.data.default_root_state[env_ids].clone()
  st[:, :3] = pos + env.scene.env_origins[env_ids]
  st[:, 3:7] = torch.stack([torch.cos(yaw / 2), torch.zeros_like(yaw), torch.zeros_like(yaw), torch.sin(yaw / 2)], -1)
  st[:, 7:] = 0.0
  box.write_root_state_to_sim(st, env_ids=env_ids)
  r, a = u(T.GOAL_D), u(T.GOAL_ANGLE)
  s = T._state(env)
  s.goal[env_ids] = st[:, :3] + torch.stack([r * torch.cos(a), r * torch.sin(a), torch.zeros_like(r)], -1)
  s.d_prev[env_ids] = r


cfg = load_env_cfg(TASK, play=True)
cfg.scene.num_envs = N
cfg.episode_length_s = EPISODE_S
geom, body = SceneEntityCfg("box", geom_names=("box_geom",)), SceneEntityCfg("box", body_names=("box",))
cfg.events["reset_box"] = EventTermCfg(func=reset_box_sized, mode="reset")
# Registered so these fields get per-env memory; the values are overwritten below.
cfg.events["size_fields"] = EventTermCfg(func=dr.geom_size, mode="startup", params={
  "asset_cfg": geom, "operation": "abs", "ranges": {0: (0.35, 0.35), 1: (0.35, 0.35), 2: (0.35, 0.35)}})
cfg.events["mass_fields"] = EventTermCfg(func=dr.pseudo_inertia, mode="startup", params={
  "asset_cfg": body, "alpha_range": (0.0, 0.0)})
agent = load_rl_cfg(TASK)
env = RslRlVecEnvWrapper(ManagerBasedRlEnv(cfg=cfg, device=dev), clip_actions=agent.clip_actions)
runner = load_runner_cls(TASK)(env, asdict(agent), device=dev)
runner.load(ckpt, load_cfg={"actor": True}, strict=True, map_location=dev)
policy = runner.get_inference_policy(device=dev)
u = env.unwrapped
geom.resolve(u.scene)
tm = u.termination_manager

for e, edge in enumerate(EDGE):       # one geom_size call per size, so it also recomputes the broadphase bounds
  for h, height in enumerate(HEIGHT):
    ids = torch.nonzero((e_i == e) & (h_i == h)).flatten()
    dr.geom_size(u, ids, asset_cfg=geom, operation="abs",
                 ranges={0: (edge / 2, edge / 2), 1: (edge / 2, edge / 2), 2: (height / 2, height / 2)})
bid = u.scene["box"].indexing.body_ids[0]
model = u.sim.model
model.body_mass[:, bid] = mass
a2, b2 = (2 * half_xy) ** 2, (2 * half_z) ** 2
model.body_inertia[:, bid] = mass[:, None] / 12.0 * torch.stack([a2 + b2, a2 + b2, a2 + a2], -1)
u.sim.recompute_constants(RecomputeLevel.set_const)

t0 = time.time()
obs, _ = env.reset()
code = torch.full((N,), -1, device=dev)   # index into OUTCOMES, -1 = not ended
for t in range(int(EPISODE_S / u.step_dt) - 2):
  with torch.no_grad():
    a = policy(obs)
  obs, _, _, _ = env.step(a)
  for i, name in enumerate(("goal_reached", "fell", "box_tip", "box_oob")):
    code = torch.where((code < 0) & tm.get_term(name).bool(), torch.full_like(code, i), code)
code = torch.where(code < 0, torch.full_like(code, 4), code)

rates = torch.stack([(code == i).float().view(n_cells, REPS).mean(1) for i in range(5)], -1)
shape = (len(EDGE), len(HEIGHT), len(MASS))
result = {"task": TASK, "checkpoint": ckpt, "edge": EDGE, "height": HEIGHT, "mass": MASS, "reps": REPS, "episode_s": EPISODE_S,
          "outcomes": {name: rates[:, i].view(shape).round(decimals=3).tolist() for i, name in enumerate(OUTCOMES)}}
json.dump(result, open(out_path, "w"))
print(f"done in {time.time() - t0:.0f} s, {N} envs. overall: " + ", ".join(f"{n} {rates[:, i].mean():.1%}" for i, n in enumerate(OUTCOMES)))
ref = (EDGE.index(0.7), HEIGHT.index(0.7), MASS.index(4))
print("trained box 0.7 m, 4 kg: " + ", ".join(f"{n} {result['outcomes'][n][ref[0]][ref[1]][ref[2]]:.1%}" for n in OUTCOMES))
