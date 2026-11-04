"""Spread of the randomised KickMA reset, and every piece inside the pitch."""
import torch
import mjlab_k1  # noqa: F401
from mjlab.envs import ManagerBasedRlEnv
from mjlab_k1 import tasks_kick_ma as T

cfg = T.kick_ma_env_cfg(play=False)
cfg.scene.num_envs = 256
env = ManagerBasedRlEnv(cfg, device="cuda:0")
env.reset()
env.sim.forward()
o = env.scene.env_origins[:, :2]
pa, pb = (env.scene[a].data.root_link_pos_w[:, :2] - o for a in ("robot_a", "robot_b"))
ball = env.scene["ball"].data.root_link_pos_w[:, :2] - o
for name, p in (("robot_a", pa), ("robot_b", pb), ("ball", ball), ("pass", T._st(env).pass_w[:, :2] - o)):
  print("LAYOUT", name, "min", p.min(0).values.tolist(), "max", p.max(0).values.tolist(), "std", p.std(0).tolist())
yaw = env.scene["robot_a"].data.heading_w
print("LAYOUT yaw_a std", yaw.std().item(), "a-b dist min/mean", (pa - pb).norm(dim=-1).min().item(), (pa - pb).norm(dim=-1).mean().item())
print("LAYOUT all inside", bool(((ball.abs() < torch.tensor([T.PITCH_X, T.PITCH_Y], device=env.device)).all() and (pb.abs() < torch.tensor([T.PITCH_X, T.PITCH_Y], device=env.device)).all())))
to_goal = T._goal_w(env)[:, :2] - env.scene["robot_b"].data.root_link_pos_w[:, :2]
err = env.scene["robot_b"].data.heading_w - torch.atan2(to_goal[:, 1], to_goal[:, 0])
err = torch.atan2(err.sin(), err.cos()).rad2deg()
d = (pa - pb).norm(dim=-1)
print("LAYOUT robot_b x min", pb[:, 0].min().item(), "facing error mean", err.mean().item(), "std", err.std().item())
print("LAYOUT a-b dist min/max", d.min().item(), d.max().item(), "b-ball min", (pb - ball).norm(dim=-1).min().item())
