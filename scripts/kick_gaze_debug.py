"""Servo trace with zero policy action: joint signs, target vs measured head joints, image error over time."""
import torch
import mjlab_k1  # noqa: F401
from mjlab.envs import ManagerBasedRlEnv
from mjlab_k1 import tasks_kick_strike as S
from mjlab_k1.ball_gaze import detect

cfg = S.kick_strike_env_cfg(play=True)
cfg.scene.num_envs = 16
cfg.actions["upper_body"].frozen_prob = 1.0
cfg.actions["gaze"].noise, cfg.actions["gaze"].dropout = 0.0, 0.0
env = ManagerBasedRlEnv(cfg, device="cuda:0")
env.reset()
gaze = env.action_manager.get_term("gaze")
r = env.scene["robot"]
print("head joints", [r.joint_names[i] for i in gaze._ids.tolist()], "limits", gaze._lo[0].tolist(), gaze._hi[0].tolist())
act = torch.zeros(env.num_envs, env.action_manager.total_action_dim, device=env.device)
with torch.no_grad():
  for t in range(150):
    env.step(act)
    box, seen = detect(env)
    if t % 10 == 0:
      q = r.data.joint_pos[0, gaze._ids]
      print(f"t{t:3d} seen {seen.float().mean():.2f} offset {box[seen][:, :2].norm(dim=-1).mean():.3f} "
            f"env0 u {box[0, 0]:+.3f} v {box[0, 1]:+.3f} target {gaze._target[0, 0]:+.3f} {gaze._target[0, 1]:+.3f} "
            f"joint {q[0]:+.3f} {q[1]:+.3f}")
