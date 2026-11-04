import torch
import mjlab_k1  # noqa: F401
from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg
from mjlab.tasks.velocity import mdp
from mjlab_k1.k1_constants import LEG_REGEX

cfg = load_env_cfg("Mjlab-Velocity-Flat-Booster-K1", play=True)
cfg.scene.num_envs = 4
env = ManagerBasedRlEnv(cfg=cfg, device="cuda:0")
obs, _ = env.reset()
a = torch.zeros(4, 12, device="cuda:0")
obs, *_ = env.step(a)
ref = obs["actor"][0]
print("ref dim", ref.shape)
r = env.scene["robot"]; d = r.data
oids, onames = r.find_joints(LEG_REGEX)
print("obs joint order", onames)
cmd = env.command_manager.get_command("twist")
parts = {
 "ang_vel": mdp.builtin_sensor(env, "robot/imu_ang_vel")[0], "grav": mdp.projected_gravity(env)[0],
 "jpos": (d.joint_pos[:, oids] - d.default_joint_pos[:, oids])[0], "jvel": d.joint_vel[:, oids][0],
 "last": a[0], "cmd": cmd[0]}
sizes = [3, 3, 12, 12, 12, 3]
H = 5
off = 0
for (k, v), n in zip(parts.items(), sizes):
  seg = ref[off:off + n * H].view(H, n)
  print(k, "newest-row err", (seg[-1] - v).abs().max().item(), "oldest-row err", (seg[0] - v).abs().max().item())
  off += n * H
