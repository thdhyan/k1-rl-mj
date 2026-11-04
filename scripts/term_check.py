import sys, torch
import mjlab_k1  # noqa: F401
from mjlab.envs import ManagerBasedRlEnv
from mjlab_k1.env_cfg import k1_velocity_env_cfg
from mjlab_k1.frozen_base import FrozenBaseActionCfg
from mjlab_k1.upper_body import RandomUpperBodyActionCfg

cfg = k1_velocity_env_cfg(rough=False, play=True)
cfg.scene.num_envs = 32
cfg.commands["twist"].ranges.lin_vel_x = (0.4, 0.4); cfg.commands["twist"].ranges.lin_vel_y = (0, 0); cfg.commands["twist"].ranges.ang_vel_z = (0, 0)
cfg.commands["twist"].rel_standing_envs = 0.0
cfg.actions = {"base": FrozenBaseActionCfg(entity_name="robot", checkpoint=sys.argv[1]),
               "upper_body": RandomUpperBodyActionCfg(entity_name="robot")}
env = ManagerBasedRlEnv(cfg=cfg, device="cuda:0")
obs, _ = env.reset()
term_ = env.action_manager.get_term("base")
act = torch.zeros(32, 3, device="cuda:0"); act[:, 0] = 0.4
falls = 0; vs = []
for i in range(300):
  obs, _, term, _, _ = env.step(act)
  if i in (20, 60):
    ref = obs["actor"]; H = 5; off = 0; out = []
    fr = term_._frame()
    # actor obs: ang_vel3, grav3, jpos12, jvel12, last_action(3 here), cmd3 ; term-major, H rows each
    for nm, n, lo in (("ang", 3, 0), ("grav", 3, 3), ("jpos", 12, 6), ("jvel", 12, 18)):
      seg = ref[:, off:off + n * H].view(-1, H, n)[:, -1]
      out.append((nm, round((seg - fr[:, lo:lo + n]).abs().max().item(), 4))); off += n * H
    print("FRAMECMP", i, out)
  falls += int(term.sum())
  if i > 100: vs.append(env.scene["robot"].data.root_link_lin_vel_b[:, 0].mean().item())
print(f"TERMCHECK falls={falls} mean vx={sum(vs)/len(vs):.3f}")
