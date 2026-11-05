"""Export a get-up policy: ONNX actor + getup_meta.json (the full 22-joint deploy contract).
Usage: python scripts/getup_export.py <checkpoint.pt> <out_dir> [task]
Deploy loop: obs = 5-step term-major history of [ang_vel 3, gravity 3, joint_pos_rel 22, joint_vel_rel 22, last_action 22]
(oldest first); a = policy(obs) clipped to action_clip; raw = default_pos + action_scale * a; target = raw slew-limited
to +-vel_cap * dt per control step (start from the measured joint pos); send target to the PD with kp/kd."""
import json, sys
from dataclasses import asdict
from pathlib import Path

import torch

import mjlab_k1  # noqa: F401
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls

ckpt, out = Path(sys.argv[1]), Path(sys.argv[2])
task = sys.argv[3] if len(sys.argv) > 3 else "Mjlab-GetUpBent-Booster-K1"
out.mkdir(parents=True, exist_ok=True)
dev = "cuda:0"
cfg = load_env_cfg(task, play=True)
cfg.scene.num_envs = 2
agent = load_rl_cfg(task)
env = RslRlVecEnvWrapper(ManagerBasedRlEnv(cfg=cfg, device=dev), clip_actions=agent.clip_actions)
runner = load_runner_cls(task)(env, asdict(agent), device=dev)
runner.load(str(ckpt), load_cfg={"actor": True}, strict=True, map_location=dev)
runner.export_policy_to_onnx(str(out), "policy.onnx")

u = env.unwrapped
term = u.action_manager.get_term("joint_pos")
robot = u.scene["robot"]
ids = term.target_ids
names = list(term.target_names)
scale = term.scale if torch.is_tensor(term.scale) else torch.full((len(names),), float(term.scale))
group = u.observation_manager
terms = list(zip(group.active_terms["actor"], group.group_obs_term_dim["actor"]))
lim = robot.data.soft_joint_pos_limits[0, ids].cpu()
kp = robot.data.joint_stiffness[0, ids].cpu() if hasattr(robot.data, "joint_stiffness") else None
kd = robot.data.joint_damping[0, ids].cpu() if hasattr(robot.data, "joint_damping") else None
meta = {
  "task": task, "checkpoint": ckpt.name, "control_hz": round(1 / u.step_dt), "physics_dt": u.cfg.sim.mujoco.timestep,
  "decimation": u.cfg.decimation, "joint_order": names,
  "default_pos": robot.data.default_joint_pos[0, ids].cpu().tolist(),  # = the goal pose (bent elbows)
  "action_scale": scale[0].cpu().tolist() if scale.ndim > 1 else scale.tolist(), "action_clip": agent.clip_actions,
  "vel_cap_rad_s": term.cfg.vel_cap, "slew": "target moves <= vel_cap*dt per control step; first step starts at measured q",
  "kp": kp.tolist() if kp is not None else None, "kd": kd.tolist() if kd is not None else None,
  "soft_joint_limits": lim.tolist(),
  "obs_terms": [{"name": n, "dim": int(d[-1]) if hasattr(d, "__len__") else int(d)} for n, d in terms],
  "history": cfg.observations["actor"].history_length, "stacking": "term-major",
  "episode_s": 12.0, "stood_def": "trunk height > 0.45 m and upright > 0.9",
}
(out / "getup_meta.json").write_text(json.dumps(meta, indent=1))
print("EXPORT", out, "joints", len(names), "obs", meta["obs_terms"], "hist", meta["history"])
