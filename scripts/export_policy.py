"""Export a trained K1 mjlab policy to ONNX + policy_meta.json for scripts/mujoco_eval.py.
Usage: uv run python scripts/export_policy.py <task> <checkpoint.pt> <out_dir>"""
import json, sys
from dataclasses import asdict
from pathlib import Path

import mjlab_k1  # noqa: F401
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab_k1.k1_constants import K1_ACTION_SCALE, K1_ACTUATORS, LEG_REGEX

task, ckpt, out = sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3])
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
names = list(term.target_names)
base = {k: j for k, j in zip(LEG_REGEX, K1_ACTUATORS[:6])}
kind = lambda n: next(r for r in LEG_REGEX if __import__("re").fullmatch(r, n))
act = {n: base[kind(n)] for n in names}
init = {"Hip_Pitch": -0.3, "Knee_Pitch": 0.6, "Ankle_Pitch": -0.3}
default = [next((v for k, v in init.items() if n.endswith(k)), 0.0) for n in names]
up = K1_ACTUATORS[6:]
UPPER = ["AAHead_yaw", "Head_pitch", "ALeft_Shoulder_Pitch", "Left_Shoulder_Roll", "Left_Elbow_Pitch", "Left_Elbow_Yaw",
         "ARight_Shoulder_Pitch", "Right_Shoulder_Roll", "Right_Elbow_Pitch", "Right_Elbow_Yaw"]
ukp = [up[1].stiffness if "Head" in n else up[0].stiffness for n in UPPER]
ukd = [up[1].damping if "Head" in n else up[0].damping for n in UPPER]
udef = [-1.3 if n == "Left_Shoulder_Roll" else 1.3 if n == "Right_Shoulder_Roll" else 0.0 for n in UPPER]
hist = cfg.observations["actor"].history_length
meta = {
  "joint_order": names, "default_pos": default,
  "action_scale": [K1_ACTION_SCALE[kind(n)] for n in names], "action_clip": agent.clip_actions,
  "kp": [act[n].stiffness for n in names], "kd": [act[n].damping for n in names],
  "effort": [act[n].effort_limit for n in names],
  "delay_physics_steps": [2, 8], "physics_dt": u.cfg.sim.mujoco.timestep, "decimation": u.cfg.decimation,
  "control_hz": 50,
  "obs_terms": [{"name": "base_ang_vel", "dim": 3}, {"name": "projected_gravity", "dim": 3},
                {"name": "joint_pos_rel", "dim": 12}, {"name": "joint_vel", "dim": 12},
                {"name": "last_action", "dim": 12}, {"name": "command", "dim": 3}],
  "history": hist, "stacking": "term-major", "init_height": 0.56, "friction": 0.6,
  "upper_joint_order": UPPER, "upper_default_pos": udef, "upper_kp": ukp, "upper_kd": ukd,
}
(out / "policy_meta.json").write_text(json.dumps(meta, indent=1))
print("exported", out, "action_clip", agent.clip_actions, "hist", hist, "dt", meta["physics_dt"], "dec", meta["decimation"])
