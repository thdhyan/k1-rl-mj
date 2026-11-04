import mjlab_k1  # noqa: F401
from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg
from mjlab_k1.k1_constants import LEG_REGEX
cfg = load_env_cfg("Mjlab-Velocity-Flat-Booster-K1", play=True); cfg.scene.num_envs = 2
env = ManagerBasedRlEnv(cfg=cfg, device="cuda:0")
r = env.scene["robot"]
print("ORD builtin", list(env.action_manager.get_term("joint_pos").target_names))
print("ORD find_true", r.find_joints(LEG_REGEX, preserve_order=True)[1])
print("ORD find_false", r.find_joints(LEG_REGEX)[1])
