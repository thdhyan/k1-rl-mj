from dataclasses import replace

from mjlab.rl import RslRlModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg
from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.velocity.rl import VelocityOnPolicyRunner

from .env_cfg import k1_velocity_env_cfg


def k1_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  # AGILE Table 4: actor [256,256,128], critic [512,256,128], ELU, lr 1e-3, 5 epochs, 4 minibatches
  return RslRlOnPolicyRunnerCfg(
    actor=RslRlModelCfg(hidden_dims=(256, 256, 128), activation="elu", obs_normalization=True,
                        distribution_cfg={"class_name": "GaussianDistribution", "init_std": 1.0, "std_type": "scalar"}),
    critic=RslRlModelCfg(hidden_dims=(512, 256, 128), activation="elu", obs_normalization=True),
    algorithm=RslRlPpoAlgorithmCfg(
      value_loss_coef=1.0, use_clipped_value_loss=True, clip_param=0.2, entropy_coef=0.005,
      num_learning_epochs=5, num_mini_batches=4, learning_rate=1.0e-3, schedule="adaptive",
      gamma=0.99, lam=0.95, desired_kl=0.01, max_grad_norm=1.0),
    experiment_name="k1_velocity", save_interval=100, num_steps_per_env=24, max_iterations=20_000)


for _id, _rough in (("Rough", True), ("Flat", False)):
  register_mjlab_task(
    task_id=f"Mjlab-Velocity-{_id}-Booster-K1",
    env_cfg=k1_velocity_env_cfg(rough=_rough),
    play_env_cfg=k1_velocity_env_cfg(rough=_rough, play=True),
    rl_cfg=k1_ppo_runner_cfg(),
    runner_cls=VelocityOnPolicyRunner)


def k1_task_ppo_cfg(name: str) -> RslRlOnPolicyRunnerCfg:
  """HRL task layers (kick/push/carry): small actor, 50 Hz env step, frozen walk base underneath."""
  return RslRlOnPolicyRunnerCfg(
    actor=RslRlModelCfg(hidden_dims=(256, 128), activation="elu", obs_normalization=True,
                        distribution_cfg={"class_name": "GaussianDistribution", "init_std": 0.5, "std_type": "scalar"}),
    critic=RslRlModelCfg(hidden_dims=(256, 128), activation="elu", obs_normalization=True),
    algorithm=RslRlPpoAlgorithmCfg(
      value_loss_coef=1.0, use_clipped_value_loss=True, clip_param=0.2, entropy_coef=0.005,
      num_learning_epochs=5, num_mini_batches=4, learning_rate=5.0e-4, schedule="adaptive",
      gamma=0.99, lam=0.95, desired_kl=0.01, max_grad_norm=1.0),
    experiment_name=name, save_interval=100, num_steps_per_env=24, max_iterations=5_000)


from mjlab.rl import MjlabOnPolicyRunner  # noqa: E402

from .tasks_kick import kick_env_cfg  # noqa: E402
from .tasks_carry import carry_env_cfg  # noqa: E402
from .tasks_push import push_env_cfg  # noqa: E402
from .tasks_head import head_env_cfg  # noqa: E402

register_mjlab_task(
  task_id="Mjlab-Kick-Booster-K1", env_cfg=kick_env_cfg(), play_env_cfg=kick_env_cfg(play=True),
  rl_cfg=k1_task_ppo_cfg("k1_kick"), runner_cls=MjlabOnPolicyRunner)
register_mjlab_task(
  task_id="Mjlab-Push-Booster-K1", env_cfg=push_env_cfg(), play_env_cfg=push_env_cfg(play=True),
  rl_cfg=k1_task_ppo_cfg("k1_push"), runner_cls=MjlabOnPolicyRunner)
register_mjlab_task(
  task_id="Mjlab-Carry-Booster-K1", env_cfg=carry_env_cfg(), play_env_cfg=carry_env_cfg(play=True),
  rl_cfg=k1_task_ppo_cfg("k1_carry"), runner_cls=MjlabOnPolicyRunner)
register_mjlab_task(
  task_id="Mjlab-HeadTrack-Booster-K1", env_cfg=head_env_cfg(), play_env_cfg=head_env_cfg(play=True),
  rl_cfg=k1_task_ppo_cfg("k1_head"), runner_cls=MjlabOnPolicyRunner)

from .tasks_skills import (height_walk_env_cfg, rough_height_walk_env_cfg, squat_env_cfg, stand_env_cfg,  # noqa: E402
                           steps_env_cfg)

register_mjlab_task(
  task_id="Mjlab-Velocity-Height-Flat-Booster-K1", env_cfg=height_walk_env_cfg(),
  play_env_cfg=height_walk_env_cfg(play=True), rl_cfg=k1_ppo_runner_cfg(), runner_cls=VelocityOnPolicyRunner)
register_mjlab_task(
  task_id="Mjlab-Velocity-Height-Rough-Booster-K1", env_cfg=rough_height_walk_env_cfg(),
  play_env_cfg=rough_height_walk_env_cfg(play=True),
  rl_cfg=replace(k1_ppo_runner_cfg(), experiment_name="k1_velocity_rough"), runner_cls=VelocityOnPolicyRunner)

register_mjlab_task(
  task_id="Mjlab-Stand-Booster-K1", env_cfg=stand_env_cfg(), play_env_cfg=stand_env_cfg(play=True),
  rl_cfg=replace(k1_ppo_runner_cfg(), experiment_name="k1_stand", max_iterations=3_000),
  runner_cls=VelocityOnPolicyRunner)
register_mjlab_task(
  task_id="Mjlab-Squat-Booster-K1", env_cfg=squat_env_cfg(), play_env_cfg=squat_env_cfg(play=True),
  rl_cfg=replace(k1_ppo_runner_cfg(), experiment_name="k1_squat"), runner_cls=VelocityOnPolicyRunner)
register_mjlab_task(
  task_id="Mjlab-Steps-Booster-K1", env_cfg=steps_env_cfg(), play_env_cfg=steps_env_cfg(play=True),
  rl_cfg=k1_task_ppo_cfg("k1_steps"), runner_cls=MjlabOnPolicyRunner)

from .tasks_kick_dir import kick_dir_env_cfg  # noqa: E402

register_mjlab_task(
  task_id="Mjlab-KickDir-Booster-K1", env_cfg=kick_dir_env_cfg(), play_env_cfg=kick_dir_env_cfg(play=True),
  rl_cfg=k1_task_ppo_cfg("k1_kick_dir"), runner_cls=MjlabOnPolicyRunner)

from .tasks_jump import jump_env_cfg  # noqa: E402

register_mjlab_task(
  task_id="Mjlab-Jump-Booster-K1", env_cfg=jump_env_cfg(), play_env_cfg=jump_env_cfg(play=True),
  rl_cfg=k1_task_ppo_cfg("k1_jump"), runner_cls=MjlabOnPolicyRunner)

from .tasks_getup import getup_env_cfg  # noqa: E402

register_mjlab_task(
  task_id="Mjlab-GetUp-Booster-K1", env_cfg=getup_env_cfg(), play_env_cfg=getup_env_cfg(play=True),
  rl_cfg=replace(k1_ppo_runner_cfg(), experiment_name="k1_getup", max_iterations=6_000),
  runner_cls=MjlabOnPolicyRunner)

from .tasks_push_ma import push_ma_env_cfg  # noqa: E402


def k1_marl_ppo_cfg(name: str) -> RslRlOnPolicyRunnerCfg:
  """Task-layer PPO with a shared per-robot actor (marl.SharedActorModel) and a centralised critic."""
  cfg = k1_task_ppo_cfg(name)
  return replace(cfg, actor=replace(cfg.actor, class_name="mjlab_k1.marl:SharedActorModel"),
                 critic=replace(cfg.critic, hidden_dims=(512, 256, 128)), max_iterations=6_000)


register_mjlab_task(
  task_id="Mjlab-PushMA-Booster-K1", env_cfg=push_ma_env_cfg(), play_env_cfg=push_ma_env_cfg(play=True),
  rl_cfg=k1_marl_ppo_cfg("k1_push_ma"), runner_cls=MjlabOnPolicyRunner)

from .tasks_push_ik import push_ik_env_cfg  # noqa: E402

def _push_ik_ppo_cfg() -> RslRlOnPolicyRunnerCfg:
  """Actions saturate at +-1, so unbounded std drifts up at no cost (push_ik_v2 went 0.5 -> 4.4)."""
  cfg = k1_marl_ppo_cfg("k1_push_ik")
  dist = {**cfg.actor.distribution_cfg, "std_range": (1e-3, 0.6)}
  # gamma 0.99 at 50 Hz sees 2 s, too short for a push that takes 10-20 s; 0.998 sees 10 s
  return replace(cfg, actor=replace(cfg.actor, distribution_cfg=dist), max_iterations=8_000, num_steps_per_env=48,
                 algorithm=replace(cfg.algorithm, entropy_coef=0.002, gamma=0.998, lam=0.97))


register_mjlab_task(
  task_id="Mjlab-PushIK-Booster-K1", env_cfg=push_ik_env_cfg(), play_env_cfg=push_ik_env_cfg(play=True),
  rl_cfg=_push_ik_ppo_cfg(), runner_cls=MjlabOnPolicyRunner)

from .tasks_track import track_env_cfg  # noqa: E402

register_mjlab_task(
  task_id="Mjlab-Track-Booster-K1", env_cfg=track_env_cfg(), play_env_cfg=track_env_cfg(play=True),
  rl_cfg=k1_task_ppo_cfg("k1_track"), runner_cls=MjlabOnPolicyRunner)

from .tasks_carry2 import carry2_env_cfg  # noqa: E402

register_mjlab_task(
  task_id="Mjlab-Carry2-Booster-K1", env_cfg=carry2_env_cfg(), play_env_cfg=carry2_env_cfg(play=True),
  rl_cfg=k1_task_ppo_cfg("k1_carry2"), runner_cls=MjlabOnPolicyRunner)

from .tasks_carry_vel import carry_vel_env_cfg  # noqa: E402

register_mjlab_task(
  task_id="Mjlab-CarryVel-Booster-K1", env_cfg=carry_vel_env_cfg(), play_env_cfg=carry_vel_env_cfg(play=True),
  rl_cfg=k1_task_ppo_cfg("k1_carry_vel"), runner_cls=MjlabOnPolicyRunner)

from .tasks_kick_ma import kick_ma_env_cfg  # noqa: E402

register_mjlab_task(
  task_id="Mjlab-KickMA-Booster-K1", env_cfg=kick_ma_env_cfg(), play_env_cfg=kick_ma_env_cfg(play=True),
  rl_cfg=k1_marl_ppo_cfg("k1_kick_ma"), runner_cls=MjlabOnPolicyRunner)

from .tasks_kick_strike import kick_strike_env_cfg  # noqa: E402

register_mjlab_task(
  task_id="Mjlab-KickStrike-Booster-K1", env_cfg=kick_strike_env_cfg(), play_env_cfg=kick_strike_env_cfg(play=True),
  rl_cfg=k1_task_ppo_cfg("k1_kick_strike"), runner_cls=MjlabOnPolicyRunner)
