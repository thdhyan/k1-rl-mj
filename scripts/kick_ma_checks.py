"""Scripted checks for Mjlab-KickMA-Booster-K1: hold before pass, receive, switch, goal, out of play.

usage: python scripts/kick_ma_checks.py
"""
import torch

import mjlab_k1  # noqa: F401
from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg
from mjlab_k1 import tasks_kick_ma as M

cfg = load_env_cfg("Mjlab-KickMA-Booster-K1")
cfg.scene.num_envs = 4
env = ManagerBasedRlEnv(cfg=cfg, device="cuda:0")
env.reset()
st, ball = M._st(env), env.scene["ball"]
zero = torch.zeros(4, env.action_manager.total_action_dim, device=env.device)
ok = lambda c, msg: print(("PASS " if c else "FAIL ") + msg)

# 1. hold: robot_b command is zero before any pass (zero actions, ball resting by robot_a)
for _ in range(20):
  env.step(zero)
cmd_b = env.action_manager.get_term("base_robot_b").command
ok(not st.switched.any() and cmd_b.abs().max() == 0, f"robot_b held before pass, |cmd| {cmd_b.abs().max().item():.3f}")

# 2. receive + switch: put the ball on the pass point
org = env.scene.env_origins
p = st.pass_w.clone()
ball.write_root_state_to_sim(torch.cat([p, torch.tensor([1.0, 0, 0, 0], device=env.device).expand(4, 4), torch.zeros(4, 6, device=env.device)], -1), env_ids=torch.arange(4, device=env.device))
env.step(zero)
ok(bool(st.received.all()), "ball on pass point -> received")
ok(bool(st.switched.all()), "ball near robot_b -> switched")
env.step(zero)
ok(float(M.pass_received(env).sum()) == 0.0, "pass_received one-shot does not repeat")

# 2b. robot_a release: not while the ball only sits far away, then latched by a kick followed by the ball 1 m off
ids = torch.arange(4, device=env.device)
quat = torch.tensor([1.0, 0, 0, 0], device=env.device).expand(4, 4)
st.a_kicked[:], st.a_released[:] = False, False
pa = env.scene["robot_a"].data.root_link_pos_w
rest = torch.cat([pa + torch.tensor([1.3, 0, 0], device=env.device), quat, torch.zeros(4, 6, device=env.device)], -1)
ball.write_root_state_to_sim(rest, env_ids=ids)
env.sim.forward()
ok(float(M._gate_a(env).min()) == 1.0, "robot_a not released with the ball far and no kick")
vel = torch.zeros(4, 6, device=env.device)
vel[:, 0] = 1.0
ball.write_root_state_to_sim(torch.cat([pa + torch.tensor([0.4, 0, 0], device=env.device), quat, vel], -1), env_ids=ids)
env.sim.forward()
ok(float(M._gate_a(env).max()) == 1.0 and bool(st.a_kicked.all()), "ball leaving robot_a's foot -> kicked, not released yet")
ball.write_root_state_to_sim(rest, env_ids=ids)
env.sim.forward()
ok(float(M._gate_a(env).min()) == 1.0 and bool(st.a_released.all()), "kicked and ball 1 m off -> released, fade starts at 1")
env.episode_length_buf += 25  # 0.5 s at 50 Hz
g = M._gate_a(env)
ok(float((g - 0.5).abs().max()) < 1e-3, f"0.5 s after release -> gate 0.5, got {g.max().item():.3f}")
env.episode_length_buf += 25
ok(float(M._gate_a(env).max()) == 0.0 and float(M._res_gate_a(env).max()) == 0.0, "1 s after release -> command and residual off")

# 3. goal: ball just over the goal line at centre, low
g = torch.tensor([M.PITCH_X - 0.05, 0.0, 0.3], device=env.device) + org
ball.write_root_state_to_sim(torch.cat([g, torch.tensor([1.0, 0, 0, 0], device=env.device).expand(4, 4), torch.zeros(4, 6, device=env.device)], -1), env_ids=torch.arange(4, device=env.device))
env.sim.forward()  # terminations reset the env inside step, so read the predicates before stepping
ok(bool(M.goal_scored(env).all()), "ball over centre of goal -> goal_scored")
ok(not bool(M.ball_oob(env).any()), "goal is not counted as out of play")

# 4. out: wide of the goal line
o = torch.tensor([M.PITCH_X + 0.5, 1.5, 0.11], device=env.device) + org
ball.write_root_state_to_sim(torch.cat([o, torch.tensor([1.0, 0, 0, 0], device=env.device).expand(4, 4), torch.zeros(4, 6, device=env.device)], -1), env_ids=torch.arange(4, device=env.device))
env.sim.forward()
ok(bool(M.ball_oob(env).all()), "ball wide past goal line -> ball_oob")
