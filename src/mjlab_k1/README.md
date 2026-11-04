# src/mjlab_k1

Part of [k1-rl-mj](https://github.com/thdhyan/k1-rl-mj).

| File | What it does |
|---|---|
| `__init__.py` |  |
| `ball_gaze.py` | Head servo that keeps the ball's bounding box centred in the head camera. No learning: bbox centre error -> head joints. |
| `cli.py` | Import the K1 tasks (registers them), then hand over to mjlab's train/play CLIs. |
| `depth_encoder.py` | Frozen ResNet18 (ImageNet weights) for depth images, up to layer3, so the feature map keeps where things are in the image. |
| `env_cfg.py` | K1 decoupled lower-body velocity tracking on mjlab: AGILE-style, small reward set. |
| `frozen_base.py` | Frozen lower-body policy as an action term: the HRL 'actuator' for every layer above it. |
| `k1_constants.py` | Booster K1 (22 DoF) for mjlab. |
| `marl.py` | Multi-agent PPO with parameter sharing (MAPPO-style CTDE) on the single-agent rsl-rl runner. |
| `tasks_carry.py` | Box carry (port of Isaac Lab `Isaac-Carry-K1-v0`, carry v1) on mjlab: an HRL task layer with arms. |
| `tasks_carry2.py` | Box carry v2: no weld. The box is a free, colliding body held only by the forearm stubs squeezing its sides. |
| `tasks_carry_vel.py` | Carry a free box of varying mass under a velocity command: residual layer on the frozen walk base. |
| `tasks_getup.py` | Get-up: from lying (prone, supine, side) or a random tumble back to the HOME stand, all 22 joints. |
| `tasks_head.py` | Head tracking (port of Isaac Lab `Isaac-HeadTrack-K1-v0`) on mjlab: the head layer of the HRL stack. |
| `tasks_jump.py` | Jump in place on the K1 (Mjlab-Jump-Booster-K1). Method follows Olsen 2025; see docs/k1_jump_strategies.md. |
| `tasks_kick.py` | Kick task (port of Isaac Lab `Isaac-Kick-*`) on mjlab: the first HRL task layer. |
| `tasks_kick_dir.py` | Directed kick (Mjlab-KickDir-Booster-K1): pass the ball to a target given relative to the robot. |
| `tasks_kick_ma.py` | Two-robot soccer kick (Mjlab-KickMA-Booster-K1): robot A passes to robot B, B chases and shoots at the goal centre. |
| `tasks_kick_strike.py` | Directed kick teacher (Mjlab-KickStrike-Booster-K1): tight line supervision plus a contact model of the strike. |
| `tasks_push.py` | Box push (port of Isaac Lab `Isaac-Push-Simple-K1-v0`) on mjlab: an HRL task layer with arms. |
| `tasks_push_ik.py` | Two K1s push a box to a goal POSE, with only what the PhaseSpace caps give (MAPPO, shared actor). |
| `tasks_push_ma.py` | Two-robot box push (MAPPO, parameter sharing): two K1s push one long box to a goal with their HANDS. |
| `tasks_skills.py` | Ports of the remaining Isaac Lab K1 tasks to mjlab: stand (P1 Basic), squat (H* command), steps (residual). |
| `tasks_track.py` | Ball tracking v2: keep the ball centred in the head camera with the head AND the body (walk base command). |
| `upper_body.py` | Policy-less upper-body driver: the decoupled-WBC 'other half'. |
