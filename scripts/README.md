# scripts

Part of [k1-rl-mj](https://github.com/thdhyan/k1-rl-mj).

| File | What it does |
|---|---|
| `base_probe.py` | Sanity: frozen base inside a task env must walk at the commanded speed. Usage: python scripts/base_probe.py <task> [vx_norm] |
| `carry_clips.py` | carry_vel_v1 clips, MuJoCo render: <ckpt> <out_dir> Fixed mass: env 0 of 8 at each mass, same command schedule inside the trained range, no  |
| `export_mjcf.py` | Write the K1 exactly as mjlab trains it (Booster 4 Hz PD gains, armature, feet-only collision, home keyframe) to a standalone MJCF next to K |
| `export_policy.py` | Export a trained K1 mjlab policy to ONNX + policy_meta.json for scripts/mujoco_eval.py. Usage: uv run python scripts/export_policy.py <task> |
| `gait_probe.py` | Roll out a trained K1 mjlab policy at fixed commands; report cadence and tracking. Usage: uv run python scripts/gait_probe.py <task> <checkp |
| `jump_motor_test.py` | Open-loop motor test: settle in a squat, drive hips, knees and ankles toward full extension, then hold a landing pose. No policy. Sweeps squ |
| `jump_probe.py` | Measure every landed jump of a checkpoint over one 20 s episode, and film env 0 without selection. Usage: k1 python scripts/jump_probe.py <c |
| `jump_smoke.py` | Check Mjlab-Jump start states with zero actions: heights per start phase, standing trunk and foot site, stale contact. Usage: k1 python scri |
| `kick_depth_probe.py` | Head-camera depth on KickStrike: render speed, encoder speed, and how often the ball is in view. <teacher.pt> <resnet.pth> <out_dir> [envs] |
| `kick_dir_record.py` | Record pass episodes from Mjlab-KickDir for the Isaac RTX replay, including the per-episode target. Usage: k1 python scripts/kick_dir_record |
| `kick_gaze_debug.py` | Servo trace with zero policy action: joint signs, target vs measured head joints, image error over time. |
| `kick_gaze_static.py` | Servo convergence on a still ball: robot stands, ball placed at a known range and bearing, residual u, v after 3 s. |
| `kick_gaze_sweep.py` | Ball in view of the head camera against a fixed head pitch and pan, geometry from rollouts of a KickStrike checkpoint. <teacher.pt> [envs] T |
| `kick_grid_render.py` | 9 parallel envs from MuJoCo Warp physics, no episode picked: scripts/kick_grid_render.py strike/ma <ckpt> <out.mp4> [seconds] [title] Each t |
| `kick_ma_checks.py` | Scripted checks for Mjlab-KickMA-Booster-K1: hold before pass, receive, switch, goal, out of play. |
| `kick_ma_eval.py` | Pass-and-shoot rates of a Mjlab-KickMA checkpoint: <ckpt> [envs] [out.json]. Each env counts its first episode only. Touches are ball veloci |
| `kick_ma_goal_video.py` | One picked episode: robot_a's pass is received and robot_b's touch is the last before the goal. <ckpt> <out.mp4> [envs] [label] Touches are  |
| `kick_ma_init.py` | Warm-start Mjlab-KickMA-Booster-K1 from a Mjlab-KickDir checkpoint (kick_dir_v1), and check it numerically. |
| `kick_ma_layout.py` | Spread of the randomised KickMA reset, and every piece inside the pitch. |
| `kick_ma_smoke.py` | Smoke test for Mjlab-KickMA-Booster-K1: build, dims, phase switch, goal, OOB, and a shot-free random run. |
| `kick_record.py` | Record kick episodes from mjlab for the Isaac RTX replay: every body's world pose per step, plus ball and goal. Usage: k1 python scripts/kic |
| `kick_spread.py` | Where a kick policy sends the ball: heading error and landing spread against the commanded target, sim only. Usage: k1 python scripts/kick_s |
| `kick_strike_checks.py` | Scripted checks of the KickStrike supervision terms. Prints PASS or FAIL per check. |
| `kick_strike_init.py` | Warm-start Mjlab-KickStrike-Booster-K1 from kick_dir_v1: <kick_dir_ckpt.pt> <out.pt>. |
| `kick_yolo_check.py` | Run the bbox head servo on KickStrike and measure centring; save RGB frames with projected boxes for the YOLO check. <teacher.pt> <out_dir>  |
| `kick_yolo_eval.py` | Real yolov8n on frames from scripts/kick_yolo_check.py, class sports ball (32). <dir>. Run in its own env: uvx --with ultralytics. |
| `mlp_check.py` |  |
| `obs_check.py` |  |
| `order_check.py` |  |
| `push_ik_eval.py` | Success rate of a Mjlab-PushIK checkpoint at the full reset ranges: <ckpt> [envs] [out.json]. Success = box within 0.2 m and 10 deg (mod 180 |
| `push_ik_probe.py` | Can the frozen walk base + hand IK move the box at all? Scripted: walk at the box, hands forward at the given reach. Prints box travel and c |
| `push_ik_render.py` | Render Mjlab-PushIK from MuJoCo Warp physics with box and goal corners and errors drawn on the frame. |
| `push_limits.py` | Empirical limits of the one-robot box push (Mjlab-Push): success and failure rate per box footprint, height and mass. Usage: k1 python scrip |
| `push_limits_plot.py` | Heatmaps of scripts/push_limits.py: success % per footprint and height, one panel per mass, main failure as a letter. Usage: python scripts/ |
| `shove_eval.py` | Fall rate of a walk checkpoint under sideways shoves on the trunk, same forces for every checkpoint. Usage: k1 python scripts/shove_eval.py  |
| `skill_eval.py` | First episode of each env, deterministic actions: share of envs per termination, mean episode seconds, twist error if the task has one. Usag |
| `smooth_probe.py` | Walk-base smoothness + top-speed table at fixed commands (flat, no pushes); same numbers for every checkpoint. Usage: k1 python scripts/smoo |
| `task_smoke.py` | Build an mjlab K1 task, step it with random actions, print dims, reward terms and terminations. |
| `term_check.py` |  |
| `velocity_grid.py` | Velocity profile: every (vx, wz) cell of a grid, per box mass, classified as tracking / fell / dropped the box. Usage: k1 python scripts/vel |
| `velocity_grid_gif.py` | Animated velocity profile: one heatmap frame per box weight, same statuses and colours as the HTML page. Usage: python scripts/velocity_grid |
| `velocity_grid_html.py` | Merge scripts/velocity_grid.py outputs into one self-contained page with a weight slider. Usage: python scripts/velocity_grid_html.py out.ht |
| `walk_profile_gif.py` | Walking velocity profile as a GIF: one tile per (vx, wz) command, robots walking, border color = tracking status. Usage: k1 python scripts/w |
| `wbc_check.py` | Closed-loop parity: wbc.DecoupledWBC leg targets == the training-time FrozenBaseAction targets. |
| `wbc_head_check.py` | Closed-loop parity of wbc.HeadModule against the trained head policy inside Mjlab-HeadTrack-Booster-K1. |
