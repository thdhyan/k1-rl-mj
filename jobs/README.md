# jobs

Part of [k1-rl-mj](https://github.com/thdhyan/k1-rl-mj).

| File | What it does |
|---|---|
| `dl_grid_carry.sh` | On dl: when carry_vel_v1 ends, velocity grid per box mass on its last checkpoint. |
| `dl_walk_v4.sh` | On dl, in tmux walk4: walk_still_v4 from v3 with sideways shoves, training videos, then the shove table for v3 and v4. |
| `zzbw_push_ik_grid.sh` | grid.sh <ckpt relative to mjlab/> <out.mp4>: 16-env curriculum render |
| `zzbw_push_ik_post_v8.sh` | After v8 training ends: eval and multi-size render with the final checkpoint. |
| `zzbw_push_ik_queue_v9.sh` | After v8's eval and renders: train v9 on tall boxes (height 0.75-2x K1 height 0.95 m), then eval and render it. |
| `zzbw_push_ik_v10.sh` | On zz-bw, in tmux pik10: PushIK v10 from v8. Two-hand reward credit, near-goal and at-face resets. Then evals and renders. |
