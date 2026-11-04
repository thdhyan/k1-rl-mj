"""Heatmaps of scripts/push_limits.py: success % per footprint and height, one panel per mass, main failure as a letter.
Usage: python scripts/push_limits_plot.py <in.json> <out.png>"""
import json
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

r = json.load(open(sys.argv[1]))
ok = np.array(r["outcomes"]["success"]) * 100
fails = ("fell", "box_tip", "box_oob", "timeout")
letters = "FTOW"
bad = np.stack([np.array(r["outcomes"][k]) for k in fails], -1)
fig, axes = plt.subplots(3, 3, figsize=(11.5, 10.5), sharex=True, sharey=True, constrained_layout=True)
for k, ax in enumerate(axes.flat):
  im = ax.imshow(ok[:, :, k], vmin=0, vmax=100, cmap="Blues", origin="lower", aspect="auto")
  ax.set_title(f"{r['mass'][k]} kg", fontsize=12, loc="left")
  for i in range(len(r["edge"])):
    for j in range(len(r["height"])):
      v = ok[i, j, k]
      ax.text(j, i, f"{v:.0f}\n{letters[bad[i, j, k].argmax()]}", ha="center", va="center", fontsize=8.5,
              color="white" if v > 55 else "#222222", linespacing=0.95)
  ax.set_xticks(range(len(r["height"])), r["height"])
  ax.set_yticks(range(len(r["edge"])), r["edge"])
  ax.tick_params(length=0)
  for s in ax.spines.values():
    s.set_visible(False)
for ax in axes[-1]:
  ax.set_xlabel("box height (m)")
for ax in axes[:, 0]:
  ax.set_ylabel("footprint edge (m)")
fig.colorbar(im, ax=axes, shrink=0.5, label="success, % of episodes")
fig.suptitle(f"One K1 pushing a box, push_v1, {r['reps']} episodes per cell, {r['episode_s']:.0f} s limit. Trained only on 0.7 m cube, 4 kg.\n"
             "Number: success %. Letter: most common failure. F robot fell, T box tipped, O box out of bounds, W timeout.", fontsize=11.5)
fig.savefig(sys.argv[2], dpi=130)
