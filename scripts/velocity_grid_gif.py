"""Animated velocity profile: one heatmap frame per box weight, same statuses and colours as the HTML page.
Usage: python scripts/velocity_grid_gif.py out.gif a.json [b.json ...]"""
import io
import json
import sys

import matplotlib
matplotlib.use("Agg")
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

ST = [("Tracks vx and yaw", "#1baf7a"), ("Tracks vx only", "#2a78d6"), ("Tracks yaw only", "#e87ba4"),
      ("Up, tracks neither", "#c3c2b7"), ("Fell", "#2b2b29"), ("Dropped box", "#eda100")]
frames = []
for path in sys.argv[2:]:
  d = json.load(open(path))
  for m, p in d["profiles"].items():
    frames.append(("No box, walk_still_v4" if m == "none" else f"{float(m):g} kg box, carry_vel_v1", np.array(p["status"]), d))
imgs = []
for title, s, d in frames:
  fig, ax = plt.subplots(figsize=(9.4, 7.2), constrained_layout=True, dpi=90)
  # x = vx, y = yaw
  ax.imshow(s.T, origin="lower", aspect="auto", cmap=ListedColormap([c for _, c in ST]), vmin=-0.5, vmax=5.5,
            extent=(d["vx"][0] - 0.05, d["vx"][-1] + 0.05, d["wz"][0] - np.pi / 32, d["wz"][-1] + np.pi / 32))
  ax.set_xlabel("commanded vx (m/s)")
  ax.set_ylabel("commanded yaw rate (rad/s)")
  counts = "   ".join(f"{name} {(s == k).sum()}" for k, (name, _) in enumerate(ST))
  ax.set_title(f"{title}\n{counts}", loc="left", fontsize=10.5)
  ax.legend(handles=[Patch(color=c, label=n) for n, c in ST], ncol=6, loc="upper center", bbox_to_anchor=(0.5, -0.09), frameon=False, fontsize=9, columnspacing=1.0, handlelength=1.2)
  buf = io.BytesIO()
  fig.savefig(buf, format="png")
  plt.close(fig)
  imgs.append(Image.open(buf).convert("P", palette=Image.ADAPTIVE, colors=64))
imgs[0].save(sys.argv[1], save_all=True, append_images=imgs[1:], duration=1800, loop=0)
print("wrote", sys.argv[1], len(imgs))
