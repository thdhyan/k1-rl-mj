"""Real yolov8n on frames from scripts/kick_yolo_check.py, class sports ball (32). <dir>. Run in its own env: uvx --with ultralytics."""
import json
import sys
from pathlib import Path

import numpy as np
from ultralytics import YOLO

d = Path(sys.argv[1])
f = np.load(d / "frames.npz")
rgb, box, seen, dist = f["rgb"], f["box"], f["seen"], f["dist"]
H, W = rgb.shape[1:3]
m = YOLO("yolov8n.pt")
found, err_px, bands = np.zeros(len(rgb), bool), np.full(len(rgb), np.nan), {}
for i in range(0, len(rgb), 64):
  res = m.predict([x[..., ::-1] for x in rgb[i:i + 64]], classes=[32], conf=0.1, imgsz=320, verbose=False)
  for j, r in enumerate(res):
    if len(r.boxes):
      b = r.boxes.xywh[r.boxes.conf.argmax()].cpu().numpy()
      px = (box[i + j, 0] + 1) / 2 * W, (1 - box[i + j, 1]) / 2 * H
      found[i + j], err_px[i + j] = True, np.hypot(b[0] - px[0], b[1] - px[1])
for name, s in (("near", dist < 0.8), ("mid", (dist >= 0.8) & (dist < 2.5)), ("far", dist >= 2.5)):
  s = s & seen
  bands[name] = dict(n=int(s.sum()), recall=float(found[s].mean()) if s.any() else None, centre_err_px_p50=float(np.nanmedian(err_px[s])) if found[s].any() else None)
fp = found & ~seen
r = dict(bands=bands, false_positive_rate=float(fp.sum() / max((~seen).sum(), 1)), image=[W, H])
print("YOLO", json.dumps(r))
json.dump(r, open(d / "yolo.json", "w"), indent=1)
