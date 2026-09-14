#!/usr/bin/env python3
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
src = ROOT / "results/20260821/real/pandar_run_130755/four_methods/run_232828_factor.npz"
dst = ROOT / "results/20260821/real/pandar_run_130755/four_methods/run_130755_factor_map_only.png"

with np.load(src) as data:
    points = np.asarray(data["map"], float)

fig, ax = plt.subplots(figsize=(12, 8), dpi=220)
fig.patch.set_alpha(0.0)
ax.set_facecolor("none")
if len(points):
    ax.scatter(points[:, 0], points[:, 1], s=0.9, c="#00e5ff", alpha=0.95, linewidths=0)
ax.set_aspect("equal", adjustable="box")
ax.axis("off")
fig.tight_layout()
fig.savefig(dst, format="png", dpi=220, bbox_inches="tight", transparent=True, pad_inches=0)
plt.close(fig)
print(dst)
