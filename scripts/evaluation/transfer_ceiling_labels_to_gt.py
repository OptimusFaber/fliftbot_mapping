#!/usr/bin/env python3
"""Convert ceiling-label coordinates from the encoder tag frame to room_real GT."""
import argparse, json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from flifltbot.mapping.real_room_gt import (
    ROOM_X, ROOM_Y, CEILING_ORIGIN_X_M, CEILING_ORIGIN_Y_M,
    tag_cm_to_room_m, point_in_room_footprint,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--alignment", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--plot", required=True)
    a = ap.parse_args()
    alignment = json.loads(Path(a.alignment).read_text())
    raw = json.loads(Path(a.labels).read_text())
    labels = []
    for ident, p in raw.items():
        x, y = tag_cm_to_room_m(float(p["x"]), float(p["y"]))
        labels.append({
            "id": str(ident),
            "encoder_tag_cm": {"x": p["x"], "y": p["y"], "h": p["h"]},
            "gt_room_xyz_m": [round(x, 5), round(y, 5), round(float(p["h"]) / 100, 5)],
            "inside_footprint": point_in_room_footprint(x, y),
        })
    labels.sort(key=lambda q: (q["gt_room_xyz_m"][1], q["gt_room_xyz_m"][0], q["id"]))
    payload = {
        "source_alignment": str(Path(a.alignment).resolve()),
        "source_labels": str(Path(a.labels).resolve()),
        "target_frame": "room_real GT (SW floor corner origin, +X east, +Y north)",
        "transform": "gt_x = 11.685 + encoder_y/100; gt_y = 3.525 - encoder_x/100",
        "origin": {"room_gt_xy_m": [0.0, 0.0], "encoder_tag_id": "98",
                   "encoder_tag_frame_xy_cm": [0.0, 0.0],
                   "room_gt_xy_for_tag_98_m": [CEILING_ORIGIN_X_M, CEILING_ORIGIN_Y_M]},
        "room_size_m": [ROOM_X, ROOM_Y],
        "alignment_context": alignment,
        "labels": labels,
    }
    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2) + "\n")
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.set_xlim(-0.5, ROOM_X + 0.5); ax.set_ylim(-0.5, ROOM_Y + 0.5)
    ax.add_patch(plt.Rectangle((0, 0), ROOM_X, ROOM_Y, fill=False, lw=2, color="black"))
    xs = [q["gt_room_xyz_m"][0] for q in labels]; ys = [q["gt_room_xyz_m"][1] for q in labels]
    ax.scatter(xs, ys, s=20, color="#1677ff", label="ceiling labels in GT")
    for q in labels: ax.text(q["gt_room_xyz_m"][0] + .03, q["gt_room_xyz_m"][1] + .03, q["id"], fontsize=5)
    ax.scatter([0], [0], marker="+", s=180, lw=2.5, color="red", label="GT origin (0, 0)")
    ax.scatter([CEILING_ORIGIN_X_M], [CEILING_ORIGIN_Y_M], marker="x", s=120, lw=2, color="darkorange", label="tag 98 / encoder origin")
    ax.set_aspect("equal"); ax.grid(alpha=.25); ax.set_xlabel("GT X, m"); ax.set_ylabel("GT Y, m")
    ax.set_title("room_real: ceiling labels transferred from encoder frame to GT")
    ax.legend(loc="upper left", fontsize=8); fig.tight_layout()
    plot = Path(a.plot); plot.parent.mkdir(parents=True, exist_ok=True); fig.savefig(plot, dpi=220); plt.close(fig)
    print(f"wrote {len(labels)} labels: {out}\nplot: {plot}\nGT origin: (0.000, 0.000) m; encoder tag 98: ({CEILING_ORIGIN_X_M:.3f}, {CEILING_ORIGIN_Y_M:.3f}) m")


if __name__ == "__main__": main()
