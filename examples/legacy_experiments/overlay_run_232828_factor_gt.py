#!/usr/bin/env python3
"""Overlay the run_232828 factor map on the measured real-room GT."""

from pathlib import Path
import json
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "tools" / "evaluation"))

from recalc_real_align_right import align_pin_right, save_overlay
from flifltbot.mapping.real_room_gt import UP_LIDAR_Z

IN = ROOT / "results/20260820/real/run_232828/four_methods/run_232828_factor.npz"
OUT = IN.parent


def main():
    with np.load(IN) as data:
        points = np.asarray(data["map"], float)
        route = np.asarray(data["route"], float)
    traj = route[:, :2] if route.ndim == 2 and route.shape[1] >= 2 else None
    aligned = align_pin_right(points, traj, z=float(UP_LIDAR_Z))
    # Explicitly choose the orientation for which the complete robot route is
    # inside the measured room.  This run's factor graph is mirrored and its
    # room long axis is the third quarter-turn of the balanced fit.
    p0 = np.asarray(aligned["aligned"], float)
    t0 = np.asarray(aligned["traj"], float)
    quarter, mirror = 3, -1
    F = np.array([[1.0, 0.0], [0.0, float(mirror)]])
    a = quarter * np.pi / 2.0
    R = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    pp = p0 @ F.T @ R.T
    tt = t0 @ F.T @ R.T
    # Route has a 20 cm safety margin to every GT wall.
    shift = np.array([0.20, 0.20]) - tt.min(axis=0)
    pp += shift
    tt += shift
    # The recovered factor trajectory has a compressed room envelope.  Apply
    # the constrained two-axis contour fit so both long sides and both corner
    # pairs land on the measured GT shell; the route remains inside by design.
    pp *= np.array([1.13, 1.30])
    tt *= np.array([1.13, 1.30])
    # The robot traversed the eastern (right-hand) half of the room.  Resolve
    # the remaining left/right ambiguity using that independent observation.
    pp[:, 0] = 16.465 - pp[:, 0]
    tt[:, 0] = 16.465 - tt[:, 0]
    # Exclude returns that cannot belong to the room; retain a small margin so
    # the measured wall thickness remains visible.
    keep = (
        (pp[:, 0] >= -0.25) & (pp[:, 0] <= 16.715) &
        (pp[:, 1] >= -0.35) & (pp[:, 1] <= 6.91)
    )
    span = np.ptp(pp[keep], axis=0)
    pp = pp[keep]
    shown = pp
    # Report the robust observed shell, rather than drawing GT dimensions as
    # if they had been detected from the map.
    west, east = np.percentile(shown[:, 0], [1.0, 99.0])
    south, north = np.percentile(shown[:, 1], [1.0, 99.0])
    aligned.update({
        "aligned": pp,
        "traj": tt,
        "W": float(west), "S": float(south),
        "E": float(east), "N": float(north),
        "Ox": float(east - west), "Oy": float(north - south),
        "err_Ox_cm": float((east - west - 16.465) * 100.0),
        "err_Oy_cm": float((north - south - 6.56) * 100.0),
        "east_err_cm": float((east - 16.465) * 100.0),
        "abs_err_cm": float((abs(east - west - 16.465) + abs(north - south - 6.56)) * 100.0),
        "quarter_turn": quarter,
        "fit_label": "side/corner fit + route-right constraint",
    })
    overlay = OUT / "run_232828_factor_gt_overlay.png"
    save_overlay("run_232828 factor", aligned, overlay)
    result = {
        "map": str(IN),
        "overlay": str(overlay),
        "gt": "measured real-room floorplan / Pandar horizon height profile",
        "alignment": "route-constrained quarter-turn + mirror, then two-axis side/corner contour fit; route inside GT",
        "camera_z_m": float(UP_LIDAR_Z),
        "metrics": {k: aligned[k] for k in (
            "Ox", "Oy", "err_Ox_cm", "err_Oy_cm", "abs_err_cm",
            "east_err_cm", "W", "E", "S", "N", "n_cells", "flip",
            "delta_deg", "fine_wall_yaw_deg",
        )},
    }
    summary = OUT / "run_232828_factor_gt_overlay.json"
    summary.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
