#!/usr/bin/env python3
"""Isolated run_192133 experiments for recovering the pre-regression sim path."""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))


VARIANTS = {
    "odom_oracle": {
        "use_absolute_odom": True,
        "rear_pivot_x": 0.0,
        "kwargs": dict(
            use_encoder=True,
            deskew_mode="full_se2",
            synthesize_beam_times=True,
            frontend_icp="wheel",
            skip_lidar_sequential=True,
            skip_loop_closures=True,
            factor_mode=True,
            map_min_hits=5,
        ),
    },
    "wheel_pivot0": {
        "use_absolute_odom": False,
        "rear_pivot_x": 0.0,
        "kwargs": dict(
            use_encoder=True,
            deskew_mode="full_se2",
            synthesize_beam_times=True,
            frontend_icp="wheel",
            skip_lidar_sequential=True,
            skip_loop_closures=True,
            factor_mode=True,
            map_min_hits=5,
        ),
    },
    "factor_pivot0": {
        "use_absolute_odom": False,
        "rear_pivot_x": 0.0,
        "kwargs": dict(
            use_encoder=True,
            deskew_mode="full_se2",
            synthesize_beam_times=True,
            frontend_icp="submap",
            lidar_seq_target="submap",
            submap_keyframes=10,
            factor_mode=True,
            estimate_scale=False,
            calibrate_dt=False,
            map_min_hits=5,
        ),
    },
    "factor_pivot0_no_loops": {
        "use_absolute_odom": False,
        "rear_pivot_x": 0.0,
        "kwargs": dict(
            use_encoder=True,
            deskew_mode="full_se2",
            synthesize_beam_times=True,
            frontend_icp="submap",
            lidar_seq_target="submap",
            submap_keyframes=10,
            factor_mode=True,
            estimate_scale=False,
            calibrate_dt=False,
            skip_loop_closures=True,
            map_min_hits=5,
        ),
    },
    # Apply the robust real-scan constraint profile to the same warehouse data.
    # factor_mode=False selects scalar scan factors with wheel-agreement gates;
    # the frontend and all geometry remain identical to factor_pivot0.
    "real_factor": {
        "use_absolute_odom": False,
        "rear_pivot_x": 0.0,
        "kwargs": dict(
            use_encoder=True,
            deskew_mode="full_se2",
            synthesize_beam_times=True,
            frontend_icp="submap",
            lidar_seq_target="submap",
            submap_keyframes=10,
            factor_mode=False,
            estimate_scale=False,
            calibrate_dt=False,
            map_min_hits=5,
        ),
    },
    "factor_scan_pivot0": {
        "use_absolute_odom": False,
        "rear_pivot_x": 0.0,
        "kwargs": dict(
            use_encoder=True,
            deskew_mode="full_se2",
            synthesize_beam_times=True,
            frontend_icp="scan",
            lidar_seq_target="scan",
            submap_keyframes=10,
            factor_mode=True,
            estimate_scale=False,
            calibrate_dt=False,
            map_min_hits=5,
        ),
    },
    "factor_pivot_old": {
        "use_absolute_odom": False,
        "rear_pivot_x": -1.235,
        "kwargs": dict(
            use_encoder=True,
            deskew_mode="full_se2",
            synthesize_beam_times=True,
            frontend_icp="submap",
            lidar_seq_target="submap",
            submap_keyframes=10,
            factor_mode=True,
            estimate_scale=False,
            calibrate_dt=False,
            map_min_hits=5,
        ),
    },
}

# Controlled ablations used by the 20260820 warehouse sweep.  They inherit the
# same graph frontend so each row changes only deskew or LiDAR/odom timing.
_BASE_FACTOR = dict(VARIANTS["factor_pivot0"])
for _mode in ("none", "wheel", "wheel+yaw", "full_se2"):
    _spec = {
        "use_absolute_odom": False,
        "rear_pivot_x": 0.0,
        "kwargs": dict(VARIANTS["factor_pivot0"]["kwargs"]),
    }
    _spec["kwargs"]["deskew_mode"] = _mode
    _spec["kwargs"]["synthesize_beam_times"] = _mode == "full_se2"
    VARIANTS[f"deskew_{_mode.replace('+', '_')}"] = _spec
for _dt_ms in (-100, -50, -25, 25, 50, 100):
    _spec = {
        "use_absolute_odom": False,
        "rear_pivot_x": 0.0,
        "kwargs": dict(VARIANTS["factor_pivot0"]["kwargs"]),
    }
    _spec["kwargs"]["encoder_dt_s"] = _dt_ms / 1000.0
    VARIANTS[f"dt_{_dt_ms:+d}ms"] = _spec


def json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("variant", choices=sorted(VARIANTS))
    parser.add_argument(
        "--run",
        type=Path,
        default=ROOT / "maps/20260818/sim/run_192133",
        help="Recorded run to replay (default: historical run_192133).",
    )
    parser.add_argument(
        "--out-root",
        type=Path,
        default=ROOT / "results/20260819/sim_restore/run_192133",
        help="Root directory; the variant name is appended.",
    )
    args = parser.parse_args()
    spec = VARIANTS[args.variant]

    # Patch experiment constants before importing modules that bind them.
    from flifltbot.mapping import config

    config.LOW_LIDAR_X = 1.71
    config.LOW_LIDAR_HALF_Y = 0.385
    config.UP_LIDAR_X = 1.16
    config.SIM_UP_LIDAR_X = 1.14
    config.UP_LIDAR_Z = 1.645
    config.REAR_PIVOT_X = float(spec["rear_pivot_x"])
    config.USE_ABSOLUTE_ODOM = bool(spec["use_absolute_odom"])
    config.CAMERA_POSE_BODY.update({
        "left": (1.71, 0.385, config.LOW_LIDAR_Z, config.LOW_LIDAR_YAW),
        "right": (1.71, -0.385, config.LOW_LIDAR_Z, -config.LOW_LIDAR_YAW),
        "up": (1.16, 0.0, 1.645, 0.0),
        "top": (1.16, 0.0, 1.645, 0.0),
    })

    from flifltbot.mapping import encoder_sync
    from flifltbot.mapping.factor_diag import (
        apply_se2,
        ate_metrics,
        load_odom_gt,
        sync_est_to_gt,
        umeyama_se2,
    )
    from flifltbot.mapping.pipeline import sew_all_local_frames

    encoder_sync.REAR_PIVOT_X = config.REAR_PIVOT_X
    encoder_sync.USE_ABSOLUTE_ODOM = config.USE_ABSOLUTE_ODOM

    run = args.run.resolve()
    out = args.out_root.resolve() / args.variant
    out.mkdir(parents=True, exist_ok=True)

    started = time.perf_counter()
    points, poses, files, stats = sew_all_local_frames(
        run,
        cameras=["up"],
        source="sim",
        up_body_x=1.14,
        **spec["kwargs"],
    )
    wall_s = time.perf_counter() - started

    gt = load_odom_gt(run)
    valid_poses = [p for p in poses if isinstance(p, dict) and p.get("pose")]
    est_xy, gt_xy, est_yaw, gt_yaw, labels = sync_est_to_gt(valid_poses, gt)
    yaw, tx, ty = umeyama_se2(est_xy, gt_xy)
    metrics = ate_metrics(est_xy, gt_xy, est_yaw, gt_yaw, labels)

    payload = {
        "variant": args.variant,
        "run": str(run),
        "geometry": {
            "left_right_x_m": 1.71,
            "left_right_y_m": [0.385, -0.385],
            "real_up_x_m": 1.16,
            "sim_up_x_m": 1.14,
            "up_z_m": 1.645,
            "rear_pivot_x_m": spec["rear_pivot_x"],
        },
        "use_absolute_odom": spec["use_absolute_odom"],
        "kwargs": spec["kwargs"],
        "alignment": "global rigid SE(2), no scale/reflection",
        "transform": {"yaw_deg": math.degrees(yaw), "tx_m": tx, "ty_m": ty},
        "metrics": metrics,
        "wall_s": wall_s,
        "stats": stats,
        "map": points,
        "poses": poses,
    }
    result_path = out / "result.json"
    result_path.write_text(
        json.dumps(payload, default=json_default, separators=(",", ":")),
        encoding="utf-8",
    )
    summary = {k: payload[k] for k in (
        "variant", "run", "geometry", "use_absolute_odom", "kwargs",
        "alignment", "transform", "metrics", "wall_s",
    )}
    summary["stats"] = {k: stats.get(k) for k in (
        "n_frames", "keyframes", "wheel_edges", "lidar_seq_edges",
        "loop_closures", "stable_cells", "loop_error_m", "loop_error_deg",
        "odom_trajectory",
    )}
    (out / "summary.json").write_text(
        json.dumps(summary, indent=2, default=json_default) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, default=json_default), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
