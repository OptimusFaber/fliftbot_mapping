#!/usr/bin/env python3
"""Replay canonical LiDAR scans on a calibrated deploy map without ROS or GT."""
from __future__ import annotations
import argparse
import csv
import json
import math
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
import direct_udp_mc_localizer as matcher
from map_bundle import validate_bundle, read_reference
from marker98_pose import pose_in_marker98_frame


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recording", type=Path, required=True, help="canonical directory or sim RUN/lidar")
    parser.add_argument("--map-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--initial", nargs=3, type=float, metavar=("X_MAP", "Y_MAP", "YAW_RAD"))
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--min-overlap", type=float, default=.5)
    parser.add_argument("--max-rmse", type=float, default=.2)
    parser.add_argument("--min-acceptance", type=float, default=.9)
    args = parser.parse_args()
    if args.stride < 1 or not 0 < args.min_acceptance <= 1:
        parser.error("stride must be positive and min-acceptance in (0,1]")
    report = validate_bundle(args.map_dir)
    root = Path(report["map_dir"])
    calibration = report["navigation_config"]
    mount = calibration["scan_to_base"]
    files = sorted((args.recording / "horizon").glob("*.npz"))[::args.stride]
    if not files:
        parser.error(f"no horizon/*.npz in {args.recording}")
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("output must be empty")
    args.output.mkdir(parents=True, exist_ok=True)
    points, field, resolution, origin, shape = matcher.load_map(root / "map.yaml")
    tree = cKDTree(points)
    reference = read_reference(root / "origin_reference.json")
    pose = tuple(args.initial) if args.initial else None
    accepted, rejected = [], []
    for number, path in enumerate(files):
        with np.load(path) as data:
            scan = np.asarray(data["xy"], dtype=float)
            stamp = float(data["stamp"])
            frame = int(data["frame"]) if "frame" in data else number
        radius = np.linalg.norm(scan, axis=1)
        valid = np.isfinite(scan).all(axis=1) & (radius >= .15) & (radius <= calibration["max_range_m"])
        base = matcher.transform(scan[valid], (mount["x_m"], mount["y_m"], math.radians(mount["yaw_deg"])))
        if len(base) < 100:
            rejected.append({"frame": frame, "reason": "sparse"})
            continue
        seed = pose if pose is not None else matcher.global_pose(base, field, resolution, origin, shape)
        candidate, rmse, overlap = matcher.icp(base, tree, seed)
        if not math.isfinite(rmse) or rmse > args.max_rmse or overlap < args.min_overlap:
            rejected.append({"frame": frame, "reason": "quality", "overlap": overlap,
                             "rmse": rmse if math.isfinite(rmse) else None})
            continue
        pose = candidate
        local = pose_in_marker98_frame(*pose, *reference)
        accepted.append((stamp, frame, *pose, *local, rmse, overlap))
    with (args.output / "trajectory.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("ts", "frame", "map_x", "map_y", "map_yaw_rad", "x", "y", "alpha_rad", "rmse_m", "overlap"))
        writer.writerows(accepted)
    ratio = len(accepted) / len(files)
    result = {"status": "accepted" if ratio >= args.min_acceptance else "rejected",
              "recording": str(args.recording.resolve()), "map_dir": str(root),
              "clock": "input stamps preserved (simulation time for Gazebo)",
              "pose_frame": "robot rotation axis in bundle local frame; metres/radians",
              "frames": len(files), "accepted": len(accepted), "acceptance_ratio": ratio,
              "rmse_median_m": float(np.median([row[-2] for row in accepted])) if accepted else None,
              "rejections": rejected}
    (args.output / "metrics.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return 0 if result["status"] == "accepted" else 3


if __name__ == "__main__":
    raise SystemExit(main())
