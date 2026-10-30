#!/usr/bin/env python3
"""Transform a Cartographer trajectory into the physical marker-98 frame."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

from marker98_pose import pose_in_marker98_frame


DOTNET_EPOCH = 62135596800.0


def yaw(pose: dict) -> float:
    return math.atan2(2.0 * (pose["qw"] * pose["qz"]), 1.0 - 2.0 * pose["qz"] ** 2)


def wrap(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def normalize(input_path: Path, output_path: Path, reference_path: Path) -> int:
    data = json.loads(input_path.read_text(encoding="utf-8"))
    poses = data.get("poses")
    if not isinstance(poses, list) or not poses:
        raise ValueError(f"{input_path}: expected non-empty poses list")
    reference = json.loads(reference_path.read_text(encoding="utf-8"))
    x0, y0 = map(float, reference["marker98_map_xy_m"])
    a0 = float(reference.get("marker98_map_yaw_rad", 0.0))
    rows = []
    for pose in poses:
        x, y, alpha = pose_in_marker98_frame(
            float(pose["x"]), float(pose["y"]), yaw(pose), x0, y0, a0)
        rows.append({
            "x": x,
            "y": y,
            "alpha": alpha,
            "ts": float(pose["timestamp_universal"]) / 1e7 - DOTNET_EPOCH,
        })
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["x", "y", "alpha", "ts"])
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trajectory", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--marker-reference", type=Path, required=True)
    args = parser.parse_args()
    print(f"exported {normalize(args.trajectory, args.output, args.marker_reference)} poses")


if __name__ == "__main__":
    main()
