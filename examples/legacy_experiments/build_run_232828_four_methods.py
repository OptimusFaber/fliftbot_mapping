#!/usr/bin/env python3
"""Build four 2D maps for the real Pandar run recorded on 2026-08-20.

The Pandar stamps in this run use the sensor's stale 1999 clock.  Frame wall
times from encoder.jsonl are therefore used as the LiDAR clock, interpreted as
Europe/Moscow (UTC+3), and the separately copied motion-controller CSV log is
converted to the mapper's encoder format on the same UTC timeline.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from flifltbot.mapping.config import METHOD_PRESETS  # noqa: E402
from flifltbot.mapping import pipeline  # noqa: E402

METHODS = ("clouds", "odom", "improved", "factor")
MSK = timezone(timedelta(hours=3))


def parse_robot_time(value: str) -> float:
    return datetime.strptime(value, "%Y_%m_%d_%H_%M_%S_%f").replace(tzinfo=MSK).timestamp()


def angle_delta_deg(a: float, b: float) -> float:
    return (a - b + 180.0) % 360.0 - 180.0


def prepare_shadow_run(run: Path, shadow: Path, odom_dir: Path | None = None) -> tuple[dict[int, float], dict]:
    shadow.mkdir(parents=True, exist_ok=True)
    horizon_link = shadow / "horizon"
    if not horizon_link.exists():
        horizon_link.symlink_to((run / "horizon").resolve(), target_is_directory=True)

    frame_clock = {}
    recorder_rows = []
    with (run / "encoder.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            frame = int(row["frame"])
            # recorder's enc_unix is the authoritative wall clock; the
            # human-readable enc_t can carry a different timezone.
            stamp = float(row["enc_unix"])
            frame_clock[frame] = stamp
            recorder_rows.append((frame, stamp))
    start, end = recorder_rows[0][1], recorder_rows[-1][1]

    candidates = sorted((odom_dir or run).glob("fliftbot_move_logger_*.log"))
    wheel_csv = (odom_dir or run) / "wheel_encoder_data.csv"
    if wheel_csv.exists():
        wheel_rows = []
        with wheel_csv.open(encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                stamp = float(row["timestamp_sec"])
                if start - 1.0 <= stamp <= end + 1.0:
                    wheel_rows.append((stamp, row))
        converted = []
        for stamp, row in wheel_rows:
            converted.append({
                "enc_t": datetime.fromtimestamp(stamp, timezone.utc).strftime("%Y_%m_%d_%H_%M_%S_%f"),
                "enc_unix": stamp,
                "enc_v": float(row["linear_velocity_m_per_s"]),
                "enc_dgr": math.degrees(float(row["wheel_angle_rad"])),
                "enc_available": True,
            })
        (shadow / "encoder.jsonl").write_text(
            "".join(json.dumps(row, separators=(",", ":")) + "\n" for row in converted),
            encoding="utf-8",
        )
        sync = {
            "timezone": "Unix timestamps (UTC)",
            "utc_correction_hours": 0,
            "lidar_clock_source": "per-frame enc_t from recorder encoder.jsonl",
            "lidar_start_utc": datetime.fromtimestamp(start, timezone.utc).isoformat(),
            "lidar_end_utc": datetime.fromtimestamp(end, timezone.utc).isoformat(),
            "odometry_source": str(wheel_csv),
            "odometry_samples": len(converted),
            "odometry_start_utc": datetime.fromtimestamp(wheel_rows[0][0], timezone.utc).isoformat(),
            "odometry_end_utc": datetime.fromtimestamp(wheel_rows[-1][0], timezone.utc).isoformat(),
        }
        return frame_clock, sync
    if not candidates:
        raise FileNotFoundError("wheel_encoder_data.csv or fliftbot_move_logger_*.log is missing")
    raw_rows = []
    with candidates[-1].open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            stamp = parse_robot_time(row["t"])
            if start - 1.0 <= stamp <= end + 1.0:
                raw_rows.append((stamp, row))
    if len(raw_rows) < 2:
        raise RuntimeError("motion log does not overlap the LiDAR recording")

    converted = []
    for stamp, row in raw_rows:
        converted.append({
            "enc_t": row["t"],
            "enc_unix": stamp,
            "enc_v": float(row["enc_v"]),
            "enc_dgr": float(row["enc_dgr"]),
            "odom_x": float(row["x"]),
            "odom_y": float(row["y"]),
            "odom_yaw": math.radians(float(row["dgr"])),
            "enc_available": True,
        })
    encoder_path = shadow / "encoder.jsonl"
    encoder_path.write_text(
        "".join(json.dumps(row, separators=(",", ":")) + "\n" for row in converted),
        encoding="utf-8",
    )
    sync = {
        "timezone": "Europe/Moscow (UTC+3)",
        "utc_correction_hours": -3,
        "lidar_clock_source": "per-frame enc_t from recorder encoder.jsonl",
        "lidar_start_utc": datetime.fromtimestamp(start, timezone.utc).isoformat(),
        "lidar_end_utc": datetime.fromtimestamp(end, timezone.utc).isoformat(),
        "odometry_samples": len(converted),
        "odometry_start_utc": datetime.fromtimestamp(raw_rows[0][0], timezone.utc).isoformat(),
        "odometry_end_utc": datetime.fromtimestamp(raw_rows[-1][0], timezone.utc).isoformat(),
    }
    return frame_clock, sync


def install_corrected_loader(frame_clock: dict[int, float], lidar_xy: tuple[float, float]):
    original = pipeline.load_horizon_npz_frame

    def corrected(path, min_points=200):
        loaded = original(path, min_points=min_points)
        if loaded is None:
            return None
        points, old_stamp, point_times, cameras, metadata = loaded
        frame = int(metadata["frame"])
        stamp = frame_clock[frame]
        times = np.asarray(point_times, dtype=float)
        if old_stamp is not None:
            times = times + (stamp - float(old_stamp))
        else:
            times = np.full(len(points), stamp, dtype=float)
        points = np.asarray(points, dtype=float) + np.asarray(lidar_xy, dtype=float)
        metadata.update({
            "t": stamp,
            "ts": datetime.fromtimestamp(stamp, timezone.utc).isoformat(),
            "point_times": [float(value) for value in times],
            "clock_repaired": True,
        })
        return points, stamp, times, cameras, metadata

    pipeline.load_horizon_npz_frame = corrected


def plot_map(path: Path, run_name: str, method: str, points, poses, stats):
    pts = np.asarray(points, float)
    route = np.asarray([item["pose"][:2] for item in poses], float)
    fig, ax = plt.subplots(figsize=(9, 7))
    if len(pts):
        ax.scatter(pts[:, 0], pts[:, 1], s=0.55, alpha=0.48, color="#1468b3")
    if len(route):
        ax.plot(route[:, 0], route[:, 1], lw=1.0, color="#d62828")
        ax.scatter(*route[0], s=32, color="#2a9d4b", zorder=3)
        ax.scatter(*route[-1], s=32, color="#8b0000", zorder=3)
    ax.set_aspect("equal")
    ax.grid(alpha=0.18)
    ax.set_xlabel("X, m")
    ax.set_ylabel("Y, m")
    ax.set_title(f"{run_name} — {method}\nframes={stats.get('n_frames')}  cells={len(pts)}  loops={stats.get('loop_closures', 0)}")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_comparison(path: Path, run_name: str, rows):
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    for ax, row in zip(axes.flat, rows):
        with np.load(row["result_npz"]) as data:
            pts, route = data["map"], data["route"]
        if len(pts):
            ax.scatter(pts[:, 0], pts[:, 1], s=0.3, alpha=0.42, color="#1468b3")
        if len(route):
            ax.plot(route[:, 0], route[:, 1], lw=0.8, color="#d62828")
        ax.set_aspect("equal")
        ax.grid(alpha=0.15)
        ax.set_title(f"{row['method']} | cells={row['stable_cells']} | loops={row['loop_closures']} | {row['wall_s']:.1f}s")
    fig.suptitle(f"{run_name}: four mapping methods", fontsize=15)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--odom-dir", type=Path)
    parser.add_argument("--out", type=Path, default=ROOT / "results/20260820/real/run_232828/four_methods")
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    run_name = args.run.resolve().name
    shadow = args.out / "synchronized_run"
    # Measured from the robot rotation axis; original notes were centimetres.
    lidar_pose = {"x_m": 0.590, "y_m": 0.035, "z_m": 1.510, "yaw_rad": 0.0}
    frame_clock, sync = prepare_shadow_run(args.run.resolve(), shadow, args.odom_dir.resolve() if args.odom_dir else None)
    install_corrected_loader(frame_clock, (lidar_pose["x_m"], lidar_pose["y_m"]))

    rows = []
    for method in args.methods:
        print(f"START {method}", flush=True)
        kwargs = dict(METHOD_PRESETS[method])
        started = time.perf_counter()
        points, poses, files, stats = pipeline.sew_all_local_frames(
            shadow, cameras=["horizon"], source="real", up_body_x=None,
            max_frames=args.max_frames, **kwargs
        )
        elapsed = time.perf_counter() - started
        pts = np.asarray(points, dtype=np.float32)
        route = np.asarray([item["pose"] for item in poses], dtype=np.float64)
        result_npz = args.out / f"{run_name}_{method}.npz"
        result_png = args.out / f"{run_name}_{method}.png"
        np.savez_compressed(result_npz, map=pts, route=route)
        plot_map(result_png, run_name, method, pts, poses, stats)
        row = {
            "method": method,
            "wall_s": round(elapsed, 3),
            "n_frames": int(stats.get("n_frames", len(files))),
            "encoder_frames": int(stats.get("encoder_frames", 0)),
            "encoder_synced_frames": int(stats.get("encoder_synced_frames", 0) or 0),
            "keyframes": int(stats.get("keyframes", 0)),
            "loop_closures": int(stats.get("loop_closures", 0)),
            "stable_cells": int(stats.get("stable_cells", len(pts))),
            "mode": stats.get("mode"),
            "result_npz": str(result_npz),
            "result_png": str(result_png),
        }
        rows.append(row)
        print(f"DONE {method}: {elapsed:.1f}s", flush=True)

    comparison = args.out / f"{run_name}_all_methods.png"
    plot_comparison(comparison, run_name, rows)
    summary = {
        "run": str(args.run.resolve()),
        "methods": rows,
        "comparison_png": str(comparison),
        "lidar_pose_rotation_center": lidar_pose,
        "time_sync": sync,
        "notes": "Z is recorded for provenance; a horizontal 2D map uses the XY lever arm only.",
    }
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
