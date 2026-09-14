#!/usr/bin/env python3
"""Plot LiDAR-time nav trajectory and nav-derived ceiling markers on room GT."""
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

import nav_marker_map_notebook_style as nav_markers


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_RECORDING = ROOT / "ros-data/recordings/real/20260908/160617_dephan_m360_room"
DEFAULT_GT_DIR = ROOT / "ros-data/gt/real_room/real_room"
DEFAULT_OUT = ROOT / "ros-data/results/real/20260914/nav_lidar_interval_floorplan"
REFERENCE_MARKER_ID = 98


def local_to_room(points: np.ndarray, reference_room_xy: np.ndarray) -> np.ndarray:
    """Convert tag-frame (+X south, +Y east) to room (+X east, +Y north)."""
    points = np.asarray(points, dtype=float)
    return np.column_stack(
        [
            reference_room_xy[0] + points[..., 1],
            reference_room_xy[1] - points[..., 0],
        ]
    )


def load_gt(path: Path) -> tuple[dict[int, np.ndarray], np.ndarray]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    markers = {
        int(row["id"]): np.asarray(row["gt_room_xyz_m"][:2], dtype=float)
        for row in payload["labels"]
    }
    if REFERENCE_MARKER_ID not in markers:
        raise ValueError(f"GT has no reference marker {REFERENCE_MARKER_ID}")
    return markers, markers[REFERENCE_MARKER_ID]


def lidar_interval(recording: Path) -> tuple[float, float, dict[str, Any]]:
    metadata = json.loads((recording / "recording_metadata.json").read_text(encoding="utf-8"))
    lidar = metadata["lidar"]
    start, end = float(lidar["first_ts"]), float(lidar["last_ts"])
    if not math.isfinite(start) or not math.isfinite(end) or end <= start:
        raise ValueError("invalid LiDAR interval in recording_metadata.json")
    return start, end, lidar


def load_floorplan_contract(gt_dir: Path) -> tuple[np.ndarray, list[float]]:
    meta = json.loads((gt_dir / "floorplan_meta.json").read_text(encoding="utf-8"))
    image = np.asarray(Image.open(gt_dir / "floorplan.png").convert("L"))
    resolution = float(meta["cell_m"])
    origin_x, origin_y = (float(v) for v in meta["origin"][:2])
    extent = [
        origin_x,
        origin_x + image.shape[1] * resolution,
        origin_y,
        origin_y + image.shape[0] * resolution,
    ]
    return image, extent


def unique_nav_in_interval(
    nav_t: np.ndarray, nav_pose: np.ndarray, start: float, end: float
) -> tuple[np.ndarray, np.ndarray]:
    mask = (nav_t >= start) & (nav_t <= end)
    times, poses = nav_t[mask], nav_pose[mask]
    if len(times) < 2:
        raise ValueError("fewer than two nav samples overlap the LiDAR interval")
    return times, poses


def filter_observations_to_interval(
    payload: dict[str, Any], frame_times: np.ndarray, start: float, end: float
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    mask = (frame_times >= start) & (frame_times <= end)
    indices = np.flatnonzero(mask)
    data = payload.get("data", [])
    filtered = {"data": [data[int(i)] for i in indices]}
    return filtered, frame_times[mask], indices


def notebook_code_centers(
    rows: list[dict[str, Any]], min_observations: int
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Follow the notebook geometry, then take one median centre per decoded code."""
    by_code: dict[int, list[np.ndarray]] = defaultdict(list)
    for row in rows:
        by_code[int(row["marker_id"])].append(np.asarray(row["point_xy_m"], dtype=float))
    points, rejected = [], []
    for marker_id, values in sorted(by_code.items()):
        if len(values) < min_observations:
            rejected.append(
                {
                    "marker_id": marker_id,
                    "reason": "below_min_observations",
                    "observations": len(values),
                }
            )
            continue
        array = np.asarray(values, dtype=float)
        center = np.median(array, axis=0)
        points.append(
            {
                "marker_id": marker_id,
                "point_xy_m": center.tolist(),
                "observations": len(values),
                "median_spread_m": float(np.median(np.linalg.norm(array - center, axis=1))),
            }
        )
    return points, rejected


def render(
    output: Path,
    floorplan: np.ndarray,
    extent: list[float],
    trajectory_room: np.ndarray,
    nav_pose: np.ndarray,
    predictions_room: np.ndarray,
    predictions: list[dict[str, Any]],
    gt: dict[int, np.ndarray],
    reference_xy: np.ndarray,
    start: float,
    end: float,
) -> None:
    fig, ax = plt.subplots(figsize=(20, 9), dpi=180, constrained_layout=True)
    ax.imshow(
        np.flipud(floorplan),
        origin="lower",
        extent=extent,
        cmap="gray_r",
        vmin=0,
        vmax=255,
        interpolation="nearest",
        alpha=0.82,
        zorder=0,
    )

    gt_xy = np.asarray(list(gt.values()), dtype=float)
    ax.scatter(
        gt_xy[:, 0],
        gt_xy[:, 1],
        marker="s",
        s=44,
        facecolors="none",
        edgecolors="#2563eb",
        linewidths=1.0,
        zorder=3,
        label=f"GT ceiling markers ({len(gt)})",
    )
    for marker_id, point in gt.items():
        ax.text(point[0] + 0.035, point[1] + 0.035, str(marker_id), fontsize=5.5, color="#1d4ed8", zorder=4)

    ax.plot(
        trajectory_room[:, 0],
        trajectory_room[:, 1],
        color="#7c3aed",
        linewidth=2.2,
        zorder=5,
        label=f"robot nav_* path during LiDAR ({len(trajectory_room)} poses)",
    )
    ax.scatter(*trajectory_room[0], marker="o", s=80, color="#16a34a", edgecolors="white", linewidths=0.8, zorder=7, label="path start")
    ax.scatter(*trajectory_room[-1], marker="X", s=90, color="#7c3aed", edgecolors="white", linewidths=0.8, zorder=7, label="path end")

    arrow_indices = np.linspace(0, len(trajectory_room) - 1, min(16, len(trajectory_room)), dtype=int)
    # A local heading vector [cos(yaw), sin(yaw)] maps to room [sin(yaw), -cos(yaw)].
    room_directions = np.column_stack([np.sin(nav_pose[:, 2]), -np.cos(nav_pose[:, 2])])
    ax.quiver(
        trajectory_room[arrow_indices, 0],
        trajectory_room[arrow_indices, 1],
        room_directions[arrow_indices, 0],
        room_directions[arrow_indices, 1],
        angles="xy",
        scale_units="xy",
        scale=1.7,
        width=0.003,
        color="#7c3aed",
        alpha=0.8,
        zorder=6,
    )

    if len(predictions_room):
        ax.scatter(
            predictions_room[:, 0],
            predictions_room[:, 1],
            marker="o",
            s=58,
            color="#dc2626",
            edgecolors="white",
            linewidths=0.8,
            zorder=8,
            label=f"algorithm marker points ({len(predictions_room)})",
        )
        for point, prediction in zip(predictions_room, predictions):
            ax.text(
                point[0] + 0.045,
                point[1] - 0.09,
                str(prediction["marker_id"]),
                fontsize=6,
                color="#991b1b",
                zorder=9,
            )

    ax.scatter(
        *reference_xy,
        marker="*",
        s=180,
        color="#f59e0b",
        edgecolors="black",
        linewidths=0.8,
        zorder=10,
        label="reference marker 98",
    )
    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("room X — east (m)")
    ax.set_ylabel("room Y — north (m)")
    ax.grid(color="#64748b", alpha=0.18, linewidth=0.5)
    ax.set_title(
        "nav_* trajectory and nav-derived ceiling-marker points on real-room floorplan\n"
        f"LiDAR interval {start:.3f} … {end:.3f} s ({end - start:.1f} s); "
        "tag-frame origin = GT marker 98; no fitted rotation or scale"
    )
    ax.legend(loc="upper left", fontsize=8, framealpha=0.92, ncol=2)
    fig.savefig(output)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recording", type=Path, default=DEFAULT_RECORDING)
    parser.add_argument("--gt-dir", type=Path, default=DEFAULT_GT_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--height-m", type=float, default=3.03)
    parser.add_argument("--min-observations", type=int, default=12)
    args = parser.parse_args()

    recording = args.recording.resolve()
    gt_dir = args.gt_dir.resolve()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    start, end, lidar_meta = lidar_interval(recording)
    observations = json.loads((recording / "observations.json").read_text(encoding="utf-8"))
    frame_times = nav_markers.load_frame_times(recording / "frame_timestamps.csv", len(observations["data"]))
    interval_observations, interval_frame_times, frame_indices = filter_observations_to_interval(
        observations, frame_times, start, end
    )
    nav_log = next(recording.glob("fliftbot_move_logger_*.log"))
    all_nav_t, all_nav_pose, nav_meta = nav_markers.load_nav(nav_log)
    nav_t, nav_pose = unique_nav_in_interval(all_nav_t, all_nav_pose, start, end)
    camera_origin = nav_markers.load_camera_origin(recording / "camera_position.txt")

    raw = nav_markers.raw_rows(
        interval_observations,
        interval_frame_times,
        all_nav_t,
        all_nav_pose,
        camera_origin,
        args.height_m,
        0.0,
    )
    predictions, rejected = notebook_code_centers(raw, args.min_observations)
    prediction_by_id = {int(row["marker_id"]): row for row in predictions}
    if REFERENCE_MARKER_ID not in prediction_by_id:
        raise ValueError(
            f"notebook-like result has no marker {REFERENCE_MARKER_ID}; cannot anchor the overlay"
        )

    gt, reference_xy = load_gt(gt_dir / "ceiling_labels_room_real_gt.json")
    prediction_local = np.asarray([row["point_xy_m"] for row in predictions], dtype=float)
    predicted_reference_local = np.asarray(
        prediction_by_id[REFERENCE_MARKER_ID]["point_xy_m"], dtype=float
    )
    # "Centre is 98": use exactly one translation for path and marker points.
    # Heading and scale remain the known notebook/tag-frame conventions.
    aligned_nav_local = nav_pose[:, :2] - predicted_reference_local
    aligned_prediction_local = prediction_local - predicted_reference_local
    trajectory_room = local_to_room(aligned_nav_local, reference_xy)
    predictions_room = local_to_room(aligned_prediction_local, reference_xy)
    floorplan, extent = load_floorplan_contract(gt_dir)

    png_path = args.out_dir / "nav_lidar_interval_floorplan.png"
    render(
        png_path,
        floorplan,
        extent,
        trajectory_room,
        nav_pose,
        predictions_room,
        predictions,
        gt,
        reference_xy,
        start,
        end,
    )

    report = {
        "schema": "fliftbot.nav_lidar_interval_floorplan.v1",
        "inputs": {
            "recording": str(recording),
            "floorplan": str((gt_dir / "floorplan.png").resolve()),
            "gt_markers": str((gt_dir / "ceiling_labels_room_real_gt.json").resolve()),
            "nav_source": nav_meta,
        },
        "lidar_interval": {
            "first_ts": start,
            "last_ts": end,
            "duration_s": end - start,
            "packets": int(lidar_meta["packets"]),
            "frames": int(lidar_meta["frames"]),
        },
        "selection": {
            "nav_poses": int(len(nav_t)),
            "camera_frames": int(len(interval_frame_times)),
            "first_camera_frame_index": int(frame_indices[0]),
            "last_camera_frame_index": int(frame_indices[-1]),
            "raw_marker_observations": int(len(raw)),
            "marker_codes_with_min_observations": int(len(predictions)),
            "plotted_marker_points": int(len(predictions)),
            "rejected_codes": rejected,
        },
        "coordinate_contract": {
            "reference_marker_id": REFERENCE_MARKER_ID,
            "reference_room_xy_m": reference_xy.tolist(),
            "nav_and_prediction_source_frame": "tag 98 local frame: +X south, +Y east",
            "room_frame": "+X east, +Y north",
            "predicted_reference_local_xy_m": predicted_reference_local.tolist(),
            "anchor_translation_local_xy_m": (-predicted_reference_local).tolist(),
            "formula": "aligned_local=local-predicted_98; room_x=11.685+aligned_local_y; room_y=3.525-aligned_local_x",
            "reference_translation_applied_to_trajectory": True,
            "reference_translation_applied_to_predictions": True,
            "fitted_rotation": False,
            "fitted_scale": False,
        },
        "marker_algorithm": {
            "method": "stored camera ray intersected with fixed ceiling plane, transformed by interpolated nav_x/nav_y/nav_dgr",
            "aggregation": "median per decoded marker code, matching the notebook-style full-run audit; no cross-code clustering",
            "ceiling_height_m": args.height_m,
            "min_observations": args.min_observations,
            "points": [
                {
                    **row,
                    "point_room_xy_m": predictions_room[index].tolist(),
                }
                for index, row in enumerate(predictions)
            ],
        },
        "output": str(png_path.resolve()),
    }
    (args.out_dir / "report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    with (args.out_dir / "nav_lidar_interval.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["timestamp_s", "nav_x", "nav_y", "nav_dgr", "room_x", "room_y"])
        for timestamp, pose, room in zip(nav_t, nav_pose, trajectory_room):
            writer.writerow([timestamp, pose[0], pose[1], math.degrees(pose[2]), room[0], room[1]])

    print(
        json.dumps(
            {
                "output": str(png_path.resolve()),
                "lidar_duration_s": end - start,
                "nav_poses": len(nav_t),
                "camera_frames": len(interval_frame_times),
                "marker_points": len(predictions),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
