#!/usr/bin/env python3
"""Focused ceiling-marker localization in the Cartographer map frame.

This is the small localization output used while validating the coordinate
contract.  It intentionally does not create per-marker images or re-run SLAM.

The geometry follows ``auto_map/build_global_marker_map.py``:

* ``center`` is ``[u=x, v=y]``;
* legacy ``tlbr`` is ``[y1, x1, y2, x2]``;
* pixels are undistorted into the optical camera frame;
* the fixed optical-to-robot convention is the ``-pi/2`` branch;
* the lateral camera offset is reflected once (sidecar -> mapping body frame);
* Cartographer poses are interpolated by Unix timestamp;
* output marker coordinates remain in Cartographer map coordinates.

The stored observation ``ray`` is retained for audit, but is not transformed
again.  Optional GT handling is strictly offline evaluation and is never used
to select or alter runtime points.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


UNIX_EPOCH_TICKS = 621355968000000000
DEFAULT_ROLL = -0.074022
DEFAULT_PITCH = -0.149217
DEFAULT_YAW = -1.548500
DEFAULT_TIME_OFFSET = -0.098153


def rotation2(yaw: float) -> np.ndarray:
    c, s = math.cos(float(yaw)), math.sin(float(yaw))
    return np.asarray([[c, -s], [s, c]], dtype=float)


def rotation3(roll: float, pitch: float, yaw: float) -> np.ndarray:
    cr, sr = math.cos(float(roll)), math.sin(float(roll))
    cp, sp = math.cos(float(pitch)), math.sin(float(pitch))
    cy, sy = math.cos(float(yaw)), math.sin(float(yaw))
    rx = np.asarray([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]])
    ry = np.asarray([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]])
    rz = np.asarray([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]])
    return rz @ ry @ rx


def load_camera_intrinsics(path: Path) -> tuple[np.ndarray, np.ndarray, int, int]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    entry = next(iter(payload.values()))
    width, height = int(entry["width"]), int(entry["height"])
    scale = float(max(width, height))
    matrix = np.asarray(
        [
            [float(entry["focal_x"]) * scale, 0.0, width / 2.0 + float(entry["c_x"]) * scale],
            [0.0, float(entry["focal_y"]) * scale, height / 2.0 + float(entry["c_y"]) * scale],
            [0.0, 0.0, 1.0],
        ],
        dtype=float,
    )
    distortion = np.asarray(
        [float(entry.get(key, 0.0)) for key in ("k1", "k2", "p1", "p2", "k3")],
        dtype=float,
    )
    return matrix, distortion, width, height


def load_camera_position(path: Path, y_sign: int) -> tuple[np.ndarray, dict[str, Any]]:
    lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(lines) < 2:
        raise ValueError(f"camera_position.txt must have two lines: {path}")
    raw_position = np.asarray([float(value) for value in lines[0].split()], dtype=float)
    raw_angles = np.asarray([float(value) for value in lines[1].split()], dtype=float)
    if raw_position.shape != (3,) or raw_angles.shape != (2,):
        raise ValueError(f"Invalid camera_position.txt: {path}")
    if not np.isfinite(raw_position).all() or not np.isfinite(raw_angles).all():
        raise ValueError(f"Non-finite camera_position.txt: {path}")
    # The sidecar is centimetres; the mapping route is metres.  The old
    # AutoMap pipeline explicitly reflected the lateral component once.
    input_m = raw_position / 100.0
    mapping_m = input_m.copy()
    mapping_m[1] *= int(y_sign)
    return mapping_m, {
        "source": str(path.resolve()),
        "raw_position_cm": raw_position.tolist(),
        "sidecar_orientation_rad": raw_angles.tolist(),
        "input_offset_m": input_m.tolist(),
        "mapping_body_offset_m": mapping_m.tolist(),
        "units_conversion": "camera_position first line cm -> m",
        "lateral_sign_applied_once": int(y_sign),
    }


def detection_center(item: dict[str, Any], width: int, height: int) -> tuple[np.ndarray | None, str]:
    """Return a pixel as [u=x,v=y].

    Existing recordings use ``tlbr=[y1,x1,y2,x2]``.  Their explicit center is
    already [x,y] and is preferred after checking it against the box center.
    """
    box = np.asarray(item.get("tlbr", []), dtype=float)
    from_box: np.ndarray | None = None
    if box.shape == (4,) and np.isfinite(box).all():
        y1, x1, y2, x2 = box
        candidate = np.asarray([(x1 + x2) / 2.0, (y1 + y2) / 2.0], dtype=float)
        if 0.0 <= candidate[0] < width and 0.0 <= candidate[1] < height:
            from_box = candidate
    center = np.asarray(item.get("center", []), dtype=float)
    if center.shape == (2,) and np.isfinite(center).all():
        if 0.0 <= center[0] < width and 0.0 <= center[1] < height:
            if from_box is not None and np.linalg.norm(center - from_box) <= 4.0:
                return center, "center_xy_verified_against_legacy_tlbr"
            if from_box is None:
                return center, "center_xy"
    if from_box is not None:
        return from_box, "legacy_tlbr_yxyx"
    return None, "invalid"


def load_frame_times(payload: dict[str, Any], observations_path: Path) -> tuple[np.ndarray, str]:
    frames = payload.get("data", [])
    embedded = payload.get("frame_timestamps")
    if embedded and len(embedded) >= len(frames):
        values = np.asarray(embedded[: len(frames)], dtype=float)
        if values.shape == (len(frames),) and np.isfinite(values).all():
            return values, "observations.json:frame_timestamps"
    sidecar = observations_path.parent / "frame_timestamps.csv"
    if not sidecar.is_file():
        raise FileNotFoundError(f"Missing frame timestamps: {sidecar}")
    by_index: dict[int, float] = {}
    with sidecar.open(newline="", encoding="utf-8") as handle:
        for ordinal, row in enumerate(csv.DictReader(handle)):
            try:
                index = int(row.get("frame_index", ordinal))
                value = float(row["time_take_img_sec"])
            except (KeyError, TypeError, ValueError):
                continue
            if math.isfinite(value):
                by_index[index] = value
    if not all(index in by_index for index in range(len(frames))):
        raise ValueError(f"Frame timestamp CSV is incomplete: {sidecar}")
    return np.asarray([by_index[index] for index in range(len(frames))], dtype=float), str(sidecar.resolve())


def load_trajectory(path: Path) -> tuple[np.ndarray, np.ndarray]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    poses = payload.get("poses", payload if isinstance(payload, list) else [])
    times: list[float] = []
    values: list[list[float]] = []
    for item in poses:
        if "timestamp_universal" in item:
            stamp = (float(item["timestamp_universal"]) - UNIX_EPOCH_TICKS) / 1.0e7
        elif "timestamp_ns" in item:
            stamp = float(item["timestamp_ns"]) / 1.0e9
        elif "timestamp_s" in item:
            stamp = float(item["timestamp_s"])
        elif "t" in item:
            # Native cloud-map exports store Unix seconds in ``t``.
            stamp = float(item["t"])
        else:
            continue
        x, y = float(item.get("x", math.nan)), float(item.get("y", math.nan))
        pose = item.get("pose")
        if isinstance(pose, list) and len(pose) >= 3:
            yaw = float(pose[2])
        elif "orientation_dgr" in item:
            yaw = math.radians(float(item["orientation_dgr"]))
        else:
            qx, qy, qz, qw = (float(item.get(key, 0.0)) for key in ("qx", "qy", "qz", "qw"))
            yaw = math.atan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))
        if math.isfinite(stamp) and all(math.isfinite(value) for value in (x, y, yaw)):
            times.append(stamp)
            values.append([x, y, yaw])
    if len(times) < 2:
        raise ValueError(f"Need at least two timed poses: {path}")
    order = np.argsort(np.asarray(times))
    times_a = np.asarray(times, dtype=float)[order]
    values_a = np.asarray(values, dtype=float)[order]
    keep = np.r_[True, np.diff(times_a) > 1.0e-9]
    return times_a[keep], values_a[keep]


def interpolate_poses(query: np.ndarray, times: np.ndarray, poses: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    query = np.asarray(query, dtype=float)
    inside = (query >= times[0]) & (query <= times[-1])
    yaw = np.interp(query, times, np.unwrap(poses[:, 2]))
    result = np.column_stack(
        [np.interp(query, times, poses[:, 0]), np.interp(query, times, poses[:, 1]), yaw]
    )
    return result, inside


def optical_rays(pixels: np.ndarray, matrix: np.ndarray, distortion: np.ndarray) -> np.ndarray:
    undistorted = cv2.undistortPoints(
        np.asarray(pixels, dtype=float).reshape(-1, 1, 2), matrix, distortion
    ).reshape(-1, 2)
    rays = np.column_stack((undistorted, np.ones(len(undistorted), dtype=float)))
    rays /= np.linalg.norm(rays, axis=1, keepdims=True)
    return rays


def project_observations(
    observations: list[dict[str, Any]],
    trajectory_times: np.ndarray,
    trajectory: np.ndarray,
    camera_offset_m: np.ndarray,
    fixed_camera_angles: tuple[float, float, float],
    time_offset_s: float,
    ceiling_z: float,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    if not observations:
        return [], {}
    query = np.asarray([row["timestamp_s"] for row in observations], dtype=float) + time_offset_s
    poses, inside = interpolate_poses(query, trajectory_times, trajectory)
    fixed_rotation = rotation3(*fixed_camera_angles)
    output: list[dict[str, Any]] = []
    counters: defaultdict[str, int] = defaultdict(int)
    for index, (row, pose, is_inside) in enumerate(zip(observations, poses, inside)):
        result = dict(row)
        result["observation_index"] = index
        result["query_timestamp_s"] = float(query[index])
        result["trajectory_inside"] = bool(is_inside)
        result["robot_pose_map"] = [float(value) for value in pose]
        if not is_inside:
            result["filter_reason"] = "outside_cartographer_trajectory"
            result["geometry_valid"] = False
            counters["outside_cartographer_trajectory"] += 1
            output.append(result)
            continue
        body_rotation = rotation2(float(pose[2]))
        camera_xy = pose[:2] + body_rotation @ camera_offset_m[:2]
        origin = np.asarray([camera_xy[0], camera_xy[1], camera_offset_m[2]], dtype=float)
        ray_body = fixed_rotation @ np.asarray(row["ray_optical_unit"], dtype=float)
        ray_map = np.asarray(
            [*(body_rotation @ ray_body[:2]), ray_body[2]], dtype=float
        )
        ray_map /= np.linalg.norm(ray_map)
        result["camera_origin_map"] = [float(value) for value in origin]
        result["ray_cartographer_map_unit"] = [float(value) for value in ray_map]
        if not np.isfinite(ray_map).all() or ray_map[2] <= 1.0e-6:
            result["filter_reason"] = "downward_or_invalid_ray"
            result["geometry_valid"] = False
            counters["downward_or_invalid_ray"] += 1
            output.append(result)
            continue
        scale = (ceiling_z - origin[2]) / ray_map[2]
        if not math.isfinite(scale) or scale <= 0.0:
            result["filter_reason"] = "invalid_plane_intersection"
            result["geometry_valid"] = False
            counters["invalid_plane_intersection"] += 1
            output.append(result)
            continue
        point = origin + scale * ray_map
        result["point_xyz_m"] = [float(value) for value in point]
        result["geometry_valid"] = True
        result["filter_reason"] = None
        counters["geometry_valid"] += 1
        output.append(result)
    counters["raw_observations"] = len(observations)
    counters["frames_outside_trajectory"] = sum(
        1 for row in output if row.get("filter_reason") == "outside_cartographer_trajectory"
    )
    return output, dict(counters)


def largest_component(points: np.ndarray, radius_m: float) -> np.ndarray:
    if len(points) == 0:
        return np.zeros(0, dtype=bool)
    adjacency = np.linalg.norm(points[:, None, :] - points[None, :, :], axis=2) <= radius_m
    remaining = set(range(len(points)))
    components: list[list[int]] = []
    while remaining:
        seed = next(iter(remaining))
        remaining.remove(seed)
        stack = [seed]
        component: list[int] = []
        while stack:
            current = stack.pop()
            component.append(current)
            neighbours = [index for index in remaining if adjacency[current, index]]
            for index in neighbours:
                remaining.remove(index)
                stack.append(index)
        components.append(component)
    best = max(
        components,
        key=lambda component: (
            len(component),
            -float(
                np.median(
                    np.linalg.norm(
                        points[component] - np.median(points[component], axis=0), axis=1
                    )
                )
            ),
        ),
    )
    mask = np.zeros(len(points), dtype=bool)
    mask[best] = True
    return mask


def filter_markers(
    projected: list[dict[str, Any]],
    min_observations: int,
    cluster_radius_m: float,
    inlier_radius_m: float,
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]], list[dict[str, Any]]]:
    by_id: defaultdict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(projected):
        if row.get("geometry_valid"):
            by_id[str(row["marker_id"])].append(index)
    markers: dict[str, dict[str, Any]] = {}
    diagnostics: dict[str, dict[str, Any]] = {}
    for marker_id in sorted(by_id, key=lambda value: int(value) if value.isdigit() else value):
        row_indices = by_id[marker_id]
        points = np.asarray([projected[index]["point_xyz_m"][:2] for index in row_indices], dtype=float)
        diag: dict[str, Any] = {
            "marker_id": int(marker_id) if marker_id.isdigit() else marker_id,
            "raw_count": sum(1 for row in projected if str(row["marker_id"]) == marker_id),
            "geometry_valid_count": len(row_indices),
            "inlier_count": 0,
            "rejected_count": len(row_indices),
            "status": "rejected",
            "spread_m": None,
            "spread_p95_m": None,
            "point_xyz_m": None,
            "raw_observation_indices": row_indices,
            "filtered_observation_indices": [],
        }
        if len(points) < min_observations:
            diag["reason"] = "below_min_observations"
            diagnostics[marker_id] = diag
            continue
        # This is the robust filter used by the already-validated AutoMap
        # reconstruction: median -> 75th-percentile gate -> median ->
        # 85th-percentile gate.  A connected-component selector is tempting,
        # but it can choose a coherent false track-switch cluster for a code.
        # Keep the proven per-ID median rule for this localization output.
        center = np.median(points, axis=0)
        distances = np.linalg.norm(points - center, axis=1)
        keep = distances <= max(inlier_radius_m, float(np.percentile(distances, 75.0)))
        if int(keep.sum()) < min_observations:
            diag["reason"] = "insufficient_robust_inliers"
            diagnostics[marker_id] = diag
            continue
        center = np.median(points[keep], axis=0)
        distances = np.linalg.norm(points - center, axis=1)
        keep &= distances <= max(inlier_radius_m, float(np.percentile(distances[keep], 85.0)))
        if int(keep.sum()) < min_observations:
            diag["reason"] = "insufficient_robust_inliers"
            diagnostics[marker_id] = diag
            continue
        center = np.median(points[keep], axis=0)
        distances = np.linalg.norm(points - center, axis=1)
        inlier_indices = np.asarray(row_indices, dtype=int)[keep]
        inlier_distances = distances[keep]
        point = np.asarray([center[0], center[1], projected[inlier_indices[0]]["point_xyz_m"][2]])
        marker = {
            "marker_id": int(marker_id) if marker_id.isdigit() else marker_id,
            "point_xyz_m": [float(value) for value in point],
            "observations": len(row_indices),
            "inliers": int(keep.sum()),
            "spread_m": float(np.median(inlier_distances)),
            "spread_p95_m": float(np.percentile(inlier_distances, 95)),
            "track_ids": sorted({str(projected[index]["track_id"]) for index in inlier_indices}),
            "raw_observation_indices": row_indices,
            "filtered_observation_indices": [int(value) for value in inlier_indices],
        }
        markers[marker_id] = marker
        diag.update(
            {
                "inlier_count": int(keep.sum()),
                "rejected_count": int(len(row_indices) - keep.sum()),
                "status": "accepted",
                "spread_m": marker["spread_m"],
                "spread_p95_m": marker["spread_p95_m"],
                "point_xyz_m": marker["point_xyz_m"],
                "filtered_observation_indices": marker["filtered_observation_indices"],
                "reason": None,
            }
        )
        diagnostics[marker_id] = diag
    for index, row in enumerate(projected):
        marker_id = str(row["marker_id"])
        info = diagnostics.get(marker_id)
        row["inlier"] = bool(info and index in set(info["filtered_observation_indices"]))
        if row.get("geometry_valid") and not row["inlier"]:
            row["filter_reason"] = "robust_per_id_outlier"
    filtered = [row for row in projected if row.get("inlier")]
    return markers, diagnostics, filtered


def read_map_yaml(path: Path) -> dict[str, Any]:
    values: dict[str, Any] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.split("#", 1)[0].strip()
        if not line or ":" not in line:
            continue
        key, value = line.split(":", 1)
        value = value.strip()
        if value.startswith("["):
            values[key.strip()] = json.loads(value)
        else:
            try:
                values[key.strip()] = float(value)
            except ValueError:
                values[key.strip()] = value.strip("'\"")
    return values


def map_to_grid(point_xy: Iterable[float], map_meta: dict[str, Any]) -> np.ndarray:
    origin = np.asarray(map_meta["origin"], dtype=float)
    yaw = float(origin[2])
    return rotation2(-yaw) @ (np.asarray(point_xy, dtype=float) - origin[:2])


def marker_color(marker_id: str) -> tuple[float, float, float, float]:
    digest = hashlib.sha256(marker_id.encode("utf-8")).digest()
    value = int.from_bytes(digest[:2], "big") / 65535.0
    return (*plt.get_cmap("turbo")(0.08 + 0.84 * value)[:3], 1.0)


def render_map(
    output: Path,
    map_dir: Path,
    markers: dict[str, dict[str, Any]],
    projected: list[dict[str, Any]],
    trajectory: np.ndarray,
    map_meta: dict[str, Any],
) -> None:
    image_path = map_dir / str(map_meta["image"])
    image = np.asarray(Image.open(image_path).convert("L"))
    height, width = image.shape
    resolution = float(map_meta["resolution"])
    extent = [0.0, width * resolution, 0.0, height * resolution]
    fig, ax = plt.subplots(figsize=(26, 18), dpi=220)
    ax.imshow(
        np.flipud(image),
        cmap="gray",
        origin="lower",
        extent=extent,
        interpolation="nearest",
        alpha=0.92,
    )
    grid_trajectory = np.asarray([map_to_grid(row[:2], map_meta) for row in trajectory])
    if len(grid_trajectory):
        ax.plot(grid_trajectory[:, 0], grid_trajectory[:, 1], color="#2563eb", lw=0.7, alpha=0.6, label="Cartographer trajectory")
    for row in projected:
        if not row.get("inlier"):
            continue
        start = map_to_grid(row["camera_origin_map"][:2], map_meta)
        end = map_to_grid(row["point_xyz_m"][:2], map_meta)
        ax.plot(
            [start[0], end[0]], [start[1], end[1]],
            color=marker_color(str(row["marker_id"])), alpha=0.035, lw=0.4,
        )
    for marker_id, marker in markers.items():
        point = map_to_grid(marker["point_xyz_m"][:2], map_meta)
        color = marker_color(marker_id)
        ax.scatter([point[0]], [point[1]], s=85, color=[color], edgecolors="white", linewidths=0.8, zorder=5)
        ax.annotate(str(marker["marker_id"]), point, xytext=(5, 5), textcoords="offset points", fontsize=9, color=color, zorder=6)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Occupancy-grid X for display (m)")
    ax.set_ylabel("Occupancy-grid Y for display (m)")
    ax.set_title(f"20260908 ceiling-marker localization v2 — {len(markers)} accepted IDs")
    ax.grid(alpha=0.18)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(output, dpi=220)
    plt.close(fig)


def offline_fit_se2(pred: dict[str, np.ndarray], gt: dict[str, np.ndarray]) -> dict[str, Any]:
    keys = sorted(set(pred).intersection(gt), key=lambda value: int(value) if value.isdigit() else value)
    if len(keys) < 2:
        return {"matched": len(keys), "error": "fewer than two common IDs"}
    a = np.asarray([pred[key] for key in keys], dtype=float)
    b = np.asarray([gt[key] for key in keys], dtype=float)
    ca, cb = a.mean(axis=0), b.mean(axis=0)
    u, _, vt = np.linalg.svd((a - ca).T @ (b - cb))
    rotation = vt.T @ u.T
    if np.linalg.det(rotation) < 0:
        vt[-1] *= -1.0
        rotation = vt.T @ u.T
    translation = cb - rotation @ ca
    before = np.linalg.norm(a - b, axis=1)
    after = np.linalg.norm(a @ rotation.T + translation - b, axis=1)
    return {
        "matched": len(keys),
        "gt_total": len(gt),
        "coverage": len(keys) / max(len(gt), 1),
        "before_rigid_alignment": {
            "rmse_m": float(np.sqrt(np.mean(before**2))),
            "median_m": float(np.median(before)),
            "p95_m": float(np.percentile(before, 95)),
        },
        "after_one_evaluation_only_se2": {
            "rmse_m": float(np.sqrt(np.mean(after**2))),
            "median_m": float(np.median(after)),
            "p95_m": float(np.percentile(after, 95)),
            "max_m": float(np.max(after)),
            "under_0_5m": int(np.count_nonzero(after < 0.5)),
            "under_1m": int(np.count_nonzero(after < 1.0)),
        },
        "evaluation_only_transform": {
            "rotation_rad": float(math.atan2(rotation[1, 0], rotation[0, 0])),
            "translation_m": [float(value) for value in translation],
        },
        "note": "GT and this SE(2) are offline diagnostics only; neither is used to create marker_points.json",
    }


def load_gt(path: Path) -> dict[str, np.ndarray]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    labels = payload.get("labels", payload if isinstance(payload, list) else [])
    return {str(item["id"]): np.asarray(item["xyz_m"][:2], dtype=float) for item in labels}


def load_marker_points(path: Path) -> dict[str, np.ndarray]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    result: dict[str, np.ndarray] = {}
    for item in payload.get("markers", []):
        marker_id = item.get("marker_id", item.get("code"))
        point = item.get("point_xyz_m")
        if point is None and item.get("x") is not None and item.get("y") is not None:
            point = [item["x"], item["y"]]
        if marker_id is not None and point is not None:
            result[str(marker_id)] = np.asarray(point[:2], dtype=float)
    return result


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[3]
    recording = root / "ros-data/recordings/real/20260908/160617_dephan_m360_room"
    cartographer = root / "ros-data/results/real/20260908/cartographer_standalone"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--observations", type=Path, default=recording / "observations.json")
    parser.add_argument("--trajectory", type=Path, default=cartographer / "trajectory.json")
    parser.add_argument("--map-dir", type=Path, default=cartographer)
    parser.add_argument("--camera-json", type=Path, default=recording / "camera.json")
    parser.add_argument("--camera-position", type=Path, default=recording / "camera_position.txt")
    parser.add_argument("--out-dir", type=Path, default=cartographer / "cartographer_marker_localization_v2")
    parser.add_argument("--ceiling-z", type=float, default=3.03)
    parser.add_argument("--camera-y-sign", type=int, choices=(-1, 1), default=-1)
    parser.add_argument("--camera-roll", type=float, default=DEFAULT_ROLL)
    parser.add_argument("--camera-pitch", type=float, default=DEFAULT_PITCH)
    parser.add_argument("--camera-yaw", type=float, default=DEFAULT_YAW)
    parser.add_argument("--time-offset", type=float, default=DEFAULT_TIME_OFFSET)
    parser.add_argument("--min-observations", type=int, default=12)
    parser.add_argument("--cluster-radius-m", type=float, default=0.35)
    parser.add_argument("--inlier-radius-m", type=float, default=0.12)
    parser.add_argument("--gt", type=Path, default=None)
    parser.add_argument("--before", type=Path, default=None)
    parser.add_argument("--baseline", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = json.loads(args.observations.read_text(encoding="utf-8"))
    frame_times, frame_source = load_frame_times(payload, args.observations)
    trajectory_times, trajectory = load_trajectory(args.trajectory)
    matrix, distortion, width, height = load_camera_intrinsics(args.camera_json)
    camera_offset, camera_meta = load_camera_position(args.camera_position, args.camera_y_sign)
    raw_rows: list[dict[str, Any]] = []
    raw_count = 0
    for frame_index, frame in enumerate(payload.get("data", [])):
        if frame_index >= len(frame_times) or not isinstance(frame, dict):
            continue
        for track_id, item in frame.items():
            if not isinstance(item, dict):
                continue
            raw_count += 1
            try:
                marker_id = int(item.get("code"))
            except (TypeError, ValueError):
                marker_id = -1
            pixel, pixel_source = detection_center(item, width, height)
            if marker_id < 0:
                continue
            if pixel is None:
                continue
            optical = optical_rays(pixel.reshape(1, 2), matrix, distortion)[0]
            stored_ray = np.asarray(item.get("ray", []), dtype=float)
            if stored_ray.shape != (3,) or not np.isfinite(stored_ray).all():
                stored_ray_list = None
            else:
                stored_ray /= max(float(np.linalg.norm(stored_ray)), 1.0e-12)
                stored_ray_list = [float(value) for value in stored_ray]
            raw_rows.append(
                {
                    "frame_index": int(frame_index),
                    "timestamp_s": float(frame_times[frame_index]),
                    "track_id": str(track_id),
                    "marker_id": marker_id,
                    "pixel_uv": [float(value) for value in pixel],
                    "pixel_source": pixel_source,
                    "ray_optical_unit": [float(value) for value in optical],
                    "stored_ray_audit_unit": stored_ray_list,
                    "stored_ray_used_for_geometry": False,
                }
            )
    projected, counters = project_observations(
        raw_rows,
        trajectory_times,
        trajectory,
        camera_offset,
        (args.camera_roll, args.camera_pitch, args.camera_yaw),
        args.time_offset,
        args.ceiling_z,
    )
    markers, per_id, filtered = filter_markers(
        projected, args.min_observations, args.cluster_radius_m, args.inlier_radius_m
    )
    map_meta = read_map_yaml(args.map_dir / "map.yaml")
    output_markers: list[dict[str, Any]] = []
    for marker_id in sorted(markers, key=lambda value: int(value) if value.isdigit() else value):
        marker = markers[marker_id]
        grid = map_to_grid(marker["point_xyz_m"][:2], map_meta)
        output_markers.append(
            {
                "marker_id": marker["marker_id"],
                "point_xyz_m": marker["point_xyz_m"],
                "occupancy_grid_xyz_m_for_display_only": [float(grid[0]), float(grid[1]), marker["point_xyz_m"][2]],
                "observations": marker["observations"],
                "inliers": marker["inliers"],
                "spread_m": marker["spread_m"],
                "spread_p95_m": marker["spread_p95_m"],
                "track_ids": marker["track_ids"],
            }
        )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    marker_payload = {
        "schema": "fliftbot.cartographer_marker_localization.v2",
        "frame": "cartographer_map",
        "point_coordinate_contract": "point_xyz_m is authoritative Cartographer map-frame; occupancy_grid_xyz_m_for_display_only is derived from map.yaml",
        "runtime_gt_used": False,
        "runtime_rigid_registration_used": False,
        "inputs": {
            "observations": str(args.observations.resolve()),
            "trajectory": str(args.trajectory.resolve()),
            "map_dir": str(args.map_dir.resolve()),
            "camera_json": str(args.camera_json.resolve()),
            "camera_position": str(args.camera_position.resolve()),
            "frame_timestamps": frame_source,
        },
        "geometry": {
            "pixel_contract": "center=[u=x,v=y]; legacy tlbr=[y1,x1,y2,x2]",
            "stored_ray_policy": "audit only; never transformed or used for geometry",
            "camera_offset": camera_meta,
            "optical_to_cartographer_fixed_angles_rad": {
                "roll": float(args.camera_roll),
                "pitch": float(args.camera_pitch),
                "yaw": float(args.camera_yaw),
            },
            "time_offset_s": float(args.time_offset),
            "ceiling_z_m": float(args.ceiling_z),
            "parameter_source": "geometry-only common-plane consistency; no GT fit",
        },
        "cartographer_map_frame": {
            "trajectory_pose_frame": "trajectory.json x/y/yaw",
            "occupancy_grid_is_rendering_only": True,
            "map_yaml_origin": map_meta.get("origin"),
        },
        "markers": output_markers,
    }
    marker_path = args.out_dir / "marker_points.json"
    marker_path.write_text(json.dumps(marker_payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    trajectory_for_display = trajectory
    render_path = args.out_dir / "localization_highres.png"
    render_map(render_path, args.map_dir, markers, filtered, trajectory_for_display, map_meta)
    diagnostics: dict[str, Any] = {
        "schema": "fliftbot.cartographer_marker_localization_diagnostics.v2",
        "runtime_gt_used": False,
        "runtime_rigid_registration_used": False,
        "raw_detection_records": raw_count,
        "raw_observations": raw_rows,
        "filtered_observations": filtered,
        "per_id": list(per_id.values()),
        "counters": {
            **counters,
            "accepted_markers": len(markers),
            "accepted_filtered_observations": len(filtered),
            "rejected_ids": sum(1 for item in per_id.values() if item["status"] != "accepted"),
        },
        "filter": {
            "min_observations": args.min_observations,
            "cluster_radius_m": args.cluster_radius_m,
            "inlier_radius_m": args.inlier_radius_m,
        },
        "marker_points": str(marker_path.resolve()),
        "render": str(render_path.resolve()),
    }
    if args.gt is not None:
        pred = {str(item["marker_id"]): np.asarray(item["point_xyz_m"][:2], dtype=float) for item in output_markers}
        gt = load_gt(args.gt)
        diagnostics["offline_gt_evaluation"] = {
            "gt": str(args.gt.resolve()),
            "after": offline_fit_se2(pred, gt),
        }
        if args.before is not None and args.before.is_file():
            diagnostics["offline_gt_evaluation"]["before"] = offline_fit_se2(load_marker_points(args.before), gt)
        if args.baseline is not None and args.baseline.is_file():
            diagnostics["offline_gt_evaluation"]["baseline_20260903"] = offline_fit_se2(load_marker_points(args.baseline), gt)
    diagnostics_path = args.out_dir / "localization_diagnostics.json"
    diagnostics_path.write_text(json.dumps(diagnostics, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"markers": len(markers), "filtered_observations": len(filtered), "marker_points": str(marker_path), "diagnostics": str(diagnostics_path), "render": str(render_path)}, ensure_ascii=False))


# Internal geometry library for V5.  There is intentionally no CLI entrypoint.
