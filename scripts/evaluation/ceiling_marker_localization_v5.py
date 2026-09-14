#!/usr/bin/env python3
"""V5 ceiling-marker localization engine with ambiguity abstention.

Runtime selection uses only image geometry, Cartographer poses and observation
metadata.  GT is loaded only after the result is written, for offline scoring.
This module is the single geometry implementation behind the public V5 CLIs.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


_V2_PATH = Path(__file__).with_name("_ceiling_marker_geometry.py")
_V2_SPEC = importlib.util.spec_from_file_location("fliftbot_ceiling_marker_geometry", _V2_PATH)
if _V2_SPEC is None or _V2_SPEC.loader is None:
    raise ImportError(f"Cannot load localization helper: {_V2_PATH}")
v2 = importlib.util.module_from_spec(_V2_SPEC)
_V2_SPEC.loader.exec_module(v2)

# Installation calibration for the measured 2026-09-10 camera/LiDAR mounts.
# These angles were selected from repeat-observation ray consistency only; room
# GT is not an input to localization or to this calibration. The previous
# roll/pitch pair was coupled to the legacy (1.015, 0.0) LiDAR lever arm and
# becomes inconsistent when the measured (1.29, -0.32) mount is used.
V5_CAMERA_ROLL = -0.14
V5_CAMERA_PITCH = 0.01
V5_CAMERA_YAW = -1.57
V5_TIME_OFFSET = -0.08802549710329016


def connected_components(indices: list[int], projected: list[dict], radius_m: float) -> list[list[int]]:
    points = np.asarray([projected[i]["point_xyz_m"][:2] for i in indices], dtype=float)
    seen: set[int] = set()
    result: list[list[int]] = []
    for seed in range(len(indices)):
        if seed in seen:
            continue
        stack = [seed]
        seen.add(seed)
        component: list[int] = []
        while stack:
            current = stack.pop()
            component.append(indices[current])
            distances = np.linalg.norm(points - points[current], axis=1)
            for neighbour in np.flatnonzero(distances <= radius_m):
                neighbour = int(neighbour)
                if neighbour not in seen:
                    seen.add(neighbour)
                    stack.append(neighbour)
        result.append(component)
    return sorted(result, key=lambda value: (-len(value), min(value)))


def fit_hypothesis(
    projected: list[dict], indices: list[int], min_observations: int, inlier_radius_m: float
) -> dict | None:
    if len(indices) < min_observations:
        return None
    points = np.asarray([projected[i]["point_xyz_m"][:2] for i in indices], dtype=float)
    center = np.median(points, axis=0)
    distances = np.linalg.norm(points - center, axis=1)
    first = distances <= max(inlier_radius_m, float(np.percentile(distances, 75.0)))
    if int(first.sum()) < min_observations:
        return None
    center = np.median(points[first], axis=0)
    distances = np.linalg.norm(points - center, axis=1)
    final = first & (distances <= max(inlier_radius_m, float(np.percentile(distances[first], 85.0))))
    if int(final.sum()) < min_observations:
        return None
    center = np.median(points[final], axis=0)
    point = np.asarray([center[0], center[1], projected[indices[0]]["point_xyz_m"][2]], dtype=float)
    inlier_indices = np.asarray(indices, dtype=int)[final]
    ray_residuals = []
    for index in inlier_indices:
        row = projected[int(index)]
        origin = np.asarray(row["camera_origin_map"], dtype=float)
        direction = np.asarray(row["ray_cartographer_map_unit"], dtype=float)
        ray_residuals.append(float(np.linalg.norm(np.cross(point - origin, direction))))
    inlier_points = points[final]
    inlier_distances = np.linalg.norm(inlier_points - center, axis=1)
    return {
        "point_xyz_m": [float(value) for value in point],
        "all_indices": [int(value) for value in indices],
        "inlier_indices": [int(value) for value in inlier_indices],
        "observations": len(indices),
        "inliers": int(final.sum()),
        "spread_m": float(np.median(inlier_distances)),
        "spread_p95_m": float(np.percentile(inlier_distances, 95.0)),
        "ray_residual_m": float(np.median(ray_residuals)),
        "ray_residual_p95_m": float(np.percentile(ray_residuals, 95.0)),
        "track_ids": sorted({str(projected[i]["track_id"]) for i in inlier_indices}),
        "frame_span": int(max(projected[i]["frame_index"] for i in indices) - min(projected[i]["frame_index"] for i in indices) + 1),
    }


def hypothesis_score(hypothesis: dict) -> float:
    """Prefer support and independent tracks, penalize geometric residuals."""
    support = float(hypothesis["inliers"])
    tracks = float(len(hypothesis["track_ids"]))
    spread = float(hypothesis["spread_m"])
    ray = float(hypothesis["ray_residual_m"])
    return support * (1.0 + 0.12 * math.log1p(tracks)) / (1.0 + spread / 0.25 + ray / 0.25)


def select_hypotheses(
    projected: list[dict],
    min_observations: int = 12,
    component_radius_m: float = 0.25,
    inlier_radius_m: float = 0.12,
    ambiguity_ratio: float = 0.50,
) -> tuple[dict[str, dict], dict[str, dict], list[dict]]:
    by_code: defaultdict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(projected):
        if row.get("geometry_valid"):
            by_code[str(row["marker_id"])].append(index)

    markers: dict[str, dict] = {}
    diagnostics: dict[str, dict] = {}
    selected_indices: set[int] = set()
    for code in sorted(by_code, key=lambda value: int(value) if value.isdigit() else value):
        candidate_list: list[dict] = []
        for component in connected_components(by_code[code], projected, component_radius_m):
            fitted = fit_hypothesis(projected, component, min_observations, inlier_radius_m)
            if fitted is not None:
                fitted["score"] = float(hypothesis_score(fitted))
                candidate_list.append(fitted)
        candidate_list.sort(key=lambda value: (-value["score"], -value["inliers"], value["all_indices"][0]))
        base = {
            "marker_id": int(code) if code.isdigit() else code,
            "candidate_count": len(candidate_list),
            "status": "rejected",
            "reason": None,
            "candidates": [
                {
                    "inliers": c["inliers"],
                    "observations": c["observations"],
                    "tracks": len(c["track_ids"]),
                    "spread_m": c["spread_m"],
                    "ray_residual_m": c["ray_residual_m"],
                    "score": c["score"],
                    "point_xyz_m": c["point_xyz_m"],
                }
                for c in candidate_list
            ],
        }
        if not candidate_list:
            base["reason"] = "no_supported_hypothesis"
            diagnostics[code] = base
            continue
        best = candidate_list[0]
        second_support = candidate_list[1]["inliers"] if len(candidate_list) > 1 else 0
        base.update(
            {
                "selected_inliers": best["inliers"],
                "selected_observations": best["observations"],
                "selected_tracks": len(best["track_ids"]),
                "selected_spread_m": best["spread_m"],
                "selected_ray_residual_m": best["ray_residual_m"],
                "second_inliers": second_support,
                "ambiguity_ratio": float(second_support / max(best["inliers"], 1)),
            }
        )
        if second_support >= min_observations and second_support / max(best["inliers"], 1) >= ambiguity_ratio:
            base["reason"] = "ambiguous_spatial_hypotheses"
            diagnostics[code] = base
            continue
        marker = dict(best)
        marker["marker_id"] = int(code) if code.isdigit() else code
        markers[code] = marker
        selected_indices.update(best["inlier_indices"])
        base["status"] = "accepted"
        diagnostics[code] = base

    for index, row in enumerate(projected):
        row["inlier"] = index in selected_indices
        if row.get("geometry_valid") and not row["inlier"]:
            row["filter_reason"] = "v3_hypothesis_not_selected"
    filtered = [row for row in projected if row.get("inlier")]
    return markers, diagnostics, filtered


def merge_coincident_markers(markers: dict[str, dict], radius_m: float = 0.30) -> dict[str, dict]:
    """Collapse decoder code switches that describe one physical location.

    Distinct ceiling labels in the measured layout are much farther apart
    than this radius.  Codes sharing essentially the same reconstructed point
    are therefore competing identities, not separate markers.
    """
    pending = set(markers)
    merged: dict[str, dict] = {}
    while pending:
        seed = min(pending, key=lambda value: int(value) if value.isdigit() else value)
        stack = [seed]
        component = []
        pending.remove(seed)
        while stack:
            current = stack.pop()
            component.append(current)
            center = np.asarray(markers[current]["point_xyz_m"][:2], dtype=float)
            neighbours = [
                code for code in pending
                if np.linalg.norm(np.asarray(markers[code]["point_xyz_m"][:2], dtype=float) - center) <= radius_m
            ]
            for code in neighbours:
                pending.remove(code)
                stack.append(code)
        ranked = sorted(component, key=lambda code: (-markers[code]["score"], -markers[code]["inliers"], int(code)))
        primary = ranked[0]
        item = dict(markers[primary])
        item["candidate_marker_ids"] = [int(code) for code in ranked]
        item["identity_ambiguous"] = len(ranked) > 1
        merged[primary] = item
    return merged


def render_highres(output: Path, map_dir: Path, markers: dict[str, dict], projected: list[dict], trajectory: np.ndarray, map_meta: dict) -> None:
    image = np.asarray(Image.open(map_dir / str(map_meta["image"])).convert("L"))
    height, width = image.shape
    resolution = float(map_meta["resolution"])
    extent = [0.0, width * resolution, 0.0, height * resolution]
    fig, ax = plt.subplots(figsize=(32, 22), dpi=240)
    ax.imshow(np.flipud(image), cmap="gray", origin="lower", extent=extent, interpolation="nearest", alpha=0.92)
    path = np.asarray([v2.map_to_grid(row[:2], map_meta) for row in trajectory])
    if len(path):
        ax.plot(path[:, 0], path[:, 1], color="#2563eb", lw=0.7, alpha=0.55, label="Cartographer trajectory")
    for row in projected:
        if not row.get("inlier"):
            continue
        start = v2.map_to_grid(row["camera_origin_map"][:2], map_meta)
        end = v2.map_to_grid(row["point_xyz_m"][:2], map_meta)
        ax.plot([start[0], end[0]], [start[1], end[1]], color="#f97316", alpha=0.035, lw=0.45)
    for code, marker in markers.items():
        point = v2.map_to_grid(marker["point_xyz_m"][:2], map_meta)
        color = v2.marker_color(str(code))
        ax.scatter([point[0]], [point[1]], s=115, color=[color], edgecolors="white", linewidths=1.0, zorder=5)
        # Use compact sequential display numbers.  Decoder IDs remain in JSON.
        display_number = marker.get("display_number", code)
        ax.annotate(str(display_number), point, xytext=(6, 6), textcoords="offset points", fontsize=10, color=color, zorder=6)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Occupancy-grid X for display (m)")
    ax.set_ylabel("Occupancy-grid Y for display (m)")
    ax.set_title(f"Cartographer marker localization v3 — {len(markers)} accepted IDs")
    ax.grid(alpha=0.18)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(output, dpi=240)
    plt.close(fig)


def render_points_only(output: Path, map_dir: Path, markers: dict[str, dict], map_meta: dict) -> None:
    """Render only the occupancy raster and sequentially numbered points."""
    image = np.asarray(Image.open(map_dir / str(map_meta["image"])).convert("L"))
    height, width = image.shape
    resolution = float(map_meta["resolution"])
    fig, ax = plt.subplots(figsize=(32, 22), dpi=240)
    ax.imshow(
        np.flipud(image), cmap="gray", origin="lower",
        extent=[0.0, width * resolution, 0.0, height * resolution],
        interpolation="nearest", alpha=0.96,
    )
    for code, marker in markers.items():
        point = v2.map_to_grid(marker["point_xyz_m"][:2], map_meta)
        number = int(marker["display_number"])
        ax.scatter([point[0]], [point[1]], s=150, color="#ef4444", edgecolors="white", linewidths=1.2, zorder=5)
        ax.annotate(str(number), point, xytext=(7, 7), textcoords="offset points", fontsize=13, color="#111827", weight="bold", zorder=6)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Occupancy-grid X for display (m)")
    ax.set_ylabel("Occupancy-grid Y for display (m)")
    ax.set_title(f"Cartographer map — localized ceiling points ({len(markers)})")
    ax.grid(alpha=0.14)
    fig.tight_layout()
    fig.savefig(output, dpi=240)
    plt.close(fig)


def render_marker_codes(output: Path, map_dir: Path, markers: dict[str, dict], map_meta: dict) -> None:
    """Render physical points labelled with decoded code candidates."""
    image = np.asarray(Image.open(map_dir / str(map_meta["image"])).convert("L"))
    height, width = image.shape
    resolution = float(map_meta["resolution"])
    fig, ax = plt.subplots(figsize=(32, 22), dpi=240)
    ax.imshow(np.flipud(image), cmap="gray", origin="lower", extent=[0, width*resolution, 0, height*resolution], interpolation="nearest", alpha=.96)
    for marker in markers.values():
        point = v2.map_to_grid(marker["point_xyz_m"][:2], map_meta)
        candidates = marker.get("candidate_marker_ids", [marker["marker_id"]])
        label = "/".join(str(value) for value in candidates)
        color = "#f59e0b" if len(candidates) > 1 else "#2563eb"
        ax.scatter([point[0]], [point[1]], s=150, color=color, edgecolors="white", linewidths=1.2, zorder=5)
        ax.annotate(label, point, xytext=(7,7), textcoords="offset points", fontsize=11, color="#111827", weight="bold", zorder=6)
    ax.set_aspect("equal", adjustable="box"); ax.grid(alpha=.14)
    ax.set_xlabel("Occupancy-grid X for display (m)"); ax.set_ylabel("Occupancy-grid Y for display (m)")
    ax.set_title(f"Cartographer map — marker-code candidates ({len(markers)} physical points)")
    fig.tight_layout(); fig.savefig(output, dpi=240); plt.close(fig)


def render_cloud_map(
    output: Path, cloud: np.ndarray, markers: dict[str, dict], trajectory: np.ndarray,
    *, labels: str, projected: list[dict] | None = None,
) -> None:
    """Render markers directly in the metric frame of a clouds.json map."""
    fig, ax = plt.subplots(figsize=(32, 22), dpi=240)
    if len(cloud):
        ax.scatter(cloud[:, 0], cloud[:, 1], s=0.55, color="#111827", alpha=0.62, rasterized=True)
    if labels == "full" and len(trajectory):
        ax.plot(trajectory[:, 0], trajectory[:, 1], color="#2563eb", lw=0.7, alpha=0.55, label="trajectory")
        for row in projected or []:
            if row.get("inlier"):
                start, end = row["camera_origin_map"], row["point_xyz_m"]
                ax.plot([start[0], end[0]], [start[1], end[1]], color="#f97316", alpha=0.035, lw=0.45)
    for marker in markers.values():
        point = marker["point_xyz_m"][:2]
        candidates = marker.get("candidate_marker_ids", [marker["marker_id"]])
        if labels == "number":
            label, color = str(marker["display_number"]), "#ef4444"
        else:
            label = "/".join(str(value) for value in candidates)
            color = "#f59e0b" if len(candidates) > 1 else "#2563eb"
        ax.scatter([point[0]], [point[1]], s=150, color=color, edgecolors="white", linewidths=1.2, zorder=5)
        ax.annotate(label, point, xytext=(7, 7), textcoords="offset points", fontsize=11,
                    color="#111827", weight="bold", zorder=6)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("map X, m"); ax.set_ylabel("map Y, m"); ax.grid(alpha=.14)
    ax.set_title(f"Cloud map — localized ceiling points ({len(markers)})")
    if labels == "full": ax.legend(loc="best")
    fig.tight_layout(); fig.savefig(output, dpi=240); plt.close(fig)


def render_gt_comparison(output: Path, predicted: dict[str, np.ndarray], gt: dict[str, np.ndarray]) -> None:
    """Offline-only comparison after one rigid SE(2) alignment."""
    keys = sorted(set(predicted).intersection(gt), key=int)
    a = np.asarray([predicted[key] for key in keys], dtype=float)
    b = np.asarray([gt[key] for key in keys], dtype=float)
    ca, cb = a.mean(axis=0), b.mean(axis=0)
    u, _, vt = np.linalg.svd((a-ca).T @ (b-cb))
    rot = vt.T @ u.T
    if np.linalg.det(rot) < 0:
        vt[-1] *= -1; rot = vt.T @ u.T
    aligned = a @ rot.T + (cb - rot @ ca)
    errors = np.linalg.norm(aligned-b, axis=1)
    fig, ax = plt.subplots(figsize=(24, 11), dpi=240)
    ax.scatter(b[:,0], b[:,1], s=60, color="#2563eb", label="GT", zorder=4)
    ax.scatter(aligned[:,0], aligned[:,1], s=45, color="#ef4444", label="localized (evaluation-aligned)", zorder=5)
    for number, (key, source, target, error) in enumerate(zip(keys, aligned, b, errors), start=1):
        ax.plot([source[0],target[0]],[source[1],target[1]], color="#f59e0b", lw=.8, alpha=.7)
        ax.annotate(f"{number}:{key} ({error:.2f}m)", source, xytext=(4,4), textcoords="offset points", fontsize=7)
    ax.set_aspect("equal", adjustable="box"); ax.grid(alpha=.2); ax.legend()
    ax.set_xlabel("room X, m"); ax.set_ylabel("room Y, m")
    ax.set_title("Offline verification: localized points vs GT (one rigid SE(2), not used at runtime)")
    fig.tight_layout(); fig.savefig(output, dpi=240); plt.close(fig)


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
    parser.add_argument("--trajectory-pose-frame", choices=("rotation_axis", "lidar"), default="lidar",
                        help="Standalone maps built with a zero lidar.config mount export an effective lidar pose")
    parser.add_argument("--lidar-x", type=float, default=1.29,
                        help="LiDAR X from robot rotation axis, metres")
    parser.add_argument("--lidar-y", type=float, default=-0.32,
                        help="LiDAR Y from robot rotation axis, metres")
    parser.add_argument("--out-dir", type=Path, default=cartographer / "marker_localization_v5")
    parser.add_argument("--output-profile", choices=("all", "points", "labels"), default="all",
                        help="Select visualization products; marker_points.json is always written")
    parser.add_argument("--ceiling-z", type=float, default=3.13)
    parser.add_argument("--camera-y-sign", type=int, choices=(-1, 1), default=-1)
    parser.add_argument("--camera-roll", type=float, default=V5_CAMERA_ROLL)
    parser.add_argument("--camera-pitch", type=float, default=V5_CAMERA_PITCH)
    parser.add_argument("--camera-yaw", type=float, default=V5_CAMERA_YAW)
    parser.add_argument("--time-offset", type=float, default=V5_TIME_OFFSET)
    parser.add_argument("--min-observations", type=int, default=12)
    parser.add_argument("--component-radius-m", type=float, default=0.25)
    parser.add_argument("--inlier-radius-m", type=float, default=0.12)
    parser.add_argument("--ambiguity-ratio", type=float, default=0.50)
    parser.add_argument("--physical-merge-radius-m", type=float, default=0.30)
    parser.add_argument("--gt", type=Path, default=None,
                        help="Optional GT used after localization, for offline validation only")
    parser.add_argument("--before", type=Path, default=cartographer / "cartographer_marker_overlay/marker_points_cartographer.json")
    parser.add_argument("--v2", dest="v2_path", type=Path, default=cartographer / "cartographer_marker_localization_v2/marker_points.json")
    parser.add_argument("--baseline", type=Path, default=root / "ros-data/results/real/20260903/nav_video_lidar_run_114127/global_marker_map_corrected.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = json.loads(args.observations.read_text(encoding="utf-8"))
    frame_times, frame_source = v2.load_frame_times(payload, args.observations)
    trajectory_times, trajectory = v2.load_trajectory(args.trajectory)
    matrix, distortion, width, height = v2.load_camera_intrinsics(args.camera_json)
    camera_offset, camera_meta = v2.load_camera_position(args.camera_position, args.camera_y_sign)
    camera_offset_axis = camera_offset.copy()
    if args.trajectory_pose_frame == "lidar":
        # Standalone currently feeds sensor-local points with a zero mount for
        # this canonical recording, so the optimized pose effectively follows
        # the lidar. Convert the camera lever arm to that pose origin.
        camera_offset[:2] -= np.asarray([args.lidar_x, args.lidar_y], dtype=float)
    camera_meta["trajectory_pose_frame"] = args.trajectory_pose_frame
    camera_meta["camera_offset_axis_m"] = [float(x) for x in camera_offset_axis]
    camera_meta["lidar_offset_axis_xy_m"] = [float(args.lidar_x), float(args.lidar_y)]
    camera_meta["camera_offset_from_trajectory_xy_m"] = [float(x) for x in camera_offset[:2]]
    raw_rows = []
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
                continue
            if marker_id < 0:
                continue
            pixel, pixel_source = v2.detection_center(item, width, height)
            if pixel is None:
                continue
            ray = v2.optical_rays(pixel.reshape(1, 2), matrix, distortion)[0]
            raw_rows.append({"frame_index": int(frame_index), "timestamp_s": float(frame_times[frame_index]), "track_id": str(track_id), "marker_id": marker_id, "pixel_uv": [float(x) for x in pixel], "pixel_source": pixel_source, "ray_optical_unit": [float(x) for x in ray], "stored_ray_used_for_geometry": False})
    projected, counters = v2.project_observations(raw_rows, trajectory_times, trajectory, camera_offset, (args.camera_roll, args.camera_pitch, args.camera_yaw), args.time_offset, args.ceiling_z)
    markers, per_id, filtered = select_hypotheses(projected, args.min_observations, args.component_radius_m, args.inlier_radius_m, args.ambiguity_ratio)
    markers = merge_coincident_markers(markers, args.physical_merge_radius_m)
    trajectory_payload = json.loads(args.trajectory.read_text(encoding="utf-8"))
    cloud = np.asarray(trajectory_payload.get("map", []), dtype=float)
    cloud_mode = cloud.ndim == 2 and cloud.shape[1] >= 2 and len(cloud) > 0
    map_meta = None if cloud_mode else v2.read_map_yaml(args.map_dir / "map.yaml")
    serial_markers = []
    ordered_codes = sorted(markers, key=lambda value: int(value) if value.isdigit() else value)
    for display_number, code in enumerate(ordered_codes, start=1):
        marker = markers[code]
        marker["display_number"] = display_number
        display = marker["point_xyz_m"][:2] if cloud_mode else v2.map_to_grid(marker["point_xyz_m"][:2], map_meta)
        serial_markers.append({"display_number": display_number, "marker_id": marker["marker_id"], "candidate_marker_ids": marker.get("candidate_marker_ids", [marker["marker_id"]]), "identity_ambiguous": marker.get("identity_ambiguous", False), "point_xyz_m": marker["point_xyz_m"], "display_xyz_m": [float(display[0]), float(display[1]), marker["point_xyz_m"][2]], "observations": marker["observations"], "inliers": marker["inliers"], "spread_m": marker["spread_m"], "spread_p95_m": marker["spread_p95_m"], "ray_residual_m": marker["ray_residual_m"], "ray_residual_p95_m": marker["ray_residual_p95_m"], "track_ids": marker["track_ids"], "hypothesis_score": marker["score"]})
    args.out_dir.mkdir(parents=True, exist_ok=True)
    marker_path = args.out_dir / "marker_points.json"
    marker_path.write_text(json.dumps({"schema": "fliftbot.ceiling_marker_localization.v5", "method_version": "v5", "frame": "map", "point_coordinate_contract": "point_xyz_m is authoritative map-frame; display_xyz_m is display-only", "runtime_gt_used": False, "runtime_rigid_registration_used": False, "inputs": {"observations": str(args.observations.resolve()), "trajectory": str(args.trajectory.resolve()), "map_dir": str(args.map_dir.resolve()), "camera_json": str(args.camera_json.resolve()), "camera_position": str(args.camera_position.resolve()), "frame_timestamps": frame_source}, "geometry": {"pixel_contract": "center=[u=x,v=y]; legacy tlbr=[y1,x1,y2,x2]", "stored_ray_policy": "audit only; never used for geometry", "camera_offset": camera_meta, "optical_to_map_fixed_angles_rad": {"roll": args.camera_roll, "pitch": args.camera_pitch, "yaw": args.camera_yaw}, "time_offset_s": args.time_offset, "ceiling_z_m": args.ceiling_z}, "selection": {"criterion": "support * independent-track bonus / (spread and ray residual penalty)", "component_radius_m": args.component_radius_m, "ambiguity_ratio": args.ambiguity_ratio, "ambiguous_codes_are_rejected": True}, "markers": serial_markers}, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    render_path = args.out_dir / "localization_highres.png"
    if args.output_profile == "all":
        if cloud_mode: render_cloud_map(render_path, cloud[:, :2], markers, trajectory, labels="full", projected=filtered)
        else: render_highres(render_path, args.map_dir, markers, filtered, trajectory, map_meta)
    points_only_path = args.out_dir / "localization_points_only_highres.png"
    if args.output_profile in ("all", "points"):
        if cloud_mode: render_cloud_map(points_only_path, cloud[:, :2], markers, trajectory, labels="number")
        else: render_points_only(points_only_path, args.map_dir, markers, map_meta)
    marker_codes_path = args.out_dir / "localization_marker_codes_highres.png"
    if args.output_profile in ("all", "labels"):
        if cloud_mode: render_cloud_map(marker_codes_path, cloud[:, :2], markers, trajectory, labels="codes")
        else: render_marker_codes(marker_codes_path, args.map_dir, markers, map_meta)
    diagnostics = {"schema": "fliftbot.ceiling_marker_localization_diagnostics.v5", "method_version": "v5", "runtime_gt_used": False, "runtime_rigid_registration_used": False, "raw_detection_records": raw_count, "raw_observations": len(raw_rows), "filtered_observations": len(filtered), "per_id": list(per_id.values()), "counters": {**counters, "accepted_markers": len(markers), "accepted_filtered_observations": len(filtered), "rejected_ids": sum(1 for x in per_id.values() if x["status"] != "accepted")}, "selection": {"component_radius_m": args.component_radius_m, "inlier_radius_m": args.inlier_radius_m, "ambiguity_ratio": args.ambiguity_ratio, "physical_merge_radius_m": args.physical_merge_radius_m}, "marker_points": str(marker_path.resolve()), "render": str(render_path.resolve()) if args.output_profile == "all" else None, "points_only_render": str(points_only_path.resolve()) if args.output_profile in ("all", "points") else None, "marker_codes_render": str(marker_codes_path.resolve()) if args.output_profile in ("all", "labels") else None}
    diagnostics_path = args.out_dir / "localization_diagnostics.json"
    diagnostics_path.write_text(json.dumps(diagnostics, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    if args.gt is not None and args.gt.is_file():
        gt = v2.load_gt(args.gt)
        pred = {str(x["marker_id"]): np.asarray(x["point_xyz_m"][:2], dtype=float) for x in serial_markers}
        metrics = {"v5": v2.offline_fit_se2(pred, gt), "note": "GT and the SE(2) transform are offline diagnostics only; neither is used to create marker_points.json"}
        (args.out_dir / "offline_metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        gt_comparison_path = args.out_dir / "localization_vs_gt_highres.png"
        render_gt_comparison(gt_comparison_path, pred, gt)
        diagnostics["offline_gt_evaluation"] = {"gt": str(args.gt.resolve()), "metrics_file": str((args.out_dir / "offline_metrics.json").resolve())}
        diagnostics["offline_gt_evaluation"]["comparison_render"] = str(gt_comparison_path.resolve())
        diagnostics_path.write_text(json.dumps(diagnostics, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"markers": len(markers), "filtered_observations": len(filtered), "out_dir": str(args.out_dir.resolve())}, ensure_ascii=False))


if __name__ == "__main__":
    main()
