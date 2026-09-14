#!/usr/bin/env python3
"""Scene-agnostic ceiling marker localization with automatic plane height.

V6 keeps the proven V5 camera ray convention, but removes the scene-specific
ceiling height from runtime.  It reconstructs repeated marker observations by
robust multi-ray triangulation and estimates one horizontal ceiling plane from
the resulting 3-D points.  GT, room dimensions and SE(2) registration are
never read by this program.

The trajectory frame is explicit.  For a Cartographer trajectory whose
tracking frame is ``lidar``, ``lidar.config`` is read and the camera offset is
converted from the rotation-axis frame exactly once.  If the geometry is
under-observed or inconsistent, V6 abstains instead of emitting a fabricated
height or point.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


HERE = Path(__file__).resolve().parent
V5_PATH = HERE / "ceiling_marker_localization_v5.py"
OVERLAY_PATH = HERE / "overlay_marker_rays_on_cartographer_map.py"


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


v5 = _load(V5_PATH, "fliftbot_ceiling_marker_v5_for_v6")
geom = v5.v2
overlay = _load(OVERLAY_PATH, "fliftbot_ceiling_marker_overlay_for_v6")


def read_lidar_offset(path: Path) -> tuple[np.ndarray, dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    position = payload.get("position_m", payload.get("position", {}))
    values = np.asarray([float(position[key]) for key in ("x", "y", "z")], dtype=float)
    if values.shape != (3,) or not np.isfinite(values).all():
        raise ValueError(f"Invalid lidar position_m in {path}")
    return values, {"source": str(path.resolve()), "position_m": values.tolist(), "format": payload.get("format")}


def raw_observations(observations: Path, camera_json: Path) -> tuple[list[dict[str, Any]], np.ndarray, np.ndarray, int, int, str]:
    payload = json.loads(observations.read_text(encoding="utf-8"))
    frame_times, frame_source = geom.load_frame_times(payload, observations)
    matrix, distortion, width, height = geom.load_camera_intrinsics(camera_json)
    rows: list[dict[str, Any]] = []
    for frame_index, frame in enumerate(payload.get("data", [])):
        if frame_index >= len(frame_times) or not isinstance(frame, dict):
            continue
        for track_id, item in frame.items():
            if not isinstance(item, dict):
                continue
            try:
                marker_id = int(item.get("code"))
            except (TypeError, ValueError):
                continue
            if marker_id < 0:
                continue
            pixel, source = geom.detection_center(item, width, height)
            if pixel is None:
                continue
            ray = geom.optical_rays(pixel.reshape(1, 2), matrix, distortion)[0]
            rows.append({
                "frame_index": int(frame_index),
                "timestamp_s": float(frame_times[frame_index]),
                "track_id": str(track_id),
                "marker_id": marker_id,
                "pixel_uv": [float(v) for v in pixel],
                "pixel_source": source,
                "ray_optical_unit": [float(v) for v in ray],
            })
    return rows, frame_times, matrix, width, height, frame_source


def project_rays(
    rows: list[dict[str, Any]], trajectory_times: np.ndarray, trajectory: np.ndarray,
    camera_offset: np.ndarray, time_offset: float,
    angles: tuple[float, float, float],
) -> list[dict[str, Any]]:
    """Use the V5 frame/timestamp contract, while omitting plane intersection."""
    # A temporary positive plane is used only to invoke V5's frame conversion;
    # all geometry is read from camera_origin_map and ray_cartographer_map_unit.
    projected, _ = geom.project_observations(
        rows, trajectory_times, trajectory, camera_offset, angles, time_offset,
        float(camera_offset[2] + 1.0),
    )
    return [r for r in projected if r.get("geometry_valid")]


def triangulate_code(rows: list[dict[str, Any]], residual_gate: float, min_inliers: int) -> dict[str, Any] | None:
    values = [{
        "camera_xyz_m": r["camera_origin_map"],
        "ray_map_unit": r["ray_cartographer_map_unit"],
        "frame_index": r["frame_index"], "timestamp_s": r["timestamp_s"],
        "track_id": r["track_id"],
    } for r in rows]
    fit = overlay.triangulate_lines(values, residual_gate)
    if fit is None or len(fit["inlier_indices"]) < min_inliers:
        return None
    point = np.asarray(fit["point_xyz_m"], dtype=float)
    return {
        "point_xyz_m": [float(v) for v in point],
        "inlier_indices": [int(v) for v in fit["inlier_indices"]],
        "inliers": int(len(fit["inlier_indices"])),
        "raw_observations": len(rows),
        "spread_m": float(fit["spread_m"]),
        "track_ids": sorted({values[i]["track_id"] for i in fit["inlier_indices"]}),
        "code": int(rows[0]["marker_id"]),
    }


def robust_height(fits: list[dict[str, Any]], camera_z: float, min_height: float, max_height: float) -> dict[str, Any] | None:
    candidates = [f for f in fits if camera_z + min_height <= f["point_xyz_m"][2] <= camera_z + max_height]
    if len(candidates) < 3:
        return None
    z = np.asarray([f["point_xyz_m"][2] for f in candidates], dtype=float)
    center = float(np.median(z))
    mad = float(np.median(np.abs(z - center)))
    gate = max(0.12, 3.0 * 1.4826 * mad)
    keep = np.abs(z - center) <= gate
    if int(keep.sum()) < 3:
        return None
    z_keep = z[keep]
    estimate = float(np.median(z_keep))
    robust_sigma = float(1.4826 * np.median(np.abs(z_keep - estimate)))
    # A plane estimate is actionable only when independent marker fits agree.
    confidence = min(1.0, float(keep.sum()) / 12.0) * math.exp(-robust_sigma / 0.10)
    if confidence < 0.25:
        return None
    return {
        "ceiling_z_m": estimate,
        "candidate_marker_fits": len(candidates),
        "inlier_marker_fits": int(keep.sum()),
        "rejected_marker_fits": int((~keep).sum()),
        "robust_sigma_m": robust_sigma,
        "confidence": float(confidence),
        "bounds_m_above_camera": [float(min_height), float(max_height)],
    }


def main() -> None:
    root = Path(__file__).resolve().parents[3]
    recording = root / "ros-data/recordings/real/20260908/160617_dephan_m360_room"
    result = root / "ros-data/results/real/20260910/160617_dephan_m360_room/cartographer_lidar_tracking_candidate"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--observations", type=Path, default=recording / "observations.json")
    parser.add_argument("--trajectory", type=Path, default=result / "trajectory.json")
    parser.add_argument("--camera-json", type=Path, default=recording / "camera.json")
    parser.add_argument("--camera-position", type=Path, default=recording / "camera_position.txt")
    parser.add_argument("--lidar-config", type=Path, default=recording / "lidar.config")
    parser.add_argument("--trajectory-pose-frame", choices=("auto", "rotation_axis", "lidar"), default="auto")
    parser.add_argument("--camera-y-sign", type=int, choices=(-1, 1), default=-1)
    parser.add_argument("--camera-roll", type=float, default=v5.V5_CAMERA_ROLL)
    parser.add_argument("--camera-pitch", type=float, default=v5.V5_CAMERA_PITCH)
    parser.add_argument("--camera-yaw", type=float, default=v5.V5_CAMERA_YAW)
    parser.add_argument("--time-offset", type=float, default=v5.V5_TIME_OFFSET)
    parser.add_argument("--out-dir", type=Path, default=result / "marker_localization_v6_auto")
    parser.add_argument("--min-observations", type=int, default=12)
    parser.add_argument("--triangulation-residual-m", type=float, default=0.20)
    parser.add_argument("--min-height-above-camera-m", type=float, default=0.50)
    parser.add_argument("--max-height-above-camera-m", type=float, default=8.0)
    args = parser.parse_args()
    trajectory_payload = json.loads(args.trajectory.read_text(encoding="utf-8"))
    tracking_frame = str(trajectory_payload.get("tracking_frame", ""))
    pose_frame = tracking_frame if args.trajectory_pose_frame == "auto" else args.trajectory_pose_frame
    if pose_frame not in ("lidar", "rotation_axis"):
        raise SystemExit("Cannot infer trajectory pose frame; pass --trajectory-pose-frame lidar|rotation_axis")
    lidar_offset, lidar_meta = read_lidar_offset(args.lidar_config)
    camera_axis, camera_meta = geom.load_camera_position(args.camera_position, args.camera_y_sign)
    camera_from_pose = camera_axis.copy()
    if pose_frame == "lidar":
        camera_from_pose[:2] -= lidar_offset[:2]
    raw, _, _, _, _, frame_source = raw_observations(args.observations, args.camera_json)
    tt, trajectory = geom.load_trajectory(args.trajectory)
    projected = project_rays(raw, tt, trajectory, camera_from_pose, args.time_offset, (args.camera_roll, args.camera_pitch, args.camera_yaw))
    by_code: dict[str, list[dict[str, Any]]] = {}
    for row in projected:
        by_code.setdefault(str(row["marker_id"]), []).append(row)
    fits: list[dict[str, Any]] = []
    rejected: dict[str, str] = {}
    for code, rows in sorted(by_code.items(), key=lambda x: int(x[0])):
        if len(rows) < args.min_observations:
            rejected[code] = "below_min_observations"; continue
        fit = triangulate_code(rows, args.triangulation_residual_m, args.min_observations)
        if fit is None:
            rejected[code] = "no_robust_3d_hypothesis"; continue
        fits.append(fit)
    height = robust_height(fits, float(camera_from_pose[2]), args.min_height_above_camera_m, args.max_height_above_camera_m)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {
        "schema": "fliftbot.ceiling_marker_localization.v6",
        "method_version": "v6",
        "runtime_gt_used": False,
        "runtime_registration_used": False,
        "status": "accepted" if height is not None else "abstained",
        "inputs": {"observations": str(args.observations.resolve()), "trajectory": str(args.trajectory.resolve()), "camera_json": str(args.camera_json.resolve()), "camera_position": str(args.camera_position.resolve()), "lidar_config": str(args.lidar_config.resolve()), "frame_timestamps": frame_source},
        "frame_contract": {"trajectory_tracking_frame": tracking_frame, "trajectory_pose_frame": pose_frame, "camera_axis_offset_m": camera_axis.tolist(), "lidar_axis_offset_m": lidar_offset.tolist(), "camera_offset_from_trajectory_m": camera_from_pose.tolist(), "camera_y_sign": args.camera_y_sign},
        "geometry": {"optical_to_map_fixed_angles_rad": {"roll": args.camera_roll, "pitch": args.camera_pitch, "yaw": args.camera_yaw}, "time_offset_s": args.time_offset, "height_estimation": "robust_multi_ray_triangulation"},
        "height": height,
        "counts": {"raw_observations": len(raw), "projected_observations": len(projected), "codes": len(by_code), "robust_fits": len(fits), "rejected_codes": len(rejected)},
        "rejected_codes": rejected,
        "marker_fits": fits,
    }
    if height is not None:
        z = float(height["ceiling_z_m"])
        # Convert the same observations to V5 plane intersections and retain
        # V5 hypothesis selection/ambiguity handling for the output points.
        projected_plane, counters = geom.project_observations(raw, tt, trajectory, camera_from_pose, (args.camera_roll, args.camera_pitch, args.camera_yaw), args.time_offset, z)
        markers, per_id, filtered = v5.select_hypotheses(projected_plane, args.min_observations, 0.25, 0.12, 0.50)
        markers = v5.merge_coincident_markers(markers, 0.30)
        serial = []
        for code in sorted(markers, key=lambda value: int(value) if value.isdigit() else value):
            marker = markers[code]
            serial.append({"marker_id": marker["marker_id"], "candidate_marker_ids": marker.get("candidate_marker_ids", [marker["marker_id"]]), "point_xyz_m": marker["point_xyz_m"], "observations": marker["observations"], "inliers": marker["inliers"], "spread_m": marker["spread_m"], "ray_residual_m": marker["ray_residual_m"], "track_ids": marker["track_ids"], "confidence": float(math.exp(-marker["spread_m"] / 0.05) * math.exp(-marker["ray_residual_m"] / 0.05))})
        report["markers"] = serial
        report["v5_selection"] = {"counters": counters, "filtered_observations": len(filtered), "per_id": list(per_id.values())}
    out = args.out_dir / "marker_points_v6.json"
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "ceiling_z_m": height["ceiling_z_m"] if height else None, "marker_fits": len(fits), "markers": len(report.get("markers", [])), "out": str(out.resolve())}, ensure_ascii=False))


if __name__ == "__main__":
    main()
