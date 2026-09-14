#!/usr/bin/env python3
"""Experimental V7 stored-ray ceiling marker localizer.

V7 reproduces the colleague notebook's useful primitive (the already stored
camera ray and exact ray/plane intersection), while keeping the production
frame contract explicit.  It reads camera_position.txt, lidar.config and a
timed trajectory, estimates the ceiling from multi-view line intersections,
aggregates observations robustly, and collapses coincident decoder identities.

No room GT, labels2coords, fixed room dimensions or GT registration is read at
runtime.  The candidate is intentionally separate from V6 until independent
regression data accepts its frame/timing contract.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import importlib.util
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


HERE = Path(__file__).resolve().parent
V5 = importlib.util.spec_from_file_location("v7_v5", HERE / "ceiling_marker_localization_v5.py")
if V5 is None or V5.loader is None:
    raise ImportError("cannot load V5 geometry")
v5 = importlib.util.module_from_spec(V5)
V5.loader.exec_module(v5)
geom = v5.v2


def read_lidar_offset(path: Path) -> np.ndarray:
    payload = json.loads(path.read_text(encoding="utf-8"))
    pos = payload.get("position_m", payload.get("position", {}))
    return np.asarray([float(pos[k]) for k in ("x", "y", "z")], dtype=float)


def parse_encoder_time(raw: Any, timezone_offset_hours: float) -> float:
    """Parse enc_t only; no wall-clock/nav fields are accepted."""
    if isinstance(raw, (int, float)):
        return float(raw)
    text = str(raw).strip().replace("Z", "+00:00")
    if "T" in text or "+" in text:
        return dt.datetime.fromisoformat(text).timestamp()
    value = dt.datetime.strptime(text, "%Y_%m_%d_%H_%M_%S_%f")
    return value.replace(tzinfo=dt.timezone(dt.timedelta(hours=timezone_offset_hours))).timestamp()


def load_encoder_log(path: Path, wheelbase_m: float, speed_sign: float, steering_sign: float, initial_yaw_deg: float, timezone_offset_hours: float) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Integrate only enc_t, enc_v and enc_dgr with the V6 bicycle contract."""
    samples = []
    if path.suffix.lower() == ".jsonl":
        source_rows = (json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip())
    else:
        source_rows = csv.DictReader(path.open(newline="", encoding="utf-8"))
    for row in source_rows:
        try:
            # Deliberately do not read x/y/dgr/nav_* or enc_unix.
            stamp = parse_encoder_time(row["enc_t"], timezone_offset_hours)
            speed = float(row["enc_v"])
            steering = float(row["enc_dgr"])
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            continue
        if all(math.isfinite(v) for v in (stamp, speed, steering)):
            samples.append((stamp, speed, steering))
    if len(samples) < 2:
        raise ValueError(f"not enough encoder samples in {path}")
    samples.sort(key=lambda x: x[0])
    poses = [[0.0, 0.0, math.radians(initial_yaw_deg)]]
    times = [samples[0][0]]
    for (t0, v0, a0), (t1, v1, a1) in zip(samples, samples[1:]):
        delta_t = min(max(t1 - t0, 0.0), 0.5)
        velocity = speed_sign * 0.5 * (v0 + v1)
        steering = math.radians(0.5 * (a0 + a1)) * steering_sign
        yaw = poses[-1][2]
        yaw_rate = velocity * math.sin(steering) / wheelbase_m
        delta_yaw = yaw_rate * delta_t
        forward = velocity * math.cos(steering)
        if abs(yaw_rate) < 1.0e-10:
            dx, dy = forward * delta_t, 0.0
        else:
            radius = forward / yaw_rate
            dx, dy = radius * math.sin(delta_yaw), radius * (1.0 - math.cos(delta_yaw))
        c, s = math.cos(yaw), math.sin(yaw)
        poses.append([poses[-1][0] + c * dx - s * dy, poses[-1][1] + s * dx + c * dy, yaw + delta_yaw])
        times.append(t1)
    return np.asarray(times), np.asarray(poses), {
        "source_type": "encoder_bicycle_integration",
        "source_fields": ["enc_t", "enc_v", "enc_dgr"],
        "forbidden_fields": ["x", "y", "dgr", "nav_t", "nav_x", "nav_y", "nav_dgr", "enc_unix"],
        "wheelbase_m": wheelbase_m, "speed_sign": speed_sign,
        "steering_sign": steering_sign, "initial_yaw_deg": initial_yaw_deg,
        "encoder_timezone_offset_hours": timezone_offset_hours,
        "sample_count": len(samples),
    }


def project_rows(
    rows: list[dict[str, Any]], times: np.ndarray, poses: np.ndarray,
    camera_offset: np.ndarray, time_offset: float,
) -> list[dict[str, Any]]:
    if not rows:
        return []
    query = np.asarray([r["timestamp_s"] for r in rows], dtype=float) + time_offset
    interpolated, inside = geom.interpolate_poses(query, times, poses)
    result = []
    for row, pose, ok, query_time in zip(rows, interpolated, inside, query):
        if not ok:
            continue
        c, s = math.cos(float(pose[2])), math.sin(float(pose[2]))
        R = np.asarray([[c, -s], [s, c]], dtype=float)
        origin_xy = pose[:2] + R @ camera_offset[:2]
        direction = np.asarray([*(R @ np.asarray(row["ray_optical_unit"], dtype=float)[:2]), row["ray_optical_unit"][2]], dtype=float)
        norm = float(np.linalg.norm(direction))
        if norm <= 1.0e-12 or not np.isfinite(norm):
            continue
        direction /= norm
        if direction[2] <= 1.0e-6:
            continue
        item = dict(row)
        item.update({
            "query_timestamp_s": float(query_time),
            "camera_origin_map": [float(origin_xy[0]), float(origin_xy[1]), float(camera_offset[2])],
            "ray_map_unit": direction.tolist(),
            "robot_pose_map": pose.tolist(),
        })
        result.append(item)
    return result


def closest_lines(o1: np.ndarray, d1: np.ndarray, o2: np.ndarray, d2: np.ndarray) -> np.ndarray | None:
    dot = float(d1 @ d2)
    den = 1.0 - dot * dot
    if den < 1.0e-5:
        return None
    w = o2 - o1
    a = float((w @ d1 - dot * (w @ d2)) / den)
    b = float((dot * (w @ d1) - w @ d2) / den)
    return 0.5 * (o1 + a * d1 + o2 + b * d2)


def triangulate(rows: list[dict[str, Any]], residual_gate: float, max_samples: int) -> dict[str, Any] | None:
    if len(rows) < 4:
        return None
    if len(rows) > max_samples:
        indices = np.linspace(0, len(rows) - 1, max_samples).astype(int)
        rows = [rows[int(i)] for i in indices]
    origins = np.asarray([r["camera_origin_map"] for r in rows], dtype=float)
    directions = np.asarray([r["ray_map_unit"] for r in rows], dtype=float)
    best: tuple[int, float, np.ndarray, np.ndarray] | None = None
    for i in range(len(rows) - 1):
        for j in range(i + 1, len(rows)):
            point = closest_lines(origins[i], directions[i], origins[j], directions[j])
            if point is None:
                continue
            residuals = np.linalg.norm(np.cross(point[None, :] - origins, directions), axis=1)
            depths = np.einsum("ij,ij->i", point[None, :] - origins, directions)
            mask = (residuals <= residual_gate) & (depths > 0.0)
            score = (int(mask.sum()), -float(np.median(residuals[mask])) if mask.any() else -1.0)
            if best is None or score > (best[0], best[1]):
                best = (score[0], score[1], point, mask)
    if best is None or best[0] < 4:
        return None
    mask = best[3]
    directions_i, origins_i = directions[mask], origins[mask]
    projectors = np.eye(3)[None, :, :] - directions_i[:, :, None] * directions_i[:, None, :]
    matrix = projectors.sum(axis=0)
    if np.linalg.cond(matrix) > 1.0e10:
        return None
    point = np.linalg.solve(matrix, np.einsum("nij,nj->i", projectors, origins_i))
    residuals = np.linalg.norm(np.cross(point[None, :] - origins, directions), axis=1)
    depths = np.einsum("ij,ij->i", point[None, :] - origins, directions)
    final = (residuals <= residual_gate) & (depths > 0.0)
    if int(final.sum()) < 4:
        return None
    return {
        "point_xyz_m": point.tolist(),
        "inliers": int(final.sum()),
        "raw_observations": len(rows),
        "ray_residual_m": float(np.median(residuals[final])),
        "spread_m": float(np.median(np.linalg.norm((origins[final] + ((point - origins[final]) * directions[final]).sum(axis=1)[:, None] * directions[final])[:, :2] - point[:2], axis=1))) if final.any() else 0.0,
    }


def estimate_height(fits: list[dict[str, Any]], camera_z: float, lo: float, hi: float) -> dict[str, Any] | None:
    valid = [f for f in fits if camera_z + lo <= f["point_xyz_m"][2] <= camera_z + hi]
    if len(valid) < 3:
        return None
    z = np.asarray([f["point_xyz_m"][2] for f in valid])
    center = float(np.median(z)); mad = float(np.median(np.abs(z - center)))
    gate = max(0.12, 3.0 * 1.4826 * mad)
    keep = np.abs(z - center) <= gate
    if int(keep.sum()) < 3:
        return None
    estimate = float(np.median(z[keep]))
    sigma = float(1.4826 * np.median(np.abs(z[keep] - estimate)))
    confidence = min(1.0, float(keep.sum()) / 12.0) * math.exp(-sigma / 0.10)
    if confidence + 1.0e-9 < 0.25:
        return None
    return {"ceiling_z_m": estimate, "candidate_fits": len(valid), "inlier_fits": int(keep.sum()), "rejected_fits": int((~keep).sum()), "robust_sigma_m": sigma, "confidence": float(confidence)}


def aggregate(projected: list[dict[str, Any]], ceiling_z: float, min_observations: int, merge_radius: float) -> list[dict[str, Any]]:
    by_code: dict[int, list[dict[str, Any]]] = {}
    for row in projected:
        code = int(row["marker_id"])
        if row["ray_map_unit"][2] <= 0:
            continue
        origin = np.asarray(row["camera_origin_map"], dtype=float)
        direction = np.asarray(row["ray_map_unit"], dtype=float)
        depth = (ceiling_z - origin[2]) / direction[2]
        if depth > 0 and np.isfinite(depth):
            item = dict(row); item["point_xyz_m"] = (origin + depth * direction).tolist()
            by_code.setdefault(code, []).append(item)
    candidates = []
    for code, rows in by_code.items():
        if len(rows) < min_observations:
            continue
        points = np.asarray([r["point_xyz_m"][:2] for r in rows])
        center = np.median(points, axis=0)
        distances = np.linalg.norm(points - center, axis=1)
        keep = distances <= max(0.12, float(np.percentile(distances, 85.0)))
        if int(keep.sum()) < min_observations:
            continue
        center = np.median(points[keep], axis=0)
        candidates.append({
            "marker_id": code, "point_xyz_m": [float(center[0]), float(center[1]), float(ceiling_z)],
            "observations": len(rows), "inliers": int(keep.sum()),
            "spread_m": float(np.median(np.linalg.norm(points[keep] - center, axis=1))),
            "ray_residual_m": float(np.median([np.linalg.norm(np.cross(np.asarray(rows[i]["point_xyz_m"]) - np.asarray(rows[i]["camera_origin_map"]), np.asarray(rows[i]["ray_map_unit"]))) for i in np.flatnonzero(keep)])),
            "track_ids": sorted({str(rows[i]["track_id"]) for i in np.flatnonzero(keep)}),
        })
    # Connected components in physical XY, independent of decoder ID.
    pending = set(range(len(candidates))); merged = []
    while pending:
        seed = min(pending); pending.remove(seed); stack = [seed]; group = [seed]
        while stack:
            i = stack.pop(); pi = np.asarray(candidates[i]["point_xyz_m"][:2])
            for j in list(pending):
                if np.linalg.norm(pi - np.asarray(candidates[j]["point_xyz_m"][:2])) <= merge_radius:
                    pending.remove(j); stack.append(j); group.append(j)
        best = max(group, key=lambda i: (candidates[i]["inliers"], candidates[i]["observations"], -candidates[i]["spread_m"]))
        item = dict(candidates[best]); item["candidate_marker_ids"] = sorted(int(candidates[i]["marker_id"]) for i in group); item["identity_ambiguous"] = len(group) > 1; merged.append(item)
    return sorted(merged, key=lambda x: int(x["marker_id"]))


def main() -> None:
    root = Path(__file__).resolve().parents[3]
    rec = root / "ros-data/recordings/real/20260908/160617_dephan_m360_room"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--observations", type=Path, default=rec / "observations.json")
    parser.add_argument("--encoder-log", type=Path, default=rec / "encoder.jsonl", help="encoder log containing enc_t, enc_v, enc_dgr")
    parser.add_argument("--camera-position", type=Path, default=rec / "camera_position.txt")
    parser.add_argument("--lidar-config", type=Path, default=rec / "lidar.config")
    parser.add_argument("--trajectory-pose-frame", choices=("rotation_axis",), default="rotation_axis")
    parser.add_argument("--camera-y-sign", type=int, choices=(-1, 1), default=1)
    parser.add_argument("--wheelbase-m", type=float, default=1.410)
    parser.add_argument("--encoder-speed-sign", type=float, default=-1.0)
    parser.add_argument("--encoder-steering-sign", type=float, default=1.0)
    parser.add_argument("--initial-yaw-deg", type=float, default=0.0)
    parser.add_argument("--encoder-timezone-offset-hours", type=float, default=0.0)
    parser.add_argument("--time-offset", type=float, default=0.0)
    parser.add_argument("--min-observations", type=int, default=12)
    parser.add_argument("--line-residual-m", type=float, default=0.20)
    parser.add_argument("--max-line-samples", type=int, default=80)
    parser.add_argument("--min-height-above-camera-m", type=float, default=0.50)
    parser.add_argument("--max-height-above-camera-m", type=float, default=8.0)
    parser.add_argument("--merge-radius-m", type=float, default=0.20)
    parser.add_argument("--out-dir", type=Path, default=root / "ros-data/results/real/20260914/ceiling_marker_localization_v7_stored_ray_candidate")
    args = parser.parse_args()
    payload = json.loads(args.observations.read_text(encoding="utf-8"))
    frame_times, frame_source = geom.load_frame_times(payload, args.observations)
    pose_times, poses, trajectory_meta = load_encoder_log(args.encoder_log, args.wheelbase_m, args.encoder_speed_sign, args.encoder_steering_sign, args.initial_yaw_deg, args.encoder_timezone_offset_hours)
    tracking_frame = "rotation_axis"
    pose_frame = args.trajectory_pose_frame
    lidar = read_lidar_offset(args.lidar_config)
    camera_axis, camera_meta = geom.load_camera_position(args.camera_position, args.camera_y_sign)
    camera_offset = camera_axis.copy()
    rows = []
    for fi, frame in enumerate(payload.get("data", [])):
        if fi >= len(frame_times) or not isinstance(frame, dict):
            continue
        for track_id, item in frame.items():
            if not isinstance(item, dict):
                continue
            try:
                code = int(item["code"]); ray = np.asarray(item["ray"], dtype=float)
            except (KeyError, TypeError, ValueError):
                continue
            if code < 0 or ray.shape != (3,) or not np.isfinite(ray).all():
                continue
            ray /= max(float(np.linalg.norm(ray)), 1.0e-12)
            rows.append({"frame_index": fi, "timestamp_s": float(frame_times[fi]), "track_id": str(track_id), "marker_id": code, "ray_optical_unit": ray.tolist()})
    projected = project_rows(rows, pose_times, poses, camera_offset, args.time_offset)
    by_code: dict[int, list[dict[str, Any]]] = {}
    for row in projected: by_code.setdefault(int(row["marker_id"]), []).append(row)
    fits = []
    for code, values in by_code.items():
        if len(values) >= args.min_observations:
            fit = triangulate(values, args.line_residual_m, args.max_line_samples)
            if fit is not None: fits.append(fit)
    height = estimate_height(fits, float(camera_offset[2]), args.min_height_above_camera_m, args.max_height_above_camera_m)
    report: dict[str, Any] = {"schema": "fliftbot.ceiling_marker_localization.v7.experimental", "method_version": "v7-stored-ray-encoder-bicycle", "runtime_gt_used": False, "runtime_registration_used": False, "status": "accepted" if height else "abstained", "inputs": {"observations": str(args.observations.resolve()), "encoder_log": str(args.encoder_log.resolve()), "camera_position": str(args.camera_position.resolve()), "lidar_config": str(args.lidar_config.resolve()), "frame_timestamps": frame_source}, "frame_contract": {"trajectory_tracking_frame": tracking_frame, "trajectory_pose_frame": pose_frame, "camera_axis_offset_m": camera_axis.tolist(), "lidar_axis_offset_m": lidar.tolist(), "camera_offset_from_trajectory_m": camera_offset.tolist(), "camera_y_sign": args.camera_y_sign, "trajectory_metadata": trajectory_meta, "camera_metadata": camera_meta}, "geometry": {"ray_source": "observations.json:item.ray", "ray_plane_formula": "P=O+((H-Oz)/dz)d", "time_offset_s": args.time_offset, "ceiling_estimation": "robust multi-view line triangulation", "aggregation": "median per code + physical coincident-cluster collapse", "merge_radius_m": args.merge_radius_m}, "height": height, "counts": {"raw_stored_ray_observations": len(rows), "projected_observations": len(projected), "codes": len(by_code), "line_fits": len(fits)}}
    if height:
        markers = aggregate(projected, float(height["ceiling_z_m"]), args.min_observations, args.merge_radius_m)
        for marker in markers:
            marker["confidence"] = float(math.exp(-marker["spread_m"] / 0.05) * math.exp(-marker["ray_residual_m"] / 0.05))
        report["markers"] = markers
        report["counts"]["accepted_physical_clusters"] = len(markers)
    else:
        report["markers"] = []
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out = args.out_dir / "marker_points_v7.json"
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "height": height, "markers": len(report["markers"]), "out": str(out.resolve())}, ensure_ascii=False))


if __name__ == "__main__":
    main()
