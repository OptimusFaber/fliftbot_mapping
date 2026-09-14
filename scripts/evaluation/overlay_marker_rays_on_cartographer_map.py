#!/usr/bin/env python3
"""Project ceiling-marker observations into a Cartographer map.

Production inputs are observations.json, camera.json, camera_position.txt,
frame_timestamps.csv, trajectory.json and map.yaml.  No room GT or
scene-specific registration is used.  Legacy observations use tlbr=[y1,x1,y2,x2];
CameraModel receives pixels as [u=x,v=y].  Rays are always recomputed through
CameraModel from pixels; the legacy stored ray is audit-only.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import shutil
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

UNIX_EPOCH_TICKS = 621355968000000000
PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CAMERA_MODEL = PROJECT_ROOT / "fliftbot_navigation-navigation_INS@7014eb114e3" / "camera_model.py"


def read_yaml(path: Path) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.split("#", 1)[0].strip()
        if not line or ":" not in line:
            continue
        key, value = line.split(":", 1)
        value = value.strip()
        if value.startswith("["):
            result[key.strip()] = json.loads(value)
        else:
            try:
                result[key.strip()] = float(value)
            except ValueError:
                result[key.strip()] = value.strip("'\"")
    return result


def rotation(yaw: float) -> np.ndarray:
    c, s = math.cos(float(yaw)), math.sin(float(yaw))
    return np.asarray([[c, -s], [s, c]], dtype=float)


def quaternion_yaw(q: dict[str, Any]) -> float:
    x, y, z, w = (float(q.get(k, 0.0)) for k in ("qx", "qy", "qz", "qw"))
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


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
        else:
            continue
        value = [float(item.get("x", math.nan)), float(item.get("y", math.nan)), quaternion_yaw(item)]
        if math.isfinite(stamp) and all(math.isfinite(v) for v in value):
            times.append(stamp); values.append(value)
    if len(times) < 2:
        raise RuntimeError(f"Need at least two timed Cartographer poses in {path}")
    order = np.argsort(np.asarray(times))
    times_a = np.asarray(times, dtype=float)[order]
    values_a = np.asarray(values, dtype=float)[order]
    keep = np.r_[True, np.diff(times_a) > 1.0e-9]
    return times_a[keep], values_a[keep]


def load_frame_times(payload: dict[str, Any], observations_path: Path, csv_path: Path | None = None) -> np.ndarray:
    frames = payload.get("data", [])
    embedded = payload.get("frame_timestamps")
    if embedded and len(embedded) >= len(frames):
        values = np.asarray(embedded[:len(frames)], dtype=float)
        if values.shape == (len(frames),) and np.isfinite(values).all():
            return values
    candidates: list[Path] = []
    if csv_path is not None:
        candidates.append(csv_path)
    if payload.get("video_path"):
        candidates.append(Path(payload["video_path"]).parent / "frame_timestamps.csv")
    candidates.append(observations_path.parent / "frame_timestamps.csv")
    for path in dict.fromkeys(candidates):
        if not path.is_file():
            continue
        with path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        by_index: dict[int, float] = {}
        for ordinal, row in enumerate(rows):
            try:
                index = int(row.get("frame_index", ordinal))
                stamp = float(row["time_take_img_sec"])
            except (KeyError, TypeError, ValueError):
                continue
            if math.isfinite(stamp):
                by_index[index] = stamp
        if len(by_index) >= len(frames) and all(i in by_index for i in range(len(frames))):
            return np.asarray([by_index[i] for i in range(len(frames))], dtype=float)
    raise RuntimeError("No complete frame timestamps in observations.json or frame_timestamps.csv")


def interpolate_poses(query: np.ndarray, times: np.ndarray, route: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    query = np.asarray(query, dtype=float)
    inside = (query >= times[0]) & (query <= times[-1])
    yaw = np.interp(query, times, np.unwrap(route[:, 2]))
    pose = np.column_stack((np.interp(query, times, route[:, 0]), np.interp(query, times, route[:, 1]), yaw))
    return pose, inside


def load_camera_model(camera_json: Path, camera_position: Path, model_path: Path | None = None):
    """Instantiate the exact CameraModel contract used by get_video_data.py."""
    path = model_path or DEFAULT_CAMERA_MODEL
    if not path.is_file():
        raise FileNotFoundError(f"CameraModel source not found: {path}")
    spec = importlib.util.spec_from_file_location("fliftbot_camera_model", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load CameraModel from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    lines = [line.strip() for line in camera_position.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(lines) < 2:
        raise ValueError(f"camera_position.txt must contain position and orientation lines: {camera_position}")
    position = np.asarray([float(v) for v in lines[0].split()], dtype=float)
    orientation = np.asarray([float(v) for v in lines[1].split()], dtype=float)
    if position.shape != (3,) or orientation.shape != (2,) or not np.isfinite(position).all() or not np.isfinite(orientation).all():
        raise ValueError(f"Invalid camera_position.txt: {camera_position}")
    offsets_m = position / 100.0
    tilts_rad = np.asarray([-orientation[0], orientation[1], math.pi / 2.0], dtype=float)
    model = module.CameraModel(offsets=tuple(offsets_m), tilts_rad=tuple(tilts_rad), camera_json_path=str(camera_json))
    return model, {
        "source": str(path.resolve()), "camera_json": str(camera_json.resolve()), "camera_position": str(camera_position.resolve()),
        "offsets_m_robot_body": offsets_m.tolist(), "tilts_rad_camera_model": tilts_rad.tolist(),
        "pixel_input_contract": "[u=x, v=y]", "ray_output_contract": "CameraModel.pixels_to_rays world/body frame at robot pose",
    }


def detection_center(item: dict[str, Any], width: int, height: int) -> tuple[np.ndarray | None, str]:
    """Return pixels as [u=x,v=y], handling legacy tlbr=[y1,x1,y2,x2]."""
    box = np.asarray(item.get("tlbr", []), dtype=float)
    from_box = None
    if box.shape == (4,) and np.isfinite(box).all():
        y1, x1, y2, x2 = box
        candidate = np.asarray([(x1 + x2) / 2.0, (y1 + y2) / 2.0], dtype=float)
        if 0.0 <= candidate[0] < width and 0.0 <= candidate[1] < height:
            from_box = candidate
    center = np.asarray(item.get("center", []), dtype=float)
    if center.shape == (2,) and np.isfinite(center).all() and 0.0 <= center[0] < width and 0.0 <= center[1] < height:
        if from_box is not None and np.linalg.norm(center - from_box) <= 4.0:
            return center, "center_xy_verified_against_legacy_tlbr"
        if from_box is None:
            return center, "center_xy"
    if from_box is not None:
        return from_box, "legacy_tlbr_yxyx"
    return None, "invalid"


def load_intrinsic_size(camera_json: Path) -> tuple[int, int]:
    entry = next(iter(json.loads(camera_json.read_text(encoding="utf-8")).values()))
    return int(entry["width"]), int(entry["height"])


def camera_ray_at_pose(camera_model: Any, pose: np.ndarray, pixel: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Use CameraModel directly, avoiding manual optical/body axis guesses."""
    origin, direction = camera_model.pixels_to_rays(np.asarray(pose[:2]), np.asarray([pose[2]]), np.asarray(pixel).reshape(1, 2))
    ray = np.asarray(direction[0], dtype=float); ray /= np.linalg.norm(ray)
    return np.asarray(origin, dtype=float), ray


def map_transform(map_meta: dict[str, Any], image_shape: tuple[int, int]) -> dict[str, Any]:
    origin = np.asarray(map_meta.get("origin", [0.0, 0.0, 0.0]), dtype=float)
    resolution = float(map_meta.get("resolution", 0.0))
    if origin.shape != (3,) or not np.isfinite(origin).all() or resolution <= 0.0:
        raise ValueError("map.yaml must contain finite origin [x,y,yaw] and positive resolution")
    height, width = image_shape
    return {"origin_xy_m": origin[:2], "origin_yaw_rad": float(origin[2]), "resolution_m": resolution, "width_px": int(width), "height_px": int(height)}


def map_to_grid(point_xy: Iterable[float], transform: dict[str, Any]) -> np.ndarray:
    return rotation(-transform["origin_yaw_rad"]) @ (np.asarray(point_xy, dtype=float) - transform["origin_xy_m"])


def map_to_grid_xyz(point_xyz: Iterable[float], transform: dict[str, Any]) -> np.ndarray:
    p = np.asarray(point_xyz, dtype=float)
    return np.asarray([*map_to_grid(p[:2], transform), p[2]], dtype=float)


def transform_trajectory(route: np.ndarray, transform: dict[str, Any]) -> np.ndarray:
    result = np.asarray(route, dtype=float).copy()
    if len(result):
        result[:, :2] = np.asarray([map_to_grid(p, transform) for p in result[:, :2]])
        result[:, 2] -= transform["origin_yaw_rad"]
    return result


def unit_vector(value: Any) -> np.ndarray | None:
    vector = np.asarray(value, dtype=float)
    if vector.shape != (3,) or not np.isfinite(vector).all():
        return None
    norm = float(np.linalg.norm(vector))
    return vector / norm if norm > 1.0e-9 else None


def collect_observations(payload: dict[str, Any], frame_times: np.ndarray, trajectory_times: np.ndarray, trajectory: np.ndarray, camera_model: Any, image_size: tuple[int, int], time_offset: float) -> tuple[dict[str, list[dict[str, Any]]], dict[str, int]]:
    width, height = image_size
    by_code: dict[str, list[dict[str, Any]]] = defaultdict(list)
    counters: dict[str, int] = defaultdict(int)
    queries = np.asarray(frame_times, dtype=float) + float(time_offset)
    poses, inside = interpolate_poses(queries, trajectory_times, trajectory)
    for frame_index, frame in enumerate(payload.get("data", [])):
        if frame_index >= len(poses) or not inside[frame_index] or not isinstance(frame, dict):
            counters["frames_outside_trajectory"] += 1; continue
        pose = poses[frame_index]
        for track_id, item in frame.items():
            if not isinstance(item, dict): continue
            try: code = int(item.get("code"))
            except (TypeError, ValueError): code = -1
            if code < 0:
                counters["invalid_or_unknown_codes"] += 1; continue
            pixel, pixel_source = detection_center(item, width, height)
            if pixel is None:
                counters["invalid_pixels"] += 1; continue
            try: origin, ray = camera_ray_at_pose(camera_model, pose, pixel)
            except (ValueError, TypeError, np.linalg.LinAlgError):
                counters["invalid_camera_rays"] += 1; continue
            if not np.isfinite(origin).all() or not np.isfinite(ray).all() or ray[2] <= 1.0e-5:
                counters["downward_or_invalid_rays"] += 1; continue
            stored = unit_vector(item.get("ray"))
            by_code[str(code)].append({
                "marker_id_raw": code, "track_id": str(track_id), "frame_index": int(frame_index),
                "timestamp_s": float(frame_times[frame_index]), "query_timestamp_s": float(queries[frame_index]),
                "pixel_uv": [float(v) for v in pixel], "pixel_source": pixel_source,
                "robot_pose_map": [float(v) for v in pose], "camera_xyz_m": [float(v) for v in origin],
                "ray_map_unit": [float(v) for v in ray], "stored_ray_audit_unit": stored.tolist() if stored is not None else None,
                "stored_ray_used_for_geometry": False,
            })
            counters["accepted_observations"] += 1
    counters["frames_total"] = len(payload.get("data", []))
    counters["frames_inside_trajectory"] = counters["frames_total"] - counters["frames_outside_trajectory"]
    return dict(by_code), dict(counters)


def _plane_points(rows: list[dict[str, Any]], ceiling_z: float) -> tuple[np.ndarray, np.ndarray]:
    points: list[np.ndarray] = []; valid: list[int] = []
    for index, row in enumerate(rows):
        origin = np.asarray(row["camera_xyz_m"], dtype=float); direction = np.asarray(row["ray_map_unit"], dtype=float)
        scale = (ceiling_z - origin[2]) / direction[2]
        if math.isfinite(scale) and scale > 0.0:
            points.append(origin + scale * direction); valid.append(index)
    return np.asarray(points, dtype=float), np.asarray(valid, dtype=int)


def _largest_spatial_cluster(points: np.ndarray, radius_m: float) -> np.ndarray:
    if len(points) == 0: return np.zeros(0, dtype=bool)
    adjacency = np.linalg.norm(points[:, None, :2] - points[None, :, :2], axis=2) <= radius_m
    remaining = set(range(len(points))); components: list[list[int]] = []
    while remaining:
        seed = next(iter(remaining)); stack = [seed]; component = []; remaining.remove(seed)
        while stack:
            current = stack.pop(); component.append(current)
            neighbours = [j for j in remaining if adjacency[current, j]]
            for j in neighbours: remaining.remove(j); stack.append(j)
        components.append(component)
    best = max(components, key=lambda c: (len(c), -float(np.median(np.linalg.norm(points[c, :2] - np.median(points[c, :2], axis=0), axis=1)))))
    mask = np.zeros(len(points), dtype=bool); mask[best] = True
    return mask


def project_to_plane(rows: list[dict[str, Any]], ceiling_z: float, cluster_radius_m: float) -> dict[str, Any] | None:
    values, valid_indices = _plane_points(rows, ceiling_z)
    if len(values) < 2: return None
    keep_cluster = _largest_spatial_cluster(values, cluster_radius_m)
    if int(keep_cluster.sum()) < 2: return None
    cluster_indices = np.flatnonzero(keep_cluster)
    center = np.median(values[cluster_indices, :2], axis=0)
    distances = np.linalg.norm(values[cluster_indices, :2] - center, axis=1)
    keep_final = distances <= max(0.08, float(np.percentile(distances, 85.0)))
    if int(keep_final.sum()) < 2: return None
    inlier_indices = valid_indices[cluster_indices[keep_final]]
    center = np.median(values[cluster_indices[keep_final], :2], axis=0)
    spread = float(np.median(np.linalg.norm(values[cluster_indices[keep_final], :2] - center, axis=1)))
    return {"point_xyz_m": [float(center[0]), float(center[1]), float(ceiling_z)], "inlier_indices": inlier_indices.astype(int), "spread_m": spread, "method": "robust_ray_plane_intersection"}


def closest_point_between_lines(a: np.ndarray, d: np.ndarray, b: np.ndarray, e: np.ndarray) -> np.ndarray | None:
    cross = np.cross(d, e); denom = float(cross @ cross)
    if denom < 1.0e-8: return None
    t = float(((b - a) @ np.cross(e, cross)) / denom); u = float(((b - a) @ np.cross(d, cross)) / denom)
    return (a + t * d + b + u * e) / 2.0


def triangulate_lines(rows: list[dict[str, Any]], residual_gate_m: float) -> dict[str, Any] | None:
    """Robust multi-ray triangulation without a known ceiling."""
    if len(rows) < 4: return None
    origins = np.asarray([r["camera_xyz_m"] for r in rows], dtype=float); directions = np.asarray([r["ray_map_unit"] for r in rows], dtype=float)
    best_point = None; best_mask = None; best_score = (-1, float("inf")); stride = max(1, len(rows) // 120)
    candidates = list(range(0, len(rows), stride))
    for i in candidates:
        for j in candidates:
            if j <= i: continue
            point = closest_point_between_lines(origins[i], directions[i], origins[j], directions[j])
            if point is None or not np.isfinite(point).all(): continue
            residuals = np.linalg.norm(np.cross(point[None, :] - origins, directions), axis=1)
            depths = np.einsum("ij,ij->i", point[None, :] - origins, directions)
            mask = (residuals <= residual_gate_m) & (depths > 0.0)
            score = (int(mask.sum()), float(np.median(residuals[mask])) if mask.any() else float("inf"))
            if score[0] > best_score[0] or (score[0] == best_score[0] and score[1] < best_score[1]):
                best_point, best_mask, best_score = point, mask, score
    if best_point is None or best_mask is None or int(best_mask.sum()) < 4: return None
    d, o = directions[best_mask], origins[best_mask]; projectors = np.eye(3)[None, :, :] - d[:, :, None] * d[:, None, :]
    matrix = projectors.sum(axis=0)
    if np.linalg.cond(matrix) > 1.0e10: return None
    point = np.linalg.solve(matrix, np.einsum("nij,nj->i", projectors, o))
    residuals = np.linalg.norm(np.cross(point[None, :] - origins, directions), axis=1); depths = np.einsum("ij,ij->i", point[None, :] - origins, directions)
    mask = (residuals <= residual_gate_m) & (depths > 0.0)
    if int(mask.sum()) < 4: return None
    return {"point_xyz_m": [float(v) for v in point], "inlier_indices": np.flatnonzero(mask).astype(int), "spread_m": float(np.median(residuals[mask])), "method": "robust_multi_ray_triangulation"}


def cluster_places(rows: list[dict[str, Any]], radius_m: float) -> list[dict[str, Any]]:
    clusters: list[dict[str, Any]] = []
    for row_index, row in enumerate(rows):
        point = np.asarray(row["camera_xyz_m"][:2], dtype=float); distances = [float(np.linalg.norm(point - c["center_xy_m"])) for c in clusters]
        if not distances or min(distances) > radius_m: clusters.append({"center_xy_m": point.copy(), "row_indices": [row_index]})
        else:
            index = int(np.argmin(distances)); clusters[index]["row_indices"].append(row_index); indices = clusters[index]["row_indices"]
            clusters[index]["center_xy_m"] = np.mean([np.asarray(rows[i]["camera_xyz_m"][:2]) for i in indices], axis=0)
    result = []
    for index, cluster in enumerate(clusters):
        indices = cluster["row_indices"]; times = [float(rows[i]["timestamp_s"]) for i in indices]
        result.append({"place_index": index, "camera_xy_m": [float(v) for v in cluster["center_xy_m"]], "observations": len(indices), "first_timestamp_s": min(times), "last_timestamp_s": max(times), "first_frame_index": int(rows[indices[0]]["frame_index"])})
    return result


def marker_color(marker_id: str) -> tuple[float, float, float, float]:
    digest = hashlib.sha256(marker_id.encode("utf-8")).digest(); value = int.from_bytes(digest[:2], "big") / 65535.0
    return (*plt.get_cmap("turbo")(0.08 + 0.84 * value)[:3], 1.0)


def render_map(output: Path, map_image: np.ndarray, transform: dict[str, Any], markers: dict[str, dict[str, Any]], trajectory: np.ndarray, rows: list[dict[str, Any]], title: str, focus: tuple[np.ndarray, np.ndarray] | None = None, max_rays: int = 7000) -> None:
    height, width = map_image.shape[:2]; resolution = float(transform["resolution_m"]); extent = [0.0, width * resolution, 0.0, height * resolution]
    fig, ax = plt.subplots(figsize=(13, 10)); ax.imshow(np.flipud(map_image), cmap="gray", origin="lower", extent=extent, interpolation="nearest", alpha=0.92)
    if len(trajectory): ax.plot(trajectory[:, 0], trajectory[:, 1], color="#2563eb", lw=0.65, alpha=0.55, label="Cartographer trajectory")
    stride = max(1, len(rows) // max_rays)
    for row in rows[::stride]:
        start = map_to_grid(row["camera_xyz_m"][:2], transform); marker = markers.get(str(row["marker_key"]))
        point = marker.get("occupancy_grid_xyz_m") if marker else None
        direction = rotation(-transform["origin_yaw_rad"]) @ np.asarray(row["ray_map_unit"][:2], dtype=float); direction /= max(float(np.linalg.norm(direction)), 1.0e-9)
        end = np.asarray(point[:2], dtype=float) if point is not None else start + direction * 3.0
        ax.plot([start[0], end[0]], [start[1], end[1]], color=marker_color(str(row["marker_key"])), alpha=0.08, lw=0.45)
    for key, marker in markers.items():
        point = np.asarray(marker["occupancy_grid_xyz_m"][:2], dtype=float); color = marker_color(key)
        ax.scatter([point[0]], [point[1]], s=50, color=[color], edgecolors="white", linewidths=0.5, zorder=5); ax.annotate(str(marker["marker_id"]), point, xytext=(4, 4), textcoords="offset points", fontsize=7, color=color)
    if focus is not None:
        point, views = focus; ax.scatter(views[:, 0], views[:, 1], s=20, color="#16a34a", edgecolors="white", linewidths=0.3, zorder=6, label="distinct view places"); ax.scatter([point[0]], [point[1]], s=180, marker="*", color="#f97316", edgecolors="black", linewidths=0.7, zorder=7, label="reconstructed marker")
        all_xy = np.vstack((views, point[None, :]))
        margin = 2.0
        lo, hi = all_xy.min(axis=0) - margin, all_xy.max(axis=0) + margin
        ax.set_xlim(float(lo[0]), float(hi[0])); ax.set_ylim(float(lo[1]), float(hi[1]))
    ax.set_aspect("equal", adjustable="box"); ax.set_xlabel("Occupancy-grid X (m)"); ax.set_ylabel("Occupancy-grid Y (m)"); ax.set_title(title); ax.grid(alpha=0.2); ax.legend(loc="best")
    output.parent.mkdir(parents=True, exist_ok=True); fig.tight_layout(); fig.savefig(output, dpi=180); plt.close(fig)


def _component_hypotheses(hypotheses: list[dict[str, Any]], merge_radius_m: float) -> list[list[dict[str, Any]]]:
    parent = list(range(len(hypotheses)))
    def find(x: int) -> int:
        while parent[x] != x: parent[x] = parent[parent[x]]; x = parent[x]
        return x
    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb: parent[rb] = ra
    for i, left in enumerate(hypotheses):
        for j in range(i):
            right = hypotheses[j]
            if set(left["track_ids"]).intersection(right["track_ids"]) and np.linalg.norm(np.asarray(left["point_xyz_m"])[:2] - np.asarray(right["point_xyz_m"])[:2]) <= merge_radius_m: union(i, j)
    groups: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for i, value in enumerate(hypotheses): groups[find(i)].append(value)
    return list(groups.values())


def _prepare_output_dirs(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True); multi = out_dir / "multi_seen_points"
    if multi.is_dir():
        for child in multi.iterdir():
            if child.name.startswith("marker_") and child.is_dir(): shutil.rmtree(child)
            elif child.name == "index.json": child.unlink()
    multi.mkdir(exist_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--map-dir", type=Path, required=True); parser.add_argument("--observations", type=Path, required=True); parser.add_argument("--camera-json", type=Path); parser.add_argument("--camera-position", type=Path); parser.add_argument("--frame-timestamps", type=Path); parser.add_argument("--trajectory", type=Path); parser.add_argument("--camera-model", type=Path); parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--time-offset-s", type=float, default=0.0, help="Explicit sensor clock offset only; never optimized from GT")
    parser.add_argument("--ceiling-z", type=float, default=None, help="Known ceiling plane in metres; if omitted use triangulation")
    parser.add_argument("--height-mode", choices=("auto", "plane", "triangulate"), default="auto"); parser.add_argument("--min-observations", type=int, default=5); parser.add_argument("--outlier-radius-m", type=float, default=0.35); parser.add_argument("--triangulation-residual-m", type=float, default=0.20); parser.add_argument("--merge-radius-m", type=float, default=0.40); parser.add_argument("--place-radius-m", type=float, default=0.75)
    args = parser.parse_args()
    if args.ceiling_z is not None and (not math.isfinite(args.ceiling_z) or args.ceiling_z <= 0): parser.error("--ceiling-z must be positive")
    if args.height_mode == "plane" and args.ceiling_z is None: parser.error("--height-mode plane requires --ceiling-z")
    map_dir = args.map_dir.resolve(); observations_path = args.observations.resolve(); map_meta = read_yaml(map_dir / "map.yaml"); image_name = str(map_meta.get("image", "map.pgm")); map_image = np.asarray(Image.open(map_dir / image_name).convert("L")); transform = map_transform(map_meta, map_image.shape[:2])
    trajectory_path = (args.trajectory or (map_dir / "trajectory.json")).resolve(); camera_json = (args.camera_json or (observations_path.parent / "camera.json")).resolve(); camera_position = (args.camera_position or (observations_path.parent / "camera_position.txt")).resolve(); payload = json.loads(observations_path.read_text(encoding="utf-8")); trajectory_times, trajectory = load_trajectory(trajectory_path); frame_times = load_frame_times(payload, observations_path, args.frame_timestamps.resolve() if args.frame_timestamps else None); camera_model, camera_meta = load_camera_model(camera_json, camera_position, args.camera_model); by_code, counters = collect_observations(payload, frame_times, trajectory_times, trajectory, camera_model, load_intrinsic_size(camera_json), args.time_offset_s)
    hypotheses: list[dict[str, Any]] = []; rejected_codes: dict[str, str] = {}
    for code, rows in sorted(by_code.items(), key=lambda pair: int(pair[0])):
        if len(rows) < args.min_observations: rejected_codes[code] = "below_min_observations"; continue
        fit = project_to_plane(rows, float(args.ceiling_z), args.outlier_radius_m) if args.height_mode == "plane" or (args.height_mode == "auto" and args.ceiling_z is not None) else triangulate_lines(rows, args.triangulation_residual_m)
        if fit is None or len(fit["inlier_indices"]) < args.min_observations: rejected_codes[code] = "no_robust_geometry"; continue
        inlier_set = {int(i) for i in fit["inlier_indices"]}
        inlier_rows = [dict(rows[int(i)], accepted_by_robust_fit=True) for i in sorted(inlier_set)]
        all_rows_for_code = [dict(row, accepted_by_robust_fit=(i in inlier_set)) for i, row in enumerate(rows)]
        hypotheses.append({"code": int(code), "point_xyz_m": fit["point_xyz_m"], "rows": inlier_rows, "rows_all": all_rows_for_code, "track_ids": sorted({r["track_id"] for r in inlier_rows}), "inliers": len(inlier_rows), "spread_m": float(fit["spread_m"]), "method": fit["method"], "raw_observations": len(rows)})
    markers: dict[str, dict[str, Any]] = {}; all_rows: list[dict[str, Any]] = []
    for group in _component_hypotheses(hypotheses, args.merge_radius_m):
        group = sorted(group, key=lambda h: (-h["inliers"], h["code"])); dominant = group[0]; codes = sorted({h["code"] for h in group}); key = str(dominant["code"]); rows = []; seen = set()
        for hypothesis in group:
            for row in hypothesis["rows_all"]:
                identity = (row["frame_index"], row["track_id"], row["marker_id_raw"])
                if identity in seen: continue
                seen.add(identity); value = dict(row); value["marker_key"] = key; rows.append(value)
        places = cluster_places(rows, args.place_radius_m)
        for place in places: place["camera_xy_occupancy_m"] = map_to_grid(place["camera_xy_m"], transform).tolist()
        point = np.asarray(dominant["point_xyz_m"], dtype=float); occupancy = map_to_grid_xyz(point, transform)
        markers[key] = {"marker_id": int(dominant["code"]), "codes": codes, "point_xyz_m": [float(v) for v in point], "occupancy_grid_xyz_m": occupancy.tolist(), "observations": len(rows), "raw_observations": sum(h["raw_observations"] for h in group), "inliers": sum(h["inliers"] for h in group), "track_ids": sorted({r["track_id"] for r in rows}), "places": places, "spread_m": float(dominant["spread_m"]), "geometry_method": dominant["method"], "aliases_merged_by_track_and_distance": codes, "rows": rows}; all_rows.extend(rows)
    all_rows.sort(key=lambda row: (row["timestamp_s"], row["marker_key"], row["track_id"])); _prepare_output_dirs(args.out_dir); trajectory_grid = transform_trajectory(trajectory, transform); render_map(args.out_dir / "map_with_marker_rays.png", map_image, transform, markers, trajectory_grid, all_rows, f"Cartographer map with marker points/rays ({len(markers)} markers)")
    repeated = []
    for key, marker in sorted(markers.items(), key=lambda pair: int(pair[0])):
        if len(marker["places"]) < 2: continue
        folder = args.out_dir / "multi_seen_points" / f"marker_{key}"; folder.mkdir(parents=True, exist_ok=True); rows = []
        for row in marker["rows"]:
            value = dict(row)
            value["place_index"] = min(
                range(len(marker["places"])),
                key=lambda i: float(np.linalg.norm(
                    np.asarray(row["camera_xyz_m"][:2]) - np.asarray(marker["places"][i]["camera_xy_m"])
                )),
            )
            value["ray_endpoint_map_xyz_m"] = marker["point_xyz_m"]
            value["ray_endpoint_occupancy_xyz_m"] = marker["occupancy_grid_xyz_m"]
            rows.append(value)
        marker_payload = {k: marker[k] for k in ("marker_id", "codes", "point_xyz_m", "occupancy_grid_xyz_m", "observations", "raw_observations", "inliers", "track_ids", "places", "spread_m", "geometry_method")}; marker_payload["rays"] = rows; marker_payload["place_definition"] = f"greedy spatial clusters with radius {args.place_radius_m:.3f} m"; (folder / "observations.json").write_text(json.dumps(marker_payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        views = np.asarray([p["camera_xy_occupancy_m"] for p in marker["places"]], dtype=float); render_map(folder / f"marker_{key}_views.png", map_image, transform, {key: marker}, trajectory_grid, rows, f"Marker {key}: {len(views)} distinct view places", focus=(np.asarray(marker["occupancy_grid_xyz_m"][:2]), views), max_rays=3000); repeated.append({"marker_id": marker["marker_id"], "codes": marker["codes"], "folder": str(folder.resolve()), "places": len(views), "observations": marker["observations"], "point_xyz_m": marker["point_xyz_m"], "occupancy_grid_xyz_m": marker["occupancy_grid_xyz_m"]})
    serial_markers = [{k: marker[k] for k in ("marker_id", "codes", "point_xyz_m", "occupancy_grid_xyz_m", "observations", "raw_observations", "inliers", "track_ids", "places", "spread_m", "geometry_method", "aliases_merged_by_track_and_distance")} for key, marker in sorted(markers.items(), key=lambda pair: int(pair[0]))]
    selected_timestamp_path = args.frame_timestamps.resolve() if args.frame_timestamps else (observations_path.parent / "frame_timestamps.csv").resolve()
    report = {"schema": "fliftbot.cartographer_marker_overlay.v2", "method": "camera_model_pixel_reprojection_cartographer_pose_robust_geometry", "gt_or_bbox_alignment_used": False, "runtime_gt_used": False, "registration_used": False, "inputs": {"map_dir": str(map_dir), "trajectory": str(trajectory_path), "observations": str(observations_path), "camera_json": str(camera_json), "camera_position": str(camera_position), "frame_timestamps": str(selected_timestamp_path)}, "camera_model": camera_meta, "ray_frame": "CameraModel world/body frame evaluated at Cartographer robot pose; optical axes are handled by CameraModel", "stored_ray_policy": "stored observations.ray is audit-only; geometry is recomputed from pixel center/tlbr through CameraModel", "cartographer_map_frame": "trajectory.json x/y/yaw; no additional yaw, lateral reflection, or scene fit", "occupancy_grid_transform": {"origin_xy_m_in_cartographer_map": [float(v) for v in transform["origin_xy_m"]], "origin_yaw_rad": transform["origin_yaw_rad"], "resolution_m": transform["resolution_m"], "size_px": [transform["width_px"], transform["height_px"]], "formula": "p_grid=R(-origin_yaw)*(p_cartographer-origin_xy); display flip is raster-only"}, "timestamp_mapping": {"frame_timestamp_source": "frame_index keyed CSV/embedded timestamps", "time_offset_s_explicit": float(args.time_offset_s), "offset_optimized": False, "cartographer_time_range_s": [float(trajectory_times[0]), float(trajectory_times[-1])], "frame_time_range_s": [float(frame_times[0]), float(frame_times[-1])]}, "height": {"mode": "known_plane" if args.ceiling_z is not None else "robust_multi_ray_triangulation", "ceiling_z_m": float(args.ceiling_z) if args.ceiling_z is not None else None, "source": "explicit CLI/config" if args.ceiling_z is not None else "estimated independently per marker from rays", "triangulation_residual_gate_m": args.triangulation_residual_m}, "robust_consensus": {"code_buckets": "raw code evidence; track is not forced to one code", "track_switch_policy": "per-code robust hypotheses are merged only when inlier hypotheses share a track and are within merge radius", "outlier_radius_m": args.outlier_radius_m, "merge_radius_m": args.merge_radius_m, "min_observations": args.min_observations, "rejected_codes": rejected_codes}, "counters": counters, "markers": serial_markers, "multi_seen_markers": repeated}
    args.out_dir.mkdir(parents=True, exist_ok=True); (args.out_dir / "marker_points_cartographer.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"); (args.out_dir / "multi_seen_points" / "index.json").write_text(json.dumps({"markers": repeated}, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (args.out_dir / "README.md").write_text("# Cartographer marker overlay\n\nProduction overlay: frame timestamp -> interpolated Cartographer pose -> CameraModel pixel reprojection -> robust ray geometry. GT, room labels, bbox/SE(2) registration and scene-specific parameter fitting are not used. The legacy stored `observations.ray` is audit-only; pixels are interpreted as `[u=x,v=y]`, while legacy `tlbr` is `[y1,x1,y2,x2]`.\n\n" + f"Markers: {len(markers)}; multi-seen: {len(repeated)}; height mode: {'known plane z=' + str(args.ceiling_z) if args.ceiling_z is not None else 'robust multi-ray triangulation'}; explicit time offset: {args.time_offset_s:.6f} s.\n\nThe map-frame coordinates are authoritative for downstream consumers. Occupancy-grid coordinates are only for rendering against map.yaml/map.pgm. `quality_report.json`, if present, is offline-only and may use GT for evaluation.\n", encoding="utf-8")
    print(json.dumps({"markers": len(markers), "multi_seen": len(repeated), "height_mode": report["height"]["mode"], "out": str(args.out_dir)}, indent=2))


if __name__ == "__main__":
    main()
