#!/usr/bin/env python3
"""Runtime-only diagnostic for repeated V6 observations of one marker.

The script intentionally reuses the V6 runtime path and its V5 hypothesis
selection.  It does not read labels2coords/GT, does not register to GT and
does not alter V6 outputs.  It writes a JSON audit, a Markdown summary, a
map view (robot path + every accepted/rejected observation) and a timeline.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import numpy as np


HERE = Path(__file__).resolve().parent


def load_v6() -> Any:
    spec = importlib.util.spec_from_file_location("fliftbot_v6_diag", HERE / "ceiling_marker_localization_v6.py")
    if spec is None or spec.loader is None:
        raise ImportError("cannot load V6")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def finite_list(values: Any) -> list[float] | None:
    a = np.asarray(values, dtype=float)
    return [float(v) for v in a.tolist()] if a.ndim == 1 and np.isfinite(a).all() else None


def ellipse_stats(points: np.ndarray, center: np.ndarray) -> dict[str, Any]:
    cov = np.cov(points.T, ddof=1) if len(points) > 1 else np.zeros((2, 2), dtype=float)
    eig, vec = np.linalg.eigh(cov)
    order = np.argsort(eig)[::-1]
    eig, vec = eig[order], vec[:, order]
    # sqrt(chi2_2(0.95)); this is only a descriptive ellipse, not a filter.
    chi95 = 2.447746830680816
    angle = math.degrees(math.atan2(vec[1, 0], vec[0, 0])) if len(points) > 1 else 0.0
    return {
        "covariance_m2": [[float(v) for v in row] for row in cov],
        "eigenvalues_m2": [float(v) for v in eig],
        "axes_1sigma_m": [float(math.sqrt(max(v, 0.0))) for v in eig],
        "axes_95pct_m": [float(chi95 * math.sqrt(max(v, 0.0))) for v in eig],
        "angle_deg": float(angle),
        "center_xy_m": [float(v) for v in center],
    }


def project_to_grid(v6: Any, rows: list[dict[str, Any]], map_meta: dict[str, Any]) -> np.ndarray:
    return np.asarray([v6.geom.map_to_grid(r[:2], map_meta) for r in rows], dtype=float)


def save_map_plot(path: Path, v6: Any, map_yaml: Path, map_dir: Path, trajectory: np.ndarray,
                  accepted: list[dict[str, Any]], rejected: list[dict[str, Any]], center: np.ndarray,
                  ellipse: dict[str, Any], marker_id: int) -> None:
    from PIL import Image

    meta = v6.geom.read_map_yaml(map_yaml)
    image = np.asarray(Image.open(map_dir / str(meta["image"])).convert("L"))
    h, w = image.shape
    extent = [0.0, w * float(meta["resolution"]), 0.0, h * float(meta["resolution"])]
    fig, ax = plt.subplots(figsize=(18, 14), dpi=180)
    ax.imshow(np.flipud(image), cmap="gray", origin="lower", extent=extent, alpha=0.90, interpolation="nearest")

    path_grid = project_to_grid(v6, trajectory, meta)
    ax.plot(path_grid[:, 0], path_grid[:, 1], color="#2563eb", lw=0.9, alpha=0.60, label="Cartographer trajectory")
    acc_xy = np.asarray([r["point_xyz_m"][:2] for r in accepted], dtype=float)
    acc_grid = project_to_grid(v6, acc_xy, meta)
    # Color by time: this makes the robot's successive viewing positions clear.
    if len(acc_grid):
        order = np.argsort([r["frame_index"] for r in accepted])
        sc = ax.scatter(acc_grid[order, 0], acc_grid[order, 1], c=np.arange(len(order)), cmap="viridis",
                        s=20, alpha=0.80, linewidths=0, label="accepted ray-plane observations")
        fig.colorbar(sc, ax=ax, shrink=0.75, label="accepted observation order")
        cams = np.asarray([r["camera_origin_map"][:2] for r in accepted], dtype=float)
        cams_grid = project_to_grid(v6, cams, meta)
        ax.scatter(cams_grid[:, 0], cams_grid[:, 1], s=8, c="#111827", alpha=0.28, label="robot camera positions")
        # A sparse set of sight lines prevents the plot becoming unreadable.
        sparse = order[:: max(1, len(order) // 35)]
        for i in sparse:
            c = project_to_grid(v6, np.asarray([accepted[i]["camera_origin_map"][:2]]), meta)[0]
            p = acc_grid[i]
            ax.plot([c[0], p[0]], [c[1], p[1]], color="#f97316", alpha=0.16, lw=0.55)
    if rejected:
        rej = [r for r in rejected if r.get("point_xyz_m") is not None]
        if rej:
            rgrid = project_to_grid(v6, np.asarray([r["point_xyz_m"][:2] for r in rej]), meta)
            ax.scatter(rgrid[:, 0], rgrid[:, 1], marker="x", s=34, c="#dc2626", alpha=0.78, label="rejected geometry-valid observations")

    center_grid = project_to_grid(v6, np.asarray([center]), meta)[0]
    ax.scatter([center_grid[0]], [center_grid[1]], marker="*", s=250, c="#ef4444", edgecolors="white", linewidths=1.1,
               zorder=10, label=f"V6 final point {marker_id}")
    axes = np.asarray(ellipse["axes_95pct_m"], dtype=float)
    # map_to_grid is an SE(2) transform; the current map has yaw=0, but rotate
    # the ellipse consistently for arbitrary map.yaml origins.
    yaw = float(np.asarray(meta["origin"], dtype=float)[2])
    width95, height95 = 2.0 * axes[0], 2.0 * axes[1]
    ax.add_patch(Ellipse(center_grid, width=width95, height=height95,
                         angle=float(ellipse["angle_deg"]) - math.degrees(yaw),
                         fill=False, color="#ef4444", lw=1.5, label="95% descriptive spread ellipse"))
    ax.annotate(f"marker {marker_id}\nV6 median", center_grid, xytext=(8, 8), textcoords="offset points",
                fontsize=10, color="#991b1b", weight="bold")
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("occupancy-grid X (m)")
    ax.set_ylabel("occupancy-grid Y (m)")
    ax.set_title(f"V6 marker {marker_id}: all runtime observations and final point")
    ax.grid(alpha=0.15)
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def save_timeline(path: Path, accepted: list[dict[str, Any]], rejected: list[dict[str, Any]], center: np.ndarray,
                  marker_id: int) -> None:
    all_rows = accepted + rejected
    all_rows.sort(key=lambda r: (r["frame_index"], r.get("track_id", "")))
    fig, axes = plt.subplots(2, 1, figsize=(17, 9), dpi=180, sharex=True)
    for rows, color, label, marker in ((accepted, "#2563eb", "accepted", "."), (rejected, "#dc2626", "rejected", "x")):
        if not rows:
            continue
        frames = np.asarray([r["frame_index"] for r in rows], dtype=float)
        pts = np.asarray([r["point_xyz_m"][:2] for r in rows if r.get("point_xyz_m") is not None], dtype=float)
        if len(pts) != len(rows):
            valid = [r for r in rows if r.get("point_xyz_m") is not None]
            frames = np.asarray([r["frame_index"] for r in valid], dtype=float)
            pts = np.asarray([r["point_xyz_m"][:2] for r in valid], dtype=float)
        if not len(pts):
            continue
        delta = (pts - center) * 100.0
        axes[0].scatter(frames, np.linalg.norm(delta, axis=1), c=color, s=13, marker=marker, alpha=0.8, label=label)
        axes[1].scatter(frames, delta[:, 0], c=color, s=13, marker=marker, alpha=0.75, label=f"{label} ΔX")
        axes[1].scatter(frames, delta[:, 1], c=color, s=13, marker=marker, alpha=0.55, label=f"{label} ΔY")
    axes[0].axhline(10.0, color="#6b7280", ls="--", lw=0.8, label="10 cm")
    axes[0].set_ylabel("distance from final point (cm)")
    axes[1].set_ylabel("offset from final point (cm)")
    axes[1].set_xlabel("video frame index")
    axes[0].set_title(f"V6 marker {marker_id}: temporal stability of each observation")
    for ax in axes:
        ax.grid(alpha=0.2)
        ax.legend(ncol=3, fontsize=8)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main() -> None:
    root = Path(__file__).resolve().parents[3]
    recording = root / "ros-data/recordings/real/20260908/160617_dephan_m360_room"
    result = root / "ros-data/results/real/20260910/160617_dephan_m360_room/cartographer_lidar_tracking_candidate"
    default_out = root / "ros-data/results/real/20260914/v6_observation_diagnostics"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--marker-id", type=int, default=98)
    parser.add_argument("--observations", type=Path, default=recording / "observations.json")
    parser.add_argument("--trajectory", type=Path, default=result / "trajectory.json")
    parser.add_argument("--camera-json", type=Path, default=recording / "camera.json")
    parser.add_argument("--camera-position", type=Path, default=recording / "camera_position.txt")
    parser.add_argument("--lidar-config", type=Path, default=recording / "lidar.config")
    parser.add_argument("--map-yaml", type=Path, default=result / "map.yaml")
    parser.add_argument("--map-dir", type=Path, default=result)
    parser.add_argument("--out-dir", type=Path, default=default_out)
    parser.add_argument("--camera-y-sign", type=int, choices=(-1, 1), default=-1)
    parser.add_argument("--camera-roll", type=float, default=-0.14)
    parser.add_argument("--camera-pitch", type=float, default=0.01)
    parser.add_argument("--camera-yaw", type=float, default=-1.57)
    parser.add_argument("--time-offset", type=float, default=-0.08802549710329016)
    parser.add_argument("--ceiling-z-m", type=float, default=3.045754621317271)
    args = parser.parse_args()
    v6 = load_v6()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    raw, *_ = v6.raw_observations(args.observations, args.camera_json)
    trajectory_payload = json.loads(args.trajectory.read_text(encoding="utf-8"))
    tracking_frame = str(trajectory_payload.get("tracking_frame", ""))
    lidar_offset, _ = v6.read_lidar_offset(args.lidar_config)
    camera_axis, _ = v6.geom.load_camera_position(args.camera_position, args.camera_y_sign)
    camera_from_pose = camera_axis.copy()
    if tracking_frame == "lidar":
        camera_from_pose[:2] -= lidar_offset[:2]
    tt, trajectory = v6.geom.load_trajectory(args.trajectory)
    angles = (args.camera_roll, args.camera_pitch, args.camera_yaw)
    projected, counters = v6.geom.project_observations(raw, tt, trajectory, camera_from_pose, angles, args.time_offset, args.ceiling_z_m)
    code_rows = [r for r in projected if int(r.get("marker_id", -1)) == args.marker_id]
    # V6's final selection is reproduced exactly.  No GT is involved.
    _, diagnostics, _ = v6.v5.select_hypotheses(projected, 12, 0.25, 0.12, 0.50)
    accepted = [r for r in code_rows if r.get("inlier")]
    rejected = [r for r in code_rows if not r.get("inlier")]
    if not accepted:
        raise SystemExit(f"V6 did not accept marker {args.marker_id}")
    center = np.median(np.asarray([r["point_xyz_m"][:2] for r in accepted], dtype=float), axis=0)
    center3 = np.asarray([center[0], center[1], args.ceiling_z_m], dtype=float)
    distances = np.linalg.norm(np.asarray([r["point_xyz_m"][:2] for r in accepted]) - center, axis=1)
    residuals = np.asarray([
        np.linalg.norm(np.cross(center3 - np.asarray(r["camera_origin_map"]), np.asarray(r["ray_cartographer_map_unit"])))
        for r in accepted
    ])
    spread = ellipse_stats(np.asarray([r["point_xyz_m"][:2] for r in accepted], dtype=float), center)
    for r in code_rows:
        if r.get("point_xyz_m") is not None:
            r["distance_to_final_m"] = float(np.linalg.norm(np.asarray(r["point_xyz_m"][:2]) - center))
        r["ray_residual_to_final_m"] = None
        if r.get("point_xyz_m") is not None and r.get("camera_origin_map") is not None:
            r["ray_residual_to_final_m"] = float(np.linalg.norm(np.cross(center3 - np.asarray(r["camera_origin_map"]), np.asarray(r["ray_cartographer_map_unit"]))))
    # Keep all row fields useful for audit, but avoid non-JSON values.
    marker_json = {"frame_index": int(r["frame_index"]), "timestamp_s": float(r["timestamp_s"]),
                   "query_timestamp_s": float(r.get("query_timestamp_s", r["timestamp_s"])),
                   "track_id": str(r.get("track_id", "")), "marker_id": int(r["marker_id"]),
                   "pixel_uv": finite_list(r.get("pixel_uv")), "geometry_valid": bool(r.get("geometry_valid", False)),
                   "accepted_by_v6": bool(r.get("inlier", False)), "filter_reason": r.get("filter_reason"),
                   "camera_origin_map_m": finite_list(r.get("camera_origin_map")),
                   "point_xyz_m": finite_list(r.get("point_xyz_m")),
                   "distance_to_final_m": r.get("distance_to_final_m"), "ray_residual_to_final_m": r.get("ray_residual_to_final_m")}
    marker_rows = [marker_json_for for marker_json_for in (
        {"frame_index": int(r["frame_index"]), "timestamp_s": float(r["timestamp_s"]),
         "query_timestamp_s": float(r.get("query_timestamp_s", r["timestamp_s"])), "track_id": str(r.get("track_id", "")),
         "marker_id": int(r["marker_id"]), "pixel_uv": finite_list(r.get("pixel_uv")),
         "geometry_valid": bool(r.get("geometry_valid", False)), "accepted_by_v6": bool(r.get("inlier", False)),
         "filter_reason": r.get("filter_reason"), "camera_origin_map_m": finite_list(r.get("camera_origin_map")),
         "point_xyz_m": finite_list(r.get("point_xyz_m")), "distance_to_final_m": r.get("distance_to_final_m"),
         "ray_residual_to_final_m": r.get("ray_residual_to_final_m")}
        for r in code_rows)]
    frame_values = [r["frame_index"] for r in code_rows]
    ts_values = [r["timestamp_s"] for r in code_rows]
    report = {
        "schema": "fliftbot.v6_marker_observation_diagnostic.v1", "marker_id": args.marker_id,
        "runtime_gt_used": False,
        "inputs": {"observations": str(args.observations.resolve()), "trajectory": str(args.trajectory.resolve()),
                   "camera_json": str(args.camera_json.resolve()), "camera_position": str(args.camera_position.resolve()),
                   "lidar_config": str(args.lidar_config.resolve()), "map_yaml": str(args.map_yaml.resolve())},
        "v6_parameters": {"ceiling_z_m": args.ceiling_z_m, "time_offset_s": args.time_offset,
                           "camera_angles_rad": list(angles), "camera_offset_from_trajectory_m": camera_from_pose.tolist(),
                           "trajectory_tracking_frame": tracking_frame},
        "counts": {"raw_marker_observations": len(code_rows), "geometry_valid": sum(bool(r.get("geometry_valid")) for r in code_rows),
                   "accepted_by_v6": len(accepted), "rejected": len(rejected), "unique_accepted_frames": len(set(r["frame_index"] for r in accepted)),
                   "unique_tracks_accepted": len(set(r["track_id"] for r in accepted))},
        "span": {"first_frame": min(frame_values), "last_frame": max(frame_values), "frame_span": max(frame_values)-min(frame_values)+1,
                 "first_timestamp_s": min(ts_values), "last_timestamp_s": max(ts_values), "duration_s": max(ts_values)-min(ts_values)},
        "final_point_selection": {"algorithm": "V5 hypothesis selection called by V6: connected components, median XY center, inlier-radius gate; final point is median of accepted ray-plane intersections",
                                  "point_xyz_m": [float(center[0]), float(center[1]), float(args.ceiling_z_m)],
                                  "v6_diagnostic_hypothesis": diagnostics.get(str(args.marker_id)),
                                  "spread_m": float(np.median(distances)), "spread_p95_m": float(np.percentile(distances, 95)),
                                  "spread_max_m": float(np.max(distances)), "ray_residual_median_m": float(np.median(residuals)),
                                  "ray_residual_p95_m": float(np.percentile(residuals, 95)), "ray_residual_max_m": float(np.max(residuals)),
                                  "ellipse": spread},
        "observations": marker_rows,
    }
    json_path = args.out_dir / f"marker_{args.marker_id}_observations.json"
    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    map_path = args.out_dir / f"marker_{args.marker_id}_observations_map.png"
    timeline_path = args.out_dir / f"marker_{args.marker_id}_observations_timeline.png"
    save_map_plot(map_path, v6, args.map_yaml, args.map_dir, trajectory, accepted, rejected, center, spread, args.marker_id)
    save_timeline(timeline_path, accepted, rejected, center, args.marker_id)
    md = args.out_dir / f"marker_{args.marker_id}_observations.md"
    c = report["counts"]; s = report["span"]; f = report["final_point_selection"]
    md.write_text(
        f"""# V6 observations: marker {args.marker_id}

This is a runtime-only diagnostic. GT/labels2coords and registration are not used.

- Raw detections with code `{args.marker_id}`: **{c['raw_marker_observations']}**
- Geometry-valid: **{c['geometry_valid']}**
- Accepted by V6: **{c['accepted_by_v6']}**; rejected: **{c['rejected']}**
- Accepted unique frames/tracks: **{c['unique_accepted_frames']} / {c['unique_tracks_accepted']}**
- Frame span: `{s['first_frame']}..{s['last_frame']}` ({s['frame_span']} frames), duration **{s['duration_s']:.3f} s**
- Final point (median of accepted plane intersections): `({f['point_xyz_m'][0]:.6f}, {f['point_xyz_m'][1]:.6f}, {f['point_xyz_m'][2]:.6f}) m`
- Spread: median **{f['spread_m']*100:.2f} cm**, p95 **{f['spread_p95_m']*100:.2f} cm**, max **{f['spread_max_m']*100:.2f} cm**
- Ray residual to final point: median **{f['ray_residual_median_m']*100:.2f} cm**, p95 **{f['ray_residual_p95_m']*100:.2f} cm**, max **{f['ray_residual_max_m']*100:.2f} cm**
- 1-sigma ellipse axes: **{f['ellipse']['axes_1sigma_m'][0]*100:.2f} × {f['ellipse']['axes_1sigma_m'][1]*100:.2f} cm**; descriptive 95% axes: **{f['ellipse']['axes_95pct_m'][0]*100:.2f} × {f['ellipse']['axes_95pct_m'][1]*100:.2f} cm**, angle **{f['ellipse']['angle_deg']:.1f}°**.

V6 does not average all detections blindly. It first forms spatial hypotheses per decoded code, rejects observations outside the selected connected component or trajectory, and sets the point to the coordinate-wise median of accepted intersections. Therefore a cloud spanning roughly 100 cm² would place the output near the robust median, not at its mean or at the last observation.

![Map diagnostic]({map_path.name})

![Timeline diagnostic]({timeline_path.name})
""",
        encoding="utf-8",
    )
    print(json.dumps({"json": str(json_path.resolve()), "markdown": str(md.resolve()), "map_png": str(map_path.resolve()),
                      "timeline_png": str(timeline_path.resolve()), "counts": c, "point_xyz_m": f["point_xyz_m"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
