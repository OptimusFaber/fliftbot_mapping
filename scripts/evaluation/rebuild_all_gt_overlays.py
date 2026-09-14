#!/usr/bin/env python3
"""Rebuild GT overlays for every map result with one rigid, no-scale fit.

Explicit exclusions requested by the user are never read or written:
  results/20260806
  results/20260807/real/run_121410
  results/20260807/real/run_140543
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(ROOT / "src"))

from recalc_real_align_right import (  # noqa: E402
    RUN_CAMERA_Z,
    UP_LIDAR_Z,
    LOW_LIDAR_Z,
    align_pin_right,
    save_overlay,
)
from flifltbot.mapping.factor_diag import (  # noqa: E402
    apply_se2,
    ate_metrics,
    load_odom_gt,
    sync_est_to_gt,
    umeyama_se2,
)
from flifltbot.common.runtime.accuracy_eval import gt_segments  # noqa: E402
from flifltbot.common.output.layout import RECORDINGS_ROOT, RESULTS_ROOT  # noqa: E402

RESULTS = RESULTS_ROOT
EXCLUDED = (
    RESULTS / "real/20260806",
    RESULTS / "real/20260807/unknown_rplidar_run_121410_legacy",
    RESULTS / "real/20260807/unknown_rplidar_run_140543_legacy",
)

# Cosmetic residuals after automatic rigid fitting, confirmed against the
# Real Room Floorplan.  They are translations only and never alter map scale.
ROOM_DISPLAY_OFFSETS_M = {"run_141753": (0.0, 0.10)}


def excluded(path: Path) -> bool:
    resolved = path.resolve()
    return any(resolved == p.resolve() or p.resolve() in resolved.parents for p in EXCLUDED)


def load_map(path: Path):
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if not isinstance(data, dict) or not isinstance(data.get("map"), list):
        return None
    if not isinstance(data.get("poses"), list) or len(data["map"]) < 20:
        return None
    points = np.asarray(data["map"], float)
    poses = []
    for item in data["poses"]:
        if isinstance(item, dict) and item.get("pose") is not None:
            poses.append({"t": item.get("t"), "pose": list(item["pose"][:3])})
        elif isinstance(item, (list, tuple)) and len(item) >= 3:
            poses.append({"t": None, "pose": list(item[:3])})
    return data, points, poses


def is_real(path: Path) -> bool:
    return "real" in path.parts or "room" in path.parts and "archive_all_methods" in path.parts


def camera_z(path: Path) -> float:
    if "low" in path.stem.lower():
        return LOW_LIDAR_Z
    for part in reversed(path.parts):
        if part.startswith("run_") and part in RUN_CAMERA_Z:
            return RUN_CAMERA_Z[part]
    return UP_LIDAR_Z


def sim_run_dir(path: Path) -> Path | None:
    parts = path.parts
    if "archive_all_methods" in parts:
        matches = sorted((RECORDINGS_ROOT / "sim/20260815").glob("*_run_004137*"))
        return matches[-1] if matches else None
    run_parts = [p for p in path.parent.parts if "_run_" in p or p.startswith("run_")]
    if not run_parts:
        return None
    run = run_parts[-1]
    date = next((p for p in parts if len(p) == 8 and p.isdigit()), None)
    if date:
        candidate = RECORDINGS_ROOT / "sim" / date / run
        if candidate.is_dir():
            return candidate
    suffix = run.split("_run_", 1)[-1] if "_run_" in run else run.removeprefix("run_")
    matches = sorted((RECORDINGS_ROOT / "sim").glob(f"*/*_run_{suffix}*"))
    return matches[-1] if matches else None


def rectangle_segments(ax):
    for index, (a, b) in enumerate(gt_segments()):
        ax.plot(
            [a[0], b[0]], [a[1], b[1]], color="black", linewidth=1.05,
            alpha=0.75, label="GT walls, shelves, columns" if index == 0 else None,
        )


def render_sim(path: Path, points: np.ndarray, poses: list[dict]):
    run = sim_run_dir(path)
    if run is None or not (run / "encoder.jsonl").is_file():
        return None
    gt = load_odom_gt(run)
    est_xy, gt_xy, est_yaw, gt_yaw, labels = sync_est_to_gt(poses, gt)
    if len(est_xy) < 2:
        return None
    yaw, tx, ty = umeyama_se2(est_xy, gt_xy)
    world_map = apply_se2(points, yaw, tx, ty)
    world_route = apply_se2(np.asarray([p["pose"][:2] for p in poses]), yaw, tx, ty)
    aligned_est = apply_se2(est_xy, yaw, tx, ty)
    metrics = ate_metrics(est_xy, gt_xy, est_yaw, gt_yaw, labels)
    gt_route = np.asarray([g["pose"][:2] for g in gt], float)

    fig, ax = plt.subplots(figsize=(10, 14))
    rectangle_segments(ax)
    step = max(1, len(world_map) // 30000)
    ax.scatter(world_map[::step, 0], world_map[::step, 1], s=0.8,
               c="#1677ff", alpha=0.38, label="pred map")
    ax.plot(world_route[:, 0], world_route[:, 1], c="#e63946", linewidth=1.0,
            label="pred route")
    ax.plot(gt_route[:, 0], gt_route[:, 1], c="#20bf3f", linewidth=1.0,
            label="Gazebo odom GT")
    ax.set_title(
        f"{path.stem}: rigid GT↔pred fit, no scale\n"
        f"ATE median {metrics['ate_median_m'] * 100:.1f} cm, "
        f"RMSE {metrics['ate_rmse_m'] * 100:.1f} cm"
    )
    ax.set_aspect("equal")
    ax.grid(alpha=0.2)
    ax.legend(loc="best")
    ax.set_xlabel("world X, m")
    ax.set_ylabel("world Y, m")
    fig.tight_layout()
    out = path.with_name(f"{path.stem}_vs_gt.png")
    fig.savefig(out, dpi=180)
    plt.close(fig)
    payload = {
        "source_map": str(path),
        "gt_run": str(run),
        "alignment": "global least-squares rigid SE(2); no scale",
        "transform": {"yaw_deg": math.degrees(yaw), "tx_m": tx, "ty_m": ty},
        "metrics": metrics,
        "gt_objects": "outer walls + all world shelves + 10 structural columns",
        "overlay_png": str(out),
        "sync_pairs": int(len(aligned_est)),
    }
    path.with_name(f"{path.stem}_gt_alignment.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    return out


def render_real(path: Path, points: np.ndarray, poses: list[dict]):
    trajectory = np.asarray([p["pose"][:2] for p in poses], float) if poses else None
    z = camera_z(path)
    aligned = align_pin_right(points, trajectory, z=z, level_left=False)
    offset = ROOM_DISPLAY_OFFSETS_M.get(path.parent.name, (0.0, 0.0))
    if offset != (0.0, 0.0):
        delta = np.asarray(offset, float)
        aligned["aligned"] = np.asarray(aligned["aligned"]) + delta
        aligned["contour"] = np.asarray(aligned["contour"]) + delta
        if aligned.get("traj") is not None:
            aligned["traj"] = np.asarray(aligned["traj"]) + delta
        for key, amount in (("W", offset[0]), ("E", offset[0]), ("S", offset[1]), ("N", offset[1])):
            aligned[key] = float(aligned[key]) + amount
        aligned["display_offset_m"] = list(offset)
    out = path.with_name(f"{path.stem}_vs_gt.png")
    save_overlay(path.stem, aligned, out)
    payload = {
        "source_map": str(path),
        "gt_source": str(ROOT.parent / "fliftbot_sim/simulation/worlds/real_room.sdf"),
        "gt_reference": "Gazebo real_room.sdf synchronized with Canvas: Real Room Floorplan",
        "alignment": "balanced rigid four-corner fit; no scale",
        "camera_z_m": z,
        "wall_faces_m": {k: float(aligned[k]) for k in ("W", "E", "S", "N")},
        "span_m": {"Ox": float(aligned["Ox"]), "Oy": float(aligned["Oy"])},
        "span_error_cm": {
            "Ox": float(aligned["err_Ox_cm"]),
            "Oy": float(aligned["err_Oy_cm"]),
        },
        "orientation": {
            "flip": list(aligned["flip"]),
            "delta_deg": float(aligned["delta_deg"]),
        },
        "display_offset_m": list(offset),
        "overlay_png": str(out),
    }
    path.with_name(f"{path.stem}_gt_alignment.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", choices=("all", "real", "sim"), default="all")
    parser.add_argument(
        "--map", dest="map_path", default=None,
        help="Process only this map JSON instead of scanning all results.",
    )
    args = parser.parse_args()
    rows = []
    paths = (
        [Path(args.map_path).expanduser().resolve()]
        if args.map_path is not None
        else sorted(RESULTS.rglob("*.json"))
    )
    for path in paths:
        if excluded(path) or path.name.endswith("_gt_alignment.json"):
            continue
        loaded = load_map(path)
        if loaded is None:
            continue
        real = is_real(path)
        if args.scene != "all" and args.scene != ("real" if real else "sim"):
            continue
        _data, points, poses = loaded
        try:
            output = (
                render_real(path, points, poses)
                if real
                else render_sim(path, points, poses)
            )
            rows.append({"map": str(path), "overlay": None if output is None else str(output)})
            print(f"OK {path} -> {output}", flush=True)
        except Exception as exc:
            rows.append({"map": str(path), "error": str(exc)})
            print(f"FAIL {path}: {exc}", flush=True)
    summary = RESULTS / "gt_overlay_rebuild_summary.json"
    summary.write_text(json.dumps({"items": rows}, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {summary}; {len(rows)} map results")


if __name__ == "__main__":
    main()
