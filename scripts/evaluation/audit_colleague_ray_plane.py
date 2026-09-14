#!/usr/bin/env python3
"""Reproduce and audit the ray-plane algorithm from ray_plane_intersection_plot.ipynb.

This is intentionally an audit tool, not a production marker localizer.  It
keeps the notebook's contract explicit: a camera-frame ray, a fixed horizontal
plane, and an interpolated SE(2) pose.  GT is optional and is used only for an
offline, marker-98-centered comparison.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


UNIX_EPOCH_TICKS = 621355968000000000


def rot(theta: float) -> np.ndarray:
    c, s = math.cos(theta), math.sin(theta)
    return np.asarray([[c, -s], [s, c]], dtype=float)


def parse_time(value: str) -> float:
    return dt.datetime.strptime(value, "%Y_%m_%d_%H_%M_%S_%f").timestamp()


def load_times(payload: dict[str, Any], observations: Path) -> np.ndarray:
    values = payload.get("frame_timestamps")
    if isinstance(values, list) and len(values) >= len(payload.get("data", [])):
        a = np.asarray(values[: len(payload["data"])], dtype=float)
        if len(a) and np.isfinite(a).all():
            return a
    path = observations.parent / "frame_timestamps.csv"
    with path.open(newline="", encoding="utf-8") as stream:
        a = np.asarray([float(row["time_take_img_sec"]) for row in csv.DictReader(stream)], dtype=float)
    if len(a) != len(payload.get("data", [])) or not np.isfinite(a).all():
        raise ValueError(f"invalid frame timestamps: {path}")
    return a


def load_move_log(path: Path) -> tuple[np.ndarray, np.ndarray]:
    rows = []
    with path.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            try:
                rows.append((parse_time(row["t"]), float(row["x"]), float(row["y"]), math.radians(float(row["dgr"]))))
            except (KeyError, TypeError, ValueError):
                continue
    if len(rows) < 2:
        raise ValueError(f"not enough valid poses in {path}")
    a = np.asarray(rows, dtype=float)
    a = a[np.argsort(a[:, 0])]
    return a[:, 0], a[:, 1:]


def load_trajectory(path: Path) -> tuple[np.ndarray, np.ndarray]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = []
    for item in payload.get("poses", payload if isinstance(payload, list) else []):
        if "timestamp_universal" in item:
            stamp = (float(item["timestamp_universal"]) - UNIX_EPOCH_TICKS) / 1.0e7
        elif "timestamp_s" in item:
            stamp = float(item["timestamp_s"])
        elif "t" in item:
            stamp = float(item["t"])
        else:
            continue
        qz, qw = float(item.get("qz", 0.0)), float(item.get("qw", 1.0))
        yaw = math.atan2(2.0 * qw * qz, 1.0 - 2.0 * qz * qz)
        rows.append((stamp, float(item["x"]), float(item["y"]), yaw))
    if len(rows) < 2:
        raise ValueError(f"not enough timed poses in {path}")
    a = np.asarray(rows, dtype=float)
    a = a[np.argsort(a[:, 0])]
    return a[:, 0], a[:, 1:]


def interpolate(query: float, times: np.ndarray, poses: np.ndarray) -> np.ndarray | None:
    i = int(np.searchsorted(times, query))
    if i == 0 or i >= len(times):
        return None
    w = (query - times[i - 1]) / (times[i] - times[i - 1])
    da = (poses[i, 2] - poses[i - 1, 2] + math.pi) % (2.0 * math.pi) - math.pi
    return np.asarray([
        poses[i - 1, 0] + w * (poses[i, 0] - poses[i - 1, 0]),
        poses[i - 1, 1] + w * (poses[i, 1] - poses[i - 1, 1]),
        poses[i - 1, 2] + w * da,
    ])


def camera_origin(payload: dict[str, Any], position_path: Path | None) -> np.ndarray:
    value = payload.get("camera_origin")
    if value is not None:
        result = np.asarray(value, dtype=float)
    elif position_path is not None:
        line = next(line for line in position_path.read_text(encoding="utf-8").splitlines() if line.strip())
        result = np.asarray([float(v) for v in line.split()], dtype=float) / 100.0
    else:
        raise ValueError("camera origin is missing; use observations.camera_origin or --camera-position")
    if result.shape != (3,) or not np.isfinite(result).all():
        raise ValueError("camera origin must be three finite values")
    return result


def _stats(values: np.ndarray) -> dict[str, Any]:
    """Stats for matched points only; unmatched points are reported separately."""
    values = np.asarray(values, dtype=float)
    if not len(values):
        return {"count": 0}
    return {
        "count": int(len(values)), "mae_m": float(np.mean(values)),
        "rmse_m": float(np.sqrt(np.mean(values ** 2))),
        "median_m": float(np.median(values)),
        "p95_m": float(np.percentile(values, 95)), "max_m": float(np.max(values)),
        "within_0.05m": int(np.sum(values <= 0.05)),
        "within_0.10m": int(np.sum(values <= 0.10)),
        "within_0.20m": int(np.sum(values <= 0.20)),
    }


def _gated_assignment(pred: np.ndarray, gt: np.ndarray, threshold: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """One-to-one assignment that maximizes in-threshold matches.

    A plain Hungarian assignment over the complete matrix can hide false
    positives: every prediction gets a GT even when it is metres away.  Here
    out-of-gate pairs have a very large cost, so they remain unmatched in the
    reported metrics.
    """
    from scipy.optimize import linear_sum_assignment

    if not len(pred) or not len(gt):
        return np.empty(0, dtype=int), np.empty(0, dtype=int), np.empty(0, dtype=float)
    distances = np.linalg.norm(pred[:, None, :] - gt[None, :, :], axis=2)
    valid = distances <= threshold
    cost = np.where(valid, distances, 1.0e6 + distances)
    rows, cols = linear_sum_assignment(cost)
    keep = valid[rows, cols]
    return rows[keep], cols[keep], distances[rows[keep], cols[keep]]


def _threshold_metrics(pred: np.ndarray, gt: np.ndarray, thresholds=(0.05, 0.10, 0.20)) -> dict[str, Any]:
    result = {}
    for threshold in thresholds:
        rows, cols, errors = _gated_assignment(pred, gt, threshold)
        matched = int(len(errors))
        result[f"{int(threshold * 100)}cm"] = {
            "threshold_m": threshold,
            "matched": matched,
            "prediction_count": int(len(pred)),
            "gt_count": int(len(gt)),
            "precision": float(matched / len(pred)) if len(pred) else 0.0,
            "recall": float(matched / len(gt)) if len(gt) else 0.0,
            "unmatched_predicted": int(len(pred) - matched),
            "unmatched_gt": int(len(gt) - matched),
            "errors": _stats(errors),
        }
    return result


def _fit_rotation(pred: np.ndarray, gt: np.ndarray) -> float:
    """Diagnostic only: fit one proper Z rotation using GT (never production)."""
    from scipy.optimize import linear_sum_assignment, minimize_scalar

    def score(theta: float) -> float:
        aligned = pred @ rot(theta).T
        distances = np.linalg.norm(aligned[:, None, :] - gt[None, :, :], axis=2)
        rows, cols = linear_sum_assignment(distances)
        return float(np.mean(distances[rows, cols]))

    if not len(pred) or not len(gt):
        return 0.0
    grid = np.linspace(-math.pi, math.pi, 3601)
    index = int(np.argmin([score(v) for v in grid]))
    return float(minimize_scalar(
        score,
        bounds=(grid[max(0, index - 2)], grid[min(len(grid) - 1, index + 2)]),
        method="bounded",
    ).x)


def compare_point_sets(pred: dict[int, np.ndarray], gt: dict[int, np.ndarray], reference: int) -> dict[str, Any]:
    """Compare physical centres, separating geometry from decoded IDs.

    If marker ``reference`` is present on both sides, it is used only after a
    strict physical sanity check (its nearest GT must be the same reference,
    within 25 cm).  The primary comparison then uses the known pose heading
    and only translates by this reference; it does *not* fit a GT-derived
    rotation or force every prediction to a GT point.  A GT-fitted rotation is
    retained as a clearly labelled diagnostic for relative-geometry studies.
    """
    pred_ids, gt_ids = list(pred), list(gt)
    p_all = np.asarray([pred[k] for k in pred_ids], dtype=float)
    g_all = np.asarray([gt[k] for k in gt_ids], dtype=float)
    ref_pred = reference in pred
    ref_gt = reference in gt
    ref_used = False
    ref_validation: dict[str, Any] = {"present_predicted": ref_pred, "present_gt": ref_gt}

    if ref_pred and ref_gt:
        p_ref = pred[reference]
        g_ref = gt[reference]
        # The two files can have different global origins (the structured GT
        # uses the room corner, while notebook-like output uses marker 98).
        # Therefore validate the *candidate* reference by its neighbourhood,
        # after the one permitted translation by marker 98.  This catches a
        # decoded ``98`` that is actually another physical marker without
        # fitting a rotation or scale.
        candidate = p_all - p_ref + g_ref
        p_without = np.asarray([candidate[i] for i, k in enumerate(pred_ids) if k != reference])
        g_without = np.asarray([g_all[i] for i, k in enumerate(gt_ids) if k != reference])
        neigh_rows, neigh_cols, neigh_errors = _gated_assignment(p_without, g_without, 0.25)
        ref_support = int(len(neigh_errors))
        ref_validation.update({
            "candidate_transform": "identity_rotation_plus_translation(pred_98 -> gt_98)",
            "neighbourhood_gate_m": 0.25,
            "neighbourhood_matches": ref_support,
            "neighbourhood_prediction_count": int(len(p_without)),
            "accepted": bool(
                ref_support >= max(3, int(math.ceil(0.5 * min(len(p_without), len(g_without)))))
            ),
        })
        if ref_validation["accepted"]:
            # One rigid transform: identity (known heading) + translation by
            # the physical reference.  This is not a fitted rotation.
            p_all = candidate
            ref_used = True

    # Remove the anchor from error/coverage counts once it has been accepted.
    if ref_used:
        p_mask = np.asarray([k != reference for k in pred_ids])
        g_mask = np.asarray([k != reference for k in gt_ids])
        p_eval, g_eval = p_all[p_mask], g_all[g_mask]
        pred_eval_ids = [k for k in pred_ids if k != reference]
        gt_eval_ids = [k for k in gt_ids if k != reference]
    else:
        p_eval, g_eval = p_all, g_all
        pred_eval_ids, gt_eval_ids = pred_ids, gt_ids

    primary = _threshold_metrics(p_eval, g_eval)
    # Strict aggregate over the 20 cm gate is useful as an explicit number,
    # while the threshold table remains the authoritative precision/recall.
    rows, cols, errors20 = _gated_assignment(p_eval, g_eval, 0.20)

    # Exact decoded-code diagnostics, never presented as physical accuracy.
    id_errors = []
    if ref_used:
        for i, code in enumerate(pred_eval_ids):
            if code in gt:
                id_errors.append(float(np.linalg.norm(p_eval[i] - (gt[code]))))
    else:
        for i, code in enumerate(pred_eval_ids):
            if code in gt:
                id_errors.append(float(np.linalg.norm(p_eval[i] - gt[code])))

    # Diagnostic relative rotation is intentionally not used in primary.
    fitted_theta = _fit_rotation(p_eval, g_eval)
    fitted_pred = p_eval @ rot(fitted_theta).T
    from scipy.optimize import linear_sum_assignment
    d_fit = np.linalg.norm(fitted_pred[:, None, :] - g_eval[None, :, :], axis=2)
    rr, cc = linear_sum_assignment(d_fit) if len(p_eval) and len(g_eval) else ([], [])

    return {
        "status": "ok",
        "prediction_count": len(pred), "gt_count": len(gt),
        "evaluation_prediction_count": int(len(p_eval)),
        "evaluation_gt_count": int(len(g_eval)),
        "reference": ref_validation,
        "primary_known_pose_identity_rotation": {
            "transform": "translation_by_reference_only" if ref_used else "none_same_input_frame",
            "rotation_deg": 0.0,
            "thresholds": primary,
            "matched_at_20cm_stats": _stats(errors20),
        },
        "diagnostic_gt_fitted_rotation": {
            "warning": "offline diagnostic only; rotation is optimized against GT and must not be used as runtime accuracy",
            "rotation_deg": math.degrees(fitted_theta),
            "forced_assignment_stats": _stats(d_fit[rr, cc]),
        },
        "id_aware_diagnostic": _stats(np.asarray(id_errors)),
    }


def load_gt(path: Path) -> dict[int, np.ndarray]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    # Both the source labels2coords format and the transformed audit GT are
    # accepted.  They describe the same physical centres in different global
    # origins/conventions; marker-98 anchoring handles the origin difference.
    if isinstance(payload, dict) and "labels" not in payload:
        return {
            int(code): np.asarray([row["x"], row["y"]], dtype=float) / 100.0
            for code, row in payload.items()
            if isinstance(row, dict) and "x" in row and "y" in row
        }
    labels = payload.get("labels", [])
    return {int(row["id"]): np.asarray(row["gt_room_xyz_m"][:2], dtype=float) for row in labels}


def main() -> int:
    root = Path(__file__).resolve().parents[3]
    default_recording = root / "ros-data/recordings/real/20260908/160617_dephan_m360_room"
    default_result = root / "ros-data/results/real/20260910/160617_dephan_m360_room/cartographer_lidar_tracking_candidate"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--observations", type=Path, default=default_recording / "observations.json")
    parser.add_argument("--camera-position", type=Path, default=default_recording / "camera_position.txt")
    parser.add_argument("--move-log", type=Path, default=next(default_recording.glob("fliftbot_move_logger_*.log")))
    parser.add_argument("--trajectory", type=Path, default=None)
    parser.add_argument("--plane-height", type=float, default=3.13)
    parser.add_argument("--time-offset", type=float, default=0.0)
    parser.add_argument("--gt", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--min-observations", type=int, default=12)
    args = parser.parse_args()

    payload = json.loads(args.observations.read_text(encoding="utf-8"))
    frame_times = load_times(payload, args.observations)
    pose_times, poses = load_trajectory(args.trajectory) if args.trajectory else load_move_log(args.move_log)
    origin = camera_origin(payload, args.camera_position)
    by_code: dict[int, list[np.ndarray]] = {}
    raw = 0
    valid = 0
    for frame, stamp in zip(payload.get("data", []), frame_times):
        pose = interpolate(float(stamp) + args.time_offset, pose_times, poses)
        if pose is None:
            continue
        for item in frame.values():
            try:
                code = int(item.get("code"))
                ray = np.asarray(item.get("ray"), dtype=float)
            except (TypeError, ValueError):
                continue
            if code < 0 or ray.shape != (3,) or not np.isfinite(ray).all():
                continue
            raw += 1
            if abs(ray[2]) < 1.0e-9:
                continue
            distance = (args.plane_height - origin[2]) / ray[2]
            if distance <= 0.0 or not np.isfinite(distance):
                continue
            local = origin + distance * ray
            global_xy = rot(float(pose[2])) @ local[:2] + pose[:2]
            by_code.setdefault(code, []).append(global_xy)
            valid += 1
    points = {code: np.median(values, axis=0) for code, values in by_code.items() if len(values) >= args.min_observations}
    args.out_dir.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {
        "schema": "fliftbot.audit.colleague_ray_plane.v1", "runtime_gt_used": False,
        "algorithm": "P=O+((H-Oz)/dz)d; global=Rz(pose_yaw)Pxy+pose_xy",
        "inputs": {"observations": str(args.observations.resolve()), "camera_position": str(args.camera_position.resolve()), "move_log": str(args.move_log.resolve()) if args.trajectory is None else None, "trajectory": str(args.trajectory.resolve()) if args.trajectory else None},
        "parameters": {"plane_height_m": args.plane_height, "time_offset_s": args.time_offset, "camera_origin_m": origin.tolist(), "min_observations": args.min_observations},
        "counts": {"frames": len(payload.get("data", [])), "raw_valid_ray_observations": raw, "plane_intersections": valid, "codes_with_min_observations": len(points)},
        "markers": [{"marker_id": int(code), "point_xy_m": [float(v) for v in point], "observations": len(by_code[code])} for code, point in sorted(points.items())],
    }
    if args.gt:
        report["offline_comparison"] = compare_point_sets(points, load_gt(args.gt), 98)
    (args.out_dir / "audit.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"out": str((args.out_dir / "audit.json").resolve()), "counts": report["counts"], "offline_comparison": report.get("offline_comparison")}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
