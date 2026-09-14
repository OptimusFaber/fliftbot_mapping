#!/usr/bin/env python3
"""Evaluate physical marker positions independently of marker IDs.

The reference marker (normally 98) is the only semantic anchor.  All other
prediction/GT points are centered on it, then a single proper 2-D rotation is
selected by one-to-one Hungarian matching of the two point sets.  No global
translation or scale is fitted.  This answers whether a physical point is in
the place occupied by *some* GT marker, even when its decoded ID is wrong.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import linear_sum_assignment, minimize_scalar


def _xy(value: Any) -> np.ndarray:
    a = np.asarray(value, dtype=float).reshape(-1)
    if a.size < 2:
        raise ValueError(f"coordinate needs x/y: {value!r}")
    return a[:2]


def _id(value: Any) -> int:
    return int(str(value))


def load_prediction(path: Path) -> dict[int, np.ndarray]:
    data = json.loads(path.read_text(encoding="utf-8"))
    records = data.get("markers", data) if isinstance(data, dict) else data
    result: dict[int, np.ndarray] = {}
    for row in records:
        if not isinstance(row, dict):
            continue
        raw = row.get("marker_id", row.get("id", row.get("code")))
        point = row.get("point_xyz_m", row.get("xyz_m", row.get("point")))
        if raw is not None and point is not None:
            result[_id(raw)] = _xy(point)
    if not result:
        raise ValueError(f"no predictions in {path}")
    return result


def load_gt(path: Path) -> dict[int, np.ndarray]:
    data = json.loads(path.read_text(encoding="utf-8"))
    records = data.get("labels", data) if isinstance(data, dict) else data
    result: dict[int, np.ndarray] = {}
    for row in records:
        if not isinstance(row, dict):
            continue
        raw = row.get("id", row.get("marker_id", row.get("code")))
        point = row.get("gt_room_xyz_m", row.get("xyz_m", row.get("point_xyz_m", row.get("point"))))
        if raw is not None and point is not None:
            result[_id(raw)] = _xy(point)
    if not result:
        raise ValueError(f"no GT points in {path}")
    return result


def rotation(theta: float) -> np.ndarray:
    c, s = math.cos(theta), math.sin(theta)
    return np.array([[c, -s], [s, c]], dtype=float)


def one_to_one(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    distances = np.linalg.norm(a[:, None, :] - b[None, :, :], axis=2)
    rows, cols = linear_sum_assignment(distances)
    return rows, cols, distances[rows, cols]


def set_objective(theta: float, pred: np.ndarray, gt: np.ndarray) -> float:
    """Mean one-to-one distance, with the smaller set fully assigned."""
    rows, cols, values = one_to_one(pred @ rotation(theta).T, gt)
    return float(np.mean(values))


def fit_rotation(pred: np.ndarray, gt: np.ndarray) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    # A coarse global scan avoids depending on the current map orientation.
    grid = np.linspace(-math.pi, math.pi, 3601)
    values = np.asarray([set_objective(t, pred, gt) for t in grid])
    best = int(np.argmin(values))
    lo, hi = grid[max(0, best - 2)], grid[min(len(grid) - 1, best + 2)]
    refined = minimize_scalar(lambda t: set_objective(t, pred, gt), bounds=(lo, hi), method="bounded")
    theta = float(refined.x)
    aligned = pred @ rotation(theta).T
    rows, cols, distances = one_to_one(aligned, gt)
    return theta, aligned, cols, distances


def stats(values: np.ndarray) -> dict[str, Any]:
    if not len(values):
        return {"count": 0}
    return {
        "count": int(len(values)),
        "mean_m": float(np.mean(values)),
        "mae_m": float(np.mean(np.abs(values))),
        "rmse_m": float(np.sqrt(np.mean(values**2))),
        "median_m": float(np.median(values)),
        "p95_m": float(np.percentile(values, 95)),
        "max_m": float(np.max(values)),
        "within_0.10m": int(np.sum(values <= 0.10)),
        "within_0.25m": int(np.sum(values <= 0.25)),
        "within_0.50m": int(np.sum(values <= 0.50)),
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = ["pred_id", "assigned_gt_id", "id_aware_gt_id", "physical_error_m",
              "id_aware_error_m", "pred_x_m", "pred_y_m", "gt_x_m", "gt_y_m", "flag"]
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def plot(path: Path, pred_ids: list[int], pred: np.ndarray, gt_ids: list[int], gt: np.ndarray,
         assigned: np.ndarray, errors: np.ndarray, theta_deg: float, ref_id: int) -> None:
    fig, ax = plt.subplots(figsize=(18, 10), constrained_layout=True)
    # All GT points are shown faintly; unmatched GT is not an error in a sparse
    # prediction and is intentionally not connected to a prediction.
    ax.scatter(gt[:, 0], gt[:, 1], marker="s", facecolors="none", edgecolors="0.55", s=68,
               label=f"GT ({len(gt)})")
    for p, q, e, pid, gid in zip(pred, assigned, errors, pred_ids, [gt_ids[i] for i in np.argmin(np.linalg.norm(assigned[:, None, :] - gt[None, :, :], axis=2), axis=1)]):
        # q is the assigned physical GT point; show the actual residual arrow.
        color = plt.cm.turbo(min(float(e) / 0.25, 1.0))
        ax.annotate("", xy=q, xytext=p, arrowprops={"arrowstyle": "->", "color": color, "lw": 1.6, "alpha": .9})
        ax.scatter(*p, color=color, s=38, zorder=3)
        ax.text(p[0], p[1], f" {pid}", fontsize=7, color="tab:blue")
        ax.text(q[0], q[1], f" [{gid}]", fontsize=7, color="tab:orange")
    ax.scatter([0], [0], color="black", marker="*", s=190, zorder=5, label=f"reference {ref_id}")
    ax.axhline(0, color="0.85", lw=.7); ax.axvline(0, color="0.85", lw=.7)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("relative X from marker 98 [m]"); ax.set_ylabel("relative Y from marker 98 [m]")
    ax.grid(True, alpha=.2)
    ax.legend(loc="best")
    ax.set_title(f"Physical marker geometry, ID-independent — rotation {theta_deg:.3f}°, no scale\n"
                 f"arrow length = position error; color scale: 0–25 cm")
    fig.savefig(path, dpi=180)
    plt.close(fig)


def compare(pred_path: Path, gt_path: Path, out_dir: Path, ref_id: int = 98,
            gate_m: float = .5) -> dict[str, Any]:
    pred_map, gt_map = load_prediction(pred_path), load_gt(gt_path)
    if ref_id not in pred_map or ref_id not in gt_map:
        raise ValueError(f"reference {ref_id} must be present in both files")
    pred_ids = list(pred_map)
    gt_ids = list(gt_map)
    pred_rel = np.asarray([pred_map[i] - pred_map[ref_id] for i in pred_ids])
    gt_rel = np.asarray([gt_map[i] - gt_map[ref_id] for i in gt_ids])
    ref_index = pred_ids.index(ref_id)
    # Do not let the anchor point influence orientation: it is exactly (0, 0).
    fit_pred_ids = [i for i in range(len(pred_ids)) if i != ref_index]
    fit_gt_ids = [i for i, gid in enumerate(gt_ids) if gid != ref_id]
    theta, aligned_nonref, _, _ = fit_rotation(pred_rel[fit_pred_ids], gt_rel[fit_gt_ids])
    aligned = np.zeros_like(pred_rel)
    aligned[fit_pred_ids] = aligned_nonref
    # The anchored point is a physical zero-error match to GT98.
    assigned_rows, assigned_cols, assigned_errors = one_to_one(aligned[fit_pred_ids], gt_rel[fit_gt_ids])
    # one_to_one rows are all rows because prediction is the smaller set.
    pred_to_col = np.full(len(pred_ids), -1, dtype=int)
    pred_to_col[np.asarray(fit_pred_ids)[assigned_rows]] = np.asarray(fit_gt_ids)[assigned_cols]
    pred_to_col[ref_index] = gt_ids.index(ref_id)
    all_errors = np.zeros(len(pred_ids), dtype=float)
    for j, col in enumerate(pred_to_col):
        all_errors[j] = float(np.linalg.norm(aligned[j] - gt_rel[col]))
    # ID-aware nearest errors use the same fitted rigid transform, but retain
    # the semantic ID.  This is diagnostic only and does not drive the fit.
    id_aware = np.full(len(pred_ids), np.nan)
    for j, pid in enumerate(pred_ids):
        if pid in gt_map:
            id_aware[j] = float(np.linalg.norm(aligned[j] - (gt_map[pid] - gt_map[ref_id])))
    matched_gt = gt_rel[pred_to_col]
    nonref_errors = all_errors[fit_pred_ids]
    rows: list[dict[str, Any]] = []
    for j, pid in enumerate(pred_ids):
        assigned_id = gt_ids[pred_to_col[j]]
        iderr = id_aware[j]
        flag = "ok" if all_errors[j] <= gate_m else "physical_outlier"
        if not np.isnan(iderr) and iderr > gate_m:
            flag += "+id_mismatch"
        rows.append({"pred_id": pid, "assigned_gt_id": assigned_id,
                     "id_aware_gt_id": pid if not np.isnan(iderr) else "",
                     "physical_error_m": f"{all_errors[j]:.6f}",
                     "id_aware_error_m": "" if np.isnan(iderr) else f"{iderr:.6f}",
                     "pred_x_m": f"{aligned[j,0]:.6f}", "pred_y_m": f"{aligned[j,1]:.6f}",
                     "gt_x_m": f"{matched_gt[j,0]:.6f}", "gt_y_m": f"{matched_gt[j,1]:.6f}", "flag": flag})
    # Symmetric nearest-neighbour metric exposes coverage without pretending
    # that the 83-label GT should all be reconstructed by 46/47 predictions.
    D = np.linalg.norm(aligned[fit_pred_ids, None, :] - gt_rel[fit_gt_ids][None, :, :], axis=2)
    pred_nearest = D.min(axis=1)
    gt_nearest = D.min(axis=0)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(out_dir / "unlabeled_marker_geometry.csv", rows)
    result: dict[str, Any] = {
        "schema": "fliftbot.marker_unlabeled_geometry.v1",
        "inputs": {"prediction": str(pred_path), "gt": str(gt_path), "reference_id": ref_id},
        "contract": "center both sets on marker 98; fit one proper rotation by one-to-one point-set matching; no translation, scale, or global-frame use",
        "counts": {"prediction": len(pred_ids), "gt": len(gt_ids), "assigned_prediction_points": len(pred_ids),
                   "unmatched_gt_points": len(gt_ids) - len(pred_ids)},
        "rotation_deg": math.degrees(theta),
        "gate_m": gate_m,
        "physical_one_to_one": stats(nonref_errors),
        "prediction_to_gt_nearest": stats(pred_nearest),
        "gt_to_prediction_nearest_coverage": stats(gt_nearest),
        "symmetric_chamfer_mean_m": float((np.mean(pred_nearest) + np.mean(gt_nearest)) / 2),
        "physical_outlier_pred_ids": [pred_ids[i] for i, e in zip(fit_pred_ids, nonref_errors) if e > gate_m],
        "id_mismatch_but_physical_match": [pred_ids[i] for i, (e, ie) in zip(fit_pred_ids, zip(nonref_errors, id_aware[fit_pred_ids]))
                                            if e <= gate_m and not np.isnan(ie) and ie > gate_m],
        "rows": rows,
    }
    (out_dir / "unlabeled_marker_geometry.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    plot(out_dir / "unlabeled_marker_geometry.png", pred_ids, aligned, gt_ids, gt_rel, matched_gt, all_errors,
         math.degrees(theta), ref_id)
    lines = ["# ID-independent physical marker geometry", "",
             "The prediction is evaluated as a point set. Marker IDs are not used to choose the corresponding GT point; only marker 98 anchors the two relative coordinate systems.", "",
             f"Prediction points: **{len(pred_ids)}**; GT points: **{len(gt_ids)}**; unmatched GT points (not an error): **{len(gt_ids)-len(pred_ids)}**.",
             f"One proper rotation: **{math.degrees(theta):.3f}°**, no translation and no scale.", "",
             f"One-to-one physical assignment (excluding the zero-error anchor 98): MAE **{np.mean(nonref_errors):.3f} m**, RMSE **{np.sqrt(np.mean(nonref_errors**2)):.3f} m**, median **{np.median(nonref_errors):.3f} m**, p95 **{np.percentile(nonref_errors,95):.3f} m**, max **{np.max(nonref_errors):.3f} m**; within 10 cm: **{np.sum(nonref_errors<=.1)}/{len(nonref_errors)}**.",
             f"Prediction→GT nearest: MAE **{np.mean(pred_nearest):.3f} m**, p95 **{np.percentile(pred_nearest,95):.3f} m**.",
             f"GT→prediction nearest (coverage only): median **{np.median(gt_nearest):.3f} m**, within 50 cm **{np.sum(gt_nearest<=.5)}/{len(gt_nearest)}**; the remaining GT labels were not detected.", "",
             f"Physical outliers at gate {gate_m:.2f} m: {result['physical_outlier_pred_ids'] or 'none'}.",
             f"Physically correct but ID-inconsistent points: {result['id_mismatch_but_physical_match'] or 'none'}.", "",
             "## Interpretation for IDs 274, 629, 1982", "",
             "A point listed in `id_mismatch_but_physical_match` is physically near another GT position and therefore is not a geometric localization failure; its decoded ID/semantic association is wrong.", "",
             "See the CSV for every one-to-one assignment and the PNG for residual arrows (arrow length is physical position error)."]
    (out_dir / "unlabeled_marker_geometry.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pred-marker-points", required=True, type=Path)
    parser.add_argument("--gt-labels", required=True, type=Path)
    parser.add_argument("--marker-id", type=int, default=98)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--gate-m", type=float, default=.5)
    args = parser.parse_args()
    result = compare(args.pred_marker_points, args.gt_labels, args.out_dir, args.marker_id, args.gate_m)
    print(json.dumps({k: result[k] for k in ("counts", "rotation_deg", "physical_one_to_one", "prediction_to_gt_nearest", "gt_to_prediction_nearest_coverage", "physical_outlier_pred_ids", "id_mismatch_but_physical_match")}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
