#!/usr/bin/env python3
"""Compare marker geometry relative to one reference marker.

The comparison deliberately removes the global coordinate frame: all points
are translated so ``marker_id`` is the origin.  The primary measurements are
the distances from that marker, which are invariant to global translation and
rotation.  A secondary 2-D Procrustes fit reports the residual vectors after a
best rigid rotation (with and without allowing a reflection); scale is never
fitted.

The input formats are intentionally small and generic.  Prediction files are
the ``marker_points.json`` files produced by the ceiling marker pipeline.  GT
files may contain a top-level ``labels`` list with ``id`` and
``gt_room_xyz_m`` (the real-room authoritative format), or a plain list of
records with ``id`` and ``xyz_m``/``point_xyz_m``.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _as_id(value: Any) -> int:
    return int(str(value))


def _xy(value: Any) -> np.ndarray:
    a = np.asarray(value, dtype=float).reshape(-1)
    if a.size < 2:
        raise ValueError(f"A marker coordinate must have at least x/y: {value!r}")
    return a[:2]


def load_prediction(path: Path) -> dict[int, np.ndarray]:
    data = json.loads(path.read_text(encoding="utf-8"))
    records = data.get("markers", data) if isinstance(data, dict) else data
    if not isinstance(records, list):
        raise ValueError(f"Prediction file has no marker list: {path}")
    result: dict[int, np.ndarray] = {}
    for row in records:
        if not isinstance(row, dict):
            continue
        raw_id = row.get("marker_id", row.get("id", row.get("code")))
        point = row.get("point_xyz_m", row.get("xyz_m", row.get("point")))
        if raw_id is None or point is None:
            continue
        result[_as_id(raw_id)] = _xy(point)
    if not result:
        raise ValueError(f"No marker coordinates found in {path}")
    return result


def load_gt(path: Path) -> tuple[dict[int, np.ndarray], dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    records: Any = data.get("labels", data) if isinstance(data, dict) else data
    if isinstance(records, dict):
        # Support labels2coords.json: this format contains encoder coordinates,
        # not room coordinates.  Refuse it explicitly rather than pretending
        # that encoder axes are the room axes.
        if records and all(isinstance(v, dict) for v in records.values()):
            raise ValueError(
                "GT mapping has no room-frame coordinates; use "
                "ceiling_labels_room_real_gt.json or a file with xyz_m"
            )
        records = list(records.values())
    if not isinstance(records, list):
        raise ValueError(f"GT file has no label list: {path}")
    result: dict[int, np.ndarray] = {}
    for row in records:
        if not isinstance(row, dict):
            continue
        raw_id = row.get("id", row.get("marker_id", row.get("code")))
        point = row.get("gt_room_xyz_m", row.get("xyz_m", row.get("point_xyz_m", row.get("point"))))
        if raw_id is None or point is None:
            continue
        result[_as_id(raw_id)] = _xy(point)
    if not result:
        raise ValueError(f"No room-frame GT coordinates found in {path}")
    metadata = {}
    if isinstance(data, dict):
        for key in ("target_frame", "transform", "room_size_m", "origin", "source_labels"):
            if key in data:
                metadata[key] = data[key]
    return result, metadata


def _fit_linear(a: np.ndarray, b: np.ndarray, reflection: bool) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fit row-vector transform b ~= a @ r.T + t, no scale."""
    ca, cb = a.mean(axis=0), b.mean(axis=0)
    aa, bb = a - ca, b - cb
    u, _, vt = np.linalg.svd(aa.T @ bb)
    r = vt.T @ u.T
    if reflection and np.linalg.det(r) > 0:
        vt[-1] *= -1.0
        r = vt.T @ u.T
    if not reflection and np.linalg.det(r) < 0:
        vt[-1] *= -1.0
        r = vt.T @ u.T
    t = cb - ca @ r.T
    fitted = a @ r.T + t
    return r, t, fitted


def _fit_about_reference(a: np.ndarray, b: np.ndarray, reflection: bool) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Rigid fit with marker 98 fixed at the origin.

    The arrays are already reference-centered.  Keeping the origin fixed is
    what makes this a comparison of the arrangement around marker 98, rather
    than a second arbitrary translation of the marker constellation.
    """
    u, _, vt = np.linalg.svd(a.T @ b)
    r = vt.T @ u.T
    if reflection and np.linalg.det(r) > 0:
        vt[-1] *= -1.0
        r = vt.T @ u.T
    if not reflection and np.linalg.det(r) < 0:
        vt[-1] *= -1.0
        r = vt.T @ u.T
    fitted = a @ r.T
    return r, np.zeros(2), fitted


def rotation_deg(r: np.ndarray) -> float:
    return float(math.degrees(math.atan2(r[1, 0], r[0, 0])))


def _metrics(errors: np.ndarray) -> dict[str, float | int]:
    if errors.size == 0:
        return {"count": 0, "mae_m": None, "rmse_m": None, "median_abs_m": None, "p95_abs_m": None, "bias_m": None}
    ae = np.abs(errors)
    return {
        "count": int(errors.size),
        "mae_m": float(np.mean(ae)),
        "rmse_m": float(np.sqrt(np.mean(errors**2))),
        "median_abs_m": float(np.median(ae)),
        "p95_abs_m": float(np.percentile(ae, 95)),
        "bias_m": float(np.mean(errors)),
    }


def _write_csv(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    rows = list(rows)
    fields = [
        "marker_id", "pred_rel_x_m", "pred_rel_y_m", "gt_rel_x_m", "gt_rel_y_m",
        "pred_distance_m", "gt_distance_m", "distance_error_m", "abs_distance_error_m",
        "proper_fit_x_m", "proper_fit_y_m", "proper_residual_m", "reflection_fit_x_m",
        "reflection_fit_y_m", "reflection_residual_m", "flag",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _plot(path: Path, ids: list[int], pred: np.ndarray, gt: np.ndarray, fitted: np.ndarray,
          ref_id: int, proper_deg: float, reflection_deg: float, metrics: dict[str, Any],
          title_suffix: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(17, 8), constrained_layout=True)
    colors = plt.cm.tab20(np.linspace(0, 1, max(len(ids), 1)))
    for ax, pred_view, title in (
        (axes[0], pred, "Centered at reference (raw axes)"),
        (axes[1], fitted, "Prediction rigidly aligned to GT (no scale)"),
    ):
        all_xy = np.vstack([pred_view, gt, np.zeros((1, 2))])
        radius = max(1.0, float(np.max(np.linalg.norm(all_xy, axis=1))) * 1.18)
        for color, mid, p, g in zip(colors, ids, pred_view, gt):
            ax.plot([0, p[0]], [0, p[1]], color=color, alpha=0.25, linewidth=0.8)
            ax.scatter(*p, color=color, s=32, marker="o", zorder=3)
            ax.scatter(*g, facecolors="none", edgecolors=color, s=58, marker="s", zorder=3)
            ax.text(p[0], p[1], f" {mid}", fontsize=7, color="tab:blue")
            ax.text(g[0], g[1], f" {mid}", fontsize=7, color="tab:orange")
        ax.scatter([0], [0], color="black", marker="*", s=150, zorder=5, label=f"reference {ref_id}")
        ax.axhline(0, color="0.85", linewidth=0.7)
        ax.axvline(0, color="0.85", linewidth=0.7)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlim(-radius, radius); ax.set_ylim(-radius, radius)
        ax.set_xlabel("relative X [m]"); ax.set_ylabel("relative Y [m]")
        ax.set_title(title)
        ax.grid(True, alpha=0.2)
        ax.legend(loc="best", fontsize=8)
    fig.suptitle(
        f"Marker geometry relative to {ref_id} — {title_suffix}\n"
        f"distance error: MAE {metrics['all']['mae_m']:.3f} m, "
        f"RMSE {metrics['all']['rmse_m']:.3f} m, p95 {metrics['all']['p95_abs_m']:.3f} m; "
        f"proper rotation {proper_deg:.2f}°; reflection {reflection_deg:.2f}°",
        fontsize=12,
    )
    fig.text(0.5, 0.01, "filled circles = prediction; hollow squares = GT; colored rays = prediction from marker 98", ha="center", fontsize=9)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def compare(pred_path: Path, gt_path: Path, out_dir: Path, ref_id: int = 98,
            outlier_threshold_m: float = 0.25) -> dict[str, Any]:
    pred, pred_meta = load_prediction(pred_path), {}
    gt, gt_meta = load_gt(gt_path)
    if ref_id not in pred:
        raise ValueError(f"Reference marker {ref_id} is absent from prediction: {pred_path}")
    if ref_id not in gt:
        raise ValueError(f"Reference marker {ref_id} is absent from GT: {gt_path}")
    common = sorted(set(pred) & set(gt))
    if len(common) < 2:
        raise ValueError("Need reference plus at least one common marker")
    missing_in_pred = sorted(set(gt) - set(pred))
    missing_in_gt = sorted(set(pred) - set(gt))
    nonref = [i for i in common if i != ref_id]
    p0, g0 = pred[ref_id], gt[ref_id]
    pred_rel = np.asarray([pred[i] - p0 for i in nonref])
    gt_rel = np.asarray([gt[i] - g0 for i in nonref])
    pred_dist = np.linalg.norm(pred_rel, axis=1)
    gt_dist = np.linalg.norm(gt_rel, axis=1)
    distance_errors = pred_dist - gt_dist
    suspicious = np.abs(distance_errors) > outlier_threshold_m
    inlier_metrics = _metrics(distance_errors[~suspicious])
    all_metrics = _metrics(distance_errors)
    # Estimate orientation from the distance-consistent correspondences.  A
    # wrong marker identity must not rotate the whole constellation, but it is
    # retained in the per-marker residuals and in all-distance metrics.
    fit_mask = ~suspicious if int((~suspicious).sum()) >= 2 else np.ones(len(nonref), dtype=bool)
    proper_all_r, proper_all_t, proper_all_fit = _fit_about_reference(pred_rel, gt_rel, reflection=False)
    reflect_all_r, reflect_all_t, reflect_all_fit = _fit_about_reference(pred_rel, gt_rel, reflection=True)
    proper_r, proper_t, proper_fit_in = _fit_about_reference(pred_rel[fit_mask], gt_rel[fit_mask], reflection=False)
    reflect_r, reflect_t, reflect_fit_in = _fit_about_reference(pred_rel[fit_mask], gt_rel[fit_mask], reflection=True)
    proper_fit = pred_rel @ proper_r.T
    reflect_fit = pred_rel @ reflect_r.T
    proper_res = np.linalg.norm(proper_fit - gt_rel, axis=1)
    reflect_res = np.linalg.norm(reflect_fit - gt_rel, axis=1)
    rows = []
    for i, p, g, pd, gd, de, pf, pr, rf, rr, bad in zip(
        nonref, pred_rel, gt_rel, pred_dist, gt_dist, distance_errors,
        proper_fit, proper_res, reflect_fit, reflect_res, suspicious,
    ):
        rows.append({
            "marker_id": i, "pred_rel_x_m": f"{p[0]:.6f}", "pred_rel_y_m": f"{p[1]:.6f}",
            "gt_rel_x_m": f"{g[0]:.6f}", "gt_rel_y_m": f"{g[1]:.6f}",
            "pred_distance_m": f"{pd:.6f}", "gt_distance_m": f"{gd:.6f}",
            "distance_error_m": f"{de:.6f}", "abs_distance_error_m": f"{abs(de):.6f}",
            "proper_fit_x_m": f"{pf[0]:.6f}", "proper_fit_y_m": f"{pf[1]:.6f}",
            "proper_residual_m": f"{pr:.6f}", "reflection_fit_x_m": f"{rf[0]:.6f}",
            "reflection_fit_y_m": f"{rf[1]:.6f}", "reflection_residual_m": f"{rr:.6f}",
            "flag": "suspicious_distance" if bad else "ok",
        })
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(out_dir / "marker_relative_geometry.csv", rows)
    proper_all_res = np.linalg.norm(proper_all_fit - gt_rel, axis=1)
    reflect_all_res = np.linalg.norm(reflect_all_fit - gt_rel, axis=1)
    proper_rmse = float(np.sqrt(np.mean(proper_res[fit_mask]**2)))
    reflect_rmse = float(np.sqrt(np.mean(reflect_res[fit_mask]**2)))
    result: dict[str, Any] = {
        "schema": "fliftbot.marker_relative_geometry.v1",
        "inputs": {"pred_marker_points": str(pred_path), "gt_labels": str(gt_path), "reference_marker_id": ref_id},
        "coordinate_contract": "all relative vectors are marker_i - marker_reference in each input frame; no global pose is used",
        "gt_metadata": gt_meta,
        "counts": {"prediction": len(pred), "gt": len(gt), "common": len(common), "compared_nonreference": len(nonref), "missing_in_prediction": len(missing_in_pred), "missing_in_gt": len(missing_in_gt)},
        "missing_in_prediction": missing_in_pred,
        "missing_in_gt": missing_in_gt,
        "reference": {"marker_id": ref_id, "pred_xy_m": p0.tolist(), "gt_xy_m": g0.tolist()},
        "distance_metrics": {"all": all_metrics, "inliers_abs_error_le_threshold": {"threshold_m": outlier_threshold_m, **inlier_metrics}},
        "rigid_procrustes_no_scale": {
            "fit_marker_ids": [i for i, keep in zip(nonref, fit_mask) if keep],
            "all_common": {
                "proper_rotation_deg": rotation_deg(proper_all_r), "proper_rmse_m": float(np.sqrt(np.mean(proper_all_res**2))),
                "reflection_rotation_deg": rotation_deg(reflect_all_r), "reflection_rmse_m": float(np.sqrt(np.mean(reflect_all_res**2))),
            },
            "proper_rotation": {"determinant": float(np.linalg.det(proper_r)), "rotation_deg": rotation_deg(proper_r), "translation_m": proper_t.tolist(), "rmse_m": proper_rmse, "p95_m": float(np.percentile(proper_res[fit_mask], 95))},
            "reflection_allowed": {"determinant": float(np.linalg.det(reflect_r)), "rotation_deg": rotation_deg(reflect_r), "translation_m": reflect_t.tolist(), "rmse_m": reflect_rmse, "p95_m": float(np.percentile(reflect_res[fit_mask], 95))},
            "selected": "reflection_allowed" if reflect_rmse < proper_rmse else "proper_rotation",
        },
        "identity_warning": {
            "description": "Large error in distance from the reference may indicate a wrong marker identity or GT semantic mismatch; it is not removed from all-metrics.",
            "threshold_m": outlier_threshold_m,
            "suspicious_marker_ids": [i for i, bad in zip(nonref, suspicious) if bad],
        },
    }
    (out_dir / "marker_relative_geometry.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    _plot(out_dir / "marker_relative_geometry.png", nonref, pred_rel, gt_rel, proper_fit, ref_id,
          rotation_deg(proper_r), rotation_deg(reflect_r), result["distance_metrics"], out_dir.name)
    _write_report(out_dir / "marker_relative_geometry.md", result, rows)
    return result


def _fmt(value: Any) -> str:
    return "n/a" if value is None else f"{float(value):.3f}"


def _write_report(path: Path, result: dict[str, Any], rows: list[dict[str, Any]]) -> None:
    all_m = result["distance_metrics"]["all"]
    in_m = result["distance_metrics"]["inliers_abs_error_le_threshold"]
    proper = result["rigid_procrustes_no_scale"]["proper_rotation"]
    refl = result["rigid_procrustes_no_scale"]["reflection_allowed"]
    lines = [
        "# Relative marker geometry report", "",
        f"Reference marker: **{result['reference']['marker_id']}**. All vectors are centered on it; no global translation, map origin, GT alignment, or scale fitting is used.", "",
        f"Compared **{result['counts']['compared_nonreference']}** non-reference markers; prediction has {result['counts']['prediction']}, GT has {result['counts']['gt']}, common IDs {result['counts']['common']}.", "",
        f"GT frame: `{result['gt_metadata'].get('target_frame', 'unspecified')}`. Only GT x/y coordinates are compared; GT z/ceiling height is not used.", "",
        "## Distance from reference", "",
        f"All common markers: MAE **{_fmt(all_m['mae_m'])} m**, RMSE **{_fmt(all_m['rmse_m'])} m**, median absolute **{_fmt(all_m['median_abs_m'])} m**, p95 **{_fmt(all_m['p95_abs_m'])} m**, bias **{_fmt(all_m['bias_m'])} m**.", "",
        f"Inliers with |distance error| ≤ {result['distance_metrics']['inliers_abs_error_le_threshold']['threshold_m']:.2f} m: MAE **{_fmt(in_m['mae_m'])} m**, RMSE **{_fmt(in_m['rmse_m'])} m**, p95 **{_fmt(in_m['p95_abs_m'])} m**.", "",
        "These distance values are invariant to global translation and rotation. A large individual error is reported as a possible identity/semantic mismatch, not silently discarded.", "",
        "## Rigid comparison of relative vectors", "",
        f"Proper rotation only, fitted on the distance-consistent IDs in `fit_marker_ids`: {proper['rotation_deg']:.3f}°, RMSE **{proper['rmse_m']:.3f} m**, p95 **{proper['p95_m']:.3f} m**. The all-ID fit is retained in JSON for audit but is intentionally not used to orient the plot.", "",
        f"Reflection allowed: determinant {refl['determinant']:.0f}, RMSE **{refl['rmse_m']:.3f} m**, p95 **{refl['p95_m']:.3f} m**. Selected: **{result['rigid_procrustes_no_scale']['selected']}**.", "",
        f"Suspicious IDs by distance threshold: {result['identity_warning']['suspicious_marker_ids'] or 'none'}.", "",
        f"Missing in prediction: {result['missing_in_prediction'] or 'none'}.", "",
        f"Missing in GT: {result['missing_in_gt'] or 'none'}.", "",
        "## Per-marker table", "",
        "| ID | pred distance (m) | GT distance (m) | error (m) | proper residual (m) | flag |",
        "|---:|---:|---:|---:|---:|:---|",
    ]
    for row in rows:
        lines.append(f"| {row['marker_id']} | {row['pred_distance_m']} | {row['gt_distance_m']} | {row['distance_error_m']} | {row['proper_residual_m']} | {row['flag']} |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pred-marker-points", required=True, type=Path)
    parser.add_argument("--gt-labels", required=True, type=Path)
    parser.add_argument("--marker-id", type=int, default=98)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--outlier-threshold-m", type=float, default=0.25)
    args = parser.parse_args()
    result = compare(args.pred_marker_points, args.gt_labels, args.out_dir, args.marker_id, args.outlier_threshold_m)
    print(json.dumps({"out_dir": str(args.out_dir), "distance_metrics": result["distance_metrics"], "suspicious_marker_ids": result["identity_warning"]["suspicious_marker_ids"]}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
