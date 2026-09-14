#!/usr/bin/env python3
"""Offline-only marker overlay quality report.

This file is deliberately separate from the production overlay.  It may use
room GT to quantify an output, but it never writes runtime calibration and it
is never imported by the runtime pipeline.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


def load_gt(path: Path) -> dict[str, np.ndarray]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    labels = payload.get("labels", payload if isinstance(payload, list) else [])
    result = {}
    for label in labels:
        xyz = label.get("xyz_m") or label.get("gt_room_xyz_m")
        if xyz is not None and len(xyz) >= 2:
            result[str(label["id"])] = np.asarray(xyz[:2], dtype=float)
    if not result:
        raise ValueError(f"No GT labels in {path}")
    return result


def load_points(path: Path) -> dict[str, np.ndarray]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    result = {}
    for marker in payload.get("markers", []):
        marker_id = marker.get("marker_id")
        point = marker.get("point_xyz_m")
        if marker_id is not None and point is not None:
            result[str(marker_id)] = np.asarray(point[:2], dtype=float)
        elif marker.get("code") is not None and marker.get("x") is not None:
            result[str(marker["code"])] = np.asarray([marker["x"], marker["y"]], dtype=float)
    return result


def fit_se2(pred: dict[str, np.ndarray], gt: dict[str, np.ndarray]) -> dict[str, Any]:
    keys = sorted(set(pred).intersection(gt), key=lambda value: int(value) if value.isdigit() else value)
    if len(keys) < 2:
        return {"matched": len(keys), "error": "fewer than two common IDs"}
    a = np.asarray([pred[key] for key in keys], dtype=float)
    b = np.asarray([gt[key] for key in keys], dtype=float)
    ca, cb = a.mean(axis=0), b.mean(axis=0)
    u, _, vt = np.linalg.svd((a - ca).T @ (b - cb))
    r = vt.T @ u.T
    if np.linalg.det(r) < 0:
        vt[-1] *= -1.0
        r = vt.T @ u.T
    t = cb - r @ ca
    errors = np.linalg.norm(a @ r.T + t - b, axis=1)
    # The robust line is diagnostic only: it exposes outliers without hiding
    # the all-matched metric used for the main comparison.
    robust = errors <= max(0.5, 3.0 * float(np.median(errors)))
    return {
        "matched": len(keys), "gt_total": len(gt), "coverage": len(keys) / max(len(gt), 1),
        "rmse_m": float(np.sqrt(np.mean(errors ** 2))), "median_m": float(np.median(errors)),
        "p95_m": float(np.percentile(errors, 95)), "max_m": float(np.max(errors)),
        "under_0_5m": int(np.count_nonzero(errors < 0.5)), "under_1m": int(np.count_nonzero(errors < 1.0)),
        "robust_inliers": int(np.count_nonzero(robust)), "robust_median_m": float(np.median(errors[robust])) if robust.any() else None,
        "se2_rotation_rad": float(math.atan2(r[1, 0], r[0, 0])), "se2_translation_m": [float(v) for v in t],
        "note": "best rigid SE(2) is offline evaluation only; it is not fed back into production runtime",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gt", type=Path, required=True)
    parser.add_argument("--after", type=Path, required=True)
    parser.add_argument("--before", type=Path)
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    gt = load_gt(args.gt)
    outputs = {"after": fit_se2(load_points(args.after), gt)}
    if args.before:
        outputs["before"] = fit_se2(load_points(args.before), gt)
    if args.baseline:
        outputs["baseline_20260903"] = fit_se2(load_points(args.baseline), gt)
    report = {"schema": "fliftbot.marker_overlay.offline_quality.v1", "runtime_optimization": False, "gt": str(args.gt.resolve()), "outputs": outputs}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
