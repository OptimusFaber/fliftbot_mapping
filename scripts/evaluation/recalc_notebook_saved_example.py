#!/usr/bin/env python3
"""Recalculate the saved seven-point output embedded in the colleague notebook.

This is an offline audit helper.  It parses the notebook's saved output,
reprojects those exact seven local intersections with the saved pose, and
evaluates them as an unlabeled point set.  A decoded code is not used to choose
the physical GT point, so a swapped code (notably 167/193) remains a valid
geometric detection if its centre is correct.
"""
from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

import numpy as np

from audit_colleague_ray_plane import compare_point_sets, load_gt


NUM = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"
LOCAL_RE = re.compile(
    rf"track/code\s+(\d+)/(\S+):.*?point=\[\s*({NUM})\s+({NUM})\s+({NUM})\s*\]"
)
POSE_RE = re.compile(rf"interpolated pose:\s*x=({NUM}),\s*y=({NUM}),\s*dgr=({NUM})")
OFFSET_RE = re.compile(r"HEADING_OFFSET_DEG\s*=\s*([+-]?[0-9.]+)")


def output_text(cell: dict) -> str:
    chunks = []
    for output in cell.get("outputs", []):
        if output.get("output_type") == "stream":
            text = output.get("text", "")
            chunks.append("".join(text) if isinstance(text, list) else str(text))
    return "\n".join(chunks)


def main() -> int:
    root = Path(__file__).resolve().parents[3]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--notebook", type=Path, default=root / "ray_plane_intersection_plot.ipynb")
    parser.add_argument(
        "--gt",
        type=Path,
        default=root / "ros-data/gt/real_room/real_room/labels2coords.json",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=root / "ros-data/results/real/20260914/colleague_notebook_audit/saved_example",
    )
    args = parser.parse_args()

    notebook = json.loads(args.notebook.read_text(encoding="utf-8"))
    # Cell 5 contains the local points printed by ray_plane_intersections().
    local_hits = {}
    for match in LOCAL_RE.finditer(output_text(notebook["cells"][5])):
        track_id, code, x, y, z = match.groups()
        try:
            code_value = int(code)
        except ValueError:
            code_value = code
        local_hits[code_value] = {
            "track_id": int(track_id),
            "local_xyz_m": [float(x), float(y), float(z)],
        }
    if len(local_hits) != 7:
        raise RuntimeError(f"expected 7 saved local hits in cell 5, got {len(local_hits)}")

    pose_match = POSE_RE.search(output_text(notebook["cells"][11]))
    if not pose_match:
        raise RuntimeError("saved interpolated pose not found in cell 11 output")
    x, y, dgr = map(float, pose_match.groups())
    source8 = "".join(notebook["cells"][8].get("source", []))
    offset_match = OFFSET_RE.search(source8)
    offset_deg = float(offset_match.group(1)) if offset_match else 0.0
    theta = math.radians(dgr + offset_deg)
    R = np.asarray([[math.cos(theta), -math.sin(theta)], [math.sin(theta), math.cos(theta)]])

    predicted = {}
    rows = []
    for code, item in sorted(local_hits.items(), key=lambda pair: pair[1]["track_id"]):
        local = np.asarray(item["local_xyz_m"], dtype=float)
        global_xy = R @ local[:2] + np.asarray([x, y])
        predicted[int(code)] = global_xy
        rows.append({
            "decoded_code": int(code),
            "track_id": item["track_id"],
            "local_xyz_m": item["local_xyz_m"],
            "predicted_xy_m": global_xy.tolist(),
        })

    gt = load_gt(args.gt)
    gt_ids = list(gt)
    gt_points = np.asarray([gt[k] for k in gt_ids])
    physical_rows = []
    for row in rows:
        point = np.asarray(row["predicted_xy_m"])
        distances = np.linalg.norm(gt_points - point, axis=1)
        nearest = int(np.argmin(distances))
        physical_rows.append({
            **row,
            "nearest_gt_id": gt_ids[nearest],
            "nearest_distance_m": float(distances[nearest]),
        })

    comparison = compare_point_sets(predicted, gt, reference=98)
    report = {
        "schema": "fliftbot.audit.colleague_notebook_saved_example.v2",
        "notebook": str(args.notebook.resolve()),
        "source_cells": {"local_intersections": 5, "global_pose_and_errors": 11, "heading_offset": 8},
        "gt": str(args.gt.resolve()),
        "evaluation": {
            "unlabeled_physical_matching": True,
            "one_to_one": True,
            "rotation": "saved pose heading only; no GT-fitted rotation",
            "translation": "none (saved notebook coordinates and labels2coords share frame)",
            "scale": "none",
            "reflection": "none",
        },
        "saved_pose": {"x_m": x, "y_m": y, "dgr": dgr, "heading_offset_deg": offset_deg},
        "points": physical_rows,
        "comparison": comparison,
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out = args.out_dir / "saved_example_unlabeled_audit.json"
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"out": str(out.resolve()), "points": physical_rows, "comparison": comparison}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
