import json
from pathlib import Path

import numpy as np

from scripts.evaluation.compare_unlabeled_marker_geometry import compare


def test_physical_assignment_survives_wrong_ids(tmp_path: Path) -> None:
    # GT is centered on the anchor 98.  Predictions are rotated and have two
    # deliberately swapped IDs; the physical point set must still be recovered.
    gt_points = {98: [0.0, 0.0], 10: [1.0, 0.0], 20: [0.0, 1.0], 30: [1.0, 1.0]}
    theta = np.deg2rad(37.0)
    r = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    pred_xy = {98: [0.0, 0.0]}
    for pid, gid in [(10, 20), (20, 10), (30, 30)]:
        # row-vector inverse of the GT->prediction rotation
        p = np.asarray(gt_points[gid]) @ r
        pred_xy[pid] = p.tolist()
    pred = {"markers": [{"marker_id": i, "point_xyz_m": [*p, 3.03]} for i, p in pred_xy.items()]}
    gt = {"labels": [{"id": i, "gt_room_xyz_m": [*p, 3.03]} for i, p in gt_points.items()]}
    pred_path, gt_path = tmp_path / "pred.json", tmp_path / "gt.json"
    pred_path.write_text(json.dumps(pred), encoding="utf-8")
    gt_path.write_text(json.dumps(gt), encoding="utf-8")
    result = compare(pred_path, gt_path, tmp_path / "out")
    assert result["physical_one_to_one"]["max_m"] < 1e-6
    assert set(result["id_mismatch_but_physical_match"]) == {10, 20}
