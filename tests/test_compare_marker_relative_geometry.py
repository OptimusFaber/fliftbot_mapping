"""Tests for global-frame-invariant marker geometry comparison."""

import importlib.util
from pathlib import Path

import numpy as np


MODULE = Path(__file__).parents[1] / "scripts/evaluation/compare_marker_relative_geometry.py"
SPEC = importlib.util.spec_from_file_location("marker_relative_geometry", MODULE)
assert SPEC is not None and SPEC.loader is not None
M = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(M)


def _rot(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s], [s, c]])


def _write_inputs(tmp_path, pred, gt):
    import json

    pp = tmp_path / "pred.json"
    gg = tmp_path / "gt.json"
    pp.write_text(json.dumps({"markers": [{"marker_id": i, "point_xyz_m": [*p, 3.0]} for i, p in pred.items()]}))
    gg.write_text(json.dumps({"labels": [{"id": str(i), "gt_room_xyz_m": [*p, 3.0]} for i, p in gt.items()]}))
    return pp, gg


def test_translation_rotation_do_not_change_reference_distances(tmp_path):
    base = {98: np.array([2.0, 3.0]), 1: np.array([3.0, 3.0]), 2: np.array([2.0, 5.0]), 3: np.array([0.5, 4.0])}
    gt = {i: p.copy() for i, p in base.items()}
    pred = {i: p @ _rot(0.71).T + np.array([40.0, -12.0]) for i, p in base.items()}
    pp, gg = _write_inputs(tmp_path, pred, gt)
    out = M.compare(pp, gg, tmp_path / "out")
    assert out["distance_metrics"]["all"]["rmse_m"] < 1e-9
    assert out["rigid_procrustes_no_scale"]["proper_rotation"]["rmse_m"] < 1e-9


def test_no_scale_fit_reports_scale_error_and_missing_ids(tmp_path):
    gt = {98: np.array([0.0, 0.0]), 1: np.array([1.0, 0.0]), 2: np.array([0.0, 2.0]), 3: np.array([3.0, 1.0])}
    pred = {98: np.array([10.0, 10.0]), 1: np.array([12.0, 10.0]), 2: np.array([10.0, 14.0]), 4: np.array([9.0, 9.0])}
    pp, gg = _write_inputs(tmp_path, pred, gt)
    out = M.compare(pp, gg, tmp_path / "out")
    assert out["counts"]["common"] == 3
    assert out["missing_in_prediction"] == [3]
    assert out["missing_in_gt"] == [4]
    assert out["distance_metrics"]["all"]["mae_m"] > 0.1
    # Scaling is forbidden: the residual cannot be zero for both vectors.
    assert out["rigid_procrustes_no_scale"]["proper_rotation"]["rmse_m"] > 0.5
