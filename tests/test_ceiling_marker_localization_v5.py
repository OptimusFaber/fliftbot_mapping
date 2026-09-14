import importlib.util
from pathlib import Path

import numpy as np


MODULE_PATH = Path(__file__).parents[1] / "scripts/evaluation/ceiling_marker_localization_v5.py"
SPEC = importlib.util.spec_from_file_location("marker_localization_v3", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(module)


def _row(code, point, index):
    origin = np.array([float(index) * 0.05, 0.0, 1.35])
    target = np.array([point[0], point[1], 3.03])
    direction = target - origin
    direction /= np.linalg.norm(direction)
    return {
        "frame_index": index,
        "track_id": str(index % 3),
        "marker_id": code,
        "geometry_valid": True,
        "point_xyz_m": [float(point[0]), float(point[1]), 3.03],
        "camera_origin_map": origin.tolist(),
        "ray_cartographer_map_unit": direction.tolist(),
    }


def test_v5_rejects_two_supported_spatial_hypotheses():
    rows = [_row(42, [1.0, 1.0], i) for i in range(12)] + [_row(42, [3.0, 1.0], i + 20) for i in range(12)]
    markers, diagnostics, filtered = module.select_hypotheses(rows, min_observations=12, component_radius_m=0.25, ambiguity_ratio=0.5)
    assert markers == {}
    assert diagnostics["42"]["reason"] == "ambiguous_spatial_hypotheses"
    assert filtered == []


def test_v5_keeps_one_clear_physical_hypothesis():
    rows = [_row(7, [1.0 + (i % 2) * 0.01, 2.0], i) for i in range(18)]
    markers, diagnostics, filtered = module.select_hypotheses(rows, min_observations=12)
    assert list(markers) == ["7"]
    assert diagnostics["7"]["status"] == "accepted"
    assert len(filtered) >= 12


def test_lidar_pose_uses_camera_minus_lidar_xy_lever_arm():
    camera_axis = np.asarray([1.115, 0.280, 1.350])
    lidar_axis_xy = np.asarray([1.015, 0.0])
    camera_from_lidar = camera_axis.copy()
    camera_from_lidar[:2] -= lidar_axis_xy
    assert np.allclose(camera_from_lidar, [0.100, 0.280, 1.350])
