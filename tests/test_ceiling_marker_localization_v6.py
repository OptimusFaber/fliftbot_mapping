import importlib.util
import json
from pathlib import Path

import numpy as np


MODULE_PATH = Path(__file__).parents[1] / "scripts/evaluation/ceiling_marker_localization_v6.py"
SPEC = importlib.util.spec_from_file_location("marker_localization_v6", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(module)


def test_lidar_config_position_is_metric_and_explicit(tmp_path):
    path = tmp_path / "lidar.config"
    path.write_text(json.dumps({"format": "lidar.config.v1", "position_m": {"x": 1.29, "y": -0.32, "z": 2.03}}), encoding="utf-8")
    position, meta = module.read_lidar_offset(path)
    assert np.allclose(position, [1.29, -0.32, 2.03])
    assert meta["format"] == "lidar.config.v1"


def test_height_is_robust_to_two_bad_marker_fits():
    fits = [{"point_xyz_m": [float(i), 0.0, z], "inliers": 20} for i, z in enumerate([3.01, 3.03, 3.04, 3.08, 2.95, 5.5, 1.9])]
    estimate = module.robust_height(fits, 1.27, 0.5, 8.0)
    assert estimate is not None
    assert abs(estimate["ceiling_z_m"] - 3.03) < 0.05
    assert estimate["rejected_marker_fits"] >= 1


def test_height_abstains_without_independent_support():
    fits = [{"point_xyz_m": [0.0, 0.0, 3.0], "inliers": 20}, {"point_xyz_m": [1.0, 0.0, 3.1], "inliers": 20}]
    assert module.robust_height(fits, 1.27, 0.5, 8.0) is None


def test_height_estimator_is_scene_independent_for_multiple_ceiling_heights():
    for expected in (2.45, 3.03, 4.20, 5.80):
        fits = [{"point_xyz_m": [float(i), float(i % 3), expected + noise], "inliers": 18}
                for i, noise in enumerate((-.025, .0, .018, .031, -.012, .009, .022))]
        estimate = module.robust_height(fits, 1.27, 0.5, 8.0)
        assert estimate is not None
        assert abs(estimate["ceiling_z_m"] - expected) < 0.05


def test_multiray_fit_tolerates_mount_rotation_and_small_ray_noise():
    rng = np.random.default_rng(7)
    target = np.asarray([2.4, -1.1, 4.2])
    rows = []
    for i, origin in enumerate(np.asarray([
        [-1.2, 0.3, 1.1], [-0.4, -0.8, 1.1], [0.5, 0.9, 1.1],
        [1.4, -0.2, 1.1], [2.0, 1.1, 1.1], [3.1, 0.0, 1.1],
    ])):
        direction = target - origin
        direction /= np.linalg.norm(direction)
        direction += rng.normal(0.0, 2.0e-4, 3)
        direction /= np.linalg.norm(direction)
        rows.append({"marker_id": 17, "camera_origin_map": origin.tolist(),
                     "ray_cartographer_map_unit": direction.tolist(),
                     "frame_index": i, "timestamp_s": float(i), "track_id": str(i % 2)})
    fit = module.triangulate_code(rows, residual_gate=0.20, min_inliers=4)
    assert fit is not None
    assert abs(fit["point_xyz_m"][2] - target[2]) < 0.05
