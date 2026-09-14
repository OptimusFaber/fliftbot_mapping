import importlib.util
from pathlib import Path

import numpy as np


MODULE_PATH = Path(__file__).parents[1] / "scripts/evaluation/_ceiling_marker_geometry.py"
SPEC = importlib.util.spec_from_file_location("marker_localization_v2", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(module)


def test_center_xy_and_legacy_tlbr_contract():
    pixel, source = module.detection_center(
        {"center": [230.0, 120.0], "tlbr": [100.0, 200.0, 140.0, 260.0]},
        1920,
        1080,
    )
    assert np.allclose(pixel, [230.0, 120.0])
    assert "verified" in source


def test_camera_position_cm_and_lateral_sign_are_applied_once(tmp_path):
    path = tmp_path / "camera_position.txt"
    path.write_text("111.5 -28.0 135.0\n0.1 0.2\n", encoding="utf-8")
    offset, meta = module.load_camera_position(path, -1)
    assert np.allclose(offset, [1.115, 0.28, 1.35])
    assert meta["raw_position_cm"] == [111.5, -28.0, 135.0]


def test_optical_to_cartographer_yaw_minus_pi_over_two():
    ray = np.array([1.0, 0.0, 1.0])
    transformed = module.rotation3(0.0, 0.0, -np.pi / 2.0) @ (ray / np.linalg.norm(ray))
    assert np.allclose(transformed[:2], [0.0, -1.0 / np.sqrt(2.0)], atol=1e-12)
    assert transformed[2] > 0.0


def test_map_frame_point_is_not_occupancy_grid_point():
    map_meta = {"origin": [-10.0, -5.0, 0.0], "resolution": 0.05}
    point = np.array([2.0, 3.0])
    grid = module.map_to_grid(point, map_meta)
    assert np.allclose(grid, [12.0, 8.0])
    assert not np.allclose(point, grid)
