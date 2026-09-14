import importlib.util
from pathlib import Path

import numpy as np


MODULE_PATH = Path(__file__).parents[1] / "scripts/evaluation/overlay_marker_rays_on_cartographer_map.py"
SPEC = importlib.util.spec_from_file_location("marker_overlay", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
marker_overlay = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(marker_overlay)


def test_legacy_bbox_is_converted_to_camera_xy():
    pixel, source = marker_overlay.detection_center(
        {"tlbr": [100.0, 200.0, 140.0, 260.0], "center": [230.0, 120.0]},
        1920,
        1080,
    )
    assert np.allclose(pixel, [230.0, 120.0])
    assert "verified" in source


def test_frame_timestamp_csv_uses_frame_index(tmp_path):
    observations = {"data": [{}, {}, {}]}
    path = tmp_path / "frame_timestamps.csv"
    path.write_text(
        "frame_index,time_take_img_sec\n2,12.0\n0,10.0\n1,11.0\n",
        encoding="utf-8",
    )
    values = marker_overlay.load_frame_times(observations, tmp_path / "observations.json", path)
    assert np.allclose(values, [10.0, 11.0, 12.0])


def test_map_yaml_origin_transform_is_lower_left_grid():
    transform = marker_overlay.map_transform(
        {"origin": [-10.0, -5.0, 0.0], "resolution": 0.05},
        (100, 200),
    )
    assert np.allclose(marker_overlay.map_to_grid([0.0, 0.0], transform), [10.0, 5.0])


def _ray_row(origin, direction):
    return {"camera_xyz_m": list(origin), "ray_map_unit": list(direction)}


def test_robust_multiray_triangulation_rejects_outlier():
    point = np.array([2.0, -1.0, 3.0])
    rows = []
    for origin in [
        np.array([0.0, 0.0, 1.0]),
        np.array([1.0, 0.0, 1.0]),
        np.array([0.0, 1.0, 1.0]),
        np.array([1.0, 1.0, 1.0]),
        np.array([2.0, 0.0, 1.0]),
    ]:
        direction = point - origin
        direction /= np.linalg.norm(direction)
        rows.append(_ray_row(origin, direction))
    rows.append(_ray_row(np.array([0.0, 0.0, 1.0]), np.array([0.0, 0.0, 1.0])))
    result = marker_overlay.triangulate_lines(rows, 0.05)
    assert result is not None
    assert np.linalg.norm(np.asarray(result["point_xyz_m"]) - point) < 1e-6
    assert len(result["inlier_indices"]) == 5


def test_track_switching_codes_merge_only_when_geometry_agrees():
    hypotheses = [
        {"code": 10, "track_ids": ["1"], "point_xyz_m": [1.0, 2.0, 3.0]},
        {"code": 11, "track_ids": ["1"], "point_xyz_m": [1.1, 2.0, 3.0]},
        {"code": 12, "track_ids": ["1"], "point_xyz_m": [4.0, 2.0, 3.0]},
    ]
    groups = marker_overlay._component_hypotheses(hypotheses, 0.2)
    assert sorted(len(group) for group in groups) == [1, 2]
