"""Synthetic checks for result.ply wall geometry diagnostics.

These tests deliberately do not use real-room dimensions or DePHAN coordinates.
"""

import importlib.util
from pathlib import Path

import numpy as np

MODULE = Path(__file__).parents[1] / "scripts/evaluation/analyze_result_ply_wall_geometry.py"
SPEC = importlib.util.spec_from_file_location("result_wall_geometry", MODULE)
M = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(M)


def rectangle_cloud(width=8.0, height=3.0, thickness=0.16, seed=4):
    rng = np.random.default_rng(seed)
    pts = []
    # Two observable physical edges per wall, with central gaps and corners
    # intentionally absent from the fitted portions.
    for x in (0.0, thickness, width - thickness, width):
        y = np.linspace(0.35, height - 0.35, 180)
        keep = ~((y > 1.25) & (y < 1.65)) if x in (0.0, width) else np.ones(len(y), bool)
        pts.extend(np.c_[np.full(keep.sum(), x), y[keep]])
    for y in (0.0, thickness, height - thickness, height):
        x = np.linspace(0.35, width - 0.35, 260)
        keep = ~((x > 3.0) & (x < 3.5)) if y in (0.0, height) else np.ones(len(x), bool)
        pts.extend(np.c_[x[keep], np.full(keep.sum(), y)])
    # Interior clutter and outliers must not become room walls.
    pts.extend(rng.uniform([1.0, 0.5], [width - 1.0, height - 0.5], size=(120, 2)))
    a = np.asarray(pts) + rng.normal(0.0, 0.006, (len(pts), 2))
    return a


def test_two_physical_edges_and_median():
    a = rectangle_cloud()
    walls = {
        "left": M.fit_wall(a, 0, "low"),
        "right": M.fit_wall(a, 0, "high"),
        "bottom": M.fit_wall(a, 1, "low"),
        "top": M.fit_wall(a, 1, "high"),
    }
    for wall in walls.values():
        assert wall["inner"] is not None
        assert wall["outer"] is not None
        assert wall["median"] is not None
        assert abs(wall["median"]["offset_center"] - (wall["inner"]["offset_center"] + wall["outer"]["offset_center"]) / 2) < 1e-9


def test_missing_outer_edge_abstains():
    a = rectangle_cloud()[:, :]
    # Retain one edge for the right wall and clutter it with interior points.
    a = a[~((a[:, 0] > 7.92) & (a[:, 0] < 8.1))]
    w = M.fit_wall(a, 0, "high")
    assert w["inner"] is not None
    assert w["outer"] is None
    assert w.get("outer_unavailable_reason")


def test_rigid_partial_marker_subset():
    src = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [2.0, 1.0], [3.0, 2.0], [4.0, 0.2]])
    ids = [10, 11, 12, 13, 14, 15]
    ang = 0.47
    c, s = np.cos(ang), np.sin(ang)
    r = np.array([[c, -s], [s, c]])
    dst = src[[0, 2, 3, 5]] @ r.T + np.array([4.0, -2.0])
    marker_json = {"markers": [{"marker_id": i, "point_xyz_m": [*p, 3.0]} for i, p in zip(ids, src)]}
    marker, diag = M.recover_marker_correspondence(np.c_[dst, np.full(len(dst), .12)], marker_json, 13)
    assert marker is not None
    assert diag["partial_json_subset"]
    assert diag["marker_available_in_ply"]
    assert diag["correspondence_rmse_m"] < 1e-8


def test_rotated_translated_manhattan_frame():
    a = rectangle_cloud(width=7.0, height=2.8, seed=9)
    ang = 0.23
    c, s = np.cos(ang), np.sin(ang)
    r = np.array([[c, s], [-s, c]])
    b = a @ r.T + np.array([-3.0, 8.0])
    estimated = M.estimate_angle(b)
    # Sign is defined by rotate_xy; allow a few tenths of a degree for noise.
    assert abs(np.rad2deg(estimated) + np.rad2deg(ang)) < 4.0


def test_density_noise_and_resolution_variation():
    """Geometry remains usable when sampling density and noise change."""
    from scipy.spatial import cKDTree
    for stride, noise in ((1, 0.003), (2, 0.008), (3, 0.012)):
        a = rectangle_cloud(width=8.0, height=3.0, seed=20 + stride)[::stride]
        rng = np.random.default_rng(100 + stride)
        a = a + rng.normal(0.0, noise, a.shape)
        spacing = float(np.median(cKDTree(a).query(a, k=2)[0][:, 1]))
        w = M.fit_wall(a, 0, "low", seed_tol=max(0.01, 0.5 * spacing), inlier_tol=max(0.008, 0.4 * spacing))
        assert w["inner"] is not None
