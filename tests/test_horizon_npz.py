#!/usr/bin/env python3
"""Check horizon npz layout vs mapping loader (no ROS required)."""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

from flifltbot.common.runtime.npz_log import (  # noqa: E402
    extract_horizon_from_xyz,
    point_times_from_azimuth,
    point_times_from_gazebo_layout,
    point_times_from_cloud,
    horizon_npz_arrays,
    laserscan_to_horizon,
)
from flifltbot.mapping.io_frames import load_horizon_npz_frame, list_run_frame_paths  # noqa: E402


def _pandar_like_cloud():
    """40×900 organized cloud: horizon row 24 has a 5 m wall, others 8 m."""
    height, width = 40, 900
    vmin, vmax = np.radians(-25.0), np.radians(15.0)
    elev = vmin + np.arange(height) * (vmax - vmin) / (height - 1)
    az = np.linspace(-np.pi, np.pi, width, endpoint=False)
    xs = np.zeros((height, width), dtype=np.float32)
    ys = np.zeros((height, width), dtype=np.float32)
    zs = np.zeros((height, width), dtype=np.float32)
    for row, el in enumerate(elev):
        r = 5.0 if row == 24 else 8.0
        rh = r * np.cos(el)
        xs[row] = rh * np.cos(az)
        ys[row] = rh * np.sin(az)
        zs[row] = r * np.sin(el)
    return xs.reshape(-1), ys.reshape(-1), zs.reshape(-1), height, width, elev


def test_extract_horizon_row():
    x, y, z, height, width, elev = _pandar_like_cloud()
    xy, xyz, inten, times, channel = extract_horizon_from_xyz(
        x, y, z, height=height, width=width, stamp_t=10.0,
        min_range=0.3, max_range=200.0,
    )
    assert channel == 25, channel  # row 24 → 1-based
    assert xy.shape[0] == 900, xy.shape
    r = np.hypot(xy[:, 0], xy[:, 1])
    # horizontal range of the 5 m slant range at elev≈−0.4°
    assert abs(float(np.median(r)) - 5.0 * np.cos(elev[24])) < 0.05, float(np.median(r))
    assert abs(float(np.median(xyz[:, 2])) - 5.0 * np.sin(elev[24])) < 0.05
    print(f'OK extract: ch={channel} n={len(xy)} r={float(np.median(r)):.3f} m')


def test_hesai_point_timestamps_are_preserved():
    n = 40
    cloud = np.zeros(n, dtype=[
        ('x', '<f4'), ('y', '<f4'), ('z', '<f4'),
        ('timestamp', '<f8'),
    ])
    cloud['x'] = 2.0
    expected = 1_700_000_000.0 + np.arange(n) * 1e-6
    cloud['timestamp'] = expected
    times, source = point_times_from_cloud(cloud, expected[0])
    assert source == 'timestamp'
    assert np.array_equal(times, expected)
    _, _, _, selected, _ = extract_horizon_from_xyz(
        cloud['x'], cloud['y'], cloud['z'], point_times=times,
        stamp_t=expected[0], min_range=0.3, max_range=200.0,
    )
    assert np.array_equal(selected, expected)
    print('OK Hesai per-point timestamps preserved')


def test_gazebo_azimuth_timestamps():
    width, height = 900, 40
    az = np.linspace(-np.pi, np.pi, width, endpoint=False)
    x = np.tile(np.cos(az), height)
    y = np.tile(np.sin(az), height)
    for hz in (5.0, 10.0, 15.0, 20.0):
        period = 1.0 / hz
        times = point_times_from_gazebo_layout(
            len(x), height, width, 123.0, period)
        grid = times.reshape(height, width)
        assert np.allclose(grid, grid[0])
        assert abs(float(grid[0, 0]) - 123.0) < 1e-12
        expected_last = 123.0 + period * (width - 1) / width
        assert abs(float(grid[0, -1]) - expected_last) < 1e-10
        assert np.all(np.diff(grid[0]) > 0)
    print('OK Gazebo azimuth timestamps: 5/10/15/20 Hz')


def test_roundtrip_mapping_loader():
    x, y, z, height, width, _elev = _pandar_like_cloud()
    xy, xyz, inten, times, channel = extract_horizon_from_xyz(
        x, y, z, height=height, width=width, stamp_t=123.456,
        min_range=0.3, max_range=200.0,
    )
    with tempfile.TemporaryDirectory() as tmp:
        run = Path(tmp)
        horizon = run / 'horizon'
        horizon.mkdir()
        path = horizon / '00001_1970-01-01T00-02-03.456000+00-00.npz'
        np.savez(path, **horizon_npz_arrays(
            xy, xyz, inten, times, 123.456, 1, channel))
        files = list_run_frame_paths(run)
        assert files == [path], files
        loaded = load_horizon_npz_frame(path, min_points=200)
        assert loaded is not None
        points, stamp, _ptimes, cams, meta = loaded
        assert cams == ['horizon']
        assert meta['source'] == 'horizon_npz'
        assert abs(stamp - 123.456) < 1e-9
        assert len(points) >= 200
        print(f'OK mapping load: n={len(points)} cameras={cams} ch={meta["horizon_channel"]}')


def test_laserscan_2d():
    n = 500
    angles = np.linspace(-np.pi, np.pi, n, endpoint=False)
    ranges = np.full(n, 4.0)
    ranges[::7] = np.inf
    xy, xyz, inten, times, channel = laserscan_to_horizon(
        ranges, float(angles[0]), float(angles[1] - angles[0]),
        0.05, 12.0, stamp_t=1.0, scan_time=0.1,
    )
    assert channel == 0
    assert xy.shape[0] == int(np.isfinite(ranges).sum())
    assert float(np.median(np.hypot(xy[:, 0], xy[:, 1]))) - 4.0 < 0.02
    print(f'OK laserscan: n={len(xy)}')


def test_existing_sim_json_as_npz():
    """Prove mapping can read npz built from a recorded sim up-scan."""
    import json
    src = (
        ROOT.parent / 'ros-data/recordings/sim/20260814/warehouse_pandar_run_110539'
        / '01001_1970-01-01T00-02-37.300000+00-00.json'
    )
    if not src.is_file():
        print('SKIP existing json (file missing)')
        return
    payload = json.loads(src.read_text())
    up = payload['cameras']['up']['local_points']
    pts = np.asarray(up, dtype=np.float32)
    assert len(pts) > 200, len(pts)
    xyz = np.column_stack((pts, np.zeros(len(pts), dtype=np.float32)))
    times = np.asarray(payload['cameras']['up']['point_times'], dtype=np.float64)
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / '00001_sim.npz'
        np.savez(path, **horizon_npz_arrays(
            pts, xyz, np.zeros(len(pts), np.float32), times,
            float(payload['t']), 1, 0))
        loaded = load_horizon_npz_frame(path, min_points=200)
        assert loaded is not None
        points, stamp, _, cams, _meta = loaded
        print(f'OK sim-json→npz: n={len(points)} t={stamp:.1f} cameras={cams}')


if __name__ == '__main__':
    test_extract_horizon_row()
    test_hesai_point_timestamps_are_preserved()
    test_gazebo_azimuth_timestamps()
    test_roundtrip_mapping_loader()
    test_laserscan_2d()
    test_existing_sim_json_as_npz()
    print('all npz horizon checks passed')
