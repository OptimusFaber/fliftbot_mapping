#!/usr/bin/env python3
"""HTTP LiDAR clock-domain checks (no robot or ROS required)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

from flifltbot.mapping.pipeline import normalize_http_lidar_timestamps  # noqa: E402


def test_boot_relative_timestamps_are_anchored_to_host_clock():
    clouds = {
        'left': {
            't': 5157.2,
            'point_times': np.array([5157.19, 5157.21]),
        },
        'right': {
            't': 5157.3,
            'point_times': np.array([5157.29, np.nan]),
        },
    }
    offset, domain = normalize_http_lidar_timestamps(
        clouds, host_time_s=1_700_000_000.0
    )
    assert domain == 'monotonic_host_anchored'
    assert np.isclose(offset, 1_699_994_842.75)
    assert np.isclose(clouds['left']['t'], 1_700_000_000.0)
    assert np.allclose(clouds['left']['point_times'], [1_699_999_999.99, 1_700_000_000.01])
    assert np.isclose(clouds['right']['t'], 1_700_000_000.05)
    assert np.isnan(clouds['right']['point_times'][1])


def test_unix_timestamps_are_not_shifted():
    clouds = {'left': {'t': 1_700_000_000.0, 'point_times': np.array([1_700_000_000.0])}}
    offset, domain = normalize_http_lidar_timestamps(
        clouds, host_time_s=1_700_000_100.0
    )
    assert offset is None
    assert domain == 'unix'
    assert clouds['left']['t'] == 1_700_000_000.0
