"""Regression tests for the standalone runner's timing/frame contract."""

from pathlib import Path
import json

import numpy as np


def _azimuth_times(xy, stamp, period, direction=-1.0):
    """Reference implementation of the deterministic RPLIDAR synthesis."""
    angles = np.unwrap(np.arctan2(xy[:, 1], xy[:, 0]))
    phase = direction * (angles - angles[0]) / (2.0 * np.pi)
    phase = np.clip(phase, 0.0, 1.0)
    phase = np.maximum.accumulate(phase)
    return stamp - period + phase * period


def test_azimuth_synthesis_is_monotonic_and_end_stamped():
    xy = np.column_stack((
        np.cos(np.linspace(np.pi, -np.pi, 360, endpoint=False)),
        np.sin(np.linspace(np.pi, -np.pi, 360, endpoint=False)),
    ))
    times = _azimuth_times(xy, 100.0, 0.1)
    assert np.all(np.diff(times) >= 0.0)
    np.testing.assert_allclose(times[0], 99.9)
    assert times[-1] <= 100.0
    assert times[-1] > 99.99


def test_representative_recording_timestamp_contracts():
    root = Path(__file__).resolve().parents[2]
    rplidar = root / "ros-data/recordings/real/20260901/182826_rplidar_room/horizon"
    pandar = root / "ros-data/recordings/real/20260820/232828_pandar40p_room/horizon"
    if not rplidar.is_dir() or not pandar.is_dir():
        return
    with np.load(sorted(rplidar.glob("*.npz"))[0]) as data:
        assert np.unique(data["t"]).size == 1
    with np.load(sorted(pandar.glob("*.npz"))[0]) as data:
        times = np.asarray(data["t"], dtype=float)
        finite = times[np.isfinite(times)]
        assert len(np.unique(finite)) > 10
        assert np.all(np.diff(finite) >= 0.0)


def test_real_rplidar_speed_sign_override_is_recorded():
    root = Path(__file__).resolve().parents[2]
    config = root / "ros-data/recordings/real/20260901/182826_rplidar_room/lidar.config"
    if not config.is_file():
        return
    data = json.loads(config.read_text())
    assert data["lidar_type"] == "rplidar"
    assert data["encoder_speed_sign"] == 1.0
