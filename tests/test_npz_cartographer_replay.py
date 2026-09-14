"""Input/geometry tests for the direct NPZ Cartographer replay adapter."""

import math
from pathlib import Path

import numpy as np

from flifltbot.common.runtime.cartographer_replay import (
    ReplayFrame,
    build_wheel_poses,
    interpolate_encoder_series,
    scan_geometry,
)


def _frame(stamp):
    angles = np.linspace(-math.pi, math.pi, 2300, endpoint=False)
    xy = np.column_stack((5.0 * np.cos(angles), 5.0 * np.sin(angles)))
    return ReplayFrame(
        Path(f'{stamp}.npz'), xy,
        np.linspace(stamp, stamp + 0.038, len(xy)), stamp, int(stamp),
    )


def test_encoder_interpolation_rejects_outside_and_unwraps():
    samples = [
        {'enc_t': 10.0, 'enc_v': 0.1, 'enc_dgr': 179.0},
        {'enc_t': 11.0, 'enc_v': 0.2, 'enc_dgr': -179.0},
    ]
    values = interpolate_encoder_series(samples, [9.0, 10.5, 12.0])
    assert values[0] is None and values[2] is None
    # The shortest interpolation crosses ±180, rather than manufacturing 0°.
    assert abs(abs(values[1]['enc_angle']) - math.pi) < 0.1


def test_scan_geometry_keeps_all_ordered_returns():
    frame = _frame(100.0)
    angle_min, increment, ranges, error = scan_geometry(frame.xy)
    assert len(ranges) == 2300
    assert abs(angle_min + math.pi) < 1e-3
    assert abs(increment - 2.0 * math.pi / 2300.0) < 1e-6
    assert error < 1e-5


def test_wheel_pose_is_integrated_for_every_scan():
    frames = [_frame(10.0), _frame(10.1), _frame(10.2)]
    samples = [
        {'enc_t': 9.0, 'enc_v': -0.3, 'enc_dgr': 0.0},
        {'enc_t': 12.0, 'enc_v': -0.3, 'enc_dgr': 0.0},
    ]
    poses, synced = build_wheel_poses(frames, samples)
    assert len(poses) == len(frames) == len(synced)
    assert poses[-1][0] > 0.0
    assert abs(poses[-1][1]) < 1e-8
