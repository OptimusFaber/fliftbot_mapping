"""Distance-windowed submap stitching for long, repetitive trajectories."""

from __future__ import annotations

import math

import numpy as np

from .geometry_icp import (
    compose, local_to_global, relative_pose, robust_icp, voxel_downsample,
)


def _travel(poses):
    out = np.zeros(len(poses), dtype=float)
    for i in range(1, len(poses)):
        out[i] = out[i - 1] + math.hypot(
            poses[i][0] - poses[i - 1][0], poses[i][1] - poses[i - 1][1]
        )
    return out


def _window_cloud(frames, wheel_poses, start, end, stride=3):
    origin = wheel_poses[start]
    chunks = []
    for i in range(start, end, max(1, int(stride))):
        if len(frames[i]) == 0:
            continue
        chunks.append(local_to_global(frames[i], relative_pose(origin, wheel_poses[i])))
    if not chunks:
        return np.empty((0, 2), dtype=float)
    return voxel_downsample(np.vstack(chunks), cell=0.08, max_points=12000)


def build_submap_trajectory(
    frames, wheel_poses, length_m=12.0, overlap_m=5.0,
    max_correction_m=0.8, max_correction_deg=8.0,
):
    """Stitch overlapping local maps, seeded only by wheel-sensor SE(2)."""
    if len(frames) != len(wheel_poses) or not frames:
        raise ValueError('frames and wheel_poses must be non-empty and equal length')
    travel = _travel(wheel_poses)
    step_m = max(2.0, float(length_m) - float(overlap_m))
    starts = [0]
    next_d = step_m
    for i, d in enumerate(travel):
        if d >= next_d:
            starts.append(i)
            next_d += step_m
    if starts[-1] != len(frames) - 1:
        starts.append(len(frames) - 1)

    transforms = []
    submaps = []
    accepted = rejected = 0
    for si, start in enumerate(starts[:-1]):
        end_distance = travel[start] + float(length_m)
        end = int(np.searchsorted(travel, end_distance, side='right'))
        end = min(len(frames), max(start + 1, end))
        cloud = _window_cloud(frames, wheel_poses, start, end)
        seed = list(wheel_poses[start])
        transform = seed
        if submaps and len(cloud) >= 80:
            target = voxel_downsample(
                np.vstack(submaps[-2:]), cell=0.08, max_points=18000
            )
            fitted, rmse, overlap = robust_icp(
                cloud, target, initial_pose=seed, max_dist=0.45,
                max_source=6000, max_target=18000, cell_m=0.08,
            )
            corr = relative_pose(seed, fitted)
            reliable = (
                np.isfinite(rmse) and rmse <= 0.10 and overlap >= 0.35
                and math.hypot(corr[0], corr[1]) <= float(max_correction_m)
                and abs(corr[2]) <= math.radians(float(max_correction_deg))
            )
            if reliable:
                transform = fitted
                accepted += 1
            else:
                rejected += 1
        transforms.append(transform)
        submaps.append(local_to_global(cloud, transform))

    poses = [None] * len(frames)
    for si, start in enumerate(starts[:-1]):
        stop = starts[si + 1]
        origin = wheel_poses[start]
        for i in range(start, stop):
            poses[i] = compose(transforms[si], relative_pose(origin, wheel_poses[i]))
    poses[-1] = compose(
        transforms[-1], relative_pose(wheel_poses[starts[-2]], wheel_poses[-1])
    )
    return poses, {
        'submaps': len(transforms),
        'submap_length_m': float(length_m),
        'submap_overlap_m': float(overlap_m),
        'submap_icp_accepted': accepted,
        'submap_icp_rejected': rejected,
    }
