"""Point-time deskew helpers."""

from __future__ import annotations

import math

import numpy as np

from .config import (
    ENCODER_STATIC_V, POINT_DESKEW_MAX_ANGLE, POINT_DESKEW_MAX_CORRECTION,
    POINT_DESKEW_MAX_DT, SYNTHETIC_SCAN_PERIOD_S,
)
from .encoder_sync import build_encoder_series, interpolate_wheel_velocity
from .geometry_icp import compose, local_to_global, normalize_angle, relative_pose
from .io_frames import load_frame_timestamp

def deskew_frame_points(points, point_times, frame_t, encoder_series):
    """
    Move each LiDAR point to the frame's median acquisition time.

    Wheel odometry supplies translation only. Angular deskew is intentionally
    omitted because enc_dgr is wheel travel direction, not chassis yaw.
    """
    result = np.asarray(points, dtype=np.float64).copy()
    stats = {'corrected_points': 0, 'max_correction_m': 0.0}
    if (
        encoder_series is None
        or frame_t is None
        or point_times is None
        or len(point_times) != len(result)
        or len(result) == 0
    ):
        return result, stats

    times = np.asarray([
        np.nan if value is None else float(value)
        for value in point_times
    ], dtype=np.float64)
    point_velocity, point_valid = interpolate_wheel_velocity(
        times, encoder_series
    )
    reference_velocity, reference_valid = interpolate_wheel_velocity(
        [frame_t], encoder_series
    )
    dt = times - float(frame_t)
    valid = (
        point_valid
        & reference_valid[0]
        & np.isfinite(dt)
        & (np.abs(dt) <= POINT_DESKEW_MAX_DT)
    )
    if not np.any(valid):
        return result, stats

    corrections = 0.5 * (
        point_velocity[valid] + reference_velocity[0]
    ) * dt[valid, None]
    lengths = np.linalg.norm(corrections, axis=1)
    safe = lengths <= POINT_DESKEW_MAX_CORRECTION
    valid_indices = np.flatnonzero(valid)[safe]
    safe_corrections = corrections[safe]
    result[valid_indices] += safe_corrections
    stats['corrected_points'] = int(len(valid_indices))
    stats['max_correction_m'] = (
        float(np.max(np.linalg.norm(safe_corrections, axis=1)))
        if len(safe_corrections)
        else 0.0
    )
    return result, stats


def deskew_all_frames(frames, metadata, encoder_samples=None):
    """Apply timestamped wheel translation compensation to every frame."""
    series = build_encoder_series(metadata, encoder_samples=encoder_samples)
    corrected = []
    corrected_frames = 0
    corrected_points = 0
    maximum = 0.0
    for points, data in zip(frames, metadata):
        adjusted, stats = deskew_frame_points(
            points,
            data.get('point_times'),
            load_frame_timestamp(data),
            series,
        )
        corrected.append(adjusted)
        if stats['corrected_points']:
            corrected_frames += 1
            corrected_points += stats['corrected_points']
            maximum = max(maximum, stats['max_correction_m'])
    return corrected, {
        'deskewed_frames': corrected_frames,
        'deskewed_points': corrected_points,
        'deskew_max_correction_m': maximum,
    }


def interpolate_pose_trajectory(query_times, trajectory_times, trajectory_poses):
    """Linear xy + unwrapped-angle interpolation of SE2 poses."""
    query = np.asarray(query_times, dtype=np.float64)
    times = np.asarray(trajectory_times, dtype=np.float64)
    poses = np.asarray(trajectory_poses, dtype=np.float64)
    xs = np.interp(query, times, poses[:, 0])
    ys = np.interp(query, times, poses[:, 1])
    angles = np.interp(query, times, np.unwrap(poses[:, 2]))
    return np.column_stack([xs, ys, angles])


def apply_relative_pose_to_points(points, relative):
    """Apply SE2 relative pose to local points."""
    if len(points) == 0:
        return points
    x, y, angle = relative
    c, s = math.cos(angle), math.sin(angle)
    rotation = np.array([[c, -s], [s, c]], dtype=np.float64)
    return points @ rotation.T + np.array([x, y], dtype=np.float64)


def full_se2_deskew_from_poses(frames, metadata, poses):
    """
    Deskew each scan by interpolating full SE2 between neighbouring poses.

    A point measured at time t in the robot frame at t is expressed in the
    median-frame robot pose by relative_pose(pose_ref, pose_t).
    """
    timed = []
    for data, pose in zip(metadata, poses):
        frame_t = load_frame_timestamp(data)
        if frame_t is not None:
            timed.append((frame_t, pose))
    if len(timed) < 2:
        return frames, {
            'se2_deskewed_frames': 0,
            'se2_deskewed_points': 0,
            'deskew_max_correction_m': 0.0,
        }
    timed.sort(key=lambda item: item[0])
    trajectory_times = np.asarray([item[0] for item in timed], dtype=np.float64)
    trajectory_poses = np.asarray([item[1] for item in timed], dtype=np.float64)

    corrected = []
    frame_count = 0
    point_count = 0
    maximum = 0.0
    for points, data in zip(frames, metadata):
        result = np.asarray(points, dtype=np.float64).copy()
        raw_times = data.get('point_times')
        frame_t = load_frame_timestamp(data)
        if (
            raw_times is None
            or frame_t is None
            or len(raw_times) != len(result)
            or len(result) == 0
            or frame_t < trajectory_times[0]
            or frame_t > trajectory_times[-1]
        ):
            corrected.append(result)
            continue
        point_times = np.asarray([
            np.nan if value is None else float(value)
            for value in raw_times
        ], dtype=np.float64)
        dt = point_times - frame_t
        valid = (
            np.isfinite(point_times)
            & (point_times >= trajectory_times[0])
            & (point_times <= trajectory_times[-1])
            & (np.abs(dt) <= POINT_DESKEW_MAX_DT)
            & (np.abs(dt) >= 1e-4)
        )
        if not np.any(valid):
            corrected.append(result)
            continue
        ref_pose = interpolate_pose_trajectory(
            [frame_t], trajectory_times, trajectory_poses
        )[0]
        acq_poses = interpolate_pose_trajectory(
            point_times[valid], trajectory_times, trajectory_poses
        )
        indices = np.flatnonzero(valid)
        selected = result[indices].copy()
        moved = np.empty_like(selected)
        max_shift = 0.0
        keep = []
        for row, acq in enumerate(acq_poses):
            relative = relative_pose(ref_pose.tolist(), acq.tolist())
            # Point was measured in acq frame; express it in ref frame.
            # relative_pose(ref, acq) maps acq-local -> ref-local for the origin,
            # but for a point fixed in acq frame we need inverse(relative)?
            # world = acq ⊕ p; p_ref = ⊖ref ⊕ world = (⊖ref ⊕ acq) ⊕ p
            # relative_pose(ref, acq) = ⊖ref ⊕ acq. Correct.
            transformed = apply_relative_pose_to_points(
                selected[row:row + 1], relative
            )[0]
            shift = float(np.linalg.norm(transformed - selected[row]))
            if shift <= POINT_DESKEW_MAX_CORRECTION:
                moved[row] = transformed
                keep.append(row)
                max_shift = max(max_shift, shift)
        if not keep:
            corrected.append(result)
            continue
        keep = np.asarray(keep, dtype=np.int64)
        result[indices[keep]] = moved[keep]
        corrected.append(result)
        frame_count += 1
        point_count += int(len(keep))
        maximum = max(maximum, max_shift)
    return corrected, {
        'se2_deskewed_frames': frame_count,
        'se2_deskewed_points': point_count,
        'deskew_max_correction_m': maximum,
    }


def rotational_deskew_from_poses(
    frames,
    metadata,
    poses,
    encoder_samples=None,
    require_wheel_motion=True,
    max_angle=None,
):
    """
    Use a first LiDAR trajectory estimate to compensate rotation within scans.

    This is a second-pass deskew: wheel data corrects translation first, while
    interpolation of optimized LiDAR poses supplies the otherwise unobserved
    chassis yaw during each scan.
    """
    max_angle = POINT_DESKEW_MAX_ANGLE if max_angle is None else max_angle
    timed_poses = {}
    for data, pose in zip(metadata, poses):
        frame_t = load_frame_timestamp(data)
        if frame_t is not None:
            timed_poses[frame_t] = pose
    if len(timed_poses) < 2:
        return frames, {
            'rotational_deskewed_frames': 0,
            'rotational_deskewed_points': 0,
            'deskew_max_angle_deg': 0.0,
        }

    trajectory_times = np.asarray(sorted(timed_poses), dtype=np.float64)
    trajectory_angles = np.unwrap(np.asarray([
        timed_poses[timestamp][2] for timestamp in trajectory_times
    ], dtype=np.float64))
    encoder_series = build_encoder_series(
        metadata, encoder_samples=encoder_samples
    )
    corrected_frames = []
    frame_count = 0
    point_count = 0
    maximum_angle = 0.0
    for points, data in zip(frames, metadata):
        result = np.asarray(points, dtype=np.float64).copy()
        raw_times = data.get('point_times')
        frame_t = load_frame_timestamp(data)
        if (
            raw_times is None
            or frame_t is None
            or len(raw_times) != len(result)
            or len(result) == 0
            or frame_t < trajectory_times[0]
            or frame_t > trajectory_times[-1]
        ):
            corrected_frames.append(result)
            continue

        point_times = np.asarray([
            np.nan if value is None else float(value)
            for value in raw_times
        ], dtype=np.float64)
        dt = point_times - frame_t
        valid = (
            np.isfinite(point_times)
            & (point_times >= trajectory_times[0])
            & (point_times <= trajectory_times[-1])
            & (np.abs(dt) <= POINT_DESKEW_MAX_DT)
        )
        if require_wheel_motion:
            point_velocity, point_velocity_valid = interpolate_wheel_velocity(
                point_times, encoder_series
            )
            _reference_velocity, reference_velocity_valid = (
                interpolate_wheel_velocity([frame_t], encoder_series)
            )
            moving = (
                np.linalg.norm(point_velocity, axis=1) >= ENCODER_STATIC_V
            )
            valid = (
                valid
                & point_velocity_valid
                & reference_velocity_valid[0]
                & moving
            )
        if not np.any(valid):
            corrected_frames.append(result)
            continue

        reference_angle = float(np.interp(
            frame_t, trajectory_times, trajectory_angles
        ))
        angle_delta = (
            np.interp(
                point_times[valid], trajectory_times, trajectory_angles
            )
            - reference_angle
        )
        safe = (
            (np.abs(angle_delta) <= max_angle)
            & (np.abs(angle_delta) >= 1e-6)
        )
        indices = np.flatnonzero(valid)[safe]
        angles = angle_delta[safe]
        if not len(indices):
            corrected_frames.append(result)
            continue
        cosine = np.cos(angles)
        sine = np.sin(angles)
        selected = result[indices].copy()
        result[indices, 0] = (
            cosine * selected[:, 0] - sine * selected[:, 1]
        )
        result[indices, 1] = (
            sine * selected[:, 0] + cosine * selected[:, 1]
        )
        corrected_frames.append(result)
        frame_count += 1
        point_count += len(indices)
        maximum_angle = max(
            maximum_angle, float(np.max(np.abs(angles)))
        )
    return corrected_frames, {
        'rotational_deskewed_frames': frame_count,
        'rotational_deskewed_points': point_count,
        'deskew_max_angle_deg': math.degrees(maximum_angle),
    }


def synthesize_point_times_from_bearings(
    points, frame_t, period_s=SYNTHETIC_SCAN_PERIOD_S
):
    """
    Recover approximate per-beam times from point bearings.

    Used when Gazebo/recorder stored identical stamps (time_increment=0).
    Assumes a full 360° scan over period_s starting at -pi.
    """
    pts = np.asarray(points, dtype=np.float64)
    if len(pts) == 0 or frame_t is None:
        return []
    angles = np.arctan2(pts[:, 1], pts[:, 0])
    # Map [-pi, pi] → [0, period]
    frac = (angles + math.pi) / (2.0 * math.pi)
    return (float(frame_t) + frac * float(period_s)).tolist()


def ensure_beam_point_times(frames, metadata, period_s=SYNTHETIC_SCAN_PERIOD_S):
    """Fill collapsed point_times so full_se2 deskew can run on sim logs."""
    fixed = 0
    for points, data in zip(frames, metadata):
        raw = data.get('point_times') or []
        frame_t = load_frame_timestamp(data)
        if len(points) == 0 or frame_t is None:
            continue
        values = [
            np.nan if value is None else float(value) for value in raw
        ]
        finite = [v for v in values if np.isfinite(v)]
        collapsed = (
            len(finite) < 2
            or (max(finite) - min(finite)) < 1e-4
            or len(finite) != len(points)
        )
        if not collapsed:
            continue
        data['point_times'] = synthesize_point_times_from_bearings(
            points, frame_t, period_s=period_s
        )
        data['point_times_synthesized'] = True
        fixed += 1
    return fixed


def frame_yaw_rate(encoder):
    if encoder is None:
        return 0.0
    if encoder.get('enc_wz') is not None:
        try:
            return abs(float(encoder['enc_wz']))
        except (TypeError, ValueError):
            pass
    return 0.0
