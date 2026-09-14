"""Encoder synchronization and odom-derived motion helpers."""

from __future__ import annotations

import math

import numpy as np

from .config import (
    ENCODER_STATIC_V, HZ, KEYFRAME_TH, KEYFRAME_XY, MAX_FRAME_XY,
    REAR_PIVOT_X, STATIC_XY, TURN_KEYFRAME_TH, TURN_KEYFRAME_XY,
    WHEEL_DIRECTION_OFFSET, WHEEL_DIRECTION_SIGN,
    WHEEL_MAX_ROT_DISAGREEMENT, WHEEL_MAX_SYNC_ERROR, WHEEL_MAX_TRANS_DISAGREEMENT,
    WHEEL_ROT_WEIGHT, WHEEL_SPEED_SCALE, WHEEL_TRANS_WEIGHT, WHEELBASE_M,
)
from .geometry_icp import compose, normalize_angle, pose_changed, relative_pose
from .io_frames import load_frame_timestamp
from .source_io import collect_run_encoder_samples, load_encoder_record

# Runtime override: calibrated per-run when sim enc_dgr frame ≠ real robot.
_WHEEL_DIRECTION_OFFSET_RUNTIME = None
# True when this log has a real yaw-rate signal (sim). Dummy enc_wz=0 on
# some real logs must not switch the pipeline into wheel-SE2 dead-reckoning.
_WHEEL_SE2_RUNTIME = False


def set_wheel_direction_offset(offset_rad):
    global _WHEEL_DIRECTION_OFFSET_RUNTIME
    _WHEEL_DIRECTION_OFFSET_RUNTIME = None if offset_rad is None else float(offset_rad)


def get_wheel_direction_offset():
    if _WHEEL_DIRECTION_OFFSET_RUNTIME is not None:
        return _WHEEL_DIRECTION_OFFSET_RUNTIME
    return WHEEL_DIRECTION_OFFSET


def set_wheel_se2(enabled):
    global _WHEEL_SE2_RUNTIME
    _WHEEL_SE2_RUNTIME = bool(enabled)


def get_wheel_se2():
    return _WHEEL_SE2_RUNTIME


def samples_have_usable_yaw_rate(samples, min_p90=0.01):
    """True when enc_wz is a real yaw-rate, not a missing/zero column."""
    if not samples:
        return False
    rates = [
        abs(float(item['enc_wz']))
        for item in samples
        if item.get('enc_wz') is not None
    ]
    if len(rates) < max(10, len(samples) // 4):
        return False
    # Gentle, long turns can stay below 0.05 rad/s while accumulating several
    # radians.  Missing yaw telemetry remains distinguishable: it is all zero.
    return float(np.percentile(rates, 90)) >= float(min_p90)


def samples_have_usable_steering(samples):
    """Whether wheel speed + actual steering angle can drive tricycle SE(2)."""
    if not samples:
        return False
    usable = [
        item for item in samples
        if item.get('enc_available') is not False
        and item.get('enc_v') is not None
        and item.get('enc_dgr') is not None
    ]
    return len(usable) >= max(10, int(0.8 * len(samples)))


def calibrate_wheel_direction_offset(samples, min_steps=40):
    """Validate the configured enc_dgr→body offset against diagnostic GT.

    Current src/real/sim convention is identity: 0° straight, +left, -right.
    GT is diagnostic only and is never used to build mapping poses.
    """
    if not samples or len(samples) < 3:
        return None
    if not all(
        'odom_x' in s and 'odom_y' in s and 'odom_yaw' in s for s in samples
    ):
        return None
    candidates = tuple(dict.fromkeys((0.0, WHEEL_DIRECTION_OFFSET)))
    errors = {c: [] for c in candidates}
    for i in range(1, len(samples)):
        a, b = samples[i - 1], samples[i]
        if abs(a.get('enc_v', 0.0)) < 0.08 and abs(b.get('enc_v', 0.0)) < 0.08:
            continue
        ta = a.get('enc_unix', a.get('enc_t'))
        tb = b.get('enc_unix', b.get('enc_t'))
        if ta is None or tb is None:
            continue
        dt = float(tb) - float(ta)
        if dt <= 0.0 or dt > 0.5:
            continue
        odom = motion_from_odom_pose(a, b)
        if odom is None:
            continue
        ol = math.hypot(odom[0], odom[1])
        if ol < 0.02:
            continue
        odom_ang = math.atan2(odom[1], odom[0])
        dist = WHEEL_SPEED_SCALE * 0.5 * (a['enc_v'] + b['enc_v']) * dt
        enc = math.radians(0.5 * (a['enc_dgr'] + b['enc_dgr']))
        for off in candidates:
            direction = WHEEL_DIRECTION_SIGN * enc + off
            wx = dist * math.cos(direction)
            wy = dist * math.sin(direction)
            wang = math.atan2(wy, wx)
            errors[off].append(abs(normalize_angle(wang - odom_ang)))
    scored = []
    for off, errs in errors.items():
        if len(errs) < min_steps:
            continue
        scored.append((float(np.median(errs)), off, len(errs)))
    if not scored:
        return None
    scored.sort()
    best_err, best_off, n = scored[0]
    # Prefer calibrated offset only when clearly better than the other.
    if len(scored) >= 2 and scored[1][0] - best_err < math.radians(20.0):
        return None
    if best_err > math.radians(25.0):
        return None
    return best_off


def motion_from_odom_pose(previous, current):
    """Body-frame motion from consecutive /odom poses (sim and replay)."""
    needed = ('odom_x', 'odom_y', 'odom_yaw')
    if any(key not in previous or key not in current for key in needed):
        return None
    dx_world = current['odom_x'] - previous['odom_x']
    dy_world = current['odom_y'] - previous['odom_y']
    dtheta = normalize_angle(current['odom_yaw'] - previous['odom_yaw'])
    th = previous['odom_yaw']
    c, s = math.cos(th), math.sin(th)
    dx = c * dx_world + s * dy_world
    dy = -s * dx_world + c * dy_world
    if (
        math.hypot(dx, dy) < STATIC_XY
        and abs(dtheta) < math.radians(0.05)
    ):
        return [0.0, 0.0, 0.0]
    length = math.hypot(dx, dy)
    if length > MAX_FRAME_XY:
        scale = MAX_FRAME_XY / length
        dx *= scale
        dy *= scale
    return [dx, dy, dtheta]


def encoders_have_odom_poses(encoders, min_fraction=0.85):
    """Always false: global /odom is validation data, never SLAM input."""
    del encoders, min_fraction
    return False


def build_trajectory_from_odom(encoders):
    """Integrate consecutive encoder SE(2) deltas into a relative trajectory."""
    if not encoders:
        return []
    poses = [[0.0, 0.0, 0.0]]
    wheel_poses = [[0.0, 0.0, 0.0]]
    for index in range(1, len(encoders)):
        delta = motion_from_encoders(encoders[index - 1], encoders[index])
        if delta is None:
            poses.append(poses[-1].copy())
            wheel_poses.append(wheel_poses[-1].copy())
        else:
            poses.append(compose(poses[-1], delta))
            wheel_poses.append(compose(wheel_poses[-1], delta))
    return poses, wheel_poses


def build_keyframes_from_poses(
    frames,
    encoders,
    poses,
    source_indices,
    keyframe_xy=None,
    keyframe_th=None,
):
    keyframe_xy = KEYFRAME_XY if keyframe_xy is None else keyframe_xy
    keyframe_th = KEYFRAME_TH if keyframe_th is None else keyframe_th
    wheel_poses = poses  # odom trajectory doubles as wheel path
    keyframes = [{
        'frame_index': 0,
        'source_frame_index': source_indices[0],
        'points': frames[0],
        'pose': list(poses[0]),
        'encoder': encoders[0] if encoders else None,
        'wheel_pose': list(wheel_poses[0]),
        'wheel_valid_prefix': 0,
    }]
    for index in range(1, len(frames)):
        if pose_changed(
            keyframes[-1]['pose'],
            poses[index],
            min_xy=keyframe_xy,
            min_angle=keyframe_th,
        ):
            keyframes.append({
                'frame_index': index,
                'source_frame_index': source_indices[index],
                'points': frames[index],
                'pose': list(poses[index]),
                'encoder': encoders[index] if index < len(encoders) else None,
                'wheel_pose': list(wheel_poses[index]),
                'wheel_valid_prefix': index,
            })
    return keyframes


def motion_from_encoders(previous, current, dt_fallback=1.0 / HZ):
    """
    Motion in the previous robot frame.

    Tricycle SE(2) from measured signed drive-wheel speed and steering angle:
      vx = v_wheel*cos(delta), wz = v_wheel*sin(delta)/wheelbase.

    The sign of ``enc_v`` must affect both translation and yaw.  A reverse arc
    with the same physical steering angle has the opposite yaw direction.
    The same model is used in simulation and on the real robot. Mapping strips
    stored enc_wz before calling this function, so yaw is derived from the
    measured wheel speed and steering angle.
    Global /odom fields are ignored unconditionally; they are GT-only.
    """
    if previous is None or current is None:
        return None
    previous_t = previous.get('frame_t', previous.get('enc_t'))
    current_t = current.get('frame_t', current.get('enc_t'))
    dt = current_t - previous_t if previous_t is not None and current_t is not None else dt_fallback
    if dt < 0.0 or dt > 0.5:
        return None
    if dt == 0.0:
        return [0.0, 0.0, 0.0]
    if previous.get('enc_available') is False or current.get('enc_available') is False:
        return None
    wheel_speed = WHEEL_SPEED_SCALE * 0.5 * (
        previous['enc_v'] + current['enc_v']
    )
    if 'enc_angle' in previous and 'enc_angle' in current:
        steering = 0.5 * (
            previous['enc_angle'] + current['enc_angle']
        )
    else:
        steering = math.radians(
            0.5 * (previous['enc_dgr'] + current['enc_dgr'])
        )
    steering = normalize_angle(
        WHEEL_DIRECTION_SIGN * steering + get_wheel_direction_offset()
    )
    vx = wheel_speed * math.cos(steering)
    # Yaw follows the same measured wheel speed and steering angle.  Do not
    # accept a body-rate channel here: it may be Gazebo ground-truth telemetry.
    wz = wheel_speed * math.sin(steering) / max(1e-6, WHEELBASE_M)
    theta = wz * dt
    if abs(wz) < 1e-9:
        dx = vx * dt
        dy = 0.0
    else:
        dx = vx * math.sin(theta) / wz
        dy = vx * (1.0 - math.cos(theta)) / wz
    dtheta = theta
    static_v = (
        abs(previous['enc_v']) < ENCODER_STATIC_V
        and abs(current['enc_v']) < ENCODER_STATIC_V
    )
    # Do not apply the scan-matching STATIC_XY deadband to wheel odometry.
    # At 20 Hz a perfectly normal 0.30 m/s motion is only 15 mm per sample,
    # below STATIC_XY=18 mm.  Dropping each such increment destroys most of a
    # long trajectory instead of merely suppressing noise.  Wheel standstill
    # is observable directly from the measured drive-wheel speed.
    if static_v and abs(dtheta) < math.radians(0.05):
        return [0.0, 0.0, 0.0]
    if static_v:
        dx, dy = 0.0, 0.0
    length = math.hypot(dx, dy)
    if length > MAX_FRAME_XY:
        scale = MAX_FRAME_XY / length
        dx *= scale
        dy *= scale
    return [dx, dy, dtheta]


def fuse_motion(wheel_delta, scan_delta):
    """Prefer measured LiDAR motion; wheel odometry is a fallback."""
    if wheel_delta is None:
        return scan_delta
    if scan_delta == [0.0, 0.0, 0.0]:
        return wheel_delta
    return scan_delta


def shift_encoder_samples(encoder_samples, delta_t):
    """Return copies with enc_t shifted by delta_t (seconds)."""
    if not encoder_samples:
        return []
    dt = float(delta_t)
    out = []
    for item in encoder_samples:
        copy = dict(item)
        copy['enc_t'] = float(item['enc_t']) + dt
        out.append(copy)
    return out


def calibrate_lidar_odom_delay(
    metadata,
    encoder_samples,
    frames=None,
    dt_min=None,
    dt_max=None,
    dt_step=None,
    max_pairs=None,
):
    """Search global δt such that encoder queried at frame_t+δt best matches ICP.

    Convention: positive δt means encoder timestamps are late vs lidar (we
    shift samples forward so they align with lidar times).

    Returns dict with best_dt_s, cost curve, and half-width stability metric.
    """
    from .config import (
        FACTOR_DT_MAX, FACTOR_DT_MAX_PAIRS, FACTOR_DT_MIN, FACTOR_DT_STEP,
    )
    from .geometry_icp import motion_from_scans

    dt_min = FACTOR_DT_MIN if dt_min is None else float(dt_min)
    dt_max = FACTOR_DT_MAX if dt_max is None else float(dt_max)
    dt_step = FACTOR_DT_STEP if dt_step is None else float(dt_step)
    max_pairs = FACTOR_DT_MAX_PAIRS if max_pairs is None else int(max_pairs)

    empty = {
        'best_dt_s': 0.0,
        'best_cost': None,
        'curve': [],
        'half_width_s': None,
        'n_pairs': 0,
    }
    if not encoder_samples or len(metadata) < 3:
        return empty

    # Subsample consecutive frame pairs with motion for speed.
    pair_indices = []
    if frames is not None and len(frames) == len(metadata):
        step = max(1, len(metadata) // max(20, max_pairs))
        for i in range(0, len(metadata) - 1, step):
            if len(pair_indices) >= max_pairs:
                break
            a, b = frames[i], frames[i + 1]
            if a is None or b is None or len(a) < 80 or len(b) < 80:
                continue
            pair_indices.append(i)
    else:
        step = max(1, len(metadata) // max(20, max_pairs))
        pair_indices = list(range(0, len(metadata) - 1, step))[:max_pairs]

    if not pair_indices:
        return empty

    candidates = []
    dt = dt_min
    while dt <= dt_max + 1e-12:
        candidates.append(round(dt, 5))
        dt += dt_step

    curve = []
    best_dt = 0.0
    best_cost = float('inf')
    for dt in candidates:
        shifted = shift_encoder_samples(encoder_samples, dt)
        synced, _stats = synchronize_encoders(metadata, shifted)
        costs = []
        for i in pair_indices:
            ea, eb = synced[i], synced[i + 1]
            if ea is None or eb is None:
                continue
            wheel = motion_from_encoders(ea, eb)
            if wheel is None:
                continue
            if frames is not None and len(frames) == len(metadata):
                scan, err, overlap = motion_from_scans(
                    frames[i], frames[i + 1],
                    initial_pose=wheel, return_quality=True,
                )
                if not np.isfinite(err) or overlap < 0.30:
                    # Fall back to wheel magnitude consistency only.
                    costs.append(abs(math.hypot(wheel[0], wheel[1])))
                    continue
                disagree = relative_pose(wheel, scan)
                costs.append(
                    math.hypot(disagree[0], disagree[1])
                    + 0.5 * abs(disagree[2])
                )
            else:
                costs.append(abs(float(ea.get('sync_error_s', 0.0))))
        if not costs:
            cost = float('inf')
        else:
            cost = float(np.median(costs))
        curve.append({'dt_s': dt, 'cost': cost, 'n': len(costs)})
        if cost < best_cost:
            best_cost = cost
            best_dt = dt

    # Stability: width where cost ≤ best + 10% of (max-best) or +0.01 m.
    finite = [c for c in curve if np.isfinite(c['cost'])]
    half_width = None
    if finite and np.isfinite(best_cost):
        costs_arr = np.asarray([c['cost'] for c in finite], float)
        thresh = best_cost + max(0.01, 0.1 * (float(costs_arr.max()) - best_cost))
        near = [c['dt_s'] for c in finite if c['cost'] <= thresh]
        if near:
            half_width = 0.5 * (max(near) - min(near))

    print(
        f'  temporal calib δt={best_dt:+.3f}s '
        f'(cost={best_cost:.4f}, half_width={half_width})'
    )
    return {
        'best_dt_s': float(best_dt),
        'best_cost': None if not np.isfinite(best_cost) else float(best_cost),
        'curve': curve,
        'half_width_s': half_width,
        'n_pairs': len(pair_indices),
    }


def synchronize_encoders(metadata, encoder_samples=None):
    """
    Interpolate encoder velocity and unwrapped wheel direction at LiDAR times.

    The returned list is indexed exactly like the point-cloud frames. Each
    record contains the LiDAR frame time and its nearest encoder time error.
    JSON file association is deliberately ignored: samples are globally sorted
    by timestamp. A frame is rejected unless encoder samples on both sides are
    close enough, preventing interpolation across telemetry outages.
    """
    if encoder_samples is None:
        unique = {}
        for data in metadata:
            encoder = load_encoder_record(data)
            if encoder is not None:
                unique[encoder['enc_t']] = encoder
        samples = sorted(unique.values(), key=lambda item: item['enc_t'])
    else:
        samples = sorted(encoder_samples, key=lambda item: item['enc_t'])
    if len(samples) < 2:
        return [None] * len(metadata), {
            'unique_samples': len(samples),
            'synced_frames': 0,
            'rejected_wide_bracket': 0,
            'median_sync_error_s': None,
            'max_sync_error_s': None,
            'max_bracket_span_s': None,
        }

    sample_times = np.asarray(
        [item['enc_t'] for item in samples], dtype=np.float64
    )
    sample_velocities = np.asarray(
        [item['enc_v'] for item in samples], dtype=np.float64
    )
    sample_angles = np.unwrap(np.radians([
        item['enc_dgr'] for item in samples
    ]))
    has_wz = all(item.get('enc_wz') is not None for item in samples)
    if has_wz:
        sample_wz = np.asarray(
            [item['enc_wz'] for item in samples], dtype=np.float64
        )
    has_odom = all(
        'odom_x' in item and 'odom_y' in item and 'odom_yaw' in item
        for item in samples
    )
    if has_odom:
        sample_odom_x = np.asarray(
            [item['odom_x'] for item in samples], dtype=np.float64
        )
        sample_odom_y = np.asarray(
            [item['odom_y'] for item in samples], dtype=np.float64
        )
        sample_odom_yaw = np.unwrap(np.asarray(
            [item['odom_yaw'] for item in samples], dtype=np.float64
        ))

    synchronized = []
    errors = []
    bracket_spans = []
    rejected_wide_bracket = 0
    for data in metadata:
        frame_t = load_frame_timestamp(data)
        if (
            frame_t is None
            or frame_t < sample_times[0]
            or frame_t > sample_times[-1]
        ):
            synchronized.append(None)
            continue
        position = int(np.searchsorted(sample_times, frame_t))
        exact_sample = (
            position < len(sample_times)
            and abs(sample_times[position] - frame_t) <= 1e-9
        )
        if exact_sample:
            sync_error = 0.0
            bracket_span = 0.0
        else:
            if position == 0 or position == len(sample_times):
                synchronized.append(None)
                continue
            lower_error = frame_t - sample_times[position - 1]
            upper_error = sample_times[position] - frame_t
            if max(lower_error, upper_error) > WHEEL_MAX_SYNC_ERROR:
                synchronized.append(None)
                rejected_wide_bracket += 1
                continue
            sync_error = min(lower_error, upper_error)
            bracket_span = lower_error + upper_error
        velocity = float(np.interp(
            frame_t, sample_times, sample_velocities
        ))
        angle = float(np.interp(frame_t, sample_times, sample_angles))
        record = {
            'frame_t': frame_t,
            'enc_t': frame_t,
            'enc_t_raw': 'interpolated',
            'enc_v': velocity,
            'enc_dgr': math.degrees(angle),
            'enc_angle': angle,
            'sync_error_s': sync_error,
        }
        if has_odom:
            record['odom_x'] = float(np.interp(
                frame_t, sample_times, sample_odom_x
            ))
            record['odom_y'] = float(np.interp(
                frame_t, sample_times, sample_odom_y
            ))
            record['odom_yaw'] = float(np.interp(
                frame_t, sample_times, sample_odom_yaw
            ))
        if has_wz:
            record['enc_wz'] = float(np.interp(
                frame_t, sample_times, sample_wz
            ))
        synchronized.append(record)
        errors.append(sync_error)
        bracket_spans.append(bracket_span)
    return synchronized, {
        'unique_samples': len(samples),
        'synced_frames': len(errors),
        'rejected_wide_bracket': rejected_wide_bracket,
        'median_sync_error_s': (
            float(np.median(errors)) if errors else None
        ),
        'max_sync_error_s': max(errors) if errors else None,
        'max_bracket_span_s': max(bracket_spans) if bracket_spans else None,
    }


def build_encoder_series(metadata, encoder_samples=None):
    """Build globally timestamp-sorted encoder arrays, ignoring JSON pairing."""
    if encoder_samples is None:
        samples = collect_run_encoder_samples(metadata)
    else:
        samples = sorted(encoder_samples, key=lambda item: item['enc_t'])
    if len(samples) < 2:
        return None
    return {
        'times': np.asarray(
            [item['enc_t'] for item in samples], dtype=np.float64
        ),
        'velocities': np.asarray(
            [item['enc_v'] for item in samples], dtype=np.float64
        ),
        'angles': np.unwrap(np.radians([
            item['enc_dgr'] for item in samples
        ])),
    }


def interpolate_wheel_velocity(query_times, series):
    """
    Return local-frame velocity vectors at arbitrary timestamps.

    Values are valid only when samples on both sides are close enough. This is
    the same strict timestamp rule used for frame-level encoder synchronization.
    """
    query = np.asarray(query_times, dtype=np.float64)
    vectors = np.zeros((len(query), 2), dtype=np.float64)
    valid = np.zeros(len(query), dtype=bool)
    if series is None or len(query) == 0:
        return vectors, valid

    times = series['times']
    positions = np.searchsorted(times, query)
    finite = np.isfinite(query)
    exact = np.zeros(len(query), dtype=bool)
    inside_right = positions < len(times)
    exact[inside_right] = (
        np.abs(times[positions[inside_right]] - query[inside_right]) <= 1e-9
    )

    bracketed = finite & (positions > 0) & (positions < len(times))
    bracket_indices = np.flatnonzero(bracketed)
    if len(bracket_indices):
        bracket_positions = positions[bracket_indices]
        lower_error = (
            query[bracket_indices] - times[bracket_positions - 1]
        )
        upper_error = (
            times[bracket_positions] - query[bracket_indices]
        )
        valid[bracket_indices] = (
            np.maximum(lower_error, upper_error) <= WHEEL_MAX_SYNC_ERROR
        )
    valid |= finite & exact
    if not np.any(valid):
        return vectors, valid

    velocities = np.interp(
        query[valid], times, series['velocities']
    )
    directions = (
        WHEEL_DIRECTION_SIGN
        * np.interp(query[valid], times, series['angles'])
        + get_wheel_direction_offset()
    )
    speeds = WHEEL_SPEED_SCALE * velocities
    vectors[valid, 0] = speeds * np.cos(directions)
    vectors[valid, 1] = speeds * np.sin(directions)
    return vectors, valid



def integrate_wheel_poses(encoders):
    """Integrate synchronized frame-to-frame odometry for keyframe edges."""
    poses = [[0.0, 0.0, 0.0]]
    valid_prefix = [0]
    for index in range(1, len(encoders)):
        delta = motion_from_encoders(encoders[index - 1], encoders[index])
        if delta is None:
            poses.append(poses[-1].copy())
            valid_prefix.append(valid_prefix[-1])
        else:
            poses.append(compose(poses[-1], delta))
            valid_prefix.append(valid_prefix[-1] + 1)
    return poses, valid_prefix
