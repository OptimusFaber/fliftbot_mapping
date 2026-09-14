"""Frame JSON decoding and camera-cloud extraction."""

from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

from .config import (
    AVAILABLE_CAMERAS, CAMERA_ALIASES, CAMERA_EXTRINSICS, CAMERA_POSE_BODY,
    LEGACY_CAMERA_POSE_BODY, MAP_CAMERAS, SIM_UP_LIDAR_X, UP_LIDAR_X,
    MAX_RANGE, MIN_RANGE, SENSOR_MIN_RANGE,
)


def _resolve_camera_pose(camera: str):
    key = str(camera).strip().lower()
    for alias in (key, CAMERA_ALIASES.get(key)):
        if alias and alias in CAMERA_POSE_BODY:
            return alias, CAMERA_POSE_BODY[alias]
    return key, (0.0, 0.0, 0.0, 0.0)


def transform_points_se2(points, x: float, y: float, yaw: float):
    """Apply SE2 sensor→body (or any fixed mount) transform."""
    pts = np.asarray(points, dtype=np.float64)
    if len(pts) == 0:
        return pts
    c, s = math.cos(yaw), math.sin(yaw)
    rot = np.array([[c, -s], [s, c]], dtype=np.float64)
    return pts @ rot.T + np.array([x, y], dtype=np.float64)


def inverse_transform_points_se2(points, x: float, y: float, yaw: float):
    """Undo SE2 body→sensor transform."""
    pts = np.asarray(points, dtype=np.float64)
    if len(pts) == 0:
        return pts
    c, s = math.cos(yaw), math.sin(yaw)
    rot = np.array([[c, -s], [s, c]], dtype=np.float64)
    return (pts - np.array([x, y], dtype=np.float64)) @ rot


def camera_cloud_to_body(points, camera: str, cloud_meta: dict | None = None):
    """Normalize a per-camera cloud into rotation-center body frame."""
    cloud_meta = cloud_meta or {}
    frame = str(cloud_meta.get('frame', 'legacy_body')).lower()
    _, pose = _resolve_camera_pose(camera)
    x, y, _z, yaw = pose

    if frame == 'body':
        return np.asarray(points, dtype=np.float64)
    if frame == 'sensor':
        return transform_points_se2(points, x, y, yaw)

    # legacy_body: recordings transformed with wrong fork-tip extrinsics
    legacy = LEGACY_CAMERA_POSE_BODY.get(str(camera).lower())
    if legacy is None:
        return transform_points_se2(points, x, y, yaw)
    lx, ly, _lz, lyaw = legacy
    sensor = inverse_transform_points_se2(points, lx, ly, lyaw)
    return transform_points_se2(sensor, x, y, yaw)


def normalize_camera_list(cameras=None):
    """Return a validated ordered camera list for mapping/recording."""
    if cameras is None:
        cameras = MAP_CAMERAS
    if isinstance(cameras, str):
        cameras = [cameras]
    ordered = []
    for name in cameras:
        key = str(name).strip().lower()
        if key in CAMERA_ALIASES:
            # Prefer the name that exists in the payload later; keep requested alias.
            pass
        if not key or key in ordered:
            continue
        ordered.append(key)
    if not ordered:
        raise ValueError('MAP_CAMERAS must contain at least one camera name')
    return ordered


def camera_payload_keys(camera):
    """Names to try when reading a camera cloud from a frame JSON."""
    key = str(camera).strip().lower()
    keys = [key]
    alias = CAMERA_ALIASES.get(key)
    if alias and alias not in keys:
        keys.append(alias)
    return keys

def load_frame_timestamp(data):
    """Load the LiDAR acquisition timestamp as Unix seconds."""
    try:
        frame_t = float(data.get('t'))
        if np.isfinite(frame_t):
            return frame_t
    except (TypeError, ValueError):
        pass
    return parse_log_timestamp(data.get('ts'))


def parse_log_timestamp(raw):
    """Parse timestamp strings from saved frame metadata into Unix seconds."""
    if raw is None:
        return None
    text = str(raw).strip()
    if not text:
        return None
    try:
        if 'T' in text or text.endswith('Z') or '+' in text:
            return datetime.fromisoformat(
                text.replace('Z', '+00:00')
            ).timestamp()
    except ValueError:
        pass
    parts = text.replace('-', '_').split('_')
    if len(parts) < 6:
        return None
    year = int(parts[0])
    month = int(parts[1])
    day = int(parts[2])
    hour = int(parts[3])
    minute = int(parts[4])
    second = int(parts[5])
    micro = int(parts[6].ljust(6, '0')[:6]) if len(parts) > 6 else 0
    robot_timezone = timezone(timedelta(hours=3))
    moment = datetime(
        year, month, day, hour, minute, second, micro, tzinfo=robot_timezone
    )
    return moment.timestamp()

def _parse_raw_point(point):
    """Parse one /raw_points entry into (x, y, t_or_nan)."""
    x = y = None
    t = float('nan')
    if isinstance(point, dict):
        try:
            x = float(point['x'])
            y = float(point['y'])
        except (KeyError, TypeError, ValueError):
            return None
        raw_t = point.get('t')
        if raw_t is not None:
            try:
                t = float(raw_t)
            except (TypeError, ValueError):
                t = float('nan')
    elif isinstance(point, (list, tuple)) and len(point) >= 2:
        try:
            x = float(point[0])
            y = float(point[1])
        except (TypeError, ValueError):
            return None
        if len(point) >= 3 and point[2] is not None:
            try:
                t = float(point[2])
            except (TypeError, ValueError):
                t = float('nan')
    else:
        return None
    if not (np.isfinite(x) and np.isfinite(y)):
        return None
    return x, y, t


def _filter_xy_points(points, point_times, origin_xy=(0.0, 0.0), min_range=None):
    if len(points) == 0:
        empty = np.empty((0, 2), dtype=np.float64)
        return empty, np.empty(0, dtype=np.float64), None
    result = np.asarray(points, dtype=np.float64)
    times = np.asarray(point_times, dtype=np.float64)
    ox, oy = origin_xy
    dx = result[:, 0] - ox
    dy = result[:, 1] - oy
    radius2 = dx * dx + dy * dy
    near_zero = (np.abs(result[:, 0]) <= 1e-9) & (np.abs(result[:, 1]) <= 1e-9)
    min_r = SENSOR_MIN_RANGE if min_range is None else float(min_range)
    keep = (
        np.isfinite(result).all(axis=1)
        & (~near_zero)
        & (radius2 >= min_r * min_r)
        & (radius2 <= MAX_RANGE * MAX_RANGE)
    )
    filtered = result[keep]
    filtered_times = times[keep]
    finite_times = filtered_times[np.isfinite(filtered_times)]
    frame_t = float(np.median(finite_times)) if len(finite_times) else None
    return filtered, filtered_times, frame_t


def extract_camera_clouds(payload, cameras=None):
    """
    Extract per-camera local clouds from /raw_points.

    Supports both shapes:
      {"left": [{"x","y","t"}, ...], ...}
      {"left": {"points": [...]}, ...}
    """
    cameras = normalize_camera_list(cameras or AVAILABLE_CAMERAS)
    clouds = {}
    if not isinstance(payload, dict):
        return clouds
    for camera in cameras:
        side_data = None
        for key in camera_payload_keys(camera):
            if key in payload:
                side_data = payload.get(key)
                break
        items = (
            side_data.get('points')
            if isinstance(side_data, dict)
            else side_data
        )
        if not isinstance(items, list):
            continue
        points = []
        point_times = []
        for point in items:
            parsed = _parse_raw_point(point)
            if parsed is None:
                continue
            x, y, t = parsed
            points.append((x, y))
            point_times.append(t)
        origin = CAMERA_EXTRINSICS.get(camera, (0.0, 0.0))
        filtered, filtered_times, frame_t = _filter_xy_points(
            points, point_times, origin_xy=origin, min_range=SENSOR_MIN_RANGE
        )
        if len(filtered) == 0:
            continue
        # /raw_points delivers sensor-local scans — transform to body frame.
        body_pts = transform_points_se2(
            filtered, origin[0], origin[1],
            _resolve_camera_pose(camera)[1][3],
        )
        clouds[camera] = {
            'local_points': body_pts,
            'point_times': filtered_times,
            't': frame_t,
            'n_points': int(len(body_pts)),
            'frame': 'body',
        }
    return clouds


def extract_points(payload, include_point_times=False, cameras=None):
    """
    Extract local (x, y) points and frame timestamp from /raw_points.
    Prefer robot timestamp t from points; fall back to None if missing.
    """
    camera_clouds = extract_camera_clouds(
        payload, cameras=cameras or AVAILABLE_CAMERAS
    )
    points, frame_t, point_times, _used = combine_camera_clouds(
        camera_clouds, cameras=cameras or MAP_CAMERAS
    )
    if include_point_times:
        return points, frame_t, point_times
    return points, frame_t


def combine_camera_clouds(camera_clouds, cameras=None):
    """Concatenate selected camera clouds into one local frame cloud."""
    cameras = normalize_camera_list(cameras or MAP_CAMERAS)
    chunks = []
    times = []
    used = []
    for camera in cameras:
        cloud = None
        for key in camera_payload_keys(camera):
            cloud = camera_clouds.get(key)
            if cloud is not None and len(cloud.get('local_points', [])) > 0:
                break
            cloud = None
        if cloud is None:
            continue
        pts = np.asarray(cloud['local_points'], dtype=np.float64)
        if cloud.get('frame') != 'body':
            pts = camera_cloud_to_body(pts, camera, cloud)
        chunks.append(pts)
        times.append(np.asarray(cloud['point_times'], dtype=np.float64))
        used.append(camera)
    if not chunks:
        empty = np.empty((0, 2), dtype=np.float64)
        return empty, None, np.empty(0, dtype=np.float64), used
    points = np.vstack(chunks)
    point_times = np.concatenate(times)
    finite_times = point_times[np.isfinite(point_times)]
    frame_t = float(np.median(finite_times)) if len(finite_times) else None
    return points, frame_t, point_times, used


def shift_up_body_x(points, recorded_x=None, target_x=None):
    """Move body-frame up/top points from recorded lever arm to another.

    Reproject an up-LiDAR cloud between recorded and target body-frame lever
    arms. Yaw is zero, so this is a pure X translation.
    """
    pts = np.asarray(points, dtype=np.float64)
    recorded_x = UP_LIDAR_X if recorded_x is None else float(recorded_x)
    target_x = SIM_UP_LIDAR_X if target_x is None else float(target_x)
    dx = target_x - recorded_x
    if len(pts) == 0 or abs(dx) < 1e-9:
        return pts
    shifted = pts.copy()
    shifted[:, 0] += dx
    return shifted


def points_from_frame_record(data, cameras=None, up_body_x=None):
    """
    Build the mapping cloud for one saved local frame.

    New recordings store per-camera clouds under `cameras`. Older recordings
    only have a combined `local_points` array and are used as-is.
    """
    cameras = normalize_camera_list(cameras or MAP_CAMERAS)
    camera_payload = data.get('cameras')
    if isinstance(camera_payload, dict) and camera_payload:
        clouds = {}
        for name, cloud in camera_payload.items():
            if not isinstance(cloud, dict):
                continue
            points = cloud.get('local_points', cloud.get('points'))
            if not points:
                continue
            times = cloud.get('point_times')
            if times is None or len(times) != len(points):
                times = [float('nan')] * len(points)
            origin = CAMERA_EXTRINSICS.get(str(name).lower(), (0.0, 0.0))
            # Prefer sensor-local min range so body hits near the mount are dropped.
            # For legacy wrong extrinsics, also drop exact (0,0) and robot-origin near hits.
            filtered, filtered_times, frame_t = _filter_xy_points(
                points, times, origin_xy=origin, min_range=SENSOR_MIN_RANGE
            )
            if len(filtered) == 0:
                continue
            body_pts = camera_cloud_to_body(filtered, name, cloud)
            if (
                up_body_x is not None
                and str(name).lower() in ('up', 'top')
            ):
                body_pts = shift_up_body_x(body_pts, target_x=up_body_x)
            clouds[name] = {
                'local_points': body_pts,
                'point_times': filtered_times,
                't': cloud.get('t', frame_t),
                'n_points': int(len(body_pts)),
                'frame': 'body',
            }
        points, frame_t, point_times, used = combine_camera_clouds(
            clouds, cameras=cameras
        )
        if frame_t is None:
            frame_t = load_frame_timestamp(data)
        return points, frame_t, point_times, used

    points = np.asarray(data.get('local_points') or [], dtype=np.float64)
    raw_times = data.get('point_times')
    if raw_times is not None and len(raw_times) == len(points):
        point_times = np.asarray([
            np.nan if value is None else float(value)
            for value in raw_times
        ], dtype=np.float64)
    else:
        point_times = np.full(len(points), np.nan, dtype=np.float64)
    filtered, filtered_times, _frame_t = _filter_xy_points(
        points, point_times, origin_xy=(0.0, 0.0), min_range=MIN_RANGE
    )
    return filtered, load_frame_timestamp(data), filtered_times, ['legacy']


def format_point_timestamp(frame_t):
    """Convert robot unix timestamp to ISO-8601 UTC string."""
    if frame_t is None:
        return datetime.now(timezone.utc).isoformat()
    return datetime.fromtimestamp(frame_t, tz=timezone.utc).isoformat()


def list_run_frame_paths(local_dir):
    """JSON frames at run root, else Pandar horizon/*.npz recordings."""
    local = Path(local_dir)
    json_files = sorted(
        path for path in local.glob('*.json') if path.name[:5].isdigit()
    )
    if json_files:
        return json_files
    horizon = local / 'horizon'
    if not horizon.is_dir():
        return []
    return sorted(
        path for path in horizon.glob('*.npz') if path.name[:5].isdigit()
    )


def load_horizon_npz_frame(path, min_points=200):
    """Load one Pandar horizon .npz as a 2D scan in sensor XY.

    Returns (points, frame_t, point_times, used_cameras, metadata) or None
    if the ring is too sparse after range filtering.
    """
    path = Path(path)
    with np.load(path) as data:
        xy = np.asarray(data['xy'], dtype=np.float64)
        if 't' in data.files:
            times = np.asarray(data['t'], dtype=np.float64)
            if times.shape[0] != xy.shape[0]:
                times = np.full(len(xy), np.nan, dtype=np.float64)
        else:
            times = np.full(len(xy), np.nan, dtype=np.float64)
        stamp = float(data['stamp']) if 'stamp' in data.files else None
        frame = int(data['frame']) if 'frame' in data.files else int(path.name[:5])
        channel = int(data['channel']) if 'channel' in data.files else None
    if stamp is None:
        finite = times[np.isfinite(times)]
        stamp = float(np.median(finite)) if len(finite) else None
    filtered, filtered_times, _frame_t = _filter_xy_points(
        xy, times, origin_xy=(0.0, 0.0), min_range=MIN_RANGE
    )
    if len(filtered) < int(min_points):
        return None
    metadata = {
        'ts': format_point_timestamp(stamp),
        't': stamp,
        'frame': frame,
        'n_points': int(len(filtered)),
        'map_cameras': ['horizon'],
        'source': 'horizon_npz',
        'horizon_channel': channel,
        'path': str(path),
        'point_times': [
            None if not np.isfinite(value) else float(value)
            for value in filtered_times
        ],
    }
    return filtered, stamp, filtered_times, ['horizon'], metadata
