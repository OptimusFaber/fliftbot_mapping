"""Robot/source I/O and encoder log access."""

from __future__ import annotations

import json
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import requests

from .config import (
    BASE_CANDIDATES, ENCODER_LOG_NAME, ENCODER_PATH, ENCODER_PATH_FALLBACK,
    ENCODER_UTC_OFFSET_HOURS, MAP_CAMERAS, POINTS_PATH, POINTS_PATH_FALLBACK,
    AVAILABLE_CAMERAS, session,
)
from .io_frames import combine_camera_clouds, extract_camera_clouds, normalize_camera_list

BASE = BASE_CANDIDATES[0]


def set_base_url(base_url):
    """Override robot HTTP base URL from an IP or full URL."""
    global BASE
    text = str(base_url or '').strip()
    if not text:
        return BASE
    # Accept URLs accidentally copied from rendered Markdown: [url](url).
    match = re.fullmatch(r'\[([^]]+)\]\(([^)]+)\)', text)
    if match:
        text = match.group(2)
    if '://' not in text:
        text = f'http://{text}'
    if ':' not in text.rsplit('/', 1)[-1]:
        text = f'{text}:10101'
    BASE = text.rstrip('/')
    return BASE

def resolve_base_url():
    """Pick the first reachable robot HTTP base URL."""
    global BASE
    for candidate in BASE_CANDIDATES:
        try:
            response = session.get(f'{candidate}/raw_points', timeout=1.0)
            if response.ok:
                BASE = candidate
                return BASE
        except requests.RequestException:
            continue
    BASE = BASE_CANDIDATES[0]
    return BASE


def parse_log_timestamp(raw):
    """
    Parse encoder time to Unix seconds.

    Undelimited robot timestamps are local robot time (UTC+3). ISO strings
    retain their explicit timezone. This puts enc_t and /raw_points t on the
    same Unix time axis.
    """
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
    robot_timezone = timezone(timedelta(hours=ENCODER_UTC_OFFSET_HOURS))
    moment = datetime(
        year, month, day, hour, minute, second, micro, tzinfo=robot_timezone
    )
    return moment.timestamp()


def fetch_encoder_log(http_session=None):
    """Read wheel odometry sample from /check_logs (fallback /chek_logs)."""
    payload = None
    client = http_session or session
    bases = [BASE] + [item for item in BASE_CANDIDATES if item != BASE]
    for base in bases:
        for path in (ENCODER_PATH, ENCODER_PATH_FALLBACK):
            try:
                response = client.get(f'{base}{path}', timeout=1.0)
                response.raise_for_status()
                payload = response.json()
                break
            except (requests.RequestException, ValueError, TypeError):
                continue
        if payload is not None:
            break
    if not isinstance(payload, dict):
        return None
    enc_t_raw = payload.get('enc_t') or payload.get('t')
    try:
        enc_v = float(payload.get('enc_v', 0.0))
        enc_dgr = float(payload.get('enc_dgr', 0.0))
    except (TypeError, ValueError):
        return None
    enc_t = parse_log_timestamp(enc_t_raw)
    if enc_t is None:
        return None
    return {
        'enc_t_raw': str(enc_t_raw),
        'enc_t': enc_t,
        'enc_v': enc_v,
        'enc_dgr': enc_dgr,
    }


def load_encoder_record(data):
    """Load encoder fields saved inside a local frame JSON."""
    if not isinstance(data, dict) or 'encoder' not in data:
        return None
    encoder = data['encoder']
    if not isinstance(encoder, dict):
        return None
    enc_t_raw = encoder.get('enc_t')
    # Prefer enc_unix when present (Gazebo / sim). Parsing enc_t with UTC+3
    # is for the real robot and skews Ignition epoch timestamps.
    enc_t = None
    if encoder.get('enc_unix') is not None:
        try:
            enc_t = float(encoder.get('enc_unix'))
        except (TypeError, ValueError):
            enc_t = None
    if enc_t is None:
        enc_t = parse_log_timestamp(enc_t_raw)
    if enc_t is None:
        return None
    try:
        enc_dgr = float(encoder.get('enc_dgr', 0.0))
        # Older Gazebo logs stored wheel *joint angle* (0..360) as enc_dgr.
        # live_map expects the physical steering angle (0 straight, ±90 limit).
        # Sim timestamps are Ignition epoch seconds (≪ 1e9). Only zero enc_dgr
        # for legacy samples — modern collector writes frame/lidar_t and true
        # steering angle (needed for turns; zeroing it smears the map).
        modern = (
            encoder.get('frame') is not None
            or encoder.get('lidar_t') is not None
            or encoder.get('lidar_encoder_dt_s') is not None
            or int(encoder.get('encoder_semantics_version', 0) or 0) >= 1
        )
        if (
            enc_t is not None
            and 0.0 <= float(enc_t) < 1.0e9
            and not modern
        ):
            enc_dgr = 0.0
        record = {
            'enc_t_raw': str(enc_t_raw) if enc_t_raw is not None else '',
            'enc_t': enc_t,
            'enc_v': float(encoder.get('enc_v', 0.0)),
            'enc_dgr': enc_dgr,
        }
        for key in ('odom_x', 'odom_y', 'odom_yaw', 'enc_wz'):
            if encoder.get(key) is not None:
                try:
                    record[key] = float(encoder.get(key))
                except (TypeError, ValueError):
                    pass
        if 'enc_available' in encoder:
            record['enc_available'] = bool(encoder.get('enc_available'))
        return record
    except (TypeError, ValueError):
        return None


def encoder_log_path(local_dir):
    return Path(local_dir) / ENCODER_LOG_NAME


def append_encoder_sample(
    local_dir, encoder, frame_index=None, frame_t=None,
    filename=ENCODER_LOG_NAME,
):
    """Append one wheel sample to a separate encoder JSONL stream."""
    if encoder is None:
        return None
    record = {
        'enc_t': encoder['enc_t_raw'],
        'enc_unix': encoder['enc_t'],
        'enc_v': encoder['enc_v'],
        'enc_dgr': encoder['enc_dgr'],
    }
    if frame_index is not None:
        record['frame'] = int(frame_index)
    if frame_t is not None:
        record['lidar_t'] = float(frame_t)
        record['lidar_encoder_dt_s'] = float(frame_t) - encoder['enc_t']
    path = Path(local_dir) / filename
    with path.open('a', encoding='utf-8') as handle:
        handle.write(json.dumps(record, separators=(',', ':')) + '\n')
    return record


def load_encoder_log_file(local_dir):
    """Load wheel samples from encoder.jsonl (preferred modern storage)."""
    path = encoder_log_path(local_dir)
    if not path.exists():
        return []
    samples = []
    for line in path.read_text(encoding='utf-8').splitlines():
        text = line.strip()
        if not text:
            continue
        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            continue
        sample = load_encoder_record({'encoder': payload})
        if sample is not None:
            samples.append(sample)
    return samples


def collect_run_encoder_samples(metadata, local_dir=None):
    """Merge encoder.jsonl with any legacy per-frame encoder blocks."""
    unique = {}
    if local_dir is not None:
        for sample in load_encoder_log_file(local_dir):
            unique[sample['enc_t']] = sample
    for data in metadata:
        sample = load_encoder_record(data)
        if sample is not None:
            unique[sample['enc_t']] = sample
    return sorted(unique.values(), key=lambda item: item['enc_t'])

def fetch_raw_points(cameras=None, record_all=False):
    """
    Return selected-camera points plus the per-camera cloud dict.

    By default only `cameras` (map subset) are parsed. Pass record_all=True
    to extract every AVAILABLE_CAMERAS cloud from /raw_points.
    """
    global BASE
    map_cameras = normalize_camera_list(cameras or MAP_CAMERAS)
    record_cameras = normalize_camera_list(
        AVAILABLE_CAMERAS if record_all else map_cameras
    )
    payload = None
    bases = [BASE] + [item for item in BASE_CANDIDATES if item != BASE]
    for base in bases:
        try:
            response = session.get(f'{base}{POINTS_PATH}', timeout=1.5)
            response.raise_for_status()
            payload = response.json()
            BASE = base
            break
        except (requests.RequestException, ValueError, TypeError):
            try:
                response = session.get(f'{base}{POINTS_PATH_FALLBACK}', timeout=1.5)
                response.raise_for_status()
                payload = response.json()
                BASE = base
                break
            except (requests.RequestException, ValueError, TypeError):
                continue
    if payload is None:
        empty = np.empty((0, 2), dtype=np.float64)
        return empty, None, np.empty(0), {}
    camera_clouds = extract_camera_clouds(payload, cameras=record_cameras)
    points, frame_t, point_times, _used = combine_camera_clouds(
        camera_clouds, cameras=map_cameras
    )
    return points, frame_t, point_times, camera_clouds
