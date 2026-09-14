"""Online/live runtime helpers used by scenario CLIs."""

from __future__ import annotations

import json
from pathlib import Path

from flifltbot.mapping.io_frames import list_run_frame_paths, load_horizon_npz_frame, points_from_frame_record
from .source_detect import detect_source


def resolve_base_url(source='auto'):
    resolved = detect_source(source)
    if resolved == 'sim':
        return 'ros2://sim'
    return 'ros2://robot'


def fetch_raw_points(frame_path, cameras=None):
    path = Path(frame_path)
    if path.suffix.lower() == '.npz':
        loaded = load_horizon_npz_frame(path)
        if loaded is None:
            raise FileNotFoundError(f'sparse or empty horizon npz: {path}')
        points, frame_t, point_times, used_cameras, _meta = loaded
        return {
            'points': points,
            'frame_t': frame_t,
            'point_times': point_times,
            'used_cameras': used_cameras,
        }
    payload = json.loads(path.read_text(encoding='utf-8'))
    points, frame_t, point_times, used_cameras = points_from_frame_record(
        payload, cameras=cameras
    )
    return {
        'points': points,
        'frame_t': frame_t,
        'point_times': point_times,
        'used_cameras': used_cameras,
    }


def latest_frame(run_dir):
    files = list_run_frame_paths(run_dir)
    if not files:
        raise FileNotFoundError(
            f'no frame JSON or horizon/*.npz files found in {run_dir}')
    return files[-1]
