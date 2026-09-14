"""Persistence and rendering helpers for local/global maps."""

from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw

from .config import AVAILABLE_CAMERAS, MAP_CAMERAS, MAP_CELL, OUT_JSON, OUT_PNG, ROOT
from .io_frames import format_point_timestamp, normalize_camera_list
from .source_io import append_encoder_sample
from output.layout import infer_results_dir

def save_local_png(path, local_points):
    size = 500
    margin = 30
    image = Image.new('RGB', (size, size), 'white')
    draw = ImageDraw.Draw(image)
    if len(local_points) == 0:
        image.save(path)
        return
    minimum = local_points.min(axis=0)
    maximum = local_points.max(axis=0)
    span = max(float(np.max(maximum - minimum)), 1.0)
    center = (minimum + maximum) / 2

    def pixel(point):
        x = margin + (point[0] - center[0] + span / 2) / span * (
            size - 2 * margin
        )
        y = size - margin - (point[1] - center[1] + span / 2) / span * (
            size - 2 * margin
        )
        return int(x), int(y)

    for point in local_points:
        x, y = pixel(point)
        draw.ellipse((x - 1, y - 1, x + 1, y + 1), fill='#2563eb')
    x, y = pixel((0.0, 0.0))
    draw.ellipse((x - 4, y - 4, x + 4, y + 4), fill='#dc2626')
    image.save(path)


def save_local_frame(
    local_dir,
    frame_index,
    timestamp,
    local_points,
    encoder=None,
    frame_t=None,
    point_times=None,
    camera_clouds=None,
    map_cameras=None,
    write_png=True,
):
    safe_timestamp = timestamp.replace(':', '-').replace('+00:00', 'Z')
    stem = f'{frame_index:05d}_{safe_timestamp}'
    cameras = normalize_camera_list(map_cameras or MAP_CAMERAS)
    record = {
        'ts': timestamp,
        'frame': frame_index,
        'local_points': np.asarray(local_points, dtype=np.float64).tolist(),
        'n_points': int(len(local_points)),
        'map_cameras': cameras,
        'recorded_cameras': normalize_camera_list(AVAILABLE_CAMERAS),
    }
    if frame_t is not None:
        record['t'] = float(frame_t)
    if point_times is not None and len(point_times) == len(local_points):
        record['point_times'] = [
            float(value) if np.isfinite(value) else None
            for value in point_times
        ]
    if camera_clouds:
        serialized = {}
        for name, cloud in camera_clouds.items():
            points = np.asarray(cloud['local_points'], dtype=np.float64)
            times = np.asarray(cloud['point_times'], dtype=np.float64)
            serialized[name] = {
                'local_points': points.tolist(),
                'point_times': [
                    float(value) if np.isfinite(value) else None
                    for value in times
                ],
                't': cloud.get('t'),
                'n_points': int(len(points)),
            }
        record['cameras'] = serialized
    # Wheel telemetry is stored separately in encoder.jsonl for cleaner
    # timestamp matching across frames.
    if encoder is not None:
        append_encoder_sample(
            local_dir, encoder, frame_index=frame_index, frame_t=frame_t
        )
    (local_dir / f'{stem}.json').write_text(
        json.dumps(record, separators=(',', ':')), encoding='utf-8'
    )
    if write_png:
        save_local_png(local_dir / f'{stem}.png', np.asarray(local_points))


def save_horizon_npz(
    local_dir,
    frame_index,
    local_points,
    *,
    frame_t=None,
    point_times=None,
    channel=0,
    writer=None,
):
    """Write one mapping scan as uncompressed horizon/*.npz (no JSON/PNG)."""
    xy = np.asarray(local_points, dtype=np.float32)
    if xy.ndim != 2 or xy.shape[0] == 0:
        return None
    n = int(xy.shape[0])
    xyz = np.column_stack((xy, np.zeros(n, dtype=np.float32)))
    if point_times is not None and len(point_times) == n:
        times = np.asarray(point_times, dtype=np.float64)
    else:
        times = np.full(n, np.nan, dtype=np.float64)
    stamp = float(frame_t) if frame_t is not None else None
    if stamp is None:
        finite = times[np.isfinite(times)]
        stamp = float(np.median(finite)) if len(finite) else 0.0
    safe_ts = (
        format_point_timestamp(stamp)
        .replace(':', '-')
        .replace('+00:00', '+00-00')
    )
    path = Path(local_dir) / 'horizon' / f'{frame_index:05d}_{safe_ts}.npz'
    arrays = {
        'xy': np.ascontiguousarray(xy, dtype=np.float32),
        'xyz': np.ascontiguousarray(xyz, dtype=np.float32),
        'intensity': np.zeros(n, dtype=np.float32),
        't': np.ascontiguousarray(times, dtype=np.float64),
        'stamp': np.asarray(stamp, dtype=np.float64),
        'frame': np.asarray(int(frame_index), dtype=np.int32),
        'channel': np.asarray(int(channel), dtype=np.int32),
    }
    if writer is not None:
        writer.submit(path, arrays)
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, **arrays)
    return path


def save_global_map(
    points,
    poses,
    tag,
    local_dir=None,
    duration=None,
    used_encoder=False,
    source='auto',
    cell_m=None,
    write_module_copy=True,
):
    results_dir = infer_results_dir(local_dir, source=source, run_name=tag)
    out_json = results_dir / f'{tag}.json'
    out_png = results_dir / f'{tag}.png'
    cell_m = MAP_CELL if cell_m is None else float(cell_m)
    result = {
        'map': points,
        'poses': poses,
        'n_frames': len(poses),
        'cell_m': cell_m,
        'mode': 'global_pose_graph+encoder' if used_encoder else 'global_pose_graph',
        'all_local_points_transformed': True,
        'transform': (
            'global_x=x+cos(alpha)*local_x-sin(alpha)*local_y; '
            'global_y=y+sin(alpha)*local_x+cos(alpha)*local_y'
        ),
        'local_maps_dir': str(local_dir) if local_dir else None,
        'duration_s': duration,
    }
    encoded = json.dumps(result, ensure_ascii=False, separators=(',', ':'))
    out_json.write_text(encoded, encoding='utf-8')
    if write_module_copy:
        OUT_JSON.write_text(encoded, encoding='utf-8')

    figure, axis = plt.subplots(figsize=(10, 10))
    if points:
        array = np.asarray(points)
        axis.scatter(
            array[:, 0], array[:, 1], s=2, c='#2563eb',
            alpha=0.8, edgecolors='none'
        )
    if poses:
        trajectory = np.asarray([item['pose'] for item in poses])
        axis.plot(
            trajectory[:, 0], trajectory[:, 1],
            color='#f97316', linewidth=1.0
        )
        x, y, angle = trajectory[-1]
        axis.scatter([x], [y], s=70, c='#dc2626')
        axis.arrow(
            x, y, 0.4 * math.cos(angle), 0.4 * math.sin(angle),
            color='#dc2626', head_width=0.12
        )
    axis.set_aspect('equal')
    axis.grid(True, alpha=0.3)
    axis.set_xlabel('global x (m)')
    axis.set_ylabel('global y (m)')
    axis.set_title(
        'global LiDAR map — pose graph + wheel odometry'
        if used_encoder else
        'global LiDAR map — pose graph + loop closure'
    )
    figure.tight_layout()
    figure.savefig(out_png, dpi=150)
    if write_module_copy:
        figure.savefig(OUT_PNG, dpi=150)
    plt.close(figure)
    return out_json, out_png
