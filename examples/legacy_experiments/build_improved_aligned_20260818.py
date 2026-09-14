#!/usr/bin/env python3
"""Build all 2026-08-18 real maps with improved and one shared room frame."""
from __future__ import annotations

import json
import math
import statistics
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from flifltbot.mapping.pipeline import build_global_map_from_run
from flifltbot.mapping.geometry_icp import extract_long_lines

ROOT = Path(__file__).resolve().parents[3]
RUN_ROOT = ROOT / 'maps/20260818/real'
def initial_nav_pose(run: Path) -> tuple[float, float, float]:
    rows = [json.loads(line)['check_logs'] for line in (run / 'encoder.jsonl').open()]
    n = max(3, len(rows) // 10)
    sample = rows[:n]
    return (
        statistics.median(float(row['nav_x']) for row in sample),
        statistics.median(float(row['nav_y']) for row in sample),
        statistics.median(float(row['nav_dgr']) for row in sample),
    )


def local_to_nav(points: np.ndarray, nav_pose) -> np.ndarray:
    """Place a zero-origin mapping result in the robot's persistent nav frame."""
    x, y, heading_deg = nav_pose
    angle = math.radians(heading_deg - 90.0)
    c, s = math.cos(angle), math.sin(angle)
    rotation = np.array([[c, -s], [s, c]])
    return np.asarray(points, float) @ rotation.T + np.array([x, y])


def manhattan_align(points: np.ndarray) -> tuple[np.ndarray, float]:
    """Rotate the consensus of long line segments onto the X/Y axes."""
    lines = extract_long_lines(
        points, min_length=2.0, min_inliers=25, max_lines=16,
    )
    if not lines:
        return np.asarray(points, float), 0.0
    # exp(4j*angle) treats horizontal/vertical and both directions as one
    # Manhattan family. Weight by segment length for robustness.
    consensus = sum(length * np.exp(4j * angle)
                    for angle, length, *_rest in lines)
    correction = -float(np.angle(consensus)) / 4.0
    c, s = math.cos(correction), math.sin(correction)
    rotation = np.array([[c, -s], [s, c]])
    return np.asarray(points, float) @ rotation.T, correction


def normalized_run_frames(world_maps: list[np.ndarray]):
    """Pin every map's robust SW corner to zero, preserving common nav axes."""
    corners = [np.percentile(points, 1.0, axis=0) for points in world_maps]
    run_extents = np.asarray([
        np.percentile(points - corner, 99.0, axis=0)
        for points, corner in zip(world_maps, corners)
    ])
    extent = np.percentile(run_extents, 95.0, axis=0)
    return corners, extent


def transform(points, matrix, offset):
    return np.asarray(points, float) @ matrix.T + offset


def render(path: Path, points: np.ndarray, poses: list[dict], title: str, extent):
    fig, ax = plt.subplots(figsize=(12, 6), dpi=150)
    if len(points):
        ax.scatter(points[:, 0], points[:, 1], s=2, c='#2563eb', alpha=.82,
                   edgecolors='none')
    if poses:
        trajectory = np.asarray([[p['x'], p['y']] for p in poses], float)
        ax.plot(trajectory[:, 0], trajectory[:, 1], c='#f97316', lw=1.2)
    ax.scatter([0], [0], c='#dc2626', s=35, zorder=4)
    ax.set_xlim(-1.0, float(extent[0]) + 1.0)
    ax.set_ylim(-1.0, float(extent[1]) + 1.0)
    ax.set_aspect('equal')
    ax.grid(True, alpha=.25)
    ax.set_xlabel('X, m')
    ax.set_ylabel('Y, m')
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main():
    builds = []
    aggregate = []
    for run in sorted(RUN_ROOT.glob('run_*')):
        if not (run / 'encoder.jsonl').is_file():
            continue
        print(f'=== improved: {run.name} ===', flush=True)
        result = build_global_map_from_run(
            run, method='improved', source='real', cameras=['up'],
            tag=f'{run.name}_improved', write_module_copy=False,
        )
        nav_pose = initial_nav_pose(run)
        world = local_to_nav(np.asarray(result['points'], float), nav_pose)
        aligned, correction = manhattan_align(world)
        builds.append((run, result, nav_pose, aligned, correction))
        aggregate.append(aligned)

    corners, extent = normalized_run_frames(aggregate)
    matrix = np.eye(2)
    manifest = {
        'method': 'improved',
        'frame': 'room_sw_corner',
        'origin_m': [0.0, 0.0],
        'axes': 'persistent nav X/Y; scene walls axis-aligned',
        'shared_matrix': matrix.tolist(),
        'origin_rule': 'per-run 1st-percentile SW corner pinned to (0,0)',
        'shared_view_extent_m': extent.tolist(),
        'runs': [],
    }
    for (run, result, nav_pose, world, correction), corner in zip(builds, corners):
        offset = -np.asarray(corner, float)
        points = transform(world, matrix, offset)
        poses = []
        nav_angle = math.radians(nav_pose[2] - 90.0)
        for source_pose in result['poses']:
            item = dict(source_pose)
            local_xy = np.asarray([[item['x'], item['y']]], float)
            world_xy = local_to_nav(local_xy, nav_pose)
            c, s = math.cos(correction), math.sin(correction)
            world_xy = world_xy @ np.array([[c, -s], [s, c]]).T
            xy = transform(world_xy, matrix, offset)[0]
            # Transform a unit heading vector, which also handles a shared flip.
            angle = float(item['pose'][2]) + nav_angle + correction
            vec = matrix @ np.array([math.cos(angle), math.sin(angle)])
            yaw = math.atan2(vec[1], vec[0])
            item['x'], item['y'] = float(xy[0]), float(xy[1])
            item['orientation_dgr'] = math.degrees(yaw)
            item['pose'] = [float(xy[0]), float(xy[1]), float(yaw)]
            poses.append(item)
        payload = {
            'map': points.tolist(),
            'poses': poses,
            'n_frames': len(poses),
            'cell_m': 0.05,
            'method': 'improved',
            'coordinate_frame': 'room_sw_corner',
            'origin_m': [0.0, 0.0],
            'axes': 'persistent nav X/Y; walls parallel to axes',
            'source_run': str(run),
            'initial_nav_pose': list(nav_pose),
            'shared_matrix': matrix.tolist(),
            'run_offset': offset.tolist(),
            'manhattan_correction_deg': math.degrees(correction),
            'stats': result['stats'],
        }
        out_json = Path(result['out_json'])
        out_png = Path(result['out_png'])
        out_json.write_text(json.dumps(payload, separators=(',', ':')), encoding='utf-8')
        render(out_png, points, poses,
               f'{run.name} — improved, shared room frame', extent)
        manifest['runs'].append({
            'run': run.name, 'map_json': str(out_json), 'map_png': str(out_png),
            'points': len(points), 'frames': len(poses),
            'sw_corner_nav_m': np.asarray(corner, float).tolist(),
        })
    manifest_path = ROOT / 'results/20260818/improved_aligned_manifest.json'
    manifest_path.write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
    print(f'wrote {manifest_path}')


if __name__ == '__main__':
    main()
