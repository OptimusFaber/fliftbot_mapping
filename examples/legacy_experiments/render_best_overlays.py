#!/usr/bin/env python3
"""Render best current map/trajectory against room and warehouse GT."""
from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial import cKDTree

from flifltbot.mapping.real_room_gt import ROOM_X, ROOM_Y, all_gt_segments
from flifltbot.common.runtime.accuracy_eval import gt_segments

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / 'results/20260818/overlays'


def densify(segments, step=0.025):
    chunks = []
    for a, b in segments:
        a, b = np.asarray(a, float), np.asarray(b, float)
        n = max(2, int(np.linalg.norm(b - a) / step) + 1)
        chunks.append(a + np.linspace(0.0, 1.0, n)[:, None] * (b - a))
    return np.vstack(chunks)


def apply_se2(points, pose):
    pts = np.asarray(points, float)
    c, s = math.cos(pose[2]), math.sin(pose[2])
    return pts @ np.array([[c, s], [-s, c]]) + np.asarray(pose[:2])


def align_room(points, gt, *, yaw_limit_deg=None):
    """Rigid-only robust alignment; deliberately never changes map scale."""
    pts = np.asarray(points, float)
    if len(pts) > 10000:
        pts = pts[np.linspace(0, len(pts) - 1, 10000).astype(int)]
    tree = cKDTree(gt)
    best = None
    if yaw_limit_deg is None:
        yaw_seeds = np.linspace(-math.pi, math.pi, 16, endpoint=False)
        bounds = (-np.inf, np.inf)
    else:
        yaw_limit = math.radians(float(yaw_limit_deg))
        yaw_seeds = np.linspace(-yaw_limit, yaw_limit, 7)
        bounds = ([-np.inf, -np.inf, -yaw_limit],
                  [np.inf, np.inf, yaw_limit])
    for yaw in yaw_seeds:
        rotated = apply_se2(pts, (0, 0, yaw))
        seed_xy = np.median(gt, axis=0) - np.median(rotated, axis=0)

        def residual(x):
            moved = apply_se2(pts, x)
            distances, _ = tree.query(moved, k=1)
            return np.minimum(distances, 0.75)

        result = least_squares(
            residual, [seed_xy[0], seed_xy[1], yaw],
            bounds=bounds, loss='huber', f_scale=0.08, max_nfev=80,
        )
        cost = float(np.median(tree.query(apply_se2(pts, result.x))[0]))
        if best is None or cost < best[0]:
            best = (cost, result.x)
    return best


def load_result(path):
    data = json.loads(Path(path).read_text(encoding='utf-8'))
    points = np.asarray(data['map'], float)
    poses = np.asarray([p['pose'] for p in data['poses']], float)
    return points, poses


def plot_segments(ax, segments, **kwargs):
    label = kwargs.pop('label', None)
    for index, (a, b) in enumerate(segments):
        ax.plot([a[0], b[0]], [a[1], b[1]],
                label=label if index == 0 else None, **kwargs)


def rotate_room_segments_180(segments):
    """Turn the measured room plan 180 degrees in its XY plane."""
    return [
        ((ROOM_X - float(a[0]), ROOM_Y - float(a[1])),
         (ROOM_X - float(b[0]), ROOM_Y - float(b[1])))
        for a, b in segments
    ]


def render_room():
    source = ROOT / 'results/20260814/real/run_190205/room190205_full_factor.json'
    points, poses = load_result(source)
    # User-verified room orientation: the measured plan must be turned by 180°
    # in XY.  Keep this orientation fixed; only a small residual yaw correction
    # is allowed while translating the SLAM result onto the plan.
    segments = rotate_room_segments_180(
        all_gt_segments(include_furniture=True, lidar_up=True)
    )
    dense = densify(segments)
    median_m, transform = align_room(points, dense, yaw_limit_deg=15.0)
    aligned_map = apply_se2(points, transform)
    aligned_route = apply_se2(poses[:, :2], transform)

    fig, ax = plt.subplots(figsize=(13, 7))
    plot_segments(ax, segments, color='black', linewidth=1.4, alpha=0.8,
                  label='GT geometry')
    ax.scatter(aligned_map[:, 0], aligned_map[:, 1], s=1.2, c='#1677ff',
               alpha=0.45, label='SLAM map')
    ax.plot(aligned_route[:, 0], aligned_route[:, 1], c='#e63946', linewidth=1.2,
            label='estimated route')
    ax.scatter(*aligned_route[0], c='limegreen', s=45, zorder=5, label='start')
    ax.scatter(*aligned_route[-1], c='darkred', s=45, zorder=5, label='finish')
    ax.set_title(f'Room run_190205: full factor map + route vs measured GT\n'
                 f'rigid alignment median residual = {median_m*100:.1f} cm (no scale)')
    ax.set_aspect('equal'); ax.grid(alpha=0.2); ax.legend(loc='best')
    ax.set_xlabel('room X, m'); ax.set_ylabel('room Y, m')
    fig.tight_layout()
    path = OUT / 'room190205_best_map_route_vs_gt.png'
    fig.savefig(path, dpi=180); plt.close(fig)
    return {'source': str(source), 'overlay': str(path),
            'rigid_alignment_median_cm': median_m * 100,
            'transform_xy_yaw': transform.tolist(),
            'gt_transform': (
                '(x, y) -> (ROOM_X - x, ROOM_Y - y), 180 deg in XY'
            ),
            'note': 'No independent GT trajectory exists for the real run.'}


def render_warehouse():
    source = ROOT / 'results/20260815/sim/run_004137/warehouse_full_factor.json'
    points, poses = load_result(source)
    enc_path = ROOT / 'maps/20260815/sim/run_004137/encoder.jsonl'
    gt_route = []
    gt_yaw = []
    for line in enc_path.read_text(encoding='utf-8').splitlines():
        row = json.loads(line)
        if row.get('odom_x') is not None and row.get('odom_y') is not None:
            gt_route.append((float(row['odom_x']), float(row['odom_y'])))
            gt_yaw.append(float(row['odom_yaw']))
    gt_route = np.asarray(gt_route, float)

    if not len(gt_route):
        raise RuntimeError(f'No Gazebo /odom GT in {enc_path}')

    # Anchor the SLAM frame to the actual first Gazebo pose.  Do not use the
    # old hard-coded SPAWN and do not fit the whole drifting trajectory: both
    # would move the warehouse plan relative to the recorded run.
    yaw = float(gt_yaw[0] - poses[0, 2])
    rotated_start = apply_se2(poses[:1, :2], (0.0, 0.0, yaw))[0]
    translation = gt_route[0] - rotated_start
    world_map = apply_se2(points, (translation[0], translation[1], yaw))
    world_route = apply_se2(poses[:, :2], (translation[0], translation[1], yaw))

    fig, ax = plt.subplots(figsize=(10, 14))
    plot_segments(ax, gt_segments(), color='black', linewidth=1.1, alpha=0.7,
                  label='GT geometry')
    ax.scatter(world_map[:, 0], world_map[:, 1], s=0.7, c='#1677ff', alpha=0.35,
               label='SLAM map')
    ax.plot(world_route[:, 0], world_route[:, 1], c='#e63946', linewidth=1.0,
            label='SLAM route')
    if len(gt_route):
        ax.plot(gt_route[:, 0], gt_route[:, 1], c='limegreen', linewidth=1.1,
                label='Gazebo /odom GT route')
    ax.set_title('Warehouse run_004137: full factor map and route vs Gazebo GT')
    ax.set_aspect('equal'); ax.grid(alpha=0.2); ax.legend(loc='best')
    ax.set_xlabel('world X, m'); ax.set_ylabel('world Y, m')
    fig.tight_layout()
    path = OUT / 'warehouse_best_map_route_vs_gt.png'
    fig.savefig(path, dpi=180); plt.close(fig)
    return {'source': str(source), 'overlay': str(path),
            'gt_route_points': int(len(gt_route)),
            'alignment': 'first synchronized Gazebo /odom pose',
            'transform_xy_yaw': [float(translation[0]), float(translation[1]), yaw]}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    summary = {'room': render_room(), 'warehouse': render_warehouse()}
    path = OUT / 'overlay_summary.json'
    path.write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
