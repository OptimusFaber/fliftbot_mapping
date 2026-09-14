"""Final hit-grid map filtering and assembly."""

from __future__ import annotations

import math

import numpy as np

from .config import HZ, MAP_CELL, MAP_MIN_COMPONENT_CELLS, MAP_MIN_HITS
from .deskew import frame_yaw_rate
from .geometry_icp import local_to_global, normalize_angle

def remove_isolated_map_noise(cells, min_component_cells=None):
    """Remove small disconnected groups caused by persistent LiDAR reflections."""
    min_component_cells = (
        MAP_MIN_COMPONENT_CELLS
        if min_component_cells is None
        else int(min_component_cells)
    )
    remaining = set(cells)
    accepted = set()
    neighbours = [
        (dx, dy)
        for dx in (-1, 0, 1)
        for dy in (-1, 0, 1)
        if dx != 0 or dy != 0
    ]
    while remaining:
        seed = remaining.pop()
        component = {seed}
        stack = [seed]
        while stack:
            x, y = stack.pop()
            for dx, dy in neighbours:
                neighbour = (x + dx, y + dy)
                if neighbour in remaining:
                    remaining.remove(neighbour)
                    component.add(neighbour)
                    stack.append(neighbour)
        if len(component) >= min_component_cells:
            accepted.update(component)
    return accepted


def build_global_map(frames, poses):
    """
    Transform ALL local points from ALL frames to global coordinates first.
    Then use occupancy hit counts to remove isolated LiDAR noise.
    """
    map_points, total_transformed, raw_cells, _stats = build_global_map_filtered(
        frames, poses, encoders=None, skip_turn_wz=None, min_hits=None
    )
    return map_points, total_transformed, raw_cells


def build_global_map_filtered(
    frames,
    poses,
    encoders=None,
    skip_turn_wz=None,
    min_hits=None,
    frame_times=None,
    cell_m=None,
    min_component_cells=None,
):
    """
    Hit-grid map builder with optional turn-frame dropout.

    Turn rate prefers encoder enc_wz; if missing, uses |Δyaw|/dt from poses.
    """
    cell_m = MAP_CELL if cell_m is None else float(cell_m)
    min_hits = MAP_MIN_HITS if min_hits is None else int(min_hits)
    if min_component_cells is None:
        min_component_cells = max(
            8, int(round(MAP_MIN_COMPONENT_CELLS * (MAP_CELL / cell_m) ** 2))
        )
    skip_turn_wz = (
        None if skip_turn_wz is None else float(skip_turn_wz)
    )
    encoders = encoders or [None] * len(frames)
    hit_counts = {}
    total_transformed = 0
    used_frames = 0
    skipped_turns = 0
    for frame_index, (local_points, pose) in enumerate(zip(frames, poses)):
        if skip_turn_wz is not None:
            wz = frame_yaw_rate(
                encoders[frame_index] if frame_index < len(encoders) else None
            )
            if wz < 1e-9 and frame_index > 0:
                prev = poses[frame_index - 1]
                # pose may be list [x,y,yaw] or dict with pose
                if isinstance(pose, dict):
                    yaw = float(pose.get('yaw', pose['pose'][2]))
                    yaw_prev = float(prev.get('yaw', prev['pose'][2]))
                else:
                    yaw = float(pose[2])
                    yaw_prev = float(prev[2])
                dt = 1.0 / HZ
                if frame_times is not None and frame_index < len(frame_times):
                    t1 = frame_times[frame_index]
                    t0 = frame_times[frame_index - 1]
                    if t0 is not None and t1 is not None and t1 > t0:
                        dt = float(t1 - t0)
                if dt > 1e-3:
                    wz = abs(normalize_angle(yaw - yaw_prev)) / dt
            if wz >= skip_turn_wz:
                skipped_turns += 1
                continue
        if local_points is None or len(local_points) == 0:
            continue
        global_points = local_to_global(local_points, pose)
        total_transformed += len(global_points)
        cells = np.round(global_points / cell_m).astype(np.int32)
        for cell in set(map(tuple, cells.tolist())):
            hit_counts[cell] = hit_counts.get(cell, 0) + 1
        used_frames += 1
        if (frame_index + 1) % 300 == 0:
            print(f'  global transform {frame_index + 1}/{len(frames)}')

    stable_cells = {
        cell for cell, hits in hit_counts.items()
        if hits >= min_hits
    }
    filtered_cells = remove_isolated_map_noise(
        stable_cells, min_component_cells=min_component_cells
    )
    map_points = [
        [round(cell[0] * cell_m, 4), round(cell[1] * cell_m, 4)]
        for cell in filtered_cells
    ]
    return map_points, total_transformed, len(hit_counts), {
        'used_frames': used_frames,
        'skipped_turn_frames': skipped_turns,
        'min_hits': min_hits,
        'skip_turn_wz': skip_turn_wz,
        'cell_m': cell_m,
        'min_component_cells': int(min_component_cells),
    }

