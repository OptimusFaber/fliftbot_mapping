"""Map accuracy evaluation helpers."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from flifltbot.mapping import pipeline
from flifltbot.mapping.real_room_gt import (
    DEFAULT_GT_DIR,
    ROOM_X,
    ROOM_Y,
    all_gt_segments,
    floorplan_dict,
)

WEST_SX = -4.41528
EAST_SX = 5.60144
WEST_YS = [-0.690987, 2.30697, 5.30708, 8.34352]
EAST_YS = [8.34352, 5.30708, -0.690987, 2.30697]
SPAWN = (13.5, -11.0, math.pi / 2)

# Static geometry instantiated by simulation/worlds/warehouse.sdf.  ``shelf``
# has a link offset of -0.5 m in X and a 3.6 x 0.6 m collision.  ``shelf_big``
# uses the same offset and a 2.1 x 18 m collision.
WAREHOUSE_SHELVES = [
    (sx - 0.5, sy, 3.6, 0.6)
    for sx, sy in (
        *[(-4.41528, y) for y in WEST_YS],
        *[(5.60144, y) for y in EAST_YS],
        *[(13.3818, y) for y in (-21.2416, -19.0028, -16.4478, -14.1028)],
    )
] + [
    (sx - 0.5, sy, 2.1, 18.0)
    for sx, sy in (
        (-9.34177, -13.5598),
        (13.9821, 15.319),
        (6.19777, -12.9647),
        (0.594376, -12.9647),
        (-5.36284, -12.9647),
    )
]

# Ten 0.495 x 0.50 m structural columns extracted from the warehouse collision
# mesh at the up-LiDAR plane.  Their centres form two X rows and five Y rows.
WAREHOUSE_COLUMNS = [
    (cx, cy, 0.495, 0.50)
    for cx in (-7.425, 7.425)
    for cy in (-15.0, -7.5, 0.0, 7.5, 15.0)
]


def shelf_x_faces(sx: float):
    cx = sx - 0.5
    return cx - 1.8, cx + 1.8


W_HI = shelf_x_faces(WEST_SX)[1]
E_LO = shelf_x_faces(EAST_SX)[0]
AISLE_GT = E_LO - W_HI


def to_world(pts: np.ndarray) -> np.ndarray:
    x0, y0, yaw0 = SPAWN
    c, s = math.cos(yaw0), math.sin(yaw0)
    wx = x0 + c * pts[:, 0] - s * pts[:, 1]
    wy = y0 + s * pts[:, 0] + c * pts[:, 1]
    return np.column_stack([wx, wy])


def wall_median(world, axis, expected, tol=1.5):
    vals = world[:, 0] if axis == 'x' else world[:, 1]
    near = vals[np.abs(vals - expected) < tol]
    if len(near) < 30:
        return None
    return float(np.median(near))


def aisle_width(world):
    band = (world[:, 1] > 2.0) & (world[:, 1] < 6.0)
    xs = world[band, 0]
    if len(xs) < 50:
        return None, None, None
    hist, edges = np.histogram(xs, bins=np.arange(-8, 10, 0.05))
    centers = 0.5 * (edges[:-1] + edges[1:])

    def peak(target, radius=0.8):
        mask = np.abs(centers - target) < radius
        if not np.any(mask):
            return None
        return float(centers[np.argmax(hist * mask)])

    west = peak(W_HI)
    east = peak(E_LO)
    if west is None or east is None:
        return None, west, east
    return east - west, west, east


def gt_segments():
    segs = [
        ((-15, -25), (15, -25)),
        ((-15, 25), (15, 25)),
        ((-15, -25), (-15, 25)),
        ((15, -25), (15, 25)),
    ]
    for cx, cy, width, depth in WAREHOUSE_SHELVES + WAREHOUSE_COLUMNS:
        x0, x1 = cx - width / 2.0, cx + width / 2.0
        y0, y1 = cy - depth / 2.0, cy + depth / 2.0
        segs += [
            ((x0, y0), (x1, y0)),
            ((x0, y1), (x1, y1)),
            ((x0, y0), (x0, y1)),
            ((x1, y0), (x1, y1)),
        ]
    return segs


def dist_seg(p, a, b):
    a = np.asarray(a)
    b = np.asarray(b)
    p = np.asarray(p)
    ab = b - a
    t = np.clip(((p - a) @ ab) / max(ab @ ab, 1e-12), 0, 1)
    return float(np.linalg.norm(p - (a + t * ab)))


def align_bbox_to_gt(pts: np.ndarray) -> np.ndarray:
    """Translate map points so AABB min → (0,0). Good if run starts near SW corner."""
    pts = np.asarray(pts, float)
    if len(pts) == 0:
        return pts
    return pts - pts.min(axis=0)


def score_map(points):
    pts = np.asarray(points, float)
    world = to_world(pts)
    walls = {}
    for name, axis, exp in [
        ('west', 'x', -15),
        ('east', 'x', 15),
        ('south', 'y', -25),
        ('north', 'y', 25),
    ]:
        med = wall_median(world, axis, exp)
        walls[name] = None if med is None else (med - exp) * 100.0

    aisle, wf, ef = aisle_width(world)
    aisle_err = None if aisle is None else (aisle - AISLE_GT) * 100.0

    span_w = float(world[:, 0].max() - world[:, 0].min())
    span_l = float(world[:, 1].max() - world[:, 1].min())

    segs = gt_segments()
    rng = np.random.default_rng(0)
    sample = world[rng.choice(len(world), min(8000, len(world)), replace=False)]
    dists = np.array([min(dist_seg(p, a, b) for a, b in segs) for p in sample])
    close = dists[dists < 1.0]
    ghost = world[
        (world[:, 0] > -2.5)
        & (world[:, 0] < 2.5)
        & (world[:, 1] > 0.0)
        & (world[:, 1] < 8.0)
    ]
    wall_errs = [abs(v) for v in walls.values() if v is not None]
    return {
        'n_cells': len(pts),
        'span_w_m': span_w,
        'span_l_m': span_l,
        'span_w_err_cm': (span_w - 30.0) * 100.0,
        'span_l_err_cm': (span_l - 50.0) * 100.0,
        'wall_err_cm': walls,
        'mean_abs_wall_cm': float(np.mean(wall_errs)) if wall_errs else None,
        'aisle_m': aisle,
        'aisle_err_cm': aisle_err,
        'aisle_faces': (wf, ef),
        'near_gt_median_cm': (
            float(np.median(close) * 100) if len(close) else None
        ),
        'near_gt_p95_cm': (
            float(np.percentile(close, 95) * 100) if len(close) else None
        ),
        'ghost_aisle_cells': int(len(ghost)),
    }


def score_map_real_room(points, *, align='bbox', gt_path: Path | None = None):
    """Score a real-robot map against the hand-measured room floorplan."""
    pts = np.asarray(points, float)
    if align == 'bbox':
        pts = align_bbox_to_gt(pts)
    elif align != 'none':
        raise ValueError(f'unknown align mode: {align}')

    span_w = float(pts[:, 0].max() - pts[:, 0].min()) if len(pts) else 0.0
    span_l = float(pts[:, 1].max() - pts[:, 1].min()) if len(pts) else 0.0

    if gt_path is not None and Path(gt_path).is_file():
        data = json.loads(Path(gt_path).read_text(encoding='utf-8'))
        # Prefer up-lidar segments (window/door recesses + tall obstacles only)
        raw = data.get('segments_up_lidar') or data.get('segments', [])
        segs = [(tuple(a), tuple(b)) for a, b in raw]
        span_gt_w = float(data.get('span_gt_m', {}).get('width_x', ROOM_X))
        span_gt_l = float(data.get('span_gt_m', {}).get('length_y', ROOM_Y))
    else:
        segs = all_gt_segments(include_furniture=True, lidar_up=True)
        span_gt_w, span_gt_l = ROOM_X, ROOM_Y

    rng = np.random.default_rng(0)
    if len(pts) == 0:
        dists = np.array([])
        close = dists
    else:
        sample = pts[rng.choice(len(pts), min(8000, len(pts)), replace=False)]
        dists = np.array([min(dist_seg(p, a, b) for a, b in segs) for p in sample])
        close = dists[dists < 1.0]

    walls = {}
    for name, axis, exp, tol in [
        ('west', 'x', 0.0, 1.0),
        ('east', 'x', span_gt_w, 1.5),
        ('south', 'y', 0.0, 1.0),
        ('north', 'y', span_gt_l, 1.5),
    ]:
        med = wall_median(pts, axis, exp, tol=tol)
        walls[name] = None if med is None else (med - exp) * 100.0
    wall_errs = [abs(v) for v in walls.values() if v is not None]

    return {
        'n_cells': len(pts),
        'gt': 'real_room',
        'align': align,
        'span_w_m': span_w,
        'span_l_m': span_l,
        'span_w_gt_m': span_gt_w,
        'span_l_gt_m': span_gt_l,
        'span_w_err_cm': (span_w - span_gt_w) * 100.0,
        'span_l_err_cm': (span_l - span_gt_l) * 100.0,
        'wall_err_cm': walls,
        'mean_abs_wall_cm': float(np.mean(wall_errs)) if wall_errs else None,
        'near_gt_median_cm': (
            float(np.median(close) * 100) if len(close) else None
        ),
        'near_gt_p95_cm': (
            float(np.percentile(close, 95) * 100) if len(close) else None
        ),
        'near_gt_mean_cm': float(np.mean(close) * 100) if len(close) else None,
        'frac_within_50cm': float(np.mean(dists < 0.5)) if len(dists) else None,
        'frac_within_1m': float(np.mean(dists < 1.0)) if len(dists) else None,
    }


def evaluate_run(
    run_dir,
    method='improved',
    source='sim',
    cameras=None,
    *,
    gt_path=None,
    align='bbox',
):
    build = pipeline.build_global_map_from_run(run_dir, method=method, cameras=cameras)
    summary = {
        'run_dir': str(run_dir),
        'method': method,
        'source': source,
        'map_json': str(build['out_json']),
        'map_png': str(build['out_png']),
    }
    if source == 'sim':
        summary['metrics'] = score_map(build['points'])
    elif source == 'real':
        gt = Path(gt_path) if gt_path else DEFAULT_GT_DIR / 'floorplan.json'
        summary['gt_path'] = str(gt)
        summary['metrics'] = score_map_real_room(
            build['points'], align=align, gt_path=gt if gt.is_file() else None
        )
        summary['gt_info'] = {
            'room_size_m': [ROOM_X, ROOM_Y],
            'default_gt_dir': str(DEFAULT_GT_DIR),
            'heights': floorplan_dict()['heights_m'],
        }
    else:
        summary['metrics'] = {
            'n_cells': len(build['points']),
            'note': 'unknown source; no GT scoring',
        }
    # Always write under results/…, never into maps/ recording dirs.
    from flifltbot.common.output.layout import infer_results_dir

    results_dir = infer_results_dir(run_dir, source=source)
    out_path = results_dir / f'accuracy_{method}_{source}.json'
    out_path.write_text(json.dumps(summary, indent=2) + '\n', encoding='utf-8')
    summary['summary_path'] = str(out_path)
    return summary
