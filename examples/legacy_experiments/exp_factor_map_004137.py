#!/usr/bin/env python3
"""Compare factor-graph mapper variants on run_004137 (horizon / Pandar)."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

from flifltbot.mapping.pipeline import build_global_map_from_run  # noqa: E402


RUN = ROOT / 'maps' / '20260815' / 'sim' / 'run_004137'
OUT = ROOT / 'results' / '20260815' / 'sim' / 'run_004137'

VARIANTS = [
    # Already built earlier — re-run for fair timing if missing, else skip rebuild.
    dict(tag='factor', method='factor', kwargs={}),
    dict(
        tag='factor_noscale',
        method='factor',
        kwargs={'estimate_scale': False},
    ),
    dict(
        tag='factor_nolidar_seq',
        method='factor',
        kwargs={'skip_lidar_sequential': True},
    ),
]


def _summary(tag, result):
    stats = result['stats']
    return {
        'tag': tag,
        'method': result['method'],
        'map_png': str(result['out_png']),
        'map_json': str(result['out_json']),
        'mode': stats.get('mode'),
        'factor_mode': stats.get('factor_mode'),
        's_v': stats.get('s_v'),
        'estimate_scale': stats.get('estimate_scale'),
        'lidar_odom_dt_s': stats.get('lidar_odom_dt_s'),
        'lidar_odom_dt_half_width_s': stats.get('lidar_odom_dt_half_width_s'),
        'n_frames': stats.get('n_frames'),
        'keyframes': stats.get('keyframes'),
        'wheel_edges': stats.get('wheel_edges'),
        'lidar_seq_edges': stats.get('lidar_seq_edges'),
        'loop_closures': stats.get('loop_closures'),
        'stable_cells': stats.get('stable_cells'),
        'raw_cells': stats.get('raw_cells'),
        'loop_error_m': stats.get('loop_error_m'),
        'loop_error_deg': stats.get('loop_error_deg'),
        'pose_graph_cost': stats.get('pose_graph_cost'),
        'timing_s': stats.get('timing_s'),
        'wall_s': result.get('wall_s'),
    }


def main():
    if not RUN.is_dir():
        raise SystemExit(f'missing run: {RUN}')
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []

    # Reference rows from prior builds if present.
    for prior_tag, prior_png in (
        ('improved', OUT / 'run_004137.png'),
        ('clouds', OUT / 'run_004137_clouds.png'),
    ):
        if prior_png.is_file():
            rows.append({
                'tag': prior_tag,
                'method': prior_tag,
                'map_png': str(prior_png),
                'note': 'prior build (not re-run)',
            })

    for variant in VARIANTS:
        tag = variant['tag']
        print('=' * 64)
        print(f'BUILD {tag}  method={variant["method"]}  kwargs={variant["kwargs"]}')
        t0 = time.perf_counter()
        result = build_global_map_from_run(
            RUN,
            method=variant['method'],
            cameras=['horizon'],
            tag=f"run_004137_{tag}" if tag != 'factor' else 'run_004137_factor',
            source='sim',
            **variant['kwargs'],
        )
        result['wall_s'] = time.perf_counter() - t0
        row = _summary(tag, result)
        rows.append(row)
        print(json.dumps(row, indent=2))

    compare_path = OUT / 'factor_compare.json'
    compare_path.write_text(
        json.dumps({'run_dir': str(RUN), 'variants': rows}, indent=2),
        encoding='utf-8',
    )
    print('=' * 64)
    print(f'wrote {compare_path}')
    for row in rows:
        print(
            f"  {row['tag']:22s}  cells={row.get('stable_cells', '?')}  "
            f"s_v={row.get('s_v', '-')}  dt={row.get('lidar_odom_dt_s', '-')}  "
            f"loops={row.get('loop_closures', '-')}  "
            f"err_m={row.get('loop_error_m', '-')}"
        )


if __name__ == '__main__':
    main()
