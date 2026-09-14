#!/usr/bin/env python3
"""Reproducible frontend/factor A/B on the recorded Gazebo trajectories."""
from __future__ import annotations

import json
import time
from pathlib import Path

from flifltbot.mapping.pipeline import build_global_map_from_run
from flifltbot.common.runtime.accuracy_eval import score_map_real_room, score_map

ROOT = Path(__file__).resolve().parents[3]
RUNS = {
    'room_190205': ROOT / 'maps/20260814/real/run_190205',
    'room_162516': ROOT / 'maps/20260807/real/run_162516',
    'warehouse': ROOT / 'maps/20260815/sim/run_004137',
}
OUT = ROOT / 'results/20260818/sim_ab'
MAX_FRAMES = 400

VARIANTS = {
    'clouds': dict(method='clouds'),
    'wheel_only': dict(method='factor', frontend_icp='wheel',
                       skip_lidar_sequential=True, skip_loop_closures=True,
                       estimate_scale=False),
    'factor_no_loops': dict(method='factor', skip_loop_closures=True,
                            estimate_scale=False),
    'factor': dict(method='factor', estimate_scale=False),
}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    report = {'max_frames': MAX_FRAMES, 'runs': {}}
    for scene, run in RUNS.items():
        if not run.is_dir():
            raise SystemExit(f'missing run: {run}')
        rows = []
        source = 'real' if scene.startswith('room_') else 'sim'
        for name, kwargs in VARIANTS.items():
            print(f'=== {scene} / {name} ===', flush=True)
            started = time.perf_counter()
            result = build_global_map_from_run(
                run, source=source, cameras=['up'], max_frames=MAX_FRAMES,
                tag=f'{scene}_{name}_20260818', **kwargs,
            )
            stats = result['stats']
            row = {
                'scene': scene,
                'variant': name,
                'method': kwargs['method'],
                'wall_s': time.perf_counter() - started,
                'map_json': str(result['out_json']),
                'map_png': str(result['out_png']),
                'n_frames': stats.get('n_frames'),
                'keyframes': stats.get('keyframes'),
                'wheel_edges': stats.get('wheel_edges'),
                'lidar_seq_edges': stats.get('lidar_seq_edges'),
                'loop_closures': stats.get('loop_closures'),
                'lidar_seq_rmse_median': stats.get('lidar_seq_rmse_median'),
                'lidar_seq_overlap_median': stats.get('lidar_seq_overlap_median'),
                'loop_rmse_median': stats.get('loop_rmse_median'),
                'loop_error_m': stats.get('loop_error_m'),
                'loop_error_deg': stats.get('loop_error_deg'),
                'stable_cells': stats.get('stable_cells'),
                'raw_cells': stats.get('raw_cells'),
                'transformed_points': stats.get('transformed_points'),
                'up_body_x': stats.get('up_body_x'),
                'deskew_mode': stats.get('deskew_mode'),
                'se2_deskewed_points': stats.get('se2_deskewed_points'),
            }
            row['map_metrics'] = (
                score_map_real_room(result['points']) if source == 'real'
                else score_map(result['points'])
            )
            rows.append(row)
            print(json.dumps(row, indent=2), flush=True)
        report['runs'][scene] = rows
    path = OUT / 'gazebo_ab_20260818.json'
    path.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(f'wrote {path}')


if __name__ == '__main__':
    main()
