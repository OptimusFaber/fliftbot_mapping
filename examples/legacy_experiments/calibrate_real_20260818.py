#!/usr/bin/env python3
"""Build calibrated maps and summarize the 2026-08-18 motion trials."""
from __future__ import annotations

import json
import math
import statistics
from pathlib import Path

from flifltbot.mapping.encoder_sync import calibrated_curvature
from flifltbot.mapping.pipeline import build_global_map_from_run

ROOT = Path(__file__).resolve().parents[3]
RUN_ROOT = ROOT / 'maps/20260818/real'
OUT = ROOT / 'results/20260818/calibration'


def median_pose(rows, tail=False):
    n = max(3, len(rows) // 10)
    sample = rows[-n:] if tail else rows[:n]
    return tuple(statistics.median(row[i] for row in sample) for i in range(3))


def motion_metrics(run):
    records = [json.loads(line) for line in (run / 'encoder.jsonl').open()]
    rows = []
    moving_steer = []
    for record in records:
        check = record['check_logs']
        rows.append((float(check['nav_x']), float(check['nav_y']), float(check['nav_dgr'])))
        if abs(float(record['enc_v'])) > 0.02:
            moving_steer.append(float(record['enc_dgr']))
    start = median_pose(rows)
    end = median_pose(rows, tail=True)
    heading = (end[2] - start[2] + 180.0) % 360.0 - 180.0
    path = sum(math.hypot(b[0] - a[0], b[1] - a[1]) for a, b in zip(rows, rows[1:]))
    steer = statistics.median(moving_steer) if moving_steer else 0.0
    direction = 'forward' if 'forward' in run.name and 'backward_forward' not in run.name else 'backward'
    return {
        'run': run.name,
        'direction': direction,
        'steering_deg': steer,
        'path_m': path,
        'chord_m': math.hypot(end[0] - start[0], end[1] - start[1]),
        'heading_deg': heading,
        'curvature_rad_m': abs(math.radians(heading)) / path if path else 0.0,
    }


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for run in sorted(RUN_ROOT.glob('run_*')):
        if not (run / 'encoder.jsonl').is_file():
            continue
        print(f'=== {run.name} ===', flush=True)
        metrics = motion_metrics(run)
        result = build_global_map_from_run(
            run, method='calibrated', source='real', cameras=['up'],
            tag=f'{run.name}_calibrated', write_module_copy=False,
        )
        metrics.update({
            'map_json': str(result['out_json']),
            'map_png': str(result['out_png']),
            'frames': result['stats'].get('n_frames'),
            'keyframes': result['stats'].get('keyframes'),
            'stable_cells': result['stats'].get('stable_cells'),
            'lidar_seq_rmse_median': result['stats'].get('lidar_seq_rmse_median'),
        })
        rows.append(metrics)

    comparison = []
    for angle in (15, 30, 45, 60):
        selected = [r for r in rows if round(abs(r['steering_deg'])) == angle]
        real_heading = statistics.mean(abs(r['heading_deg']) for r in selected)
        real_path = statistics.mean(r['path_m'] for r in selected)
        sim_heading = math.degrees(2.0 * math.tan(math.radians(angle)) / 1.41)
        calibrated = []
        for direction in ('forward', 'backward'):
            curve = calibrated_curvature(math.radians(angle), forward=direction == 'forward')
            calibrated.append(math.degrees(2.0 * curve))
        comparison.append({
            'steering_deg': angle,
            'real_heading_mean_deg': real_heading,
            'real_path_mean_m': real_path,
            'sim_nominal_heading_for_2m_deg': sim_heading,
            'sim_minus_real_deg': sim_heading - real_heading,
            'calibrated_heading_for_2m_deg': statistics.mean(calibrated),
        })
    report = {
        'dataset': str(RUN_ROOT),
        'method': 'calibrated',
        'calibration_source': '24 constant-steer real runs; 3 repeats per direction and angle',
        'runs': rows,
        'real_vs_sim': comparison,
    }
    path = OUT / 'calibrated_report.json'
    path.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(f'wrote {path}')


if __name__ == '__main__':
    main()
