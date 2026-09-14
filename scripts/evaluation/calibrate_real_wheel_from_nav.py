#!/usr/bin/env python3
"""Evaluate src bicycle odometry against nav_* GT in real encoder.jsonl runs."""
from __future__ import annotations

import argparse, glob, json, math
from pathlib import Path

import numpy as np


def wrap(a):
    return math.atan2(math.sin(a), math.cos(a))


def load_run(path):
    rows = []
    for line in Path(path).read_text().splitlines():
        record = json.loads(line)
        gt = record.get('check_logs') or {}
        if not str(gt.get('nav_t', '')).startswith('2026_'):
            continue
        try:
            rows.append({
                't': float(record.get('enc_unix')),
                'v': float(record['enc_v']),
                'steer': math.radians(float(record['enc_dgr'])),
                'gx': float(gt['nav_x']), 'gy': float(gt['nav_y']),
                'gyaw': math.radians(float(gt['nav_dgr'])),
            })
        except (KeyError, TypeError, ValueError):
            continue
    rows.sort(key=lambda r: r['t'])
    return rows


def evaluate(rows, wheelbase, speed_sign, speed_scale=1.0, steering_offset=0.0):
    x, y, yaw = rows[0]['gx'], rows[0]['gy'], rows[0]['gyaw']
    pos_errors = []; yaw_errors = []
    for a, b in zip(rows, rows[1:]):
        dt = b['t'] - a['t']
        if not 0.0 < dt <= 0.5:
            continue
        speed = speed_sign * speed_scale * 0.5 * (a['v'] + b['v'])
        steering = 0.5 * (a['steer'] + b['steer']) + steering_offset
        vx = speed * math.cos(steering)
        wz = speed * math.sin(steering) / wheelbase
        dtheta = wz * dt
        if abs(wz) < 1e-10:
            dx, dy = vx * dt, 0.0
        else:
            dx = vx * math.sin(dtheta) / wz
            dy = vx * (1.0 - math.cos(dtheta)) / wz
        c, s = math.cos(yaw), math.sin(yaw)
        x += c * dx - s * dy
        y += s * dx + c * dy
        yaw = wrap(yaw + dtheta)
        pos_errors.append(math.hypot(x - b['gx'], y - b['gy']))
        yaw_errors.append(abs(wrap(yaw - b['gyaw'])))
    return {
        'samples': len(rows),
        'duration_s': rows[-1]['t'] - rows[0]['t'],
        'gt_distance_m': float(sum(math.hypot(b['gx']-a['gx'], b['gy']-a['gy'])
                                   for a, b in zip(rows, rows[1:]))),
        'position_rmse_m': float(np.sqrt(np.mean(np.square(pos_errors)))) if pos_errors else None,
        'position_final_m': float(pos_errors[-1]) if pos_errors else None,
        'yaw_rmse_deg': math.degrees(float(np.sqrt(np.mean(np.square(yaw_errors))))) if yaw_errors else None,
        'yaw_final_deg': math.degrees(float(yaw_errors[-1])) if yaw_errors else None,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('roots', nargs='+')
    ap.add_argument('--wheelbase', type=float, default=1.410)
    ap.add_argument('--speed-sign', type=float, default=-1.0)
    ap.add_argument('--fixed-steering-offset-deg', type=float, default=-1.35)
    ap.add_argument('--auto-steering-offset', action='store_true')
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    files = []
    for root in args.roots:
        files.extend(glob.glob(str(Path(root) / '**/encoder.jsonl'), recursive=True))
    runs = []
    for path in sorted(set(files)):
        rows = load_run(path)
        if len(rows) < 5:
            continue
        result = evaluate(rows, args.wheelbase, args.speed_sign)
        fixed = evaluate(
            rows, args.wheelbase, args.speed_sign,
            steering_offset=math.radians(args.fixed_steering_offset_deg),
        )
        result['baseline'] = dict(result)
        result['fixed_offset_deg'] = args.fixed_steering_offset_deg
        result['fixed'] = fixed
        if args.auto_steering_offset:
            candidates = np.linspace(-2.5, 0.0, 251)
            scored = [
                (evaluate(rows, args.wheelbase, args.speed_sign,
                          steering_offset=math.radians(float(deg))), float(deg))
                for deg in candidates
            ]
            auto, auto_deg = min(
                scored, key=lambda item: item[0]['position_rmse_m'])
            result['auto_offset_deg'] = auto_deg
            result['auto'] = auto
        result['run'] = str(Path(path).parent)
        runs.append(result)
    moving = [r for r in runs if r['gt_distance_m'] >= 0.5]
    report = {
        'model': {
            'wheelbase_m': args.wheelbase,
            'wheel_diameter_m': 0.23,
            'body_speed': f'{args.speed_sign:+g} * enc_v',
            'steering': 'radians(enc_dgr), 0 straight, +left',
        },
        'runs': runs,
        'summary_moving_runs': {
            'count': len(moving),
            'median_final_position_error_m': float(np.median(
                [r['position_final_m'] for r in moving])) if moving else None,
            'p90_final_position_error_m': float(np.percentile(
                [r['position_final_m'] for r in moving], 90)) if moving else None,
            'median_final_yaw_error_deg': float(np.median(
                [r['yaw_final_deg'] for r in moving])) if moving else None,
            'p90_final_yaw_error_deg': float(np.percentile(
                [r['yaw_final_deg'] for r in moving], 90)) if moving else None,
        },
    }
    if moving:
        def mean_metric(section, key):
            vals = [r[section][key] for r in moving if section in r]
            return float(np.mean(vals)) if vals else None
        report['comparison_moving_runs'] = {
            'count': len(moving),
            'baseline': {
                'mean_position_rmse_m': mean_metric('baseline', 'position_rmse_m'),
                'mean_final_position_error_m': mean_metric('baseline', 'position_final_m'),
                'mean_yaw_rmse_deg': mean_metric('baseline', 'yaw_rmse_deg'),
            },
            'fixed': {
                'offset_deg': args.fixed_steering_offset_deg,
                'mean_position_rmse_m': mean_metric('fixed', 'position_rmse_m'),
                'mean_final_position_error_m': mean_metric('fixed', 'position_final_m'),
                'mean_yaw_rmse_deg': mean_metric('fixed', 'yaw_rmse_deg'),
            },
        }
        if args.auto_steering_offset:
            report['comparison_moving_runs']['auto'] = {
                'offset_range_deg': [-2.5, 0.0],
                'mean_position_rmse_m': mean_metric('auto', 'position_rmse_m'),
                'mean_final_position_error_m': mean_metric('auto', 'position_final_m'),
                'mean_yaw_rmse_deg': mean_metric('auto', 'yaw_rmse_deg'),
                'median_offset_deg': float(np.median([
                    r['auto_offset_deg'] for r in moving
                ])),
            }
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report.get('comparison_moving_runs', {}), indent=2))
    print(out)


if __name__ == '__main__':
    main()
