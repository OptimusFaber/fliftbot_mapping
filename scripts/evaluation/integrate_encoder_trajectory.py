#!/usr/bin/env python3
"""Integrate a real move logger using only enc_t, enc_v and enc_dgr."""
import argparse, csv, json, math
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt

WHEELBASE_M = 1.410

def timestamp(value):
    return datetime.strptime(value, '%Y_%m_%d_%H_%M_%S_%f').timestamp()

def wrap(a):
    return math.atan2(math.sin(a), math.cos(a))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--log', required=True)
    ap.add_argument('--start', required=True, help='logger enc_t timestamp')
    ap.add_argument('--end', required=True, help='logger enc_t timestamp')
    ap.add_argument('--out', required=True, help='output trajectory JSON')
    ap.add_argument('--plot', required=True)
    ap.add_argument('--wheelbase', type=float, default=WHEELBASE_M)
    ap.add_argument('--steering-sign', type=float, default=1.0)
    ap.add_argument('--speed-sign', type=float, default=-1.0,
                    help='src convention: body speed = -enc_v')
    args = ap.parse_args()
    lo, hi = timestamp(args.start), timestamp(args.end)
    samples = []; nav = []
    with Path(args.log).open(newline='') as f:
        for row in csv.DictReader(f):
            raw = row.get('enc_t', '')
            if not raw or raw.startswith('1970_'):
                continue
            try: t = timestamp(raw)
            except ValueError: continue
            if lo <= t <= hi:
                samples.append((t, float(row['enc_v']), float(row['enc_dgr'])))
                try:
                    nav.append((float(row['nav_x']), float(row['nav_y'])))
                except (KeyError, TypeError, ValueError):
                    pass
    samples.sort()
    if len(samples) < 2: raise SystemExit('not enough encoder samples')
    x = y = yaw = 0.0
    out = [{'t': samples[0][0], 'x': x, 'y': y, 'yaw': yaw,
            'enc_v': samples[0][1], 'enc_dgr': samples[0][2]}]
    for (t0, v0, a0), (t1, v1, a1) in zip(samples, samples[1:]):
        dt = max(0.0, min(t1 - t0, 0.5))
        v = args.speed_sign * 0.5 * (v0 + v1)
        steer = math.radians(0.5 * (a0 + a1)) * args.steering_sign
        wz = v * math.sin(steer) / args.wheelbase
        dtheta = wz * dt
        if abs(wz) < 1e-10:
            dx, dy = v * math.cos(steer) * dt, 0.0
        else:
            radius_v = v * math.cos(steer) / wz
            dx, dy = radius_v * math.sin(dtheta), radius_v * (1-math.cos(dtheta))
        c, s = math.cos(yaw), math.sin(yaw)
        x += c*dx - s*dy; y += s*dx + c*dy; yaw = wrap(yaw+dtheta)
        out.append({'t': t1, 'x': x, 'y': y, 'yaw': yaw,
                    'enc_v': v1, 'enc_dgr': a1})
    out_path = Path(args.out); out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({'source_fields':['enc_t','enc_v','enc_dgr'],
                                    'wheelbase_m':args.wheelbase,
                                    'speed_sign':args.speed_sign,
                                    'steering_sign':args.steering_sign,
                                    'trajectory':out}, indent=2))
    plt.figure(figsize=(10,8)); plt.plot([p['x'] for p in out], [p['y'] for p in out])
    if nav:
        plt.plot([p[0] for p in nav], [p[1] for p in nav], alpha=.7, label='nav')
        plt.legend()
    plt.axis('equal'); plt.grid(True); plt.xlabel('x, m'); plt.ylabel('y, m')
    plt.title('Encoder trajectory integrated from enc_t + enc_v + enc_dgr')
    plt.savefig(args.plot, dpi=180, bbox_inches='tight'); plt.close()
    print(f'wrote {len(out)} poses: {out_path}\nplot: {args.plot}')

if __name__ == '__main__': main()
