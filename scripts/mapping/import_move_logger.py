#!/usr/bin/env python3
"""Replace a placeholder encoder.jsonl with samples from fliftbot_move_logger."""
import argparse, csv, json
from datetime import datetime
from pathlib import Path

def ts(value):
    return datetime.strptime(value, '%Y_%m_%d_%H_%M_%S_%f').timestamp()

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--log', required=True)
    p.add_argument('--run-dir', required=True)
    p.add_argument('--start', required=True, help='first logger timestamp, YYYY_MM_DD_HH_MM_SS_fff')
    p.add_argument('--lidar-start', type=float, default=None,
                   help='timestamp of first LiDAR frame; remap encoder time to that clock')
    args = p.parse_args()
    run = Path(args.run_dir); start = ts(args.start)
    rows = []
    with Path(args.log).open(newline='') as f:
        for row in csv.DictReader(f):
            raw = row.get('enc_t', '')
            if not raw or raw.startswith('1970_'):
                continue
            try: t = ts(raw)
            except ValueError: continue
            if t < start: continue
            rows.append({'enc_t': raw, 'enc_unix': t,
                         'enc_v': float(row['enc_v']),
                         'enc_dgr': float(row['enc_dgr']),
                         'enc_available': True})
    if not rows: raise SystemExit('no logger samples selected')
    if args.lidar_start is not None:
        origin = ts(args.start)
        for row in rows:
            row['enc_unix'] = float(args.lidar_start) + (row['enc_unix'] - origin)
    out = run / 'encoder.jsonl'; backup = run / 'encoder.placeholder.jsonl'
    if out.exists() and not backup.exists(): out.replace(backup)
    with out.open('w', encoding='utf-8') as f:
        for row in rows: f.write(json.dumps(row, separators=(',', ':')) + '\n')
    print(f'wrote {len(rows)} samples to {out}; backup={backup if backup.exists() else "none"}')

if __name__ == '__main__': main()
