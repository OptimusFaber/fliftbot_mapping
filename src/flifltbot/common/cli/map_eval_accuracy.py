from __future__ import annotations

import argparse
import json
from pathlib import Path

from flifltbot.common.runtime.accuracy_eval import evaluate_run
from ._common import parse_cameras, resolve_path, resolve_run_dir


def build_parser():
    parser = argparse.ArgumentParser(description='Evaluate map accuracy for a recorded run.')
    parser.add_argument('--run-dir', required=True)
    parser.add_argument(
        '--method',
        choices=[
            'clouds', 'odom', 'improved', 'factor', 'factor_no_loops',
            'odom_fixed_v2', 'submap',
        ],
        default='improved',
        help='mapping method to build before scoring',
    )
    parser.add_argument('--source', choices=['auto', 'sim', 'real'], default='sim')
    parser.add_argument('--cameras', default='up')
    parser.add_argument(
        '--gt-path',
        default=None,
        help='Real-room GT floorplan.json (default: ros-data/gt/real_room/real_room/floorplan.json)',
    )
    parser.add_argument(
        '--align',
        choices=['bbox', 'none'],
        default='bbox',
        help='How to align real map frame to room GT (default: bbox)',
    )
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    run_dir = resolve_run_dir(args.run_dir)
    source = args.source
    if source == 'auto':
        # Prefer path hint; fall back to env heuristics.
        parts = {p.lower() for p in Path(run_dir).parts}
        if 'real' in parts:
            source = 'real'
        elif 'sim' in parts:
            source = 'sim'
        else:
            from flifltbot.common.runtime.source_detect import detect_source

            detected = detect_source('auto', run_dir=run_dir)
            source = 'real' if detected == 'real' else 'sim'
    gt_path = resolve_path(args.gt_path) if args.gt_path else None
    result = evaluate_run(
        run_dir,
        method=args.method,
        source=source,
        cameras=parse_cameras(args.cameras),
        gt_path=gt_path,
        align=args.align,
    )
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
