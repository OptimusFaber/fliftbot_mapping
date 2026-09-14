from __future__ import annotations

import argparse
import json

from flifltbot.mapping.pipeline import build_global_map_from_run
from ._common import parse_cameras, resolve_run_dir


def build_parser():
    parser = argparse.ArgumentParser(description='Build a global map from a recorded run.')
    parser.add_argument('--run-dir', required=True)
    parser.add_argument(
        '--method',
        choices=[
            'clouds', 'odom', 'improved', 'factor', 'factor_no_loops',
            'odom_fixed_v2', 'submap',
        ],
        default='improved',
    )
    parser.add_argument('--source', choices=['auto', 'sim', 'real'], default='auto')
    parser.add_argument('--cameras', default=None)
    parser.add_argument('--tag', default=None)
    parser.add_argument(
        '--no-loop-closures', action='store_true',
        help='Disable place-recognition loop edges (useful in repetitive aisles).',
    )
    parser.add_argument(
        '--fork-state', choices=['auto', 'down', 'up'], default='auto',
        help='Fork position for the up-LiDAR lever arm (default: run metadata).',
    )
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    run_dir = resolve_run_dir(args.run_dir)
    # Do not overwrite a method preset with argparse's default False.  An
    # explicit --no-loop-closures still takes priority for every method.
    overrides = {}
    if args.no_loop_closures:
        overrides['skip_loop_closures'] = True
    result = build_global_map_from_run(
        run_dir,
        method=args.method,
        cameras=parse_cameras(args.cameras),
        tag=args.tag,
        source=args.source,
        fork_state=args.fork_state,
        **overrides,
    )
    print(json.dumps({
        'run_dir': str(result['run_dir']),
        'method': result['method'],
        'map_json': str(result['out_json']),
        'map_png': str(result['out_png']),
        'stats': result['stats'],
    }, indent=2))


if __name__ == '__main__':
    main()
