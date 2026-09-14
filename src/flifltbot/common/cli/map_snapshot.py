from __future__ import annotations

import argparse
import json

from flifltbot.mapping.pipeline import build_snapshot_map
from flifltbot.common.runtime.live_runtime import latest_frame
from ._common import parse_cameras, resolve_path, resolve_run_dir


def build_parser():
    parser = argparse.ArgumentParser(description='Build a local snapshot map from one frame.')
    parser.add_argument('--frame', default=None)
    parser.add_argument('--run-dir', default=None)
    parser.add_argument('--source', choices=['auto', 'sim', 'real'], default='auto')
    parser.add_argument('--cameras', default='up')
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.frame:
        frame = resolve_path(args.frame, must_exist=True)
    elif args.run_dir:
        frame = latest_frame(resolve_run_dir(args.run_dir))
    else:
        frame = latest_frame(None)
    result = build_snapshot_map(frame, cameras=parse_cameras(args.cameras))
    print(json.dumps({
        'frame': str(frame),
        'snapshot_json': str(result['out_json']),
        'snapshot_png': str(result['out_png']),
        'used_cameras': result['used_cameras'],
    }, indent=2))


if __name__ == '__main__':
    main()
