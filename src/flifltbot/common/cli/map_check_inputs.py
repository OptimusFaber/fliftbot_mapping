from __future__ import annotations

import argparse
import json

from flifltbot.common.runtime.health import check_inputs, summary_lines


def build_parser():
    parser = argparse.ArgumentParser(description='Check whether mapping inputs are live.')
    parser.add_argument('--source', choices=['auto', 'sim', 'real'], default='auto')
    parser.add_argument('--json', action='store_true')
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    report = check_inputs(args.source)
    if args.json:
        print(json.dumps(report, indent=2))
        return
    for line in summary_lines(report):
        print(line)


if __name__ == '__main__':
    main()
