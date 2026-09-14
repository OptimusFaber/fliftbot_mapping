#!/usr/bin/env python3
"""Factor-graph diagnostics vs warehouse GT (/odom = GT only, never in SLAM)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

from flifltbot.mapping.config import USE_ABSOLUTE_ODOM  # noqa: E402
from flifltbot.mapping.factor_diag import (  # noqa: E402
    compare_scan_vs_submap,
    probe_icp_anisotropy,
    run_ablation,
    run_full_diagnostics,
    sweep_scale,
    sweep_temporal_offset,
    warehouse_gt_meta,
)
from flifltbot.common.cli._common import parse_cameras, resolve_path  # noqa: E402


def build_parser():
    p = argparse.ArgumentParser(
        description=(
            'Warehouse GT diagnostics for factor mapper. '
            '/odom is evaluation GT only (USE_ABSOLUTE_ODOM must stay False). '
            'Does not change the default factor preset.'
        )
    )
    p.add_argument('--run-dir', required=True)
    p.add_argument('--source', choices=['sim', 'real', 'auto'], default='sim')
    p.add_argument('--cameras', default='horizon')
    p.add_argument(
        '--mode',
        choices=[
            'ablation', 'full', 'dt', 'scale', 'submap', 'anisotropy',
        ],
        default='ablation',
    )
    p.add_argument('--out-dir', default=None)
    p.add_argument('--max-frames', type=int, default=None)
    p.add_argument('--no-loops', action='store_true',
                   help='submap mode: disable loop closures (seq-only)')
    p.add_argument('--no-dt-sweep', action='store_true')
    p.add_argument('--no-submap', action='store_true')
    p.add_argument('--no-scale', action='store_true')
    p.add_argument('--no-anisotropy', action='store_true')
    return p


def _write(out_dir, name, report):
    if not out_dir:
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / name
    path.write_text(json.dumps(report, indent=2, default=str) + '\n', encoding='utf-8')
    print(f'wrote {path}')


def main(argv=None):
    args = build_parser().parse_args(argv)
    if USE_ABSOLUTE_ODOM:
        raise SystemExit(
            'Refuse: USE_ABSOLUTE_ODOM=True would put /odom into SLAM. '
            'Keep it False; /odom is GT-only for these experiments.'
        )
    run_dir = resolve_path(args.run_dir, must_exist=True)
    cameras = parse_cameras(args.cameras)
    out_dir = resolve_path(args.out_dir) if args.out_dir else None
    print(json.dumps(warehouse_gt_meta(), indent=2))

    if args.mode == 'ablation':
        report = run_ablation(
            run_dir,
            cameras=cameras,
            source=args.source,
            out_dir=out_dir,
            max_frames=args.max_frames,
        )
    elif args.mode == 'full':
        report = run_full_diagnostics(
            run_dir,
            cameras=cameras,
            source=args.source,
            out_dir=out_dir,
            do_dt_sweep=not args.no_dt_sweep,
            do_submap=not args.no_submap,
            do_scale=not args.no_scale,
            do_anisotropy=not args.no_anisotropy,
            max_frames=args.max_frames,
        )
    elif args.mode == 'dt':
        report = sweep_temporal_offset(
            run_dir,
            cameras=cameras,
            source=args.source,
            max_frames=args.max_frames,
        )
        _write(out_dir, 'dt_sweep.json', report)
    elif args.mode == 'scale':
        report = sweep_scale(
            run_dir,
            cameras=cameras,
            source=args.source,
            max_frames=args.max_frames,
        )
        _write(out_dir, 'factor_scale.json', report)
    elif args.mode == 'submap':
        report = compare_scan_vs_submap(
            run_dir,
            cameras=cameras,
            source=args.source,
            max_frames=args.max_frames,
            with_loops=not args.no_loops,
        )
        _write(out_dir, 'factor_submap.json', report)
    else:
        report = probe_icp_anisotropy(
            run_dir, cameras=cameras, source=args.source,
        )
        _write(out_dir, 'factor_anisotropy.json', report)

    print(json.dumps(report, indent=2, default=str)[:4000])
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
