"""Export hand-measured real-room GT as JSON / PNG / YAML."""

from __future__ import annotations

import argparse
import json

from flifltbot.mapping.real_room_gt import DEFAULT_GT_DIR, save_gt_bundle
from flifltbot.common.output.layout import PROJECT_ROOT


def build_parser():
    parser = argparse.ArgumentParser(
        description="Export real-room ground-truth floorplan (JSON + occupancy PNG/YAML)."
    )
    parser.add_argument(
        "--out-dir",
        default=str(DEFAULT_GT_DIR),
        help="Output directory (default: ros-data/gt/real_room/real_room)",
    )
    parser.add_argument("--cell-m", type=float, default=0.05)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    out = Path_resolve(args.out_dir)
    result = save_gt_bundle(out, cell_m=args.cell_m)
    print(json.dumps(result, indent=2))


def Path_resolve(path: str):
    from pathlib import Path

    p = Path(path).expanduser()
    if not p.is_absolute():
        p = (PROJECT_ROOT / p).resolve()
    return p


if __name__ == "__main__":
    main()
