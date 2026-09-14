#!/usr/bin/env python3
"""Reject ground-truth inputs before an offline Cartographer run.

The standalone runner has a deliberately small input contract: ``horizon/*.npz``
for recorded lidar packets, ``encoder.jsonl`` for ``enc_*`` samples, and the
optional ``lidar.config`` mount description.  This guard is run by the shell
launcher and can also be used in CI.  Evaluation files (``nav_*``, floorplans,
alignment files, and GT poses) may live beside a recording, but they must not
be selected as SLAM inputs.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np


# Deliberately match field/file names, not the parent directory.  Some valid
# recordings are named nav_video_* even though their horizon contains only
# lidar packets and encoder samples.
FORBIDDEN = re.compile(
    r"(?:^|[_./-])(?:nav|gt|groundtruth|ground_truth|floorplan|alignment)"
    r"(?:$|[_./-])",
    re.IGNORECASE,
)
FORBIDDEN_KEY = re.compile(
    r"(?:^|_)(?:nav|gt|groundtruth|ground_truth|floorplan|alignment)"
    r"(?:$|_)",
    re.IGNORECASE,
)
def _fail(errors: list[str], message: str) -> None:
    errors.append(message)


def audit(run_dir: Path, *, require_encoder: bool = True) -> list[str]:
    errors: list[str] = []
    run_dir = run_dir.expanduser().resolve()
    horizon = run_dir / "horizon"
    encoder = run_dir / "encoder.jsonl"
    lidar_config = run_dir / "lidar.config"

    if not horizon.is_dir() or horizon.is_symlink():
        _fail(errors, f"horizon must be a real directory: {horizon}")
        return errors
    if require_encoder and (not encoder.is_file() or encoder.is_symlink()):
        _fail(errors, f"encoder.jsonl must be a real file: {encoder}")
    if lidar_config.exists() and (not lidar_config.is_file() or lidar_config.is_symlink()):
        _fail(errors, f"lidar.config must be a regular file when present: {lidar_config}")

    npz_files = sorted(horizon.glob("*.npz"))
    if not npz_files:
        _fail(errors, f"no lidar packets in {horizon}/*.npz")
    for path in npz_files:
        if FORBIDDEN.search(path.name):
            _fail(errors, f"forbidden GT-like lidar filename: {path.name}")
        if path.is_symlink() or not path.is_file():
            _fail(errors, f"lidar packet must be a regular file: {path}")
            continue
        try:
            with np.load(path, allow_pickle=False) as arrays:
                keys = set(arrays.files)
                forbidden = sorted(k for k in keys if FORBIDDEN_KEY.search(k))
                if forbidden:
                    _fail(errors, f"{path.name}: GT/navigation arrays are not SLAM inputs: {forbidden}")
                if "xy" not in keys or "stamp" not in keys:
                    _fail(errors, f"{path.name}: required lidar arrays xy and stamp are missing")
        except Exception as exc:  # noqa: BLE001 - report the offending packet
            _fail(errors, f"{path.name}: cannot inspect lidar NPZ: {exc}")

    if encoder.is_file() and not encoder.is_symlink():
        for lineno, line in enumerate(encoder.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                _fail(errors, f"encoder.jsonl:{lineno}: invalid JSON: {exc.msg}")
                continue
            if not isinstance(item, dict):
                _fail(errors, f"encoder.jsonl:{lineno}: record is not an object")
                continue
            forbidden = sorted(k for k in item if FORBIDDEN_KEY.search(k))
            if forbidden:
                _fail(errors, f"encoder.jsonl:{lineno}: GT/navigation fields are forbidden: {forbidden}")
            required = {"enc_unix", "enc_v", "enc_dgr"}
            missing = sorted(required - item.keys())
            if missing:
                _fail(errors, f"encoder.jsonl:{lineno}: missing encoder fields: {missing}")

    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", "--run_dir", dest="run_dir", type=Path, required=True)
    parser.add_argument(
        "--use_odometry", "--use-odometry", dest="use_odometry",
        type=lambda value: str(value).lower() not in {"0", "false", "no"},
        default=True,
    )
    args, unknown = parser.parse_known_args()
    # The launcher forwards Cartographer flags as well; only this guard's own
    # run directory is relevant.  Keep unknown flags visible for diagnostics,
    # but do not treat them as data sources.
    errors = audit(args.run_dir, require_encoder=args.use_odometry)
    if errors:
        print("standalone input contract: REJECTED", file=sys.stderr)
        for error in errors:
            print(f"  - {error}", file=sys.stderr)
        return 2
    print(f"standalone input contract: OK ({args.run_dir})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
