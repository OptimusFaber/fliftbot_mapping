#!/usr/bin/env python3
"""Audit the on-disk horizon NPZ contract used by standalone Cartographer.

This is intentionally independent of the C++ runner.  It reports facts about
the bytes in a recording (shape, order, timestamps and coordinate frame) and
also evaluates the exact Python wheel model used by the regular mapper.  It is
useful for distinguishing a bad recording from a loader/replay bug.

Example::

  python3 scripts/evaluation/audit_npz_contract.py \
      /path/to/recording --sample-count 7

The report is JSON on stdout, so it can be checked into an experiment result
or piped to ``jq``.  No input files are modified.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def _finite_stats(values: np.ndarray) -> dict:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if not len(values):
        return {"count": 0}
    return {
        "count": int(len(values)),
        "min": float(np.min(values)),
        "p50": float(np.percentile(values, 50)),
        "p99": float(np.percentile(values, 99)),
        "max": float(np.max(values)),
    }


def _angle_order(xy: np.ndarray) -> dict:
    """Describe the bearing order without sorting/reconstructing the cloud."""
    angle = np.arctan2(xy[:, 1], xy[:, 0]).astype(float)
    if len(angle) < 2:
        return {"count": int(len(angle))}
    # np.unwrap is deliberately applied to acquisition order: this answers
    # whether the recorder preserved the native sweep order.
    unwrapped = np.unwrap(angle)
    delta = np.diff(unwrapped)
    abs_sweep = float(np.sum(np.abs(delta)))
    median_delta = float(np.median(delta))
    sign = "increasing" if median_delta > 0 else "decreasing" if median_delta < 0 else "flat"
    expected = int(max(1, round(abs_sweep / (2.0 * math.pi))))
    # A large jump after unwrapping is a likely dropped/out-of-order sample,
    # not a real angular increment.
    return {
        "count": int(len(angle)),
        "first_rad": float(angle[0]),
        "last_rad": float(angle[-1]),
        "unwrapped_sweep_rad": abs_sweep,
        "unwrapped_signed_sweep_rad": float(unwrapped[-1] - unwrapped[0]),
        "median_delta_rad": median_delta,
        "p01_abs_delta_rad": float(np.percentile(np.abs(delta), 1)),
        "p99_abs_delta_rad": float(np.percentile(np.abs(delta), 99)),
        "native_order": sign,
        "increasing_fraction": float(np.mean(delta > 0.0)),
        "decreasing_fraction": float(np.mean(delta < 0.0)),
        "estimated_revolutions": expected,
    }


def _parse_json_number(text: str, key: str):
    """Replicate runner.cc's first-occurrence lookup for diagnostics.

    This intentionally does *not* search for a later occurrence.  The C++
    helper finds the first ``"key"`` and calls ``strtod`` immediately after
    its colon; a string value therefore returns its fallback (zero).
    """
    needle = '"' + key + '"'
    key_pos = text.find(needle)
    if key_pos < 0:
        return None
    colon = text.find(':', key_pos + len(needle))
    if colon < 0:
        return None
    match = re.match(r'\s*([-+0-9.eE]+)', text[colon + 1:])
    return float(match.group(1)) if match else 0.0


def _extrinsics(path: Path) -> dict:
    if not path.exists():
        return {"file": str(path), "exists": False}
    text = path.read_text(encoding="utf-8")
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        return {"file": str(path), "exists": True, "json_error": str(exc)}
    position = data.get("position_m", {}) if isinstance(data, dict) else {}
    orientation = data.get("orientation_deg", {}) if isinstance(data, dict) else {}
    declared = {
        "x": float(position.get("x", 0.0)),
        "y": float(position.get("y", 0.0)),
        "z": float(position.get("z", 0.0)),
        "yaw_deg": float(orientation.get("yaw", 0.0)),
    }
    # runner.cc's JsonNumber() only sees a numeric value after the *first*
    # occurrence of each key.  In lidar.config, coordinate_system.x/y/z are
    # strings, so this explicitly exposes the resulting zero fallback.
    first_numeric = {key: _parse_json_number(text, key) for key in ("x", "y", "z", "yaw")}
    return {
        "file": str(path),
        "exists": True,
        "lidar_type": data.get("lidar_type"),
        "source": data.get("source"),
        "declared_position_m": declared,
        "runner_first_numeric_lookup": first_numeric,
        "runner_parser_matches_declared": (
            first_numeric.get("x") == declared["x"]
            and first_numeric.get("y") == declared["y"]
            and first_numeric.get("z") == declared["z"]
            and first_numeric.get("yaw") == declared["yaw_deg"]
        ),
    }


def _load_encoders(path: Path) -> list[dict]:
    values = []
    if not path.exists():
        return values
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            item = json.loads(line)
            t = float(item["enc_unix"])
            values.append({
                "t": t,
                "v": float(item.get("enc_v", 0.0)),
                "dgr": float(item.get("enc_dgr", 0.0)),
            })
        except (ValueError, TypeError, KeyError, json.JSONDecodeError):
            continue
    values.sort(key=lambda x: x["t"])
    return values


def _integrate_python(encoders: list[dict], wheelbase: float) -> dict:
    """Integrate the project's encoder convention at its native timestamps."""
    # Keep this formula local and transparent, rather than importing the
    # mapper (which has process-global configuration overrides).
    x = y = yaw = 0.0
    valid_steps = 0
    distances = []
    for a, b in zip(encoders, encoders[1:]):
        dt = b["t"] - a["t"]
        if not (0.0 < dt <= 0.5):
            continue
        v = -0.5 * (a["v"] + b["v"])
        steer = math.radians(0.5 * (a["dgr"] + b["dgr"]))
        wz = v * math.sin(steer) / wheelbase
        theta = wz * dt
        vx = v * math.cos(steer)
        if abs(wz) < 1e-9:
            dx, dy = vx * dt, 0.0
        else:
            dx, dy = vx * math.sin(theta) / wz, vx * (1.0 - math.cos(theta)) / wz
        x, y = x + math.cos(yaw) * dx - math.sin(yaw) * dy, y + math.sin(yaw) * dx + math.cos(yaw) * dy
        yaw += theta
        valid_steps += 1
        distances.append(math.hypot(dx, dy))
    return {
        "valid_steps": valid_steps,
        "pose_x_m": x,
        "pose_y_m": y,
        "pose_yaw_rad": yaw,
        "distance_m": float(np.sum(distances)) if distances else 0.0,
    }


def audit(run: Path, sample_count: int) -> dict:
    horizon = run / "horizon"
    paths = sorted(horizon.glob("*.npz"))
    if not paths:
        raise SystemExit(f"no horizon/*.npz in {run}")
    picks = sorted(set([0, len(paths) // 2, len(paths) - 1] + list(range(min(sample_count, len(paths))))) )
    records = []
    all_stamps = []
    counts = []
    for index, path in enumerate(paths):
        with np.load(path, allow_pickle=False) as data:
            xy = np.asarray(data["xy"])
            stamp = float(np.asarray(data["stamp"]).reshape(()))
            times = np.asarray(data["t"], dtype=float) if "t" in data.files else np.empty(0)
            frame = int(np.asarray(data["frame"]).reshape(())) if "frame" in data.files else None
            channel = int(np.asarray(data["channel"]).reshape(())) if "channel" in data.files else None
            if xy.ndim != 2 or xy.shape[1] != 2:
                raise ValueError(f"{path}: xy is {xy.shape}, expected Nx2")
            counts.append(len(xy)); all_stamps.append(stamp)
            if index in picks:
                angle = _angle_order(xy.astype(float))
                point_monotonic = bool(len(times) < 2 or np.all(np.diff(times) >= -1e-9))
                finite = times[np.isfinite(times)]
                records.append({
                    "file": path.name,
                    "frame": frame,
                    "channel": channel,
                    "arrays": {name: {"shape": list(np.asarray(data[name]).shape), "dtype": str(np.asarray(data[name]).dtype)} for name in data.files},
                    "xy_equals_xyz_prefix": bool("xyz" not in data.files or np.allclose(np.asarray(data["xyz"])[:, :2], xy, equal_nan=True)),
                    "bearing": angle,
                    "point_time": {
                        "count": int(len(times)),
                        "finite_count": int(len(finite)),
                        "span_s": float(np.ptp(finite)) if len(finite) else None,
                        "min_minus_stamp_s": float(np.min(finite) - stamp) if len(finite) else None,
                        "max_minus_stamp_s": float(np.max(finite) - stamp) if len(finite) else None,
                        "median_minus_stamp_s": float(np.median(finite) - stamp) if len(finite) else None,
                        "monotonic_in_file_order": point_monotonic,
                    },
                })
    stamps = np.asarray(all_stamps)
    enc = _load_encoders(run / "encoder.jsonl")
    enc_t = np.asarray([e["t"] for e in enc], dtype=float)
    return {
        "run": str(run),
        "npz_count": len(paths),
        "npz_frame_id_first_last": [records[0].get("frame"), records[-1].get("frame")],
        "stamp": {
            "first": float(stamps[0]), "last": float(stamps[-1]), "duration_s": float(stamps[-1] - stamps[0]),
            "dt": _finite_stats(np.diff(stamps)), "strictly_increasing": bool(np.all(np.diff(stamps) > 0.0)),
            "point_count": _finite_stats(np.asarray(counts)),
        },
        "samples": records,
        "encoder": {
            "count": len(enc),
            "first": float(enc_t[0]) if len(enc) else None,
            "last": float(enc_t[-1]) if len(enc) else None,
            "duration_s": float(enc_t[-1] - enc_t[0]) if len(enc) > 1 else None,
            "dt": _finite_stats(np.diff(enc_t)) if len(enc) > 1 else {"count": 0},
            "strictly_increasing": bool(len(enc) < 2 or np.all(np.diff(enc_t) > 0.0)),
            "integration_wheelbase_1.410": _integrate_python(enc, 1.410),
        },
        "overlap": {
            "scan_inside_encoder_interval": int(np.sum((stamps >= enc_t[0]) & (stamps <= enc_t[-1]))) if len(enc) else 0,
            "scan_before_encoder_s": float(max(0.0, enc_t[0] - stamps[0])) if len(enc) else None,
            "scan_after_encoder_s": float(max(0.0, stamps[-1] - enc_t[-1])) if len(enc) else None,
        },
        "extrinsics": _extrinsics(run / "lidar.config"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--sample-count", type=int, default=5)
    args = parser.parse_args()
    print(json.dumps(audit(args.run_dir.resolve(), max(1, args.sample_count)), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
