#!/usr/bin/env python3
"""Sanity-check a standalone Cartographer occupancy map.

The map frame is arbitrary, so this checker deliberately does not compare raw
pixel coordinates.  It reports geometry in metres (occupied bounding box,
dominant wall extents and trajectory extent) and, when a real-room GT image is
available, creates a bbox-aligned overlay.  This makes it useful for rejecting
maps that technically exported but are clearly blown up, collapsed, or mostly
unstructured.

The checker only depends on Pillow, NumPy and SciPy (OpenCV is optional for
the wall-line diagnostic).  It accepts the trajectory JSON emitted by the
existing Cartographer trajectory exporter, as well as a simple list of
``{"x": ..., "y": ...}`` records.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw


def _yaml_scalar_map(path: Path) -> dict[str, Any]:
    """Read the tiny map YAML without requiring PyYAML."""
    out: dict[str, Any] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if ":" not in line or line.lstrip().startswith("#"):
            continue
        key, value = line.split(":", 1)
        value = value.strip()
        if value.startswith("["):
            try:
                out[key.strip()] = json.loads(value)
            except json.JSONDecodeError:
                pass
        else:
            try:
                out[key.strip()] = float(value)
            except ValueError:
                out[key.strip()] = value.strip('"\'')
    return out


def _read_map(map_path: Path, yaml_path: Path | None, occupied_threshold: int) -> tuple[np.ndarray, float, tuple[float, float]]:
    image = np.asarray(Image.open(map_path).convert("L"))
    metadata = _yaml_scalar_map(yaml_path) if yaml_path else {}
    resolution = float(metadata.get("resolution", 0.05))
    origin = metadata.get("origin", [0.0, 0.0, 0.0])
    if not isinstance(origin, list) or len(origin) < 2:
        origin = [0.0, 0.0, 0.0]
    # Cartographer exports unknown as grey (~128/205) and free as white.  A
    # strict threshold avoids treating all unknown cells as occupied.
    return image <= occupied_threshold, resolution, (float(origin[0]), float(origin[1]))


def _world_points(mask: np.ndarray, resolution: float, origin: tuple[float, float]) -> np.ndarray:
    rows, cols = np.nonzero(mask)
    if len(rows) == 0:
        return np.empty((0, 2), dtype=np.float64)
    h = mask.shape[0]
    return np.column_stack((
        origin[0] + (cols + 0.5) * resolution,
        origin[1] + (h - rows - 0.5) * resolution,
    ))


def _extent(points: np.ndarray) -> dict[str, float | int]:
    if len(points) == 0:
        return {"n": 0, "min_x_m": math.nan, "max_x_m": math.nan,
                "min_y_m": math.nan, "max_y_m": math.nan,
                "width_m": 0.0, "height_m": 0.0, "area_bbox_m2": 0.0}
    lo = points.min(axis=0)
    hi = points.max(axis=0)
    width, height = hi - lo
    return {"n": int(len(points)), "min_x_m": float(lo[0]), "max_x_m": float(hi[0]),
            "min_y_m": float(lo[1]), "max_y_m": float(hi[1]),
            "width_m": float(width), "height_m": float(height),
            "area_bbox_m2": float(width * height)}


def _trajectory_points(path: Path | None) -> np.ndarray:
    if not path or not path.exists():
        return np.empty((0, 2), dtype=np.float64)
    data = json.loads(path.read_text(encoding="utf-8"))
    records = data.get("poses", data) if isinstance(data, dict) else data
    result = []
    if not isinstance(records, list):
        return np.empty((0, 2), dtype=np.float64)
    for item in records:
        if not isinstance(item, dict):
            continue
        pose = item.get("global", item.get("pose", item))
        if isinstance(pose, dict) and "x" in pose and "y" in pose:
            try:
                result.append((float(pose["x"]), float(pose["y"])))
            except (TypeError, ValueError):
                pass
    return np.asarray(result, dtype=np.float64).reshape((-1, 2))


def _gt_points(path: Path, resolution: float) -> np.ndarray:
    image = np.asarray(Image.open(path).convert("L"))
    # GT is a binary floorplan (black walls/obstacles, white free space).
    return _world_points(image <= 100, resolution, (0.0, 0.0))


def _bbox_overlay(map_points: np.ndarray, gt_points: np.ndarray, out: Path, size: tuple[int, int] = (900, 520)) -> None:
    """Draw bbox-normalized map (red) and GT (blue), preserving aspect ratio."""
    if not len(map_points) or not len(gt_points):
        return
    all_points = np.vstack((map_points, gt_points))
    lo, hi = all_points.min(axis=0), all_points.max(axis=0)
    span = np.maximum(hi - lo, 1e-6)
    margin = 25
    scale = min((size[0] - 2 * margin) / span[0], (size[1] - 2 * margin) / span[1])
    def pix(points: np.ndarray) -> list[tuple[int, int]]:
        q = (points - lo) * scale
        # Cartesian y grows up, image y grows down.
        return [(int(margin + x), int(size[1] - margin - y)) for x, y in q]
    canvas = Image.new("RGB", size, "white")
    draw = ImageDraw.Draw(canvas)
    for x, y in pix(gt_points):
        draw.ellipse((x - 1, y - 1, x + 1, y + 1), fill=(40, 90, 220))
    for x, y in pix(map_points):
        draw.ellipse((x - 1, y - 1, x + 1, y + 1), fill=(220, 45, 45))
    draw.text((10, 8), "GT blue / standalone map red (bbox-aligned)", fill=(0, 0, 0))
    out.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out)


def _structure(mask: np.ndarray) -> dict[str, float | int]:
    # A map with only unknown cells or a giant filled blob is not useful even
    # if its bounding box happens to have the right dimensions.
    occupied = int(mask.sum())
    total = int(mask.size)
    components = 0
    try:
        import cv2  # type: ignore
        _, _, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
        components = max(0, int(len(stats) - 1))
    except ImportError:
        # Keep the script usable in minimal environments.
        components = -1
    return {"occupied_cells": occupied, "total_cells": total,
            "occupied_fraction": float(occupied / total) if total else 0.0,
            "connected_components": components}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("map", type=Path, help="map.pgm or map.png")
    parser.add_argument("--map-yaml", type=Path, help="map.yaml with resolution/origin")
    parser.add_argument("--trajectory", type=Path)
    parser.add_argument("--gt", type=Path, help="binary GT floorplan image")
    parser.add_argument("--gt-resolution", type=float, default=0.05)
    # Cartographer's probability_grid_filler exports unknown cells as exactly
    # 128; occupied cells can be any value below that depending on probability.
    parser.add_argument("--occupied-threshold", type=int, default=127)
    parser.add_argument("--json", type=Path, help="write metrics JSON")
    parser.add_argument("--overlay", type=Path, help="write bbox-aligned PNG")
    args = parser.parse_args()

    mask, resolution, origin = _read_map(args.map, args.map_yaml, args.occupied_threshold)
    map_points = _world_points(mask, resolution, origin)
    trajectory = _trajectory_points(args.trajectory)
    report: dict[str, Any] = {
        "map": str(args.map), "resolution_m": resolution,
        "occupied_threshold": args.occupied_threshold,
        "map_extent": _extent(map_points), "map_structure": _structure(mask),
        "trajectory_extent": _extent(trajectory),
    }
    if args.gt:
        gt = _gt_points(args.gt, args.gt_resolution)
        report["gt_extent"] = _extent(gt)
        mw, mh = report["map_extent"]["width_m"], report["map_extent"]["height_m"]
        gw, gh = report["gt_extent"]["width_m"], report["gt_extent"]["height_m"]
        report["bbox_error"] = {
            "width_cm": float((mw - gw) * 100), "height_cm": float((mh - gh) * 100),
            "width_ratio": float(mw / gw) if gw else math.nan,
            "height_ratio": float(mh / gh) if gh else math.nan,
        }
        if args.overlay:
            _bbox_overlay(map_points, gt, args.overlay)
            report["overlay"] = str(args.overlay)
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(report, indent=2, allow_nan=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, allow_nan=True))
    return 0 if map_points.size else 2


if __name__ == "__main__":
    raise SystemExit(main())
