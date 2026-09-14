#!/usr/bin/env python3
"""Measure straight occupied-map segments and align a Cartographer map to GT."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_MAP = ROOT / "ros-data/results/real/20260914/154006_pandar_room/cartographer_best"
DEFAULT_GT = ROOT / "ros-data/gt/real_room/real_room"
DEFAULT_OUT = ROOT / "ros-data/results/real/20260914/154006_pandar_room/cartographer_best/geometry_overlay"


def rot(points: np.ndarray, angle: float) -> np.ndarray:
    c, s = math.cos(angle), math.sin(angle)
    return np.asarray(points) @ np.array([[c, s], [-s, c]], dtype=float)


def load_map(map_dir: Path) -> tuple[np.ndarray, np.ndarray, float, np.ndarray]:
    image = np.asarray(Image.open(map_dir / "map.pgm").convert("L"))
    yaml = (map_dir / "map.yaml").read_text(encoding="utf-8").splitlines()
    resolution = next(float(line.split(":", 1)[1]) for line in yaml if line.startswith("resolution:"))
    origin = np.asarray(
        json.loads(next(line.split(":", 1)[1].strip().replace("'", '"') for line in yaml if line.startswith("origin:"))),
        dtype=float,
    )
    occupied = image < 100
    rows, cols = np.nonzero(occupied)
    xy = np.column_stack([origin[0] + (cols + 0.5) * resolution, origin[1] + (image.shape[0] - rows - 0.5) * resolution])
    return image, xy, resolution, origin


def load_gt(gt_dir: Path) -> tuple[np.ndarray, list[float]]:
    image = np.asarray(Image.open(gt_dir / "floorplan.png").convert("L"))
    meta = json.loads((gt_dir / "floorplan_meta.json").read_text(encoding="utf-8"))
    cell = float(meta["cell_m"])
    origin = np.asarray(meta["origin"][:2], dtype=float)
    return image, [origin[0], origin[0] + image.shape[1] * cell, origin[1], origin[1] + image.shape[0] * cell]


def hough_segments(image: np.ndarray, resolution: float, origin: np.ndarray) -> list[dict[str, Any]]:
    mask = (image < 100).astype(np.uint8) * 255
    lines = cv2.HoughLinesP(mask, 1, np.pi / 720.0, threshold=20, minLineLength=35, maxLineGap=8)
    if lines is None:
        return []
    result = []
    for x1, y1, x2, y2 in lines.reshape(-1, 4).astype(float):
        dx, dy = (x2 - x1) * resolution, -(y2 - y1) * resolution
        length = math.hypot(dx, dy)
        if length < 1.0:
            continue
        p1 = np.asarray([origin[0] + (x1 + 0.5) * resolution, origin[1] + (image.shape[0] - y1 - 0.5) * resolution])
        p2 = np.asarray([origin[0] + (x2 + 0.5) * resolution, origin[1] + (image.shape[0] - y2 - 0.5) * resolution])
        angle = math.atan2(p2[1] - p1[1], p2[0] - p1[0])
        result.append({"p1_m": p1.tolist(), "p2_m": p2.tolist(), "length_m": length, "angle_deg": math.degrees(angle)})
    result.sort(key=lambda row: row["length_m"], reverse=True)
    # Keep distinct segments; Hough frequently returns the same wall several times.
    selected = []
    for row in result:
        p = np.asarray(row["p1_m"])
        q = np.asarray(row["p2_m"])
        duplicate = False
        for old in selected:
            a = np.asarray(old["p1_m"])
            b = np.asarray(old["p2_m"])
            parallel = abs(math.sin(math.radians(row["angle_deg"] - old["angle_deg"]))) < 0.035
            close = min(np.linalg.norm(p - a), np.linalg.norm(p - b), np.linalg.norm(p - b), np.linalg.norm(q - a)) < 0.35
            overlap = max(np.linalg.norm(p - a), np.linalg.norm(q - b)) < max(row["length_m"], old["length_m"]) + 0.4
            if parallel and close and overlap:
                duplicate = True
                break
        if not duplicate:
            selected.append(row)
        if len(selected) >= 24:
            break
    return selected


def align_points(points: np.ndarray, segments: list[dict[str, Any]], gt_extent: list[float]) -> tuple[np.ndarray, list[dict[str, Any]], dict[str, Any]]:
    if not segments:
        raise ValueError("no straight wall candidates found")
    dominant = float(segments[0]["angle_deg"])
    # Map coordinates have already been converted to world coordinates, so rotate
    # the dominant wall to +X.
    theta = -math.radians(dominant)
    rotated = rot(points, theta)
    rotated_segments = []
    for row in segments:
        a = rot(np.asarray(row["p1_m"]), theta)
        b = rot(np.asarray(row["p2_m"]), theta)
        rotated_segments.append({**row, "aligned_p1_m": a.tolist(), "aligned_p2_m": b.tolist()})

    # Similarity alignment: preserve one uniform scale, using the longest observed
    # wall as the room's 16.465 m east-west reference. The report exposes this
    # scale because the map is not metrically compatible with GT at scale=1.
    observed_long = float(segments[0]["length_m"])
    gt_long = 16.465
    scale = gt_long / observed_long
    scaled = rotated * scale
    # Anchor the longest detected wall to the GT south wall. This is more
    # meaningful than centering the entire occupied cloud, because the cloud
    # contains scan rays and interior clutter outside the room footprint.
    longest_start = np.asarray(rotated_segments[0]["aligned_p1_m"]) * scale
    translation = np.asarray([gt_extent[0], gt_extent[2]]) - longest_start
    aligned = scaled + translation
    for row in rotated_segments:
        row["aligned_p1_m"] = (np.asarray(row["aligned_p1_m"]) * scale + translation).tolist()
        row["aligned_p2_m"] = (np.asarray(row["aligned_p2_m"]) * scale + translation).tolist()
        row["aligned_length_m"] = row["length_m"] * scale
    transform = {
        "dominant_wall_angle_deg_in_map": dominant,
        "rotation_deg_applied": math.degrees(theta),
        "uniform_scale": scale,
        "translation_xy_m": translation.tolist(),
        "scale_reference": "longest detected straight segment -> GT south wall width 16.465 m",
        "translation_reference": "start of longest wall -> GT south-west corner (0, 0)",
        "scale_one_overlay_warning": "scale=1 leaves the Pandar map metrically smaller than the GT room",
    }
    return aligned, rotated_segments, transform


def render(output: Path, map_image: np.ndarray, gt_image: np.ndarray, gt_extent: list[float], aligned: np.ndarray, segments: list[dict[str, Any]], transform: dict[str, Any]) -> None:
    fig, ax = plt.subplots(figsize=(20, 9), dpi=180, constrained_layout=True)
    ax.imshow(np.flipud(gt_image), origin="lower", extent=gt_extent, cmap="Blues", vmin=0, vmax=255, alpha=0.30, interpolation="nearest", zorder=0)
    ax.scatter(aligned[:, 0], aligned[:, 1], s=2, c="#111827", alpha=0.25, label="Pandar occupied cells", zorder=2)
    in_room = []
    for index, row in enumerate(segments):
        a, b = np.asarray(row["aligned_p1_m"]), np.asarray(row["aligned_p2_m"])
        midpoint = (a + b) / 2.0
        if not (
            gt_extent[0] - 0.25 <= midpoint[0] <= gt_extent[1] + 0.25
            and gt_extent[2] - 0.25 <= midpoint[1] <= gt_extent[3] + 0.25
        ):
            continue
        in_room.append(row)
        ax.plot([a[0], b[0]], [a[1], b[1]], color="#dc2626", linewidth=1.3, zorder=4)
        mid = (a + b) / 2
        ax.text(mid[0], mid[1], f"{len(in_room)}: {row['aligned_length_m']:.2f} m", fontsize=6, color="#991b1b", zorder=5)
    ax.set_xlim(gt_extent[0] - 0.4, gt_extent[1] + 0.4)
    ax.set_ylim(gt_extent[2] - 0.4, gt_extent[3] + 0.4)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("GT room X — east (m)")
    ax.set_ylabel("GT room Y — north (m)")
    ax.grid(alpha=0.2)
    ax.set_title(
        "Pandar Cartographer map aligned to real-room GT\n"
        f"rotation {transform['rotation_deg_applied']:.2f}°, uniform scale {transform['uniform_scale']:.3f}; "
        f"red lines = {len(in_room)} detected in-room walls/protrusions with lengths"
    )
    ax.legend(loc="upper left", fontsize=8)
    fig.savefig(output)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--map-dir", type=Path, default=DEFAULT_MAP)
    parser.add_argument("--gt-dir", type=Path, default=DEFAULT_GT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    map_image, points, resolution, origin = load_map(args.map_dir)
    gt_image, gt_extent = load_gt(args.gt_dir)
    segments = hough_segments(map_image, resolution, origin)
    aligned, aligned_segments, transform = align_points(points, segments, gt_extent)
    output = args.out_dir / "pandar_map_aligned_to_gt.png"
    in_room_segments = []
    for row in aligned_segments:
        midpoint = (np.asarray(row["aligned_p1_m"]) + np.asarray(row["aligned_p2_m"])) / 2.0
        if (
            gt_extent[0] - 0.25 <= midpoint[0] <= gt_extent[1] + 0.25
            and gt_extent[2] - 0.25 <= midpoint[1] <= gt_extent[3] + 0.25
        ):
            in_room_segments.append(row)
    render(output, map_image, gt_image, gt_extent, aligned, aligned_segments, transform)
    report = {
        "schema": "fliftbot.pandar_map_wall_geometry_alignment.v1",
        "inputs": {"map_dir": str(args.map_dir.resolve()), "map_pgm": str((args.map_dir / "map.pgm").resolve()), "gt_floorplan": str((args.gt_dir / "floorplan.png").resolve())},
        "map": {"resolution_m": resolution, "origin_xy_m": origin.tolist(), "occupied_cells": int(len(points))},
        "transform": transform,
        "in_room_segment_count": len(in_room_segments),
        "straight_segments": aligned_segments,
        "output": str(output.resolve()),
    }
    (args.out_dir / "wall_geometry_alignment.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output.resolve()), "segments": len(segments), **transform}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
