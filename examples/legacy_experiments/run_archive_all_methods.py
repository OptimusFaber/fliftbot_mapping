#!/usr/bin/env python3
"""Run the archived mapping presets without modifying the archived project."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ARCHIVE = Path("/home/optimus/ros-projects/archieve")
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ARCHIVE / "src"))

from flifltbot.mapping.config import METHOD_PRESETS  # noqa: E402
from flifltbot.mapping.pipeline import sew_all_local_frames  # noqa: E402


SCENES = {
    "room": {
        "run": Path("/home/optimus/Downloads/run_20260814_190205"),
        "source": "real",
    },
    "warehouse": {
        "run": ROOT / "maps/20260815/sim/run_004137",
        "source": "sim",
    },
}
METHODS = ("clouds", "odom", "improved", "factor")


def json_default(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def plot_one(path: Path, scene: str, method: str, points, poses, stats):
    pts = np.asarray(points, float)
    route = np.asarray([p["pose"][:2] for p in poses], float)
    fig, ax = plt.subplots(figsize=(9, 7))
    if len(pts):
        ax.scatter(pts[:, 0], pts[:, 1], s=0.7, alpha=0.45, color="#1677ff")
    if len(route):
        ax.plot(route[:, 0], route[:, 1], lw=1.0, color="#e63946")
        ax.scatter(*route[0], s=35, color="limegreen", zorder=4)
        ax.scatter(*route[-1], s=35, color="darkred", zorder=4)
    ax.set_aspect("equal")
    ax.grid(alpha=0.2)
    ax.set_xlabel("SLAM X, m")
    ax.set_ylabel("SLAM Y, m")
    ax.set_title(
        f"{scene}: archived {method}\n"
        f"frames={stats.get('n_frames')}  keyframes={stats.get('keyframes')}  "
        f"loops={stats.get('loop_closures')}  cells={len(pts)}"
    )
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def plot_compare(path: Path, scene: str, rows):
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    for ax, row in zip(axes.flat, rows):
        data = json.loads(Path(row["result_json"]).read_text(encoding="utf-8"))
        pts = np.asarray(data["map"], float)
        route = np.asarray([p["pose"][:2] for p in data["poses"]], float)
        if len(pts):
            ax.scatter(pts[:, 0], pts[:, 1], s=0.35, alpha=0.4, color="#1677ff")
        if len(route):
            ax.plot(route[:, 0], route[:, 1], lw=0.8, color="#e63946")
        ax.set_aspect("equal")
        ax.grid(alpha=0.15)
        ax.set_title(
            f"{row['method']} | cells={row['stable_cells']} | "
            f"loops={row['loop_closures']} | {row['wall_s']:.1f}s"
        )
    fig.suptitle(f"{scene}: all archived mapping methods", fontsize=15)
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", required=True, choices=sorted(SCENES))
    parser.add_argument(
        "--out-root",
        type=Path,
        default=ROOT / "results/20260818/archive_all_methods",
        help="root directory for this reproduction (default: original archive output)",
    )
    args = parser.parse_args()
    cfg = SCENES[args.scene]
    out = args.out_root / args.scene
    out.mkdir(parents=True, exist_ok=True)
    rows = []

    for method in METHODS:
        print(f"START {args.scene} {method}", flush=True)
        kwargs = dict(METHOD_PRESETS[method])
        started = time.perf_counter()
        points, poses, files, stats = sew_all_local_frames(
            cfg["run"], cameras=["up"], source=cfg["source"], **kwargs
        )
        wall_s = time.perf_counter() - started
        result_json = out / f"{args.scene}_{method}.json"
        result_png = out / f"{args.scene}_{method}.png"
        result_json.write_text(
            json.dumps(
                {
                    "scene": args.scene,
                    "method": method,
                    "archive": str(ARCHIVE),
                    "run": str(cfg["run"]),
                    "map": points,
                    "poses": poses,
                    "stats": stats,
                },
                default=json_default,
                separators=(",", ":"),
            ),
            encoding="utf-8",
        )
        plot_one(result_png, args.scene, method, points, poses, stats)
        row = {
            "method": method,
            "wall_s": wall_s,
            "n_frames": int(stats.get("n_frames", len(files))),
            "keyframes": int(stats.get("keyframes", 0)),
            "loop_closures": int(stats.get("loop_closures", 0)),
            "stable_cells": int(stats.get("stable_cells", len(points))),
            "loop_error_m": stats.get("loop_error_m"),
            "loop_error_deg": stats.get("loop_error_deg"),
            "result_json": str(result_json),
            "result_png": str(result_png),
        }
        rows.append(row)
        print(f"DONE {args.scene} {method}: {wall_s:.1f}s", flush=True)

    compare = out / f"{args.scene}_all_methods.png"
    plot_compare(compare, args.scene, rows)
    summary = {
        "scene": args.scene,
        "run": str(cfg["run"]),
        "archive": str(ARCHIVE),
        "methods": rows,
        "comparison_png": str(compare),
    }
    summary_path = out / "summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, default=json_default) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, default=json_default), flush=True)


if __name__ == "__main__":
    main()
