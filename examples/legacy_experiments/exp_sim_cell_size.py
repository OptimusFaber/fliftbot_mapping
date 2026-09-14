#!/usr/bin/env python3
"""Hit-grid cell-size A/B on sim run_000407: 5, 4, 3, 2, 1 cm.

Measures wall time, peak RSS, CPU, map size, and warehouse GT accuracy.
Writes maps + GT overlays under results/20260811/sim/run_000407/cell_grid/.
"""

from __future__ import annotations

import argparse
import json
import resource
import subprocess
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

_SCRIPTS = Path(__file__).resolve().parent
_ROOT = _SCRIPTS.parent
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from flifltbot.common.runtime.accuracy_eval import (  # noqa: E402
    AISLE_GT,
    EAST_SX,
    EAST_YS,
    WEST_SX,
    WEST_YS,
    score_map,
    shelf_x_faces,
    to_world,
)

RUN_DIR = _ROOT / "maps" / "20260811" / "sim" / "run_000407"
OUT_DIR = _ROOT / "results" / "20260811" / "sim" / "run_000407" / "cell_grid"
CELL_CM = (5, 4, 3, 2, 1)
WAREHOUSE = dict(x0=-15.0, y0=-25.0, w=30.0, l=50.0)


def _rss_mb() -> float:
    # Linux ru_maxrss is kilobytes.
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def draw_gt(ax):
    wh = WAREHOUSE
    ax.add_patch(
        Rectangle(
            (wh["x0"], wh["y0"]),
            wh["w"],
            wh["l"],
            fill=False,
            edgecolor="black",
            lw=1.8,
            label="GT warehouse",
        )
    )
    for sx, sy in [(WEST_SX, y) for y in WEST_YS] + [(EAST_SX, y) for y in EAST_YS]:
        x0, x1 = shelf_x_faces(sx)
        ax.add_patch(
            Rectangle(
                (x0, sy - 0.3),
                x1 - x0,
                0.6,
                facecolor="#666666",
                edgecolor="k",
                alpha=0.35,
                lw=0.4,
            )
        )
    w_hi = shelf_x_faces(WEST_SX)[1]
    e_lo = shelf_x_faces(EAST_SX)[0]
    ax.axvline(w_hi, color="tab:green", ls="--", lw=0.8, alpha=0.7)
    ax.axvline(e_lo, color="tab:green", ls="--", lw=0.8, alpha=0.7)


def save_overlay(
    world_pts: np.ndarray,
    cell_cm: int,
    metrics: dict,
    out_png: Path,
    world_traj: np.ndarray | None = None,
):
    fig, ax = plt.subplots(figsize=(8.2, 10.2), dpi=120)
    ax.set_aspect("equal")
    draw_gt(ax)
    if len(world_pts):
        step = max(1, len(world_pts) // 12000)
        ax.scatter(
            world_pts[::step, 0],
            world_pts[::step, 1],
            s=1.6,
            c="tab:blue",
            alpha=0.28,
            edgecolors="none",
            label="map",
        )
    if world_traj is not None and len(world_traj):
        ax.plot(
            world_traj[:, 0],
            world_traj[:, 1],
            color="tab:orange",
            lw=0.9,
            alpha=0.85,
            label="traj",
        )
    med = metrics.get("near_gt_median_cm")
    wall = metrics.get("mean_abs_wall_cm")
    ax.set_title(
        f"cell {cell_cm} cm   n={metrics.get('n_cells')}   "
        f"med {None if med is None else round(med, 1)} cm   "
        f"walls {None if wall is None else round(wall, 1)} cm"
    )
    ax.set_xlim(-17, 17)
    ax.set_ylim(-27, 27)
    ax.set_xlabel("world x (m)")
    ax.set_ylabel("world y (m)")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_png, dpi=130, bbox_inches="tight")
    plt.close(fig)


def run_one(cell_cm: int) -> dict:
    cell_m = cell_cm / 100.0
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tag = f"run_000407_cell{cell_cm}cm"

    from flifltbot.mapping.pipeline import build_global_map_from_run, scaled_min_hits_for_cell

    min_hits = scaled_min_hits_for_cell(cell_m)
    t0 = time.perf_counter()
    rss0 = _rss_mb()
    ru0 = resource.getrusage(resource.RUSAGE_SELF)
    build = build_global_map_from_run(
        RUN_DIR,
        method="improved",
        cameras=["up"],
        tag=tag,
        source="sim",
        cell_m=cell_m,
        map_min_hits=min_hits,
        write_module_copy=False,
    )
    elapsed = time.perf_counter() - t0
    ru1 = resource.getrusage(resource.RUSAGE_SELF)
    peak_rss_mb = _rss_mb()
    pts = np.asarray(build["points"], float)
    metrics = score_map(pts)
    world = to_world(pts) if len(pts) else pts
    pose_xy = []
    for item in build.get("poses") or []:
        if isinstance(item, dict):
            pose = item.get("pose") or (item.get("x"), item.get("y"))
            pose_xy.append([float(pose[0]), float(pose[1])])
        else:
            pose_xy.append([float(item[0]), float(item[1])])
    pose_xy = np.asarray(pose_xy, dtype=float)
    world_traj = to_world(pose_xy) if len(pose_xy) else None
    overlay = OUT_DIR / f"overlay_{cell_cm}cm.png"
    overlay_traj = OUT_DIR / f"overlay_{cell_cm}cm_with_traj.png"
    save_overlay(world, cell_cm, metrics, overlay)
    save_overlay(world, cell_cm, metrics, overlay_traj, world_traj=world_traj)

    map_json = Path(build["out_json"])
    map_png = Path(build["out_png"])
    # keep copies in cell_grid/
    grid_map_png = OUT_DIR / f"map_{cell_cm}cm.png"
    if map_png.is_file():
        grid_map_png.write_bytes(map_png.read_bytes())

    stats = build.get("stats") or {}
    timing = stats.get("timing_s") or {}
    row = {
        "cell_cm": cell_cm,
        "cell_m": cell_m,
        "ok": True,
        "wall_time_s": round(elapsed, 3),
        "timing_s": {k: round(float(v), 3) for k, v in timing.items()},
        "cpu_user_s": round(ru1.ru_utime - ru0.ru_utime, 3),
        "cpu_sys_s": round(ru1.ru_stime - ru0.ru_stime, 3),
        "peak_rss_mb": round(peak_rss_mb, 1),
        "rss_delta_mb": round(peak_rss_mb - rss0, 1),
        "n_cells": int(metrics["n_cells"]),
        "n_frames": int(stats.get("n_frames", 0)),
        "raw_cells": int(stats.get("raw_cells", 0)),
        "stable_cells": int(stats.get("stable_cells", metrics["n_cells"])),
        "transformed_points": int(stats.get("transformed_points", 0)),
        "loop_closures": int(stats.get("loop_closures", 0)),
        "min_hits": int(min_hits),
        "map_json_mb": round(map_json.stat().st_size / 1e6, 3) if map_json.is_file() else None,
        "map_json": str(map_json),
        "map_png": str(map_png),
        "overlay_png": str(overlay),
        "overlay_traj_png": str(overlay_traj),
        "grid_map_png": str(grid_map_png),
        "metrics": {
            "span_w_m": metrics["span_w_m"],
            "span_l_m": metrics["span_l_m"],
            "span_w_err_cm": metrics["span_w_err_cm"],
            "span_l_err_cm": metrics["span_l_err_cm"],
            "mean_abs_wall_cm": metrics["mean_abs_wall_cm"],
            "wall_err_cm": metrics["wall_err_cm"],
            "aisle_m": metrics["aisle_m"],
            "aisle_err_cm": metrics["aisle_err_cm"],
            "aisle_gt_m": AISLE_GT,
            "near_gt_median_cm": metrics["near_gt_median_cm"],
            "near_gt_p95_cm": metrics["near_gt_p95_cm"],
            "ghost_aisle_cells": metrics["ghost_aisle_cells"],
        },
    }
    (OUT_DIR / f"metrics_{cell_cm}cm.json").write_text(
        json.dumps(row, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({k: row[k] for k in (
        "cell_cm", "min_hits", "wall_time_s", "peak_rss_mb", "n_cells",
        "cpu_user_s",
    )} | {"med_cm": row["metrics"]["near_gt_median_cm"]}))
    return row


def save_compare_figure(rows: list[dict], out_png: Path):
    ok = [r for r in rows if r.get("ok")]
    n = len(ok)
    if n == 0:
        return
    fig, axes = plt.subplots(1, n, figsize=(4.4 * n + 0.6, 7.2), dpi=120)
    if n == 1:
        axes = [axes]
    for ax, row in zip(axes, ok):
        data = json.loads(Path(row["map_json"]).read_text(encoding="utf-8"))
        pts = np.asarray(data["map"], float)
        world = to_world(pts) if len(pts) else pts
        ax.set_aspect("equal")
        draw_gt(ax)
        if len(world):
            step = max(1, len(world) // 8000)
            ax.scatter(
                world[::step, 0], world[::step, 1],
                s=1.0, c="tab:blue", alpha=0.25, edgecolors="none",
            )
        m = row["metrics"]
        ax.set_title(
            f'{row["cell_cm"]} cm\n'
            f'{row["wall_time_s"]:.0f}s  {row["peak_rss_mb"]:.0f} MB\n'
            f'n={row["n_cells"]}  med {m["near_gt_median_cm"]:.1f} cm'
        )
        ax.set_xlim(-17, 17)
        ax.set_ylim(-27, 27)
        ax.set_xlabel("x (m)")
        ax.grid(True, alpha=0.2)
    axes[0].set_ylabel("y (m)")
    fig.suptitle(
        "sim run_000407 — hit-grid cell size vs warehouse GT",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(out_png, dpi=130, bbox_inches="tight")
    plt.close(fig)


def save_bars(rows: list[dict], out_png: Path):
    ok = [r for r in rows if r.get("ok")]
    if not ok:
        return
    xs = [str(r["cell_cm"]) for r in ok]
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.6), dpi=120)
    axes[0].bar(xs, [r["wall_time_s"] for r in ok], color="tab:blue")
    axes[0].set_title("Build wall time")
    axes[0].set_xlabel("cell (cm)")
    axes[0].set_ylabel("seconds")
    axes[1].bar(xs, [r["peak_rss_mb"] for r in ok], color="tab:orange")
    axes[1].set_title("Peak RSS")
    axes[1].set_xlabel("cell (cm)")
    axes[1].set_ylabel("MB")
    axes[2].bar(xs, [r["metrics"]["near_gt_median_cm"] or 0 for r in ok], color="tab:green")
    axes[2].set_title("Median dist to GT")
    axes[2].set_xlabel("cell (cm)")
    axes[2].set_ylabel("cm")
    fig.suptitle("sim run_000407 cell-size tradeoff")
    fig.tight_layout()
    fig.savefig(out_png, dpi=130, bbox_inches="tight")
    plt.close(fig)


def worker_main(cell_cm: int) -> int:
    try:
        run_one(cell_cm)
        return 0
    except Exception as exc:
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        fail = {"cell_cm": cell_cm, "ok": False, "error": str(exc)}
        (OUT_DIR / f"metrics_{cell_cm}cm.json").write_text(
            json.dumps(fail, indent=2) + "\n", encoding="utf-8"
        )
        print(f"FAILED cell={cell_cm}cm: {exc}", file=sys.stderr)
        raise


def driver_main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for cell_cm in CELL_CM:
        print("=" * 64)
        print(f"cell {cell_cm} cm")
        metrics_path = OUT_DIR / f"metrics_{cell_cm}cm.json"
        cmd = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--worker",
            "--cell-cm",
            str(cell_cm),
        ]
        env = dict(**{k: v for k, v in __import__("os").environ.items()})
        env["PYTHONPATH"] = str(_SRC) + (
            (":" + env["PYTHONPATH"]) if env.get("PYTHONPATH") else ""
        )
        proc = subprocess.run(cmd, cwd=str(_ROOT), env=env)
        if proc.returncode != 0 or not metrics_path.is_file():
            rows.append({"cell_cm": cell_cm, "ok": False, "error": f"exit {proc.returncode}"})
            continue
        rows.append(json.loads(metrics_path.read_text(encoding="utf-8")))

    compare_png = OUT_DIR / "compare_overlays.png"
    bars_png = OUT_DIR / "compare_bars.png"
    save_compare_figure(rows, compare_png)
    save_bars(rows, bars_png)
    summary = {
        "run_dir": str(RUN_DIR),
        "out_dir": str(OUT_DIR),
        "note": (
            "Same pipeline as real: enc_v + angle/yaw-rate + LiDAR ICP "
            "(USE_ABSOLUTE_ODOM=False). Sim uses rear-fork ICR: "
            "vy=-wz*REAR_PIVOT_X from enc_v/enc_wz (no /odom poses). "
            "MAP_CELL changes; min_hits scaled linearly with cell size "
            "(5→5, 4→4, 3→3, 2→2, 1→1). "
            "Isolated-component min cells scale with area (~0.125 m²)."
        ),
        "compare_overlays_png": str(compare_png),
        "compare_bars_png": str(bars_png),
        "runs": rows,
    }
    summary_path = OUT_DIR / "cell_size_compare.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print("\n" + "=" * 64)
    print(
        f"{'cm':>4}  {'hits':>5}  {'time_s':>8}  {'RSS_MB':>8}  {'cells':>8}  "
        f"{'json_MB':>8}  {'med_cm':>8}  {'wall_cm':>8}"
    )
    for r in rows:
        if not r.get("ok"):
            print(f"{r['cell_cm']:4d}  FAILED  {r.get('error')}")
            continue
        m = r["metrics"]
        print(
            f"{r['cell_cm']:4d}  {r.get('min_hits', '?'):5}  {r['wall_time_s']:8.1f}  "
            f"{r['peak_rss_mb']:8.1f}  {r['n_cells']:8d}  {r['map_json_mb']:8.3f}  "
            f"{m['near_gt_median_cm']:8.2f}  {m['mean_abs_wall_cm'] or 0:8.2f}"
        )
    print("wrote", summary_path)
    print("overlays", compare_png)
    return 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--cell-cm", type=int, default=None)
    args = parser.parse_args()
    if args.worker:
        if args.cell_cm is None:
            parser.error("--cell-cm required with --worker")
        return worker_main(args.cell_cm)
    return driver_main()


if __name__ == "__main__":
    raise SystemExit(main())
