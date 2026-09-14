#!/usr/bin/env python3
"""Build maps and GT metrics for real runs; write only under ros-data/results/."""

from __future__ import annotations

import argparse
import json
import shutil
import traceback
from pathlib import Path

import numpy as np

from flifltbot.mapping.pipeline import build_global_map_from_run
from flifltbot.mapping.real_room_gt import ROOM_X, ROOM_Y, UP_LIDAR_Z
from flifltbot.common.output.layout import MAPS_ROOT, RESULTS_ROOT, infer_results_dir
from flifltbot.common.runtime.accuracy_eval import score_map_real_room

import sys

_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))
from recalc_real_align_right import (  # noqa: E402
    LOW_LIDAR_Z,
    RUN_CAMERA_Z,
    UP_LIDAR_Z,
    align_pin_right,
    save_overlay as save_overlay_height,
)

# Cameras used when (re)building the global map offline.
RUN_MAP_CAMERAS = {
    'run_122309': ('left', 'right'),
    'run_121410': ('left', 'right'),
    'run_140543': ('left', 'right'),
    'run_144042': ('left', 'right'),
    'run_162252': ('up',),
    'run_162516': ('up',),
    'run_164805': ('up',),
}


def legacy_run_id(name: str) -> str:
    marker = name.rfind("run_")
    return name[marker:] if marker >= 0 else name

# Eval artifacts that historically leaked into maps/ — move to results/.
MOVE_FROM_MAPS = (
    "accuracy_contour_align_up160.json",
    "accuracy_improved_real.json",
    "accuracy_improved_real_pca.json",
    "accuracy_improved_sim.json",
    "accuracy_walls_midline_up160.json",
    "compare_contour_align_up160.png",
    "compare_gt_overlay_pca.png",
    "compare_walls_midline_up160.png",
    "global_map.json",
    "global_map.png",
)


def discover_real_runs() -> list[Path]:
    runs = sorted(MAPS_ROOT.glob("real/*/*_run_*"))
    return [
        p for p in runs
        if p.is_dir() and (any(p.glob("0*.json")) or any((p / "horizon").glob("0*.npz")))
    ]


# Alignment / overlays: see scripts/recalc_real_align_right.py

def migrate_maps_artifacts(run_dir: Path):
    """Move historical eval/compare files from maps/ into results/."""
    results_dir = infer_results_dir(run_dir, source="real")
    moved = []
    for name in MOVE_FROM_MAPS:
        src = run_dir / name
        if src.is_file():
            dst = results_dir / name
            if not dst.exists():
                shutil.move(str(src), str(dst))
            else:
                src.unlink()
            moved.append(name)
    # improve_exp folder
    exp = run_dir / "improve_exp"
    if exp.is_dir():
        dst = results_dir / "improve_exp"
        if dst.exists():
            shutil.rmtree(dst)
        shutil.move(str(exp), str(dst))
        moved.append("improve_exp/")
    return moved


def process_run(run_dir: Path, cameras=None, method="improved") -> dict:
    results_dir = infer_results_dir(run_dir, source="real")
    n_frames = len(list(run_dir.glob("0*.json")))
    run_id = legacy_run_id(run_dir.name)
    if cameras is None:
        cameras = RUN_MAP_CAMERAS.get(run_id, ("up",))
    print("=" * 64)
    print(f"RUN {run_dir}  frames={n_frames}  cameras={cameras}  → {results_dir}")

    moved = migrate_maps_artifacts(run_dir)
    if moved:
        print(f"  migrated from maps/: {moved}")

    build = build_global_map_from_run(
        run_dir,
        method=method,
        cameras=list(cameras),
        tag=run_dir.name,
        source="real",
    )
    pts = np.asarray(build["points"], float)
    traj = np.asarray([[p["x"], p["y"]] for p in build["poses"]], float)

    # bbox-style metrics (existing scorer)
    bbox_metrics = score_map_real_room(pts, align="bbox")

    z = float(RUN_CAMERA_Z.get(run_id, UP_LIDAR_Z))
    cam = "up" if z >= 1.0 else "low"
    wall = align_pin_right(pts, traj, z=z)
    overlay = results_dir / f"{run_dir.name}_gt_walls_{cam}.png"
    save_overlay_height(run_dir.name, wall, overlay)
    if cam == "up":
        save_overlay_height(run_dir.name, wall, results_dir / f"{run_dir.name}_gt_walls_up160.png")

    gap_left_cm = max(0.0, -wall["err_Ox_cm"])  # incomplete coverage when pin-right
    summary = {
        "run_dir": str(run_dir),
        "results_dir": str(results_dir),
        "method": method,
        "cameras": list(cameras),
        "camera_z_m": z,
        "camera": cam,
        "align": "pin_east_to_ROOM_X + prefer traj on right",
        "n_frames": n_frames,
        "map_json": str(build["out_json"]),
        "map_png": str(build["out_png"]),
        "overlay_png": str(overlay),
        "loop_closures": int(build["stats"].get("loop_closures", 0)),
        "metrics_bbox": bbox_metrics,
        "metrics_walls_robust": {
            "Ox_m": wall["Ox"],
            "Oy_m": wall["Oy"],
            "err_Ox_cm": wall["err_Ox_cm"],
            "err_Oy_cm": wall["err_Oy_cm"],
            "abs_err_cm": wall["abs_err_cm"],
            "east_err_cm": wall["east_err_cm"],
            "gap_left_cm": gap_left_cm,
            "W": wall["W"],
            "E": wall["E"],
            "S": wall["S"],
            "N": wall["N"],
            "n_cells": wall["n_cells"],
            "flip": wall["flip"],
            "delta_deg": wall["delta_deg"],
            "ok": True,
            "camera_z_m": z,
            "method": "pin RIGHT; robust faces; height-aware niches",
        },
        "gt_room_m": [ROOM_X, ROOM_Y],
    }
    out_json = results_dir / f"accuracy_{method}_real.json"
    out_json.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    # also refresh the improved name used by overlays
    (results_dir / "accuracy_improved_real.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(
        f'  {cam}@{z:.2f}m  Ox {wall["err_Ox_cm"]:+.1f}cm  Oy {wall["err_Oy_cm"]:+.1f}cm  '
        f'E {wall["east_err_cm"]:+.1f}  gapL={gap_left_cm:.0f}  → {out_json.name}'
    )
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', action='append', type=Path, help='Process only this run (repeatable)')
    args = parser.parse_args(argv)
    runs = [p.expanduser().resolve() for p in args.run] if args.run else discover_real_runs()
    print(f"Found {len(runs)} real runs")
    rows = []
    for run in runs:
        try:
            summary = process_run(run, method="improved")
            mw = summary["metrics_walls_robust"]
            rows.append(
                {
                    "run": run.name,
                    "date": run.parts[-3],
                    "camera": summary["camera"],
                    "camera_z_m": summary["camera_z_m"],
                    "n_frames": summary["n_frames"],
                    "err_Ox_cm": round(mw["err_Ox_cm"], 1),
                    "err_Oy_cm": round(mw["err_Oy_cm"], 1),
                    "east_err_cm": round(mw["east_err_cm"], 1),
                    "gap_left_cm": round(mw["gap_left_cm"], 1),
                    "abs_err_cm": round(mw["abs_err_cm"], 1),
                    "n_cells": mw["n_cells"],
                    "loop_closures": summary["loop_closures"],
                    "results_dir": summary["results_dir"],
                    "overlay_png": summary["overlay_png"],
                    "ok": True,
                }
            )
        except Exception as exc:
            print(f"  FAILED {run}: {exc}")
            traceback.print_exc()
            rows.append({"run": run.name, "ok": False, "error": str(exc)})

    summary_path = RESULTS_ROOT / "real_runs_accuracy_summary.json"
    payload = {
        "note": "pin RIGHT (east→ROOM_X); objects by camera height; robot drives on right; "
        "negative Ox / gap_left ≈ incomplete left coverage (not scale error)",
        "metric": "robust wall faces; height-aware niches; abs_err=|dOx|+|dOy|",
        "gt_room_m": [ROOM_X, ROOM_Y],
        "runs": rows,
    }
    summary_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    print("\n" + "=" * 64)
    print("SUMMARY (pin RIGHT, height-aware)")
    for r in rows:
        if not r.get("ok"):
            print(f'{r["run"]:16s} FAILED  {r.get("error")}')
            continue
        print(
            f'{r["date"]}/{r["run"]:12s}  {r["camera"]:3s}@{r["camera_z_m"]:.2f}  '
            f'Ox {r["err_Ox_cm"]:+6.1f}  Oy {r["err_Oy_cm"]:+6.1f}  '
            f'gapL {r["gap_left_cm"]:5.0f}  |err| {r["abs_err_cm"]:6.1f}'
        )
    print("wrote", summary_path)


if __name__ == "__main__":
    main()
