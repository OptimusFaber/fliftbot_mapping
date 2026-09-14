#!/usr/bin/env python3
"""Build Pandar horizon maps for real-room runs and score them vs GT."""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path

import numpy as np

_SCRIPTS = Path(__file__).resolve().parent
_ROOT = _SCRIPTS.parents[1]
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from flifltbot.mapping.pipeline import build_global_map_from_run
from flifltbot.mapping.real_room_gt import ROOM_X, ROOM_Y, UP_LIDAR_Z
from flifltbot.common.output.layout import MAPS_ROOT, RESULTS_ROOT, infer_results_dir
from flifltbot.common.runtime.accuracy_eval import score_map_real_room
from recalc_real_align_right import align_pin_right, save_overlay as save_overlay_height


ROOM_RUNS = [
    MAPS_ROOT / "real" / "20260813" / "room_pandar_run_141753",
    MAPS_ROOT / "real" / "20260813" / "room_pandar_run_150507",
]


def process_run(run_dir: Path, method="improved") -> dict:
    results_dir = infer_results_dir(run_dir, source="real")
    n_npz = len(list((run_dir / "horizon").glob("0*.npz"))) if (run_dir / "horizon").is_dir() else 0
    print("=" * 64)
    print(f"RUN {run_dir}  horizon_npz={n_npz}  → {results_dir}")

    build = build_global_map_from_run(
        run_dir,
        method=method,
        cameras=["up"],
        tag=run_dir.name,
        source="real",
    )
    pts = np.asarray(build["points"], float)
    traj = np.asarray([[p["x"], p["y"]] for p in build["poses"]], float)

    bbox_metrics = score_map_real_room(pts, align="bbox")
    z = float(UP_LIDAR_Z)
    wall = align_pin_right(pts, traj, z=z)
    overlay = results_dir / f"{run_dir.name}_gt_walls_up.png"
    save_overlay_height(run_dir.name, wall, overlay)
    save_overlay_height(run_dir.name, wall, results_dir / f"{run_dir.name}_gt_walls_up160.png")

    gap_left_cm = max(0.0, -wall["err_Ox_cm"])
    summary = {
        "run_dir": str(run_dir),
        "results_dir": str(results_dir),
        "method": method,
        "cameras": ["horizon"],
        "camera_z_m": z,
        "camera": "up",
        "sensor": "Pandar40P horizon ch13",
        "align": "pin_east_to_ROOM_X + prefer traj on right",
        "n_frames": int(build["stats"].get("n_frames", n_npz)),
        "n_horizon_npz": n_npz,
        "map_json": str(build["out_json"]),
        "map_png": str(build["out_png"]),
        "overlay_png": str(overlay),
        "loop_closures": int(build["stats"].get("loop_closures", 0)),
        "stats": {
            k: build["stats"][k]
            for k in (
                "n_frames",
                "keyframes",
                "loop_closures",
                "stable_cells",
                "mode",
                "deskew_mode",
                "timing_s",
            )
            if k in build["stats"]
        },
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
    (results_dir / "accuracy_improved_real.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(
        f'  up@{z:.2f}m  Ox {wall["err_Ox_cm"]:+.1f}cm  Oy {wall["err_Oy_cm"]:+.1f}cm  '
        f'E {wall["east_err_cm"]:+.1f}  gapL={gap_left_cm:.0f}  cells={wall["n_cells"]}  '
        f'→ {out_json.name}'
    )
    return summary


def _row(summary: dict) -> dict:
    mw = summary["metrics_walls_robust"]
    bb = summary.get("metrics_bbox") or {}
    return {
        "run": Path(summary["run_dir"]).name,
        "n_frames": summary["n_frames"],
        "n_cells": mw["n_cells"],
        "loop_closures": summary["loop_closures"],
        "err_Ox_cm": round(mw["err_Ox_cm"], 1),
        "err_Oy_cm": round(mw["err_Oy_cm"], 1),
        "abs_err_cm": round(mw["abs_err_cm"], 1),
        "east_err_cm": round(mw["east_err_cm"], 1),
        "gap_left_cm": round(mw["gap_left_cm"], 1),
        "near_gt_median_cm": (
            None if bb.get("near_gt_median_cm") is None
            else round(bb["near_gt_median_cm"], 1)
        ),
        "near_gt_p95_cm": (
            None if bb.get("near_gt_p95_cm") is None
            else round(bb["near_gt_p95_cm"], 1)
        ),
        "frac_within_50cm": bb.get("frac_within_50cm"),
        "overlay_png": summary["overlay_png"],
        "map_png": summary["map_png"],
        "results_dir": summary["results_dir"],
        "ok": True,
    }


def save_compare_png(rows: list[dict], summaries: list[dict], out_png: Path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon, Rectangle
    from flifltbot.mapping.real_room_gt import (
        CUT_H,
        CUT_W,
        DOOR,
        WIN_OUT,
        room_wall_vertices_at_z,
        visible_obstacle_rects_at_z,
    )

    n = len(summaries)
    fig, axes = plt.subplots(1, n, figsize=(12.4 * n / 2 + 0.4, 6.6), dpi=130)
    if n == 1:
        axes = [axes]
    z = UP_LIDAR_Z
    gt = np.asarray(room_wall_vertices_at_z(z))
    for ax, summary, row in zip(axes, summaries, rows):
        acc = json.loads(
            (Path(summary["results_dir"]) / f"{Path(summary['run_dir']).name}.json").read_text()
        )
        pts = np.asarray(acc["map"], float)
        traj = np.asarray([[p["x"], p["y"]] for p in acc.get("poses", [])], float)
        wall = align_pin_right(pts, traj, z=z)
        ax.set_aspect("equal")
        ax.add_patch(
            Polygon(gt, closed=True, fill=False, edgecolor="black", lw=2.0, label="GT")
        )
        ax.add_patch(
            Rectangle((0, 0), CUT_W, CUT_H, facecolor="#eee", edgecolor="gray", ls="--", alpha=0.7)
        )
        for x, y, w, d in visible_obstacle_rects_at_z(z):
            ax.add_patch(
                Rectangle((x, y), w, d, facecolor="#666666", edgecolor="k", alpha=0.35, lw=0.4)
            )
        P = wall["aligned"]
        step = max(1, len(P) // 8000)
        ax.scatter(P[::step, 0], P[::step, 1], s=1.0, c="tab:blue", alpha=0.28, label="map")
        if wall.get("traj") is not None and len(wall["traj"]):
            ax.plot(wall["traj"][:, 0], wall["traj"][:, 1], color="orange", lw=1.0, alpha=0.9)
        ax.set_title(
            f'{row["run"]}  n={row["n_frames"]}\n'
            f'Ox {row["err_Ox_cm"]:+.0f} cm   Oy {row["err_Oy_cm"]:+.0f} cm   '
            f'|err| {row["abs_err_cm"]:.0f} cm'
        )
        ax.set_xlim(-1.2, ROOM_X + 1.6)
        ax.set_ylim(-DOOR["recess"] - 0.7, ROOM_Y + WIN_OUT + 0.6)
        ax.set_xlabel("X room (m)")
        ax.set_ylabel("Y room (m)")
        ax.grid(True, alpha=0.25)
        ax.legend(loc="lower left", fontsize=8)
    fig.suptitle(
        "Pandar40P horizon maps vs real-room GT  (rigid corner alignment, no scale)",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(out_png, dpi=140, bbox_inches="tight")
    plt.close(fig)


def pick_winner(rows: list[dict]) -> dict:
    ok = [r for r in rows if r.get("ok")]
    if not ok:
        return {"winner": None, "reason": "no successful maps"}
    ranked = sorted(
        ok,
        key=lambda r: (
            r["abs_err_cm"],
            r["near_gt_median_cm"] if r["near_gt_median_cm"] is not None else 1e9,
            -r["n_cells"],
        ),
    )
    winner = ranked[0]
    reason = (
        f"smaller wall-span error |ΔOx|+|ΔOy| = {winner['abs_err_cm']:.1f} cm "
        f"(Ox {winner['err_Ox_cm']:+.1f}, Oy {winner['err_Oy_cm']:+.1f})"
    )
    if len(ranked) > 1:
        other = ranked[1]
        reason += (
            f"; runner-up {other['run']} has {other['abs_err_cm']:.1f} cm"
        )
    return {"winner": winner["run"], "reason": reason, "ranked": [r["run"] for r in ranked]}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', action='append', type=Path, help='Override room run (repeatable)')
    args = parser.parse_args(argv)
    selected_runs = [p.expanduser().resolve() for p in args.run] if args.run else ROOM_RUNS
    rows = []
    summaries = []
    for run in selected_runs:
        if not run.is_dir():
            print(f"MISSING {run}")
            rows.append({"run": run.name, "ok": False, "error": "missing run dir"})
            continue
        try:
            summary = process_run(run, method="improved")
            summaries.append(summary)
            rows.append(_row(summary))
        except Exception as exc:
            print(f"  FAILED {run}: {exc}")
            traceback.print_exc()
            rows.append({"run": run.name, "ok": False, "error": str(exc)})

    verdict = pick_winner(rows)
    out_dir = RESULTS_ROOT / "real" / "20260813" / "room_pandar_comparison"
    out_dir.mkdir(parents=True, exist_ok=True)
    compare_png = out_dir / "compare_run_141753_vs_run_150507.png"
    if len(summaries) >= 2:
        save_compare_png(rows[: len(summaries)], summaries, compare_png)
    payload = {
        "note": (
            "Pandar40P horizon channel 13 (~0°) maps, LiDAR ICP (no wheel odom). "
            "Rigidly aligned by room corners to hand-measured GT @ 1.645 m; no scale."
        ),
        "gt_room_m": [ROOM_X, ROOM_Y],
        "metric": "robust wall faces; abs_err=|dOx|+|dOy|",
        "verdict": verdict,
        "runs": rows,
        "compare_png": str(compare_png) if compare_png.exists() else None,
    }
    summary_path = out_dir / "pandar_room_compare.json"
    summary_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print("\n" + "=" * 64)
    print("PANDAR ROOM COMPARE")
    for r in rows:
        if not r.get("ok"):
            print(f'{r["run"]:16s} FAILED  {r.get("error")}')
            continue
        print(
            f'{r["run"]:16s}  n={r["n_frames"]:4d}  Ox {r["err_Ox_cm"]:+6.1f}  '
            f'Oy {r["err_Oy_cm"]:+6.1f}  |err| {r["abs_err_cm"]:6.1f}  '
            f'med {r["near_gt_median_cm"]}'
        )
    print("winner:", verdict.get("winner"), "—", verdict.get("reason"))
    print("wrote", summary_path)


if __name__ == "__main__":
    main()
