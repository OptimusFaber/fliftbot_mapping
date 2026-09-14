#!/usr/bin/env python3
"""Build and evaluate four mapping variants on Gazebo run_192133."""

from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts/evaluation"))

from flifltbot.mapping.factor_diag import (  # noqa: E402
    apply_se2, ate_metrics, load_odom_gt, sync_est_to_gt, umeyama_se2,
)
from flifltbot.mapping.pipeline import build_global_map_from_run  # noqa: E402
from flifltbot.common.runtime.accuracy_eval import gt_segments  # noqa: E402

RUN = ROOT / "maps/20260818/sim/run_192133"
OUT = ROOT / "results/20260818/sim/run_192133/method_compare"
VARIANTS = {
    "factor": dict(method="factor"),
    "wheel_only": dict(
        method="factor", frontend_icp="wheel", skip_lidar_sequential=True,
        skip_loop_closures=True, estimate_scale=False,
    ),
    "cloud_only": dict(method="clouds"),
    "improved": dict(method="improved"),
}


def _poses(payload):
    return [p for p in payload["poses"] if isinstance(p, dict) and p.get("pose")]


def evaluate(payload):
    poses = _poses(payload)
    gt = load_odom_gt(RUN)
    est_xy, gt_xy, est_yaw, gt_yaw, labels = sync_est_to_gt(poses, gt)
    yaw, tx, ty = umeyama_se2(est_xy, gt_xy)
    metrics = ate_metrics(est_xy, gt_xy, est_yaw, gt_yaw, labels)
    points = np.asarray(payload["map"], float)
    route = np.asarray([p["pose"][:2] for p in poses], float)
    return {
        "metrics": metrics,
        "transform": {"yaw_deg": math.degrees(yaw), "tx_m": tx, "ty_m": ty},
        "map_world": apply_se2(points, yaw, tx, ty),
        "route_world": apply_se2(route, yaw, tx, ty),
        "gt_route": np.asarray([g["pose"][:2] for g in gt], float),
        "sync_pairs": len(est_xy),
    }


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    rendered = []
    for name, kwargs in VARIANTS.items():
        print(f"START {name}", flush=True)
        started = time.perf_counter()
        result = build_global_map_from_run(
            RUN, source="sim", cameras=["up"],
            tag=f"run_192133_{name}", write_module_copy=False, **kwargs,
        )
        source_json = Path(result["out_json"])
        target_json = OUT / f"run_192133_{name}.json"
        target_png = OUT / f"run_192133_{name}.png"
        source_png = Path(result["out_png"])
        source_json.replace(target_json)
        source_png.replace(target_png)
        payload = json.loads(target_json.read_text(encoding="utf-8"))
        ev = evaluate(payload)
        stats = result["stats"]
        row = {
            "variant": name,
            "method": kwargs["method"],
            "wall_s": time.perf_counter() - started,
            "map_json": str(target_json),
            "map_png": str(target_png),
            "sync_pairs": ev["sync_pairs"],
            "alignment": "global rigid SE(2), no scale",
            "transform": ev["transform"],
            "metrics": ev["metrics"],
            "stats": {k: stats.get(k) for k in (
                "n_frames", "keyframes", "wheel_edges", "lidar_seq_edges",
                "loop_closures", "stable_cells", "loop_error_m", "loop_error_deg",
            )},
        }
        rows.append(row)
        rendered.append((name, ev))
        print(f"DONE {name}: RMSE={ev['metrics']['ate_rmse_m'] * 100:.1f} cm", flush=True)

    fig, axes = plt.subplots(2, 2, figsize=(16, 16), sharex=True, sharey=True)
    for ax, (name, ev) in zip(axes.flat, rendered):
        for a, b in gt_segments():
            ax.plot([a[0], b[0]], [a[1], b[1]], color="black", lw=0.75, alpha=0.7)
        pts = ev["map_world"]
        step = max(1, len(pts) // 18000)
        ax.scatter(pts[::step, 0], pts[::step, 1], s=0.55, c="#1677ff", alpha=0.35)
        ax.plot(ev["route_world"][:, 0], ev["route_world"][:, 1], c="#e63946", lw=0.8)
        ax.plot(ev["gt_route"][:, 0], ev["gt_route"][:, 1], c="#20a83a", lw=0.8)
        m = ev["metrics"]
        ax.set_title(f"{name}: median {m['ate_median_m']*100:.1f} cm, RMSE {m['ate_rmse_m']*100:.1f} cm")
        ax.set_aspect("equal")
        ax.grid(alpha=0.18)
    fig.suptitle("run_192133: map and trajectory vs Gazebo GT\nblack=GT scene, blue=map, red=prediction, green=GT")
    fig.tight_layout()
    comparison = OUT / "run_192133_four_methods_vs_gt.png"
    fig.savefig(comparison, dpi=180)
    plt.close(fig)

    report = {
        "run": str(RUN),
        "gt": "Gazebo /odom + warehouse.sdf walls, all shelves and columns",
        "alignment": "global least-squares rigid SE(2); no scale",
        "variants": rows,
        "comparison_png": str(comparison),
    }
    report_path = OUT / "metrics.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"WROTE {report_path}", flush=True)


if __name__ == "__main__":
    main()
