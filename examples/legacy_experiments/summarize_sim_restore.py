#!/usr/bin/env python3
"""Aggregate and render the isolated run_192133 restore experiments."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from flifltbot.mapping.factor_diag import apply_se2, load_odom_gt  # noqa: E402
from flifltbot.common.runtime.accuracy_eval import dist_seg, gt_segments  # noqa: E402

BASE = ROOT / "results/20260819/sim_restore/run_192133"
ORDER = (
    "odom_oracle",
    "wheel_pivot0",
    "factor_pivot_old",
    "factor_pivot0",
    "factor_pivot0_no_loops",
    "real_factor",
    "factor_scan_pivot0",
)


def main() -> None:
    rows = []
    rendered = []
    segments = gt_segments()
    gt_route = np.asarray([x["pose"][:2] for x in load_odom_gt(
        ROOT / "maps/20260818/sim/run_192133"
    )], float)
    for name in ORDER:
        result_path = BASE / name / "result.json"
        if not result_path.is_file():
            continue
        data = json.loads(result_path.read_text(encoding="utf-8"))
        tr = data["transform"]
        yaw = math.radians(tr["yaw_deg"])
        pts = apply_se2(np.asarray(data["map"], float), yaw, tr["tx_m"], tr["ty_m"])
        route = apply_se2(
            np.asarray([p["pose"][:2] for p in data["poses"]], float),
            yaw, tr["tx_m"], tr["ty_m"],
        )
        rng = np.random.default_rng(0)
        sample = pts[rng.choice(len(pts), min(2000, len(pts)), replace=False)]
        dists = np.asarray([
            min(dist_seg(point, a, b) for a, b in segments) for point in sample
        ])
        near = dists[dists < 1.0]
        map_metrics = {
            "point_to_gt_median_cm_under_1m": (
                float(np.median(near) * 100) if len(near) else None
            ),
            "point_to_gt_p95_cm_under_1m": (
                float(np.percentile(near, 95) * 100) if len(near) else None
            ),
            "fraction_within_20cm": float(np.mean(dists < 0.20)),
            "fraction_within_50cm": float(np.mean(dists < 0.50)),
        }
        row = {
            "variant": name,
            "oracle": bool(data["use_absolute_odom"]),
            "metrics": data["metrics"],
            "map_metrics": map_metrics,
            "geometry": data["geometry"],
            "kwargs": data["kwargs"],
            "result_json": str(result_path),
        }
        rows.append(row)
        rendered.append((name, data, pts, route, map_metrics))

    fig, axes = plt.subplots(2, 3, figsize=(18, 12), sharex=True, sharey=True)
    for ax, (name, data, pts, route, mm) in zip(axes.flat, rendered):
        for a, b in segments:
            ax.plot([a[0], b[0]], [a[1], b[1]], color="black", lw=0.65, alpha=0.65)
        step = max(1, len(pts) // 16000)
        ax.scatter(pts[::step, 0], pts[::step, 1], s=0.45, alpha=0.30, color="#1677ff")
        ax.plot(route[:, 0], route[:, 1], lw=0.8, color="#e63946")
        ax.plot(gt_route[:, 0], gt_route[:, 1], lw=0.75, color="#20a83a")
        m = data["metrics"]
        ax.set_title(
            f"{name}\nATE med {m['ate_median_m']*100:.1f} cm, "
            f"RMSE {m['ate_rmse_m']*100:.1f} cm; "
            f"map med {mm['point_to_gt_median_cm_under_1m']:.1f} cm"
        )
        ax.set_aspect("equal")
        ax.grid(alpha=0.15)
    fig.suptitle(
        "run_192133 restore experiments — black GT scene, blue map, "
        "red estimate, green Gazebo GT"
    )
    fig.tight_layout()
    comparison = BASE / "comparison_vs_gt.png"
    fig.savefig(comparison, dpi=180)
    plt.close(fig)

    report = {
        "run": "maps/20260818/sim/run_192133",
        "alignment": "trajectory least-squares rigid SE(2); no scale/reflection",
        "warning": "odom_oracle uses Gazebo odom and is not a valid SLAM result",
        "variants": rows,
        "comparison_png": str(comparison),
    }
    (BASE / "summary.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
