#!/usr/bin/env python3
"""A/B: real run_162516 with wheel odometry vs lidar-only.

Real logs have enc_v/enc_dgr (used as motion prior + pose-graph edges), NOT
Gazebo absolute odom_x/y/yaw (that path bypasses ICP entirely).
"""

from __future__ import annotations

import json
import resource
import sys
import time
from pathlib import Path

import numpy as np

_SCRIPTS = Path(__file__).resolve().parent
_ROOT = _SCRIPTS.parent
_SRC = _ROOT / "src"
for p in (_SRC, _SCRIPTS):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from flifltbot.mapping.config import METHOD_PRESETS, UP_LIDAR_Z  # noqa: E402
from flifltbot.mapping.output import save_global_map  # noqa: E402
from flifltbot.mapping.pipeline import sew_all_local_frames  # noqa: E402
from flifltbot.mapping.real_room_gt import ROOM_X, ROOM_Y  # noqa: E402
from flifltbot.common.runtime.accuracy_eval import score_map_real_room  # noqa: E402
from recalc_real_align_right import (  # noqa: E402
    _indoor_mask,
    align_pin_right,
    save_overlay,
)

RUN = _ROOT / "maps" / "20260807" / "real" / "run_162516"
OUT = _ROOT / "results" / "20260807" / "real" / "run_162516" / "odom_ab"
CAMERAS = ["up"]
Z = UP_LIDAR_Z


def _rss_mb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def build_variant(name: str, *, use_encoder: bool) -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    tag = f"run_162516_{name}"
    kwargs = dict(METHOD_PRESETS["improved"])
    kwargs["cameras"] = CAMERAS
    kwargs["use_encoder"] = use_encoder

    t0 = time.perf_counter()
    rss0 = _rss_mb()
    ru0 = resource.getrusage(resource.RUSAGE_SELF)
    points, poses, files, stats = sew_all_local_frames(RUN, **kwargs)
    wall_s = time.perf_counter() - t0
    ru1 = resource.getrusage(resource.RUSAGE_SELF)
    peak_rss = _rss_mb()

    out_json, out_png = save_global_map(
        points,
        poses,
        tag,
        RUN,
        used_encoder=use_encoder and bool(stats.get("encoder_frames")),
        source="real",
        write_module_copy=False,
    )
    # also keep copies under odom_ab/
    local_map = OUT / f"{tag}.png"
    local_map.write_bytes(Path(out_png).read_bytes())

    pts = np.asarray(points, float)
    traj = np.asarray([[p["x"], p["y"]] for p in poses], float)
    wall = align_pin_right(pts, traj if len(traj) else None, z=Z)
    overlay = OUT / f"{tag}_gt_walls_up.png"
    save_overlay(tag, wall, overlay)

    inn = np.asarray(wall["aligned"], float)
    inn = inn[_indoor_mask(inn, Z)]
    aligned = score_map_real_room(
        inn if len(inn) >= 80 else wall["aligned"], align="none"
    )

    row = {
        "variant": name,
        "use_encoder": use_encoder,
        "ok": True,
        "n_frames": int(stats.get("n_frames", len(files))),
        "wall_time_s": round(wall_s, 3),
        "timing_s": {
            k: round(float(v), 3) for k, v in (stats.get("timing_s") or {}).items()
        },
        "cpu_user_s": round(ru1.ru_utime - ru0.ru_utime, 3),
        "cpu_sys_s": round(ru1.ru_stime - ru0.ru_stime, 3),
        "peak_rss_mb": round(peak_rss, 1),
        "rss_delta_mb": round(peak_rss - rss0, 1),
        "mode": stats.get("mode"),
        "deskew_mode": stats.get("deskew_mode"),
        "encoder_frames": int(stats.get("encoder_frames", 0)),
        "encoder_synced_frames": int(stats.get("encoder_synced_frames", 0) or 0),
        "keyframes": int(stats.get("keyframes", 0)),
        "loop_closures": int(stats.get("loop_closures", 0)),
        "wheel_edges": int(stats.get("wheel_edges", 0)),
        "stable_cells": int(stats.get("stable_cells", len(pts))),
        "map_json": str(out_json),
        "map_png": str(local_map),
        "overlay_png": str(overlay),
        "metrics_walls_robust": {
            "Ox_m": wall["Ox"],
            "Oy_m": wall["Oy"],
            "err_Ox_cm": wall["err_Ox_cm"],
            "err_Oy_cm": wall["err_Oy_cm"],
            "abs_err_cm": wall["abs_err_cm"],
            "east_err_cm": wall["east_err_cm"],
            "gap_left_cm": max(0.0, -float(wall["err_Ox_cm"])),
            "n_cells": wall["n_cells"],
            "delta_deg": wall.get("delta_deg"),
            "level_left_deg": wall.get("level_left_deg"),
        },
        "metrics_aligned": {
            "near_gt_median_cm": aligned.get("near_gt_median_cm"),
            "near_gt_p95_cm": aligned.get("near_gt_p95_cm"),
            "frac_within_50cm": aligned.get("frac_within_50cm"),
            "mean_abs_wall_cm": aligned.get("mean_abs_wall_cm"),
        },
        "gt_room_m": [ROOM_X, ROOM_Y],
        "note": (
            "with_odom: improved + enc_v/enc_dgr wheel prior + wheel pose-graph edges. "
            "no_odom: same improved settings but use_encoder=False (pure LiDAR ICP). "
            "Neither uses Gazebo absolute odom_x/y/yaw — real log has none."
        ),
    }
    (OUT / f"{tag}_metrics.json").write_text(
        json.dumps(row, indent=2) + "\n", encoding="utf-8"
    )
    mw = row["metrics_walls_robust"]
    print(
        f"{name}: {wall_s:.1f}s  RSS={peak_rss:.0f}MB  "
        f"loops={row['loop_closures']}  cells={row['stable_cells']}  "
        f"Ox={mw['err_Ox_cm']:+.1f} Oy={mw['err_Oy_cm']:+.1f} "
        f"|err|={mw['abs_err_cm']:.1f}  med={aligned.get('near_gt_median_cm')}"
    )
    return row


def save_compare(rows: list[dict]):
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

    ok = [r for r in rows if r.get("ok")]
    n = len(ok)
    fig, axes = plt.subplots(1, n, figsize=(11.5, 5.8), dpi=130)
    if n == 1:
        axes = [axes]
    gt = np.asarray(room_wall_vertices_at_z(Z))
    for ax, row in zip(axes, ok):
        data = json.loads(Path(row["map_json"]).read_text(encoding="utf-8"))
        pts = np.asarray(data["map"], float)
        traj = np.asarray([[p["x"], p["y"]] for p in data.get("poses", [])], float)
        wall = align_pin_right(pts, traj if len(traj) else None, z=Z)
        ax.set_aspect("equal")
        ax.add_patch(Polygon(gt, closed=True, fill=False, edgecolor="k", lw=1.8))
        ax.add_patch(
            Rectangle((0, 0), CUT_W, CUT_H, facecolor="#eee", edgecolor="gray", ls="--", alpha=0.7)
        )
        for x, y, w, d in visible_obstacle_rects_at_z(Z):
            ax.add_patch(
                Rectangle((x, y), w, d, facecolor="#666", edgecolor="k", alpha=0.35, lw=0.4)
            )
        P = wall["aligned"]
        inn = P[_indoor_mask(P, Z)]
        show = inn if len(inn) >= 80 else P
        step = max(1, len(show) // 8000)
        ax.scatter(show[::step, 0], show[::step, 1], s=1.0, c="tab:blue", alpha=0.28)
        if wall.get("traj") is not None and len(wall["traj"]):
            ax.plot(wall["traj"][:, 0], wall["traj"][:, 1], color="orange", lw=0.9, alpha=0.85)
        mw = row["metrics_walls_robust"]
        al = row["metrics_aligned"]
        ax.set_title(
            f'{row["variant"]}  ({"wheel odom" if row["use_encoder"] else "lidar only"})\n'
            f'{row["wall_time_s"]:.0f}s   Ox {mw["err_Ox_cm"]:+.0f}  Oy {mw["err_Oy_cm"]:+.0f}  '
            f'|err| {mw["abs_err_cm"]:.0f} cm\n'
            f'med {al["near_gt_median_cm"]:.1f} cm   loops {row["loop_closures"]}'
        )
        ax.set_xlim(-1.2, ROOM_X + 1.6)
        ax.set_ylim(-DOOR["recess"] - 0.7, ROOM_Y + WIN_OUT + 0.6)
        ax.grid(True, alpha=0.25)
        ax.set_xlabel("X room (m)")
    axes[0].set_ylabel("Y room (m)")
    fig.suptitle(
        "run_162516 — wheel odometry ON vs OFF  (pin RIGHT, up@1.60 m)",
        fontsize=12,
    )
    fig.tight_layout()
    out = OUT / "compare_odom_ab.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return out


def main():
    rows = []
    for name, use_enc in (("with_odom", True), ("no_odom", False)):
        print("=" * 64)
        print(f"variant={name}  use_encoder={use_enc}")
        rows.append(build_variant(name, use_encoder=use_enc))

    compare = save_compare(rows)
    summary = {
        "run_dir": str(RUN),
        "out_dir": str(OUT),
        "clarification": (
            "Real encoder.jsonl has enc_v + enc_dgr (wheel speed/direction). "
            "That IS used when use_encoder=True (ICP prior + wheel edges + deskew). "
            "Sim also logs odom_x/y/yaw absolute poses; only that triggers "
            "'scan-matching SLAM bypassed'. Real run never takes that shortcut."
        ),
        "compare_png": str(compare),
        "gt_room_m": [ROOM_X, ROOM_Y],
        "runs": rows,
    }
    path = OUT / "odom_ab_compare.json"
    path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print("\n" + "=" * 64)
    print(f"{'variant':12s}  {'time':>7}  {'RSS':>6}  {'loops':>5}  {'cells':>6}  "
          f"{'Ox':>7}  {'Oy':>7}  {'|err|':>7}  {'med':>6}")
    for r in rows:
        mw = r["metrics_walls_robust"]
        al = r["metrics_aligned"]
        print(
            f"{r['variant']:12s}  {r['wall_time_s']:7.1f}  {r['peak_rss_mb']:6.0f}  "
            f"{r['loop_closures']:5d}  {r['stable_cells']:6d}  "
            f"{mw['err_Ox_cm']:+7.1f}  {mw['err_Oy_cm']:+7.1f}  "
            f"{mw['abs_err_cm']:7.1f}  {al['near_gt_median_cm']:6.1f}"
        )
    print("wrote", path)
    print("compare", compare)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
