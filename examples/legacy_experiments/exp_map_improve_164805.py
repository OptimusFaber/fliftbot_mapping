#!/usr/bin/env python3
"""Offline A/B: aggressive loop-closure + multi-camera builds on run_164805."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
from scipy import ndimage

import flifltbot.mapping.config as cfg
import flifltbot.mapping.trajectory_graph as tg
from flifltbot.mapping.pipeline import build_global_map_from_run
from flifltbot.mapping.real_room_gt import (
    CUT_H,
    CUT_W,
    DOOR,
    ROOM_X,
    ROOM_Y,
    UP_LIDAR_Z,
    WINDOWS,
    WIN_OUT,
    room_wall_vertices_up,
    up_invisible_obstacle_rects,
    up_visible_obstacle_rects,
    PILLAR_SW,
    desk_k1_polygon_m,
)

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "maps/20260807/real/run_164805"
OUT_DIR = RUN / "improve_exp"
CELL = 0.05
EDGE_PCT = 5
WIN_MARGIN = 0.15


def apply_loop_preset(name: str):
    if name == "default":
        vals = dict(
            LOOP_RADIUS=1.3,
            LOOP_FRAME_GAP=120,
            LOOP_MAX_ERROR=0.14,
            LOOP_MIN_OVERLAP=0.50,
            LOOP_MAX_CORR_XY=0.55,
            LOOP_MAX_CORR_TH=math.radians(15.0),
        )
    elif name == "aggro":
        vals = dict(
            LOOP_RADIUS=cfg.LOOP_RADIUS_AGGRO,
            LOOP_FRAME_GAP=cfg.LOOP_FRAME_GAP_AGGRO,
            LOOP_MAX_ERROR=cfg.LOOP_MAX_ERROR_AGGRO,
            LOOP_MIN_OVERLAP=cfg.LOOP_MIN_OVERLAP_AGGRO,
            LOOP_MAX_CORR_XY=cfg.LOOP_MAX_CORR_XY_AGGRO,
            LOOP_MAX_CORR_TH=cfg.LOOP_MAX_CORR_TH_AGGRO,
        )
    else:
        raise ValueError(name)
    for mod in (cfg, tg):
        for k, v in vals.items():
            setattr(mod, k, v)
    return vals


def rotate(P, ang):
    c, s = np.cos(ang), np.sin(ang)
    R = np.array([[c, -s], [s, c]])
    return (R @ P.T).T


def densify(poly, step=0.08):
    out = []
    for a, b in zip(poly[:-1], poly[1:]):
        a, b = np.asarray(a, float), np.asarray(b, float)
        n = max(int(np.linalg.norm(b - a) / step), 1)
        for i in range(n):
            out.append(a + (b - a) * (i / n))
    out.append(poly[-1])
    return np.asarray(out)


def align_and_score(pts: np.ndarray, traj: np.ndarray | None = None):
    """PCA/RANSAC straighten + robust wall extremes (niches excluded)."""
    rng = np.random.default_rng(0)
    pad = 0.25
    xmin, ymin = pts.min(0) - pad
    Wg = int(np.ceil((pts[:, 0].max() + pad - xmin) / CELL))
    Hg = int(np.ceil((pts[:, 1].max() + pad - ymin) / CELL))
    grid = np.zeros((Hg, Wg), np.uint8)
    ci = ((pts[:, 0] - xmin) / CELL).astype(int)
    ri = ((pts[:, 1] - ymin) / CELL).astype(int)
    ok = (ci >= 0) & (ci < Wg) & (ri >= 0) & (ri < Hg)
    grid[ri[ok], ci[ok]] = 1
    grid = ndimage.binary_closing(grid, iterations=2)
    grid = ndimage.binary_fill_holes(grid)
    lab, n = ndimage.label(grid)
    sizes = ndimage.sum(grid, lab, range(1, n + 1))
    blob = lab == (1 + int(np.argmax(sizes)))
    er = ndimage.binary_erosion(blob, iterations=max(1, int(0.20 / CELL)))
    use = er if er.sum() > 800 else blob
    bnd = use & ~ndimage.binary_erosion(use)
    br, bc = np.where(bnd)
    contour = np.column_stack([xmin + (bc + 0.5) * CELL, ymin + (br + 0.5) * CELL])

    best = None
    for _ in range(1500):
        i, j = rng.choice(len(contour), 2, replace=False)
        d = contour[j] - contour[i]
        if np.linalg.norm(d) < 0.5:
            continue
        tang = d / np.linalg.norm(d)
        normal = np.array([-tang[1], tang[0]])
        c0 = -normal @ contour[i]
        nin = int((np.abs(contour @ normal + c0) < 0.12).sum())
        if best is None or nin > best[0]:
            best = (nin, float(np.arctan2(tang[1], tang[0])))
    _, ang_t = best

    pts1 = rotate(pts, -ang_t)
    ctr1 = rotate(contour, -ang_t)
    tr1 = None if traj is None else rotate(traj, -ang_t)
    if (pts1.max(0) - pts1.min(0))[0] < (pts1.max(0) - pts1.min(0))[1]:
        pts1 = rotate(pts1, np.pi / 2)
        ctr1 = rotate(ctr1, np.pi / 2)
        if tr1 is not None:
            tr1 = rotate(tr1, np.pi / 2)

    gt_d = densify(np.asarray(room_wall_vertices_up()))

    def rough_x(cloud, side):
        xs = cloud[:, 0]
        span = xs.max() - xs.min()
        band = min(1.5, 0.18 * span)
        if side == "west":
            edge = np.percentile(xs, 0.5)
            return float(np.percentile(cloud[xs <= edge + band, 0], 25))
        edge = np.percentile(xs, 99.5)
        return float(np.percentile(cloud[xs >= edge - band, 0], 75))

    def rough_y(cloud, side):
        ys = cloud[:, 1]
        span = ys.max() - ys.min()
        band = min(1.5, 0.22 * span)
        if side == "south":
            edge = np.percentile(ys, 0.5)
            return float(np.percentile(cloud[ys <= edge + band, 1], 25))
        edge = np.percentile(ys, 99.5)
        return float(np.percentile(cloud[ys >= edge - band, 1], 75))

    pick = None
    for fx in (1, -1):
        for fy in (1, -1):
            for da in np.linspace(-4, 4, 13):
                c, s = np.cos(np.radians(da)), np.sin(np.radians(da))
                Rd = np.array([[c, -s], [s, c]])
                P = (Rd @ (pts1 * [fx, fy]).T).T
                C = (Rd @ (ctr1 * [fx, fy]).T).T
                T = None if tr1 is None else (Rd @ (tr1 * [fx, fy]).T).T
                Wm, Em = rough_x(P, "west"), rough_x(P, "east")
                Sm, Nm = rough_y(P, "south"), rough_y(P, "north")
                shift = np.array([Wm, Sm])
                P, C = P - shift, C - shift
                if T is not None:
                    T = T - shift
                Ox, Oy = Em - Wm, Nm - Sm
                span_pen = abs(Ox - ROOM_X) + abs(Oy - ROOM_Y)
                west = P[(P[:, 0] > -0.3) & (P[:, 0] < 1.0) & (P[:, 1] > 0.5) & (P[:, 1] < Oy - 0.5)]
                tilt = abs(np.polyfit(west[:, 1], west[:, 0], 1)[0]) * Oy if len(west) > 40 else 0.0
                cut = ((P[:, 0] > 0) & (P[:, 0] < CUT_W) & (P[:, 1] > 0) & (P[:, 1] < CUT_H)).mean()
                idx = np.linspace(0, len(C) - 1, 250).astype(int)
                ch = np.mean([np.min(np.sum((gt_d - C[i]) ** 2, axis=1)) for i in idx])
                score = 3.0 * span_pen + 1.5 * tilt + 2.0 * cut + 0.3 * np.sqrt(ch)
                if pick is None or score < pick[0]:
                    pick = (score, P, C, T, tilt)
    _, map_f, ctr_f, traj_f, tilt = pick

    def in_windows(x):
        m = np.zeros(len(x), bool)
        for w in WINDOWS:
            m |= (x >= w["x"] - WIN_MARGIN) & (x <= w["x"] + w["w"] + WIN_MARGIN)
        return m

    def in_door(x):
        return (x >= DOOR["x"] - WIN_MARGIN) & (x <= DOOR["x"] + DOOR["w"] + WIN_MARGIN)

    def mad_keep(vals, k=3.5):
        vals = np.asarray(vals, float)
        if len(vals) < 8:
            return np.ones(len(vals), bool)
        med = np.median(vals)
        mad = np.median(np.abs(vals - med)) + 1e-9
        return np.abs(vals - med) <= k * 1.4826 * mad

    def wall_x(cloud, side, band_m=1.6):
        xs, ys = cloud[:, 0], cloud[:, 1]
        y_ok = (ys > np.percentile(ys, 8)) & (ys < np.percentile(ys, 92))
        if side == "west":
            edge = np.percentile(xs[y_ok], 1)
            lobe = cloud[y_ok & (xs <= edge + band_m)]
            keep = mad_keep(lobe[:, 0])
            face = float(np.percentile(lobe[keep, 0], EDGE_PCT))
        else:
            edge = np.percentile(xs[y_ok], 99)
            lobe = cloud[y_ok & (xs >= edge - band_m)]
            keep = mad_keep(lobe[:, 0])
            face = float(np.percentile(lobe[keep, 0], 100 - EDGE_PCT))
        return face

    def wall_y(cloud, side, band_m=1.6):
        xs, ys = cloud[:, 0], cloud[:, 1]
        x_ok = (xs > np.percentile(xs, 5)) & (xs < np.percentile(xs, 95))
        if side == "north":
            x_ok = x_ok & ~in_windows(xs)
        else:
            x_ok = x_ok & ~in_door(xs)
        if side == "south":
            edge = np.percentile(ys[x_ok], 1) if x_ok.any() else np.percentile(ys, 1)
            lobe = cloud[x_ok & (ys <= edge + band_m)]
            if len(lobe) < 20:
                lobe = cloud[(xs > np.percentile(xs, 5)) & (xs < np.percentile(xs, 95)) & (ys <= edge + band_m)]
            keep = mad_keep(lobe[:, 1])
            face = float(np.percentile(lobe[keep, 1], EDGE_PCT))
        else:
            edge = np.percentile(ys[x_ok], 99) if x_ok.any() else np.percentile(ys, 99)
            lobe = cloud[x_ok & (ys >= edge - band_m)]
            if len(lobe) < 20:
                lobe = cloud[(xs > np.percentile(xs, 5)) & (xs < np.percentile(xs, 95)) & (ys >= edge - band_m)]
            keep = mad_keep(lobe[:, 1])
            face = float(np.percentile(lobe[keep, 1], 100 - EDGE_PCT))
        return face

    W, E = wall_x(map_f, "west"), wall_x(map_f, "east")
    S, N = wall_y(map_f, "south"), wall_y(map_f, "north")
    shift = np.array([W, S])
    map_a = map_f - shift
    ctr_a = ctr_f - shift
    traj_a = None if traj_f is None else traj_f - shift
    Ox, Oy = (E - W), (N - S)
    return {
        "Ox": Ox,
        "Oy": Oy,
        "err_Ox_cm": (Ox - ROOM_X) * 100,
        "err_Oy_cm": (Oy - ROOM_Y) * 100,
        "abs_err_cm": abs(Ox - ROOM_X) * 100 + abs(Oy - ROOM_Y) * 100,
        "tilt_m": float(tilt),
        "n_cells": int(len(pts)),
        "aligned": map_a,
        "contour": ctr_a,
        "traj": traj_a,
        "faces": {"W": 0.0, "E": Ox, "S": 0.0, "N": Oy},
    }


def plot_compare(rows, path: Path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon, Rectangle

    n = len(rows)
    fig, axes = plt.subplots(1, n, figsize=(5.2 * n, 5.8), dpi=120)
    if n == 1:
        axes = [axes]
    gt = np.asarray(room_wall_vertices_up())
    for ax, row in zip(axes, rows):
        m = row["metrics"]
        ax.set_aspect("equal")
        ax.set_title(
            f'{row["name"]}\n'
            f'Ox {m["Ox"]:.2f} ({m["err_Ox_cm"]:+.0f}cm)  '
            f'Oy {m["Oy"]:.2f} ({m["err_Oy_cm"]:+.0f}cm)\n'
            f'loops={row.get("loop_closures", "?")}  cells={m["n_cells"]}',
            fontsize=9,
        )
        ax.add_patch(Polygon(gt, closed=True, fill=False, edgecolor="black", lw=1.8))
        ax.add_patch(Rectangle((0, 0), CUT_W, CUT_H, facecolor="#eee", edgecolor="gray", ls="--", alpha=0.6))
        for x, y, w, d in up_visible_obstacle_rects():
            ax.add_patch(Rectangle((x, y), w, d, facecolor="#777", edgecolor="k", alpha=0.3, lw=0.4))
        for w in WINDOWS:
            ax.add_patch(Rectangle((w["x"], ROOM_Y), w["w"], WIN_OUT, facecolor="tab:green", alpha=0.15, edgecolor="tab:green", lw=0.8))
        ax.add_patch(
            Rectangle(
                (DOOR["x"], -DOOR["recess"]),
                DOOR["w"],
                DOOR["recess"],
                facecolor="tab:purple",
                alpha=0.15,
                edgecolor="tab:purple",
                lw=0.8,
            )
        )
        pts = m["aligned"]
        ax.scatter(pts[::2, 0], pts[::2, 1], s=1, c="tab:blue", alpha=0.25)
        f = m["faces"]
        rect = np.array([[f["W"], f["S"]], [f["E"], f["S"]], [f["E"], f["N"]], [f["W"], f["N"]], [f["W"], f["S"]]])
        ax.plot(rect[:, 0], rect[:, 1], "c-", lw=2.0)
        if m["traj"] is not None and len(m["traj"]):
            ax.plot(m["traj"][:, 0], m["traj"][:, 1], color="orange", lw=0.9, alpha=0.85)
        ax.set_xlim(-1.0, ROOM_X + 1.2)
        ax.set_ylim(-DOOR["recess"] - 0.8, ROOM_Y + WIN_OUT + 0.6)
        ax.grid(True, alpha=0.25)
    fig.suptitle(f"run_164805 improve experiment · up@{UP_LIDAR_Z}m · robust extremes (−niches)", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    experiments = [
        dict(name="baseline_up", loop="default", cameras=["up"], tag="exp_baseline_up"),
        dict(name="aggro_loop_up", loop="aggro", cameras=["up"], tag="exp_aggro_loop_up"),
        dict(name="baseline_up_left", loop="default", cameras=["up", "left"], tag="exp_up_left"),
        dict(name="aggro_up_left", loop="aggro", cameras=["up", "left"], tag="exp_aggro_up_left"),
        dict(name="baseline_up_right", loop="default", cameras=["up", "right"], tag="exp_up_right"),
    ]

    results = []
    for exp in experiments:
        print("\n" + "=" * 60)
        print("EXPERIMENT", exp["name"], exp)
        apply_loop_preset(exp["loop"])
        build = build_global_map_from_run(
            RUN,
            method="improved",
            cameras=exp["cameras"],
            tag=exp["tag"],
            source="real",
        )
        pts = np.asarray(build["points"], float)
        traj = np.asarray([[p["x"], p["y"]] for p in build["poses"]], float)
        metrics = align_and_score(pts, traj)
        row = {
            "name": exp["name"],
            "cameras": exp["cameras"],
            "loop": exp["loop"],
            "tag": exp["tag"],
            "map_json": str(build["out_json"]),
            "map_png": str(build["out_png"]),
            "loop_closures": int(build["stats"].get("loop_closures", 0)),
            "keyframes": int(build["stats"].get("keyframes", 0) or build["stats"].get("pose_graph_keyframes", 0) or 0),
            "metrics": {
                k: (v if k not in ("aligned", "contour", "traj", "faces") else v)
                for k, v in metrics.items()
            },
            "summary": {
                "Ox": metrics["Ox"],
                "Oy": metrics["Oy"],
                "err_Ox_cm": metrics["err_Ox_cm"],
                "err_Oy_cm": metrics["err_Oy_cm"],
                "abs_err_cm": metrics["abs_err_cm"],
                "tilt_m": metrics["tilt_m"],
                "n_cells": metrics["n_cells"],
                "loop_closures": int(build["stats"].get("loop_closures", 0)),
            },
        }
        # keep arrays only for plotting
        row["_plot"] = metrics
        results.append(row)
        print(
            f'  loops={row["loop_closures"]}  '
            f'Ox err={metrics["err_Ox_cm"]:+.1f}cm  Oy err={metrics["err_Oy_cm"]:+.1f}cm  '
            f'|err|={metrics["abs_err_cm"]:.1f}cm'
        )

    # restore defaults
    apply_loop_preset("default")

    plot_rows = []
    for r in results:
        plot_rows.append(
            {
                "name": r["name"],
                "loop_closures": r["loop_closures"],
                "metrics": r["_plot"],
            }
        )
    overlay = OUT_DIR / "compare_improve_variants.png"
    plot_compare(plot_rows, overlay)

    table = []
    for r in results:
        s = r["summary"]
        table.append(
            {
                "name": r["name"],
                "cameras": r["cameras"],
                "loop": r["loop"],
                "loop_closures": r["loop_closures"],
                "err_Ox_cm": round(s["err_Ox_cm"], 1),
                "err_Oy_cm": round(s["err_Oy_cm"], 1),
                "abs_err_cm": round(s["abs_err_cm"], 1),
                "n_cells": s["n_cells"],
                "map_png": r["map_png"],
            }
        )
    best = min(table, key=lambda t: t["abs_err_cm"])
    out_json = {
        "run": str(RUN),
        "metric": "robust wall extremes; niches excluded; abs_err = |dOx|+|dOy|",
        "variants": table,
        "best": best["name"],
        "overlay": str(overlay),
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(out_json, indent=2) + "\n")
    print("\n" + "=" * 60)
    print("SUMMARY")
    for t in table:
        mark = " <<<" if t["name"] == best["name"] else ""
        print(
            f'{t["name"]:22s} loops={t["loop_closures"]:3d}  '
            f'Ox {t["err_Ox_cm"]:+6.1f}  Oy {t["err_Oy_cm"]:+6.1f}  '
            f'|err| {t["abs_err_cm"]:5.1f}{mark}'
        )
    print("best:", best["name"])
    print("overlay:", overlay)
    print("summary:", OUT_DIR / "summary.json")


if __name__ == "__main__":
    main()
