#!/usr/bin/env python3
"""Measure the four main wall lines in a CloudCompare result.ply.

The input PLY used by the DePHAN workflow contains a large map z-cluster and a
small marker z-cluster.  Marker identities are recovered from marker_points.json
using distance fingerprints followed by a 2-D rigid Kabsch fit; PLY row order is
therefore not trusted.  Wall lines are found with a robust normal-coordinate
Hough scan after estimating the common wall angle.  An ``outer`` line is only
reported when it has independent, sufficiently long support.  In particular,
the width of a noisy occupied band is not silently called a physical wall
thickness.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import linear_sum_assignment, minimize_scalar
from scipy.spatial import cKDTree
from scipy.stats import theilslopes


def read_ascii_ply(path: Path) -> np.ndarray:
    lines = path.read_text().splitlines()
    try:
        end = lines.index("end_header")
    except ValueError as exc:
        raise ValueError(f"Only ASCII PLY is supported: {path}") from exc
    rows = []
    for line in lines[end + 1 :]:
        f = line.split()
        if len(f) >= 3:
            rows.append([float(f[0]), float(f[1]), float(f[2])])
    a = np.asarray(rows, dtype=float)
    if len(a) < 10:
        raise ValueError(f"Too few vertices in {path}")
    return a


def marker_cloud(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (map_points, marker_points), using robust z clusters."""
    z = points[:, 2]
    # CloudCompare exports used here have one big map plane and a small marker
    # plane.  Quantising avoids treating floating point noise as many planes.
    q = np.round(z, 3)
    values, counts = np.unique(q, return_counts=True)
    if len(values) < 2:
        raise ValueError("PLY has no separable map/marker z clusters")
    map_z = values[np.argmax(counts)]
    marker_candidates = [(int(c), float(v)) for v, c in zip(values, counts) if v != map_z]
    marker_candidates.sort()
    marker_z = marker_candidates[0][1]
    m = np.abs(q - marker_z) < 1e-6
    return points[~m], points[m]


def recover_marker_correspondence(
    ply_markers: np.ndarray, marker_json: dict[str, Any], marker_id: int
) -> tuple[np.ndarray, dict[str, Any]]:
    src = np.asarray([m["point_xyz_m"][:2] for m in marker_json["markers"]], dtype=float)
    dst = ply_markers[:, :2]
    if len(src) < len(dst):
        raise ValueError(f"PLY has more marker points than marker_points.json: JSON={len(src)} PLY={len(dst)}")
    if len(src) != len(dst):
        # CloudCompare may contain a manually selected subset.  Recover the
        # subset by two-point rigid RANSAC and one-to-one assignment; this also
        # makes absence of marker 98 explicit instead of silently substituting
        # another point.
        best = None
        for i, j in itertools.combinations(range(len(dst)), 2):
            vd = dst[j] - dst[i]
            ld = float(np.linalg.norm(vd))
            if ld < 0.20:
                continue
            for a, b in itertools.combinations(range(len(src)), 2):
                vs = src[b] - src[a]
                ls = float(np.linalg.norm(vs))
                if abs(ls - ld) > 0.01:
                    continue
                for di, dj in ((i, j), (j, i)):
                    ca, cb = src[[a, b]].mean(0), dst[[di, dj]].mean(0)
                    u, _, vt = np.linalg.svd((src[[a, b]] - ca).T @ (dst[[di, dj]] - cb))
                    rr = vt.T @ u.T
                    if np.linalg.det(rr) < 0:
                        vt[-1] *= -1
                        rr = vt.T @ u.T
                    tt = cb - ca @ rr.T
                    pred = src @ rr.T + tt
                    c = np.linalg.norm(dst[:, None] - pred[None, :], axis=2)
                    ri, ci = linear_sum_assignment(c)
                    ds = c[ri, ci]
                    inlier = ds < 0.01
                    nin = int(inlier.sum())
                    rm = float(np.sqrt(np.mean(ds[inlier] ** 2))) if nin else 99.0
                    score = (nin, -rm)
                    if best is None or score > best[0]:
                        best = (score, rr, tt, ri, ci, ds)
        if best is None or best[0][0] < len(dst):
            raise ValueError("Could not recover a rigid correspondence for the partial PLY marker subset")
        _, r, t, ri, ci, residual = best
        assignment = np.empty(len(dst), dtype=int)
        assignment[ri] = ci
        target = next((i for i, m in enumerate(marker_json["markers"]) if int(m["marker_id"]) == marker_id), None)
        if target is None:
            raise ValueError(f"marker_id={marker_id} is absent from marker_points.json")
        matched = np.flatnonzero(assignment == target)
        ply_idx = int(matched[0]) if len(matched) else None
        diag = {
            "json_to_ply_rotation": r.tolist(),
            "json_to_ply_translation_m": t.tolist(),
            "correspondence_rmse_m": float(np.sqrt(np.mean(residual**2))),
            "correspondence_max_residual_m": float(residual.max()),
            "matched_marker_count": int(len(dst)),
            "json_marker_count": int(len(src)),
            "partial_json_subset": True,
            "marker_id": marker_id,
            "marker_available_in_ply": ply_idx is not None,
            "ply_marker_index": ply_idx,
            "matched_marker_ids": [int(marker_json["markers"][x]["marker_id"]) for x in assignment],
        }
        if ply_idx is not None:
            diag["ply_marker_point_m"] = dst[ply_idx].tolist()
            diag["json_marker_point_m"] = src[target].tolist()
            return dst[ply_idx], diag
        return None, diag
    da = np.linalg.norm(src[:, None] - src[None, :], axis=2)
    db = np.linalg.norm(dst[:, None] - dst[None, :], axis=2)
    # Distance fingerprints are invariant to translation/rotation and are
    # sufficient for this non-symmetric marker arrangement.
    cost = np.empty((len(dst), len(src)))
    for i in range(len(dst)):
        sdb = np.sort(db[i])[1:]
        for j in range(len(src)):
            cost[i, j] = np.mean(np.abs(sdb - np.sort(da[j])[1:]))
    ri, ci = linear_sum_assignment(cost)
    assignment = np.empty(len(dst), dtype=int)
    assignment[ri] = ci
    # Refine with the rigid fit and nearest-neighbour reassignment. This makes
    # the procedure independent of the order in which CloudCompare wrote rows.
    for _ in range(3):
        A, B = src[assignment], dst
        ca, cb = A.mean(0), B.mean(0)
        u, _, vt = np.linalg.svd((A - ca).T @ (B - cb))
        r = vt.T @ u.T
        if np.linalg.det(r) < 0:
            vt[-1] *= -1
            r = vt.T @ u.T
        t = cb - ca @ r.T
        predicted = src @ r.T + t
        # one-to-one nearest matching, with fingerprint assignment as tie-break
        c = np.linalg.norm(dst[:, None] - predicted[None, :], axis=2)
        rr, cc = linear_sum_assignment(c)
        assignment[rr] = cc
    A, B = src[assignment], dst
    ca, cb = A.mean(0), B.mean(0)
    u, _, vt = np.linalg.svd((A - ca).T @ (B - cb))
    r = vt.T @ u.T
    if np.linalg.det(r) < 0:
        vt[-1] *= -1
        r = vt.T @ u.T
    t = cb - ca @ r.T
    residual = np.linalg.norm(src[assignment] @ r.T + t - dst, axis=1)
    target = next((i for i, m in enumerate(marker_json["markers"]) if int(m["marker_id"]) == marker_id), None)
    if target is None:
        raise ValueError(f"marker_id={marker_id} is absent from marker_points.json")
    ply_idx = int(np.flatnonzero(assignment == target)[0])
    diag = {
        "json_to_ply_rotation": r.tolist(),
        "json_to_ply_translation_m": t.tolist(),
        "correspondence_rmse_m": float(np.sqrt(np.mean(residual**2))),
        "correspondence_max_residual_m": float(residual.max()),
        "ply_marker_index": ply_idx,
        "ply_marker_point_m": dst[ply_idx].tolist(),
        "json_marker_point_m": src[target].tolist(),
        "marker_id": marker_id,
        "matched_marker_count": int(len(dst)),
        "json_marker_count": int(len(src)),
        "partial_json_subset": False,
        "marker_available_in_ply": True,
    }
    return dst[ply_idx], diag


def rotate_xy(a: np.ndarray, theta: float) -> np.ndarray:
    c, s = math.cos(theta), math.sin(theta)
    return a @ np.array([[c, s], [-s, c]], dtype=float).T


def wall_score(map_xy: np.ndarray, theta: float, central_q: float = 0.12, side_q: float = 0.14, seed_tol: float = 0.025, inlier_tol: float = 0.020) -> float:
    b = rotate_xy(map_xy, theta)
    total = 0.0
    for axis in (0, 1):
        normal, along = b[:, axis], b[:, 1 - axis]
        lo, hi = np.quantile(along, [central_q, 1.0 - central_q])
        central = (along >= lo) & (along <= hi)
        q0, q1 = np.quantile(normal[central], [side_q, 1.0 - side_q])
        for q, low in ((q0, True), (q1, False)):
            sel = central & ((normal <= q) if low else (normal >= q))
            v = normal[sel]
            if len(v) < 8:
                return 1e3
            edges = np.arange(v.min() - seed_tol, v.max() + seed_tol * 1.1, seed_tol)
            hist, _ = np.histogram(v, edges)
            smooth = gaussian_filter1d(hist.astype(float), 1.4)
            mode = (edges[int(np.argmax(smooth))] + edges[int(np.argmax(smooth)) + 1]) / 2
            total += float(np.median(np.abs(v - mode)))
    return total


def estimate_angle(map_xy: np.ndarray, central_q: float = 0.12, side_q: float = 0.14, seed_tol: float = 0.025, inlier_tol: float = 0.020) -> float:
    # Estimate Manhattan direction from local-neighbour vectors. This is not
    # tied to a known room yaw or a hard-coded +/-10 degree search window.
    tree = cKDTree(map_xy)
    k = min(8, len(map_xy))
    idx = tree.query(map_xy, k=k)[1][:, 1:]
    vec = map_xy[idx] - map_xy[:, None, :]
    ang = (np.arctan2(vec[..., 1], vec[..., 0]) % np.pi).ravel()
    # Ignore unusually long neighbour jumps (interior clutter/outliers).
    length = np.linalg.norm(vec, axis=2).ravel()
    keep = length <= np.quantile(length, 0.75)
    ang = ang[keep]
    hist, edges = np.histogram(ang, bins=180, range=(0.0, np.pi))
    # Keep several orientation hypotheses: diagonal scan sampling can be more
    # frequent than wall-parallel neighbours. Refine each with robust wall
    # residuals instead of trusting one histogram bin.
    order = np.argsort(hist)[::-1]
    peaks = []
    for idx in order:
        centre = (edges[idx] + edges[idx + 1]) / 2.0
        if all(abs(np.angle(np.exp(2j * (centre - q)))) > np.deg2rad(12.0) for q in peaks):
            peaks.append(centre)
        if len(peaks) == 6:
            break
    refined = []
    for centre in peaks:
        local = np.abs(np.angle(np.exp(2j * (ang - centre)))) < np.deg2rad(8.0)
        if local.sum() >= 10:
            centre = 0.5 * math.atan2(float(np.sin(2 * ang[local]).sum()), float(np.cos(2 * ang[local]).sum()))
        for candidate in (centre, centre - np.pi / 2.0):
            candidate = ((candidate + np.pi / 2) % np.pi) - np.pi / 2
            lo, hi = candidate - np.deg2rad(8.0), candidate + np.deg2rad(8.0)
            res = minimize_scalar(lambda x: wall_score(map_xy, x, central_q, side_q, seed_tol, inlier_tol), bounds=(lo, hi), method="bounded")
            refined.append((float(res.fun), float(res.x)))
    if not refined:
        return 0.0
    # Resolve the 90-degree ambiguity using the longer dimension on X.
    best_score = min(x[0] for x in refined)
    tied = [x[1] for x in refined if x[0] <= best_score + 1e-4]
    return float(max(tied, key=lambda x: np.ptp(rotate_xy(map_xy, x)[:, 0]) / max(np.ptp(rotate_xy(map_xy, x)[:, 1]), 1e-9)))


def fit_candidate(normal: np.ndarray, along: np.ndarray, offset: float, seed_tol: float = 0.025, inlier_tol: float = 0.020) -> dict[str, Any] | None:
    seed = np.abs(normal - offset) <= seed_tol
    if seed.sum() < 8:
        return None
    slope, intercept, _, _ = theilslopes(normal[seed], along[seed])
    pred = slope * along[seed] + intercept
    residual = normal[seed] - pred
    inlier = np.abs(residual) <= inlier_tol
    if inlier.sum() < 8:
        return None
    t = along[seed]
    bins = np.linspace(along.min(), along.max(), 25)
    coverage = sum(np.any(inlier[(t >= bins[i]) & (t < bins[i + 1])]) for i in range(24)) / 24.0
    n = normal[seed][inlier]
    tt = t[inlier]
    if len(n) >= 8:
        slope, intercept, _, _ = theilslopes(n, tt)
    center = float(np.median(tt))
    offset_center = float(slope * center + intercept)
    absres = np.abs(normal[seed] - (slope * t + intercept))
    return {
        "slope": float(slope),
        "intercept": float(intercept),
        "along_center": center,
        "offset_center": offset_center,
        "support": int(inlier.sum()),
        "coverage": float(coverage),
        "residual_median_m": float(np.median(absres[inlier])),
        "residual_p95_m": float(np.quantile(absres[inlier], 0.95)),
        "normal_min_m": float(normal[seed][inlier].min()),
        "normal_max_m": float(normal[seed][inlier].max()),
    }


def fit_wall(b: np.ndarray, axis: int, side: str, central_q: float = 0.12, side_q: float = 0.35, seed_tol: float = 0.025, inlier_tol: float = 0.020, min_support: int = 20, min_coverage: float = 0.40, outer_min_fraction: float = 0.03, outer_min_coverage: float = 0.60) -> dict[str, Any]:
    normal, along = b[:, axis], b[:, 1 - axis]
    central = (along >= np.quantile(along, central_q)) & (along <= np.quantile(along, 1.0 - central_q))
    side_cut = np.quantile(normal[central], side_q if side == "low" else 1.0 - side_q)
    side_mask = central & ((normal <= side_cut) if side == "low" else (normal >= side_cut))
    vv = normal[side_mask]
    # Hough scan gives line candidates; no percentile is used as a fitted line.
    candidates = []
    scan_step = max(seed_tol * 0.1, 0.001)
    for off in np.arange(vv.min(), vv.max() + scan_step, scan_step):
        c = fit_candidate(normal[side_mask], along[side_mask], float(off), seed_tol, inlier_tol)
        if c and c["support"] >= min_support and c["coverage"] >= min_coverage:
            candidates.append(c)
    # Deduplicate the overlapping scan offsets around one fitted line.
    candidates.sort(key=lambda x: x["support"] * x["coverage"], reverse=True)
    uniq = []
    for c in candidates:
        if all(abs(c["offset_center"] - q["offset_center"]) > 0.045 for q in uniq):
            uniq.append(c)
    uniq.sort(key=lambda x: x["offset_center"])
    if not uniq:
        return {"axis": axis, "side": side, "inner": None, "outer": None, "candidates": []}
    # The room-facing boundary is geometrically the edge toward the room, not
    # necessarily the most populated Hough peak (filled/noisy bands can make
    # the opposite edge denser).  Select the supported extreme in that
    # direction, then independently select the opposite extreme.
    inner = max(uniq, key=lambda x: x["offset_center"]) if side == "low" else min(uniq, key=lambda x: x["offset_center"])
    direction = -1 if side == "low" else 1  # direction from inner to outer
    outer = None
    others = [x for x in uniq if direction * (x["offset_center"] - inner["offset_center"]) > 0.045]
    if others:
        outer = min(others, key=lambda x: x["offset_center"]) if side == "low" else max(others, key=lambda x: x["offset_center"])
    # A second line with short or patchy support is not a physical boundary.
    if outer and (outer["support"] < outer_min_fraction * len(normal) or outer["coverage"] < outer_min_coverage):
        outer = None
    def decorate(line: dict[str, Any] | None, role: str) -> dict[str, Any] | None:
        if line is None:
            return None
        strong = line["support"] >= 70 and line["coverage"] >= 0.75 and line["residual_p95_m"] <= 0.025
        line["available"] = True
        line["role"] = role
        line["confidence"] = "high" if strong else "medium"
        line["physical_interpretation"] = "room-facing fitted edge" if role == "inner" else "second independently supported parallel edge; thickness is observed, not assumed"
        return line
    inner = decorate(inner, "inner")
    outer = decorate(outer, "outer")
    out = {
        "axis": axis,
        "side": side,
        "inner": inner,
        "outer": outer,
        "median": (
            {k: (inner[k] + outer[k]) / 2 for k in ("slope", "intercept", "offset_center")}
            if inner and outer
            else None
        ),
        "candidates": uniq,
    }
    if outer is None:
        out["outer_unavailable_reason"] = f"No second parallel edge met support >= {outer_min_fraction:.1%} of map points and longitudinal coverage >= {outer_min_coverage:.0%}; wall thickness is not identifiable from this PLY."
    return out


def line_point_distance(p: np.ndarray, line: dict[str, Any], axis: int) -> float:
    # line normal coordinate n = slope * along + intercept
    along_axis = 1 - axis
    return float(abs(p[axis] - (line["slope"] * p[along_axis] + line["intercept"])) / math.sqrt(1 + line["slope"] ** 2))


def line_segment(line: dict[str, Any], axis: int, along0: float, along1: float) -> np.ndarray:
    t = np.array([along0, along1])
    n = line["slope"] * t + line["intercept"]
    out = np.zeros((2, 2))
    out[:, axis] = n
    out[:, 1 - axis] = t
    return out


def gt_size(gt_path: Path) -> tuple[float, float]:
    d = json.loads(gt_path.read_text())
    return float(d["room_size_m"][0]), float(d["room_size_m"][1])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ply", type=Path, required=True)
    ap.add_argument("--marker-points", type=Path, required=True)
    ap.add_argument("--gt-floorplan", type=Path, required=True)
    ap.add_argument("--measurement-anchor-json", type=Path, default=None, help="optional independent marker-98-to-wall measurements")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--marker-id", type=int, default=98)
    ap.add_argument("--central-quantile", type=float, default=0.12, help="fraction removed at each end of wall segments")
    ap.add_argument("--wall-side-quantile", type=float, default=0.35, help="quantile separating each extreme wall corridor")
    ap.add_argument("--seed-tolerance", type=float, default=None, help="line seed tolerance in metres; default derives from point spacing")
    ap.add_argument("--inlier-tolerance", type=float, default=None, help="line inlier tolerance in metres; default derives from point spacing")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    all_points = read_ascii_ply(args.ply)
    map_points, ply_markers = marker_cloud(all_points)
    marker_json = json.loads(args.marker_points.read_text())
    marker_ply, corr = recover_marker_correspondence(ply_markers, marker_json, args.marker_id)
    nn = cKDTree(map_points[:, :2]).query(map_points[:, :2], k=2)[0][:, 1]
    point_spacing = float(np.median(nn))
    seed_tol = float(args.seed_tolerance if args.seed_tolerance is not None else max(0.01, 0.5 * point_spacing))
    inlier_tol = float(args.inlier_tolerance if args.inlier_tolerance is not None else max(0.008, 0.4 * point_spacing))
    theta = estimate_angle(map_points[:, :2], args.central_quantile, 0.14, seed_tol, inlier_tol)
    xy = rotate_xy(map_points[:, :2], theta)
    p = rotate_xy(marker_ply[None, :2], theta)[0] if marker_ply is not None else None
    walls = {
        "left": fit_wall(xy, 0, "low", args.central_quantile, args.wall_side_quantile, seed_tol, inlier_tol),
        "right": fit_wall(xy, 0, "high", args.central_quantile, args.wall_side_quantile, seed_tol, inlier_tol),
        "bottom": fit_wall(xy, 1, "low", args.central_quantile, args.wall_side_quantile, seed_tol, inlier_tol),
        "top": fit_wall(xy, 1, "high", args.central_quantile, args.wall_side_quantile, seed_tol, inlier_tol),
    }
    # Native rigid room coordinates: inner contour's lower-left is (0,0).
    inners = [w["inner"] for w in walls.values() if w["inner"]]
    xinner = [w["inner"]["offset_center"] for w in (walls["left"], walls["right"])]
    yinner = [w["inner"]["offset_center"] for w in (walls["bottom"], walls["top"])]
    # Low-normal sides are the left/bottom inner walls.  Use them as (0, 0),
    # so the native rigid diagnostic has the same handedness as the GT room.
    origin = np.array([min(xinner), min(yinner)], dtype=float)
    native_size = [float(abs(xinner[1] - xinner[0])), float(abs(yinner[1] - yinner[0]))]
    gt_w, gt_h = gt_size(args.gt_floorplan)
    scale = [gt_w / native_size[0], gt_h / native_size[1]]

    # Distances in native aligned coordinates; median is strictly arithmetic
    # mean of two fitted line equations. Missing edges remain unavailable.
    distance_rows = []
    for name, w in walls.items():
        row = {"wall": name}
        for label in ("inner", "median", "outer"):
            line = w.get(label)
            native_d = line_point_distance(p, line, int(w["axis"])) if (line and p is not None) else None
            row[f"distance_{label}_m"] = native_d
            row[f"distance_{label}_gt_normalized_m"] = native_d * scale[int(w["axis"])] if native_d is not None else None
        distance_rows.append(row)
    anchor_comparison = None
    if args.measurement_anchor_json is not None:
        anchor = json.loads(args.measurement_anchor_json.read_text())
        measured = anchor["inner_wall_distances_m"]
        direction_for_wall = {"left": "west", "right": "east", "bottom": "south", "top": "north"}
        anchor_rows = []
        for row in distance_rows:
            direction = direction_for_wall[row["wall"]]
            spec = measured[direction]
            lo, hi = (float(spec[0]), float(spec[1])) if isinstance(spec, list) else (float(spec), float(spec))
            estimate = row["distance_inner_m"]
            anchor_rows.append({"wall": row["wall"], "direction": direction, "measured_interval_m": [lo, hi], "native_estimate_m": estimate, "error_to_interval_m": ([estimate - hi, estimate - lo] if estimate is not None else None), "inside_interval": (bool(estimate is not None and lo <= estimate <= hi))})
        by = {r["direction"]: r for r in anchor_rows}
        def interval_value(value: Any) -> tuple[float, float]:
            return (float(value[0]), float(value[-1])) if isinstance(value, list) else (float(value), float(value))
        west_lo, west_hi = interval_value(measured["west"])
        east_lo, east_hi = interval_value(measured["east"])
        south_lo, south_hi = interval_value(measured["south"])
        north_lo, north_hi = interval_value(measured["north"])
        ew_est = by["west"]["native_estimate_m"] + by["east"]["native_estimate_m"] if by["west"]["native_estimate_m"] is not None and by["east"]["native_estimate_m"] is not None else None
        ns_est = by["south"]["native_estimate_m"] + by["north"]["native_estimate_m"] if by["south"]["native_estimate_m"] is not None and by["north"]["native_estimate_m"] is not None else None
        anchor_comparison = {"source": str(args.measurement_anchor_json), "walls": anchor_rows, "sums": {"ew_native_m": ew_est, "ew_measured_interval_m": [west_lo + east_lo, west_hi + east_hi], "ns_native_m": ns_est, "ns_measured_interval_m": [south_lo + north_lo, south_hi + north_hi]}}

    # Render native aligned view and GT-normalized view. Lines are labeled in a
    # side panel to keep all 12 entries legible even when edges are unavailable.
    for mode, (sx, sy) in {"native_rigid": (1.0, 1.0), "gt_normalized": (scale[0], scale[1])}.items():
        fig, ax = plt.subplots(figsize=(16, 10), dpi=180)
        q = xy.copy()
        q[:, 0] = sx * (q[:, 0] - origin[0]); q[:, 1] = sy * (q[:, 1] - origin[1])
        ax.scatter(q[:, 0], q[:, 1], s=2, c="#b8b8b8", alpha=0.45, label="map points")
        mp = np.array([sx * (p[0] - origin[0]), sy * (p[1] - origin[1])]) if p is not None else None
        if mp is not None:
            ax.scatter([mp[0]], [mp[1]], s=90, c="red", edgecolors="black", zorder=5)
            ax.annotate(f"98 ({mp[0]:.2f}, {mp[1]:.2f})", mp, xytext=(8, 8), textcoords="offset points", fontsize=11, weight="bold")
        colors = {"inner": "#16803c", "median": "#e58f00", "outer": "#c52222"}
        for name, w in walls.items():
            axis = int(w["axis"])
            along = xy[:, 1 - axis]
            a0, a1 = np.quantile(along, [.10, .90])
            for label_i, label in enumerate(("inner", "median", "outer")):
                line = w.get(label)
                if not line:
                    continue
                seg = line_segment(line, axis, a0, a1)
                seg[:, 0] = sx * (seg[:, 0] - origin[0]); seg[:, 1] = sy * (seg[:, 1] - origin[1])
                ax.plot(seg[:, 0], seg[:, 1], color=colors[label], lw=2.0, zorder=3)
                if mp is not None:
                    # Perpendicular arrow to the fitted line.  The arrow is
                    # drawn only for observed lines; unavailable edges are
                    # represented solely in the side-panel text.
                    axis_sign = 1 if (axis == 0 and name == "left") or (axis == 1 and name == "bottom") else -1
                    foot = mp.copy()
                    # In normalized display coordinates, project using the
                    # fitted line in the native frame then apply its scales.
                    pn = p[axis]
                    pa = p[1 - axis]
                    nline = line["slope"] * pa + line["intercept"]
                    foot_native = p.copy(); foot_native[axis] = nline
                    foot = np.array([sx * (foot_native[0] - origin[0]), sy * (foot_native[1] - origin[1])])
                    ax.annotate("", xy=foot, xytext=mp, arrowprops=dict(arrowstyle="->", color=colors[label], lw=1.2, alpha=.85))
                    d_native = line_point_distance(p, line, axis)
                    d_show = d_native * scale[axis] if mode == "gt_normalized" else d_native
                    mid = (mp + foot) / 2.0
                    # Move labels along the wall tangent so the three labels
                    # belonging to one wall remain separate and readable.
                    tangent_offset = (label_i - 1) * 1.50
                    label_pos = mid.copy()
                    label_pos[1 - axis] += tangent_offset
                    ax.annotate(f"{label}: {d_show:.2f} m", label_pos, fontsize=8, color=colors[label], ha="center", va="center", bbox=dict(boxstyle="round,pad=.15", fc="white", ec=colors[label], alpha=.82))
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.grid(alpha=.2)
        ax.set_title(f"result.ply wall geometry — {mode}; marker {args.marker_id}")
        dist_suffix = "_gt_normalized_m" if mode == "gt_normalized" else "_m"
        if p is None:
            lines = ["marker 98 is absent from this PLY", "distances: not computable"]
            lines += [f"{name}: inner={'fitted' if w['inner'] else 'unavailable'}, median={'fitted' if w.get('median') else 'unavailable'}, outer={'fitted' if w.get('outer') else 'unavailable'}" for name, w in walls.items()]
        else:
            lines = [f"{name}: inner={r[f'distance_inner{dist_suffix}']:.3f} m" if r[f"distance_inner{dist_suffix}"] is not None else f"{name}: inner=unavailable" for name, r in zip(walls, distance_rows)]
            for name, r in zip(walls, distance_rows):
                lines.append(f"{name}: median={r[f'distance_median{dist_suffix}']:.3f} m" if r[f"distance_median{dist_suffix}"] is not None else f"{name}: median=unavailable")
                lines.append(f"{name}: outer={r[f'distance_outer{dist_suffix}']:.3f} m" if r[f"distance_outer{dist_suffix}"] is not None else f"{name}: outer=unavailable")
        ax.text(1.02, .98, "Distances from marker 98\n" + "\n".join(lines) + "\n\nGreen inner, orange median, red outer\n(unavailable = not independently observed)", transform=ax.transAxes, va="top", fontsize=10, family="monospace", bbox=dict(facecolor="white", alpha=.9, edgecolor="#777"))
        fig.tight_layout()
        fig.savefig(args.out_dir / f"wall_geometry_{mode}.png", bbox_inches="tight")
        plt.close(fig)

    # CSV is intentionally flat and easy to inspect in CloudCompare/spreadsheets.
    with (args.out_dir / "wall_distances.csv").open("w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=["wall", "distance_inner_m", "distance_median_m", "distance_outer_m", "distance_inner_gt_normalized_m", "distance_median_gt_normalized_m", "distance_outer_gt_normalized_m"])
        wr.writeheader(); wr.writerows(distance_rows)
    result = {
        "schema": "fliftbot.result_ply_wall_geometry.v1",
        "inputs": {"ply": str(args.ply), "marker_points": str(args.marker_points), "gt_floorplan": str(args.gt_floorplan)},
        "point_counts": {"total": int(len(all_points)), "map": int(len(map_points)), "markers": int(len(ply_markers))},
        "marker_correspondence": corr,
        "alignment": {"rotation_to_wall_axes_rad": theta, "rotation_to_wall_axes_deg": math.degrees(theta), "origin_native_m": origin.tolist(), "native_inner_size_m": native_size, "gt_room_size_m": [gt_w, gt_h], "gt_normalized_scale_xy": scale, "scale_applied": {"native_rigid": False, "gt_normalized": True}},
        "method_parameters": {"point_spacing_m": point_spacing, "central_quantile": args.central_quantile, "wall_side_quantile": args.wall_side_quantile, "seed_tolerance_m": seed_tol, "inlier_tolerance_m": inlier_tol, "outer_min_support_fraction": 0.03, "outer_min_coverage": 0.60},
        "walls": walls,
        "marker_98_aligned_native_m": ((p - origin).tolist() if p is not None else None),
        "marker_98_aligned_wall_frame_m": (p.tolist() if p is not None else None),
        "distances_from_marker_98_native_m": distance_rows,
        "measurement_anchor_comparison": anchor_comparison,
        "notes": ["A line is fitted from Hough/RANSAC-supported wall points, not a percentile boundary.", "Median is available only when both independent physical boundaries pass support/coverage thresholds.", "Unobserved outer boundaries are reported as null; no wall thickness is invented."] + (["Marker 98 is absent from the supplied PLY subset; all marker distances are null."] if p is None else []),
    }
    (args.out_dir / "wall_geometry_diagnostics.json").write_text(json.dumps(result, indent=2) + "\n")
    (args.out_dir / "README.md").write_text("""# result.ply wall geometry\n\n`wall_geometry_native_rigid.png` shows the cloud after a robust wall-angle rotation only. `wall_geometry_gt_normalized.png` additionally maps the fitted inner contour to `room_size_m` from the supplied GT using explicit X/Y scale factors recorded in `wall_geometry_diagnostics.json`; it is not a claim that the original cloud had that scale.\n\nThe four wall lines are found by a normal-coordinate Hough scan and robust Theil-Sen fits over the long central portions. Corners, windows/openings and interior points are excluded by the central-span and line-support tests. `outer` and `median` are null when a second independently supported physical boundary is not observable in `result.ply`. Each observed line has a perpendicular arrow from marker 98 with a boxed distance label.\n\nIf the PLY contains a manually selected marker subset, the JSON-to-PLY rigid correspondence is recovered by two-point RANSAC. If marker 98 is absent from that subset, the images still show the fitted wall geometry, but correctly show that marker distances are not computable.\n\nDistances are in `wall_distances.csv`.\n""")
    print(json.dumps({"out_dir": str(args.out_dir), "angle_deg": math.degrees(theta), "native_inner_size_m": native_size, "scale_xy": scale, "distances": distance_rows}, indent=2))


if __name__ == "__main__":
    main()
