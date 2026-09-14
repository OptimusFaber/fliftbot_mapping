#!/usr/bin/env python3
"""Re-align real-run maps to GT with a rigid four-corner fit.

Writes ONLY under results/<date>/real/<run>/.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
from scipy import ndimage

from flifltbot.mapping.real_room_gt import (
    COL1_W_BOT,
    COL1_X,
    COL2_W_BOT,
    COL2_X,
    COL_PROTRUDE,
    CUT_H,
    CUT_W,
    DOOR,
    LOW_LIDAR_Z,
    ROOM_X,
    ROOM_Y,
    UP_LIDAR_Z,
    WIN_OUT,
    WIN_Z0,
    WINDOWS,
    PILLAR_SW,
    TV_SCREEN,
    desk_k1_polygon_m,
    invisible_obstacle_rects_at_z,
    oriented_box_corners,
    room_wall_segments,
    room_wall_vertices_at_z,
    visible_obstacle_rects_at_z,
    window_glass_rects,
    window_street_wall_rects,
)
from flifltbot.common.output.layout import RESULTS_ROOT
from flifltbot.common.runtime.accuracy_eval import score_map_real_room

CELL = 0.05
EDGE_PCT = 5
WIN_MARGIN = 0.15

# User-specified camera height per run.
# low ≈ left/right lidars @ 0.10 m; up ≈ top lidar @ 1.60 m
RUN_CAMERA_Z = {
    "run_122309": LOW_LIDAR_Z,
    "run_121410": LOW_LIDAR_Z,
    "run_140543": LOW_LIDAR_Z,
    "run_144042": LOW_LIDAR_Z,
    "run_162252": UP_LIDAR_Z,
    "run_162516": UP_LIDAR_Z,
    "run_164805": UP_LIDAR_Z,
    # Pandar40P horizon @ 1.60 m (maps/20260813/real/room/)
    "run_141753": UP_LIDAR_Z,
    "run_150507": UP_LIDAR_Z,
}


def rotate(P, ang):
    c, s = np.cos(ang), np.sin(ang)
    R = np.array([[c, -s], [s, c]])
    return (R @ P.T).T


def rotate_about(P, pivot, ang):
    P = np.asarray(P, float)
    if P is None or len(P) == 0:
        return P
    return rotate(P - np.asarray(pivot, float), ang) + np.asarray(pivot, float)


def densify(poly, step=0.08):
    out = []
    for a, b in zip(poly[:-1], poly[1:]):
        a, b = np.asarray(a, float), np.asarray(b, float)
        n = max(int(np.linalg.norm(b - a) / step), 1)
        for i in range(n):
            out.append(a + (b - a) * (i / n))
    out.append(poly[-1])
    return np.asarray(out)


def mad_keep(vals, k=3.5):
    vals = np.asarray(vals, float)
    if len(vals) < 8:
        return np.ones(len(vals), bool)
    med = np.median(vals)
    mad = np.median(np.abs(vals - med)) + 1e-9
    return np.abs(vals - med) <= k * 1.4826 * mad


def in_windows(x):
    m = np.zeros(len(x), bool)
    for w in WINDOWS:
        m |= (x >= w["x"] - WIN_MARGIN) & (x <= w["x"] + w["w"] + WIN_MARGIN)
    return m


def in_door(x):
    return (x >= DOOR["x"] - WIN_MARGIN) & (x <= DOOR["x"] + DOOR["w"] + WIN_MARGIN)


def wall_face_x(cloud, side, band_m=1.6):
    xs, ys = cloud[:, 0], cloud[:, 1]
    y_ok = (ys > np.percentile(ys, 8)) & (ys < np.percentile(ys, 92))
    if side == "west":
        edge = np.percentile(xs[y_ok], 1)
        lobe = cloud[y_ok & (xs <= edge + band_m)]
        keep = mad_keep(lobe[:, 0])
        return float(np.percentile(lobe[keep, 0], EDGE_PCT))
    edge = np.percentile(xs[y_ok], 99)
    lobe = cloud[y_ok & (xs >= edge - band_m)]
    keep = mad_keep(lobe[:, 0])
    return float(np.percentile(lobe[keep, 0], 100 - EDGE_PCT))


def wall_face_y(cloud, side, band_m=1.6, z=UP_LIDAR_Z):
    xs, ys = cloud[:, 0], cloud[:, 1]
    x_ok = (xs > np.percentile(xs, 5)) & (xs < np.percentile(xs, 95))
    # Only exclude window niches when beam can see them (≥ sill).
    if side == "north" and z >= WIN_Z0 - 1e-6:
        x_ok = x_ok & ~in_windows(xs)
    if side == "south":
        x_ok = x_ok & ~in_door(xs)
        edge = np.percentile(ys[x_ok], 1) if x_ok.any() else np.percentile(ys, 1)
        lobe = cloud[x_ok & (ys <= edge + band_m)]
        if len(lobe) < 20:
            lobe = cloud[
                (xs > np.percentile(xs, 5))
                & (xs < np.percentile(xs, 95))
                & (ys <= edge + band_m)
            ]
        keep = mad_keep(lobe[:, 1])
        return float(np.percentile(lobe[keep, 1], EDGE_PCT))
    edge = np.percentile(ys[x_ok], 99) if x_ok.any() else np.percentile(ys, 99)
    lobe = cloud[x_ok & (ys >= edge - band_m)]
    if len(lobe) < 20:
        lobe = cloud[
            (xs > np.percentile(xs, 5))
            & (xs < np.percentile(xs, 95))
            & (ys >= edge - band_m)
        ]
    keep = mad_keep(lobe[:, 1])
    return float(np.percentile(lobe[keep, 1], 100 - EDGE_PCT))


def _indoor_mask(cloud, z=UP_LIDAR_Z):
    pad_n = WIN_OUT + 0.6 if z >= WIN_Z0 - 1e-6 else 0.6
    return (
        (cloud[:, 0] > -0.6)
        & (cloud[:, 0] < ROOM_X + 0.6)
        & (cloud[:, 1] > -DOOR["recess"] - 0.5)
        & (cloud[:, 1] < ROOM_Y + pad_n)
    )


def _face_y_in_x(cloud, side, x0, x1, z=UP_LIDAR_Z):
    band = cloud[(cloud[:, 0] >= x0) & (cloud[:, 0] <= x1)]
    if len(band) < 25:
        return None
    try:
        return wall_face_y(band, side, z=z)
    except Exception:
        return None


def level_left_about_east(m, z=UP_LIDAR_Z):
    """Rotate about the pinned east/south-right corner so the left side sits on GT.

    Pin-right measures south from the whole wall, so a CCW ICP drift on the
    west half stays in the overlay (left walls float up). 1-DoF rotation
    around (E, S) ≈ (ROOM_X, 0) drops the left side without moving the right.
    Faces / cyan box are scored on in-room points so through-window ghosts
    do not yank north/south.
    """
    P0 = np.asarray(m["aligned"], float)
    C0 = None if m.get("contour") is None else np.asarray(m["contour"], float)
    T0 = None if m.get("traj") is None else np.asarray(m["traj"], float)
    pivot = np.array([float(m["E"]), float(m["S"])], float)

    x_s0, x_s1 = CUT_W + 0.25, min(8.0, ROOM_X * 0.50)
    x_n0, x_n1 = 0.40, min(6.5, ROOM_X * 0.42)

    def score(ang):
        P = rotate_about(P0, pivot, ang)
        inn = P[_indoor_mask(P, z)]
        if len(inn) < 200:
            return 1e9, P, None, None, None, None
        sl = _face_y_in_x(inn, "south", x_s0, x_s1, z)
        nl = _face_y_in_x(inn, "north", x_n0, x_n1, z)
        sr = _face_y_in_x(inn, "south", ROOM_X * 0.70, ROOM_X + 0.4, z)
        nr = _face_y_in_x(inn, "north", ROOM_X * 0.70, ROOM_X + 0.4, z)
        cut = float(
            (
                (inn[:, 0] > 0.05)
                & (inn[:, 0] < CUT_W - 0.05)
                & (inn[:, 1] > 0.05)
                & (inn[:, 1] < CUT_H - 0.05)
            ).mean()
        )
        wing = inn[
            (inn[:, 0] >= 0.15)
            & (inn[:, 0] <= CUT_W - 0.15)
            & (inn[:, 1] > 0.35)
            & (inn[:, 1] < CUT_H + 1.6)
        ]
        wing_s = (
            float(np.percentile(wing[:, 1], EDGE_PCT)) if len(wing) >= 20 else None
        )
        s = 12.0 * cut
        if sl is not None:
            s += abs(sl) * 4.0
        if nl is not None:
            s += abs(nl - ROOM_Y) * 1.2
        if sr is not None:
            s += abs(sr) * 2.5
        if nr is not None:
            s += abs(nr - ROOM_Y) * 1.0
        if wing_s is not None:
            s += abs(wing_s - CUT_H) * 3.0
        return s, P, sl, nl, sr, wing_s

    best = None
    for deg in np.linspace(-1.0, 4.0, 101):
        sc, P, sl, nl, sr, wing_s = score(np.radians(deg))
        if best is None or sc < best[0]:
            best = (sc, deg, P, sl, nl, sr, wing_s)
    _sc, deg, P, sl, nl, sr, wing_s = best
    ang = np.radians(deg)
    C = None if C0 is None else rotate_about(C0, pivot, ang)
    T = None if T0 is None else rotate_about(T0, pivot, ang)

    inn = P[_indoor_mask(P, z)]
    faces = inn if len(inn) >= 200 else P
    shiftx = wall_face_x(faces, "east") - ROOM_X
    P = P.copy()
    P[:, 0] -= shiftx
    if C is not None:
        C = np.asarray(C, float).copy()
        C[:, 0] -= shiftx
    if T is not None:
        T = np.asarray(T, float).copy()
        T[:, 0] -= shiftx
    faces = P[_indoor_mask(P, z)]
    if len(faces) < 200:
        faces = P

    W = wall_face_x(faces, "west")
    E = wall_face_x(faces, "east")
    S = wall_face_y(faces, "south", z=z)
    N = wall_face_y(faces, "north", z=z)
    Ox, Oy = E - W, N - S
    out = dict(m)
    out.update(
        {
            "aligned": P,
            "contour": C,
            "traj": T,
            "W": float(W),
            "E": float(E),
            "S": float(S),
            "N": float(N),
            "Ox": float(Ox),
            "Oy": float(Oy),
            "err_Ox_cm": float((Ox - ROOM_X) * 100),
            "err_Oy_cm": float((Oy - ROOM_Y) * 100),
            "abs_err_cm": float(abs(Ox - ROOM_X) * 100 + abs(Oy - ROOM_Y) * 100),
            "east_err_cm": float((E - ROOM_X) * 100),
            "level_left_deg": float(deg),
            "level_left_south_m": None if sl is None else float(sl),
            "level_left_north_m": None if nl is None else float(nl),
            "delta_deg": float(m.get("delta_deg", 0.0)) + float(deg),
        }
    )
    return out


def align_pin_right(pts, traj=None, z=UP_LIDAR_Z, level_left=False):
    """Rigidly align map wall corners to the measured room, without scaling.

    The historical function name is kept for callers.  Alignment is now
    balanced over all four detected wall faces, rather than pinning only the
    east wall; this prevents a small yaw/translation error from moving the
    opposite corners away from GT.
    """
    rng = np.random.default_rng(0)
    pts = np.asarray(pts, float)
    if len(pts) < 50:
        raise ValueError("too few map cells")

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
    # Keep every substantial wall component. Real maps often contain gaps at
    # doors/corners, so selecting only the largest component can reduce a full
    # room to one wall (notably archive improved on run_190205).
    keep_labels = [i + 1 for i, size in enumerate(sizes) if size >= max(16, 0.03 * max(sizes))]
    blob = np.isin(lab, keep_labels)
    er = ndimage.binary_erosion(blob, iterations=max(1, int(0.20 / CELL)))
    use = er if er.sum() > 800 else blob
    bnd = use & ~ndimage.binary_erosion(use)
    br, bc = np.where(bnd)
    contour = np.column_stack([xmin + (bc + 0.5) * CELL, ymin + (br + 0.5) * CELL])
    if len(contour) < 30:
        contour = pts.copy()

    best = None
    for _ in range(min(1500, max(200, len(contour) * 2))):
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
    if best is None:
        cxy = contour - contour.mean(0)
        _, _, vh = np.linalg.svd(cxy, full_matrices=False)
        ang_t = float(np.arctan2(vh[0, 1], vh[0, 0]))
    else:
        ang_t = best[1]

    pts1 = rotate(pts, -ang_t)
    ctr1 = rotate(contour, -ang_t)
    tr1 = None if traj is None else rotate(np.asarray(traj, float), -ang_t)
    # ``ang_t`` is the direction of the strongest wall and is already rotated
    # onto +X.  Do not swap axes from the full-cloud AABB: sparse far returns
    # made run_150507 look taller than wide and rotated it by a false 90°.

    gt_d = densify(np.asarray(room_wall_vertices_at_z(z)))

    def rough_x(cloud, side):
        xs = cloud[:, 0]
        span = max(xs.max() - xs.min(), 1e-3)
        band = min(1.5, 0.18 * span)
        if side == "west":
            edge = np.percentile(xs, 0.5)
            return float(np.percentile(cloud[xs <= edge + band, 0], 25))
        edge = np.percentile(xs, 99.5)
        return float(np.percentile(cloud[xs >= edge - band, 0], 75))

    def rough_y(cloud, side):
        ys = cloud[:, 1]
        span = max(ys.max() - ys.min(), 1e-3)
        band = min(1.5, 0.22 * span)
        if side == "south":
            edge = np.percentile(ys, 0.5)
            return float(np.percentile(cloud[ys <= edge + band, 1], 25))
        edge = np.percentile(ys, 99.5)
        return float(np.percentile(cloud[ys >= edge - band, 1], 75))

    pick = None
    for fx in (1, -1):
        for fy in (1, -1):
            for da in np.linspace(-6, 6, 25):
                c, s = np.cos(np.radians(da)), np.sin(np.radians(da))
                Rd = np.array([[c, -s], [s, c]])
                P = (Rd @ (pts1 * [fx, fy]).T).T
                C = (Rd @ (ctr1 * [fx, fy]).T).T
                T = None if tr1 is None else (Rd @ (tr1 * [fx, fy]).T).T

                # Provisional faces, then align the centre of all four wall
                # corners.  A rigid transform cannot force a distorted map to
                # match every corner; centring distributes residuals honestly.
                Wm, Em = rough_x(C, "west"), rough_x(C, "east")
                Sm, Nm = rough_y(C, "south"), rough_y(C, "north")
                shift = np.array([
                    0.5 * (Wm + Em - ROOM_X),
                    0.5 * (Sm + Nm - ROOM_Y),
                ])
                P = P - shift
                C = C - shift
                if T is not None:
                    T = T - shift

                Ox = float(wall_face_x(C, "east") - wall_face_x(C, "west"))
                Oy = float(
                    wall_face_y(C, "north", z=z) - wall_face_y(C, "south", z=z)
                )
                # Re-centre after refined wall-face estimates.
                E = wall_face_x(C, "east")
                W = wall_face_x(C, "west")
                S = wall_face_y(C, "south", z=z)
                N = wall_face_y(C, "north", z=z)
                shift2 = np.array([
                    0.5 * (W + E - ROOM_X),
                    0.5 * (S + N - ROOM_Y),
                ])
                P = P - shift2
                C = C - shift2
                if T is not None:
                    T = T - shift2

                W = wall_face_x(C, "west")
                E = wall_face_x(C, "east")
                S = wall_face_y(C, "south", z=z)
                N = wall_face_y(C, "north", z=z)
                Ox, Oy = E - W, N - S

                # Score all four corners symmetrically.  No scale is fitted.
                # Distinguishes mirrors: east must be the solid wall; south has
                # door recess + low furniture; SW cutout stays empty.
                span_pen = abs(Ox - ROOM_X) + abs(Oy - ROOM_Y)
                corner_pen = (
                    abs(W) + abs(E - ROOM_X) + abs(S) + abs(N - ROOM_Y)
                )
                if T is not None and len(T):
                    traj_pen = max(0.0, ROOM_X * 0.55 - float(np.mean(T[:, 0]))) * 2.0
                else:
                    traj_pen = max(0.0, ROOM_X * 0.55 - float(np.mean(P[:, 0]))) * 2.0
                cut = (
                    (P[:, 0] > 0)
                    & (P[:, 0] < CUT_W)
                    & (P[:, 1] > 0)
                    & (P[:, 1] < CUT_H)
                ).mean()
                idx = np.linspace(0, len(C) - 1, min(300, len(C))).astype(int)
                ch = np.mean([np.min(np.sum((gt_d - C[i]) ** 2, axis=1)) for i in idx])

                # East lobe should look like a real wall (dense, low x-std);
                # west of an incomplete run is ragged.
                east_band = P[(P[:, 0] > E - 0.9) & (P[:, 0] < E + 0.4)]
                west_band = P[(P[:, 0] > W - 0.4) & (P[:, 0] < W + 0.9)]
                def _wall_qual(band):
                    if len(band) < 20:
                        return 0.0
                    return float(len(band)) / (float(np.std(band[:, 0])) + 0.04)
                wall_pen = 0.004 * (
                    _wall_qual(west_band) - _wall_qual(east_band)
                )

                # Door recess is on the SOUTH wall only.
                door_m = (
                    (P[:, 0] >= DOOR["x"] - 0.10)
                    & (P[:, 0] <= DOOR["x"] + DOOR["w"] + 0.10)
                    & (P[:, 1] < -0.04)
                    & (P[:, 1] > -DOOR["recess"] - 0.25)
                )
                door_pen = -8.0 * float(door_m.mean())

                # South-half right crates stick out 40 cm; north-half 34 cm.
                # Reward points in the extra 6 cm strip on the south half.
                extra = P[
                    (P[:, 0] >= ROOM_X - 0.40)
                    & (P[:, 0] <= ROOM_X - 0.30)
                    & (P[:, 1] >= 0.05)
                    & (P[:, 1] <= ROOM_Y * 0.5)
                ]
                crate_pen = -0.015 * float(len(extra))

                # Visible furniture occupancy (columns, south tables/panels).
                furn_hit = 0.0
                n_furn = 0
                for x, y, w, d in visible_obstacle_rects_at_z(z):
                    if x + w < W - 0.5:
                        continue  # outside mapped span
                    n_furn += 1
                    inside = (
                        (P[:, 0] >= x - 0.12)
                        & (P[:, 0] <= x + w + 0.12)
                        & (P[:, 1] >= y - 0.12)
                        & (P[:, 1] <= y + d + 0.12)
                    )
                    furn_hit += 1.0 if inside.any() else 0.0
                furn_pen = -2.0 * (furn_hit / max(n_furn, 1))

                # Columns near south wall are a strong Y-orientation cue.
                col_pen = 0.0
                for cx, cw in ((COL1_X, COL1_W_BOT), (COL2_X, COL2_W_BOT)):
                    south = (
                        (P[:, 0] >= cx - 0.08)
                        & (P[:, 0] <= cx + cw + 0.08)
                        & (P[:, 1] >= 0.0)
                        & (P[:, 1] <= COL_PROTRUDE + 0.20)
                    )
                    north = (
                        (P[:, 0] >= cx - 0.08)
                        & (P[:, 0] <= cx + cw + 0.08)
                        & (P[:, 1] >= ROOM_Y - COL_PROTRUDE - 0.20)
                        & (P[:, 1] <= ROOM_Y)
                    )
                    col_pen += 0.002 * (float(north.sum()) - float(south.sum()))

                score = (
                    2.5 * span_pen
                    + 2.0 * corner_pen
                    + traj_pen
                    + 6.0 * cut
                    + 0.4 * np.sqrt(ch)
                    + wall_pen
                    + door_pen
                    + crate_pen
                    + furn_pen
                    + col_pen
                )
                if pick is None or score < pick[0]:
                    pick = (score, P, C, T, W, E, S, N, Ox, Oy, fx, fy, da)

    score, map_a, ctr_a, traj_a, W, E, S, N, Ox, Oy, fx, fy, da = pick

    # Refine the remaining sub-degree yaw from all long wall sections.  The
    # coarse candidate search above is intentionally robust but its 0.5° step
    # is visible over a 16 m wall.  This axial fit removes that quantisation
    # without scaling the map or forcing its dimensions to GT.
    residuals = []
    bands = (
        ctr_a[(np.abs(ctr_a[:, 1] - S) < 0.30) & (ctr_a[:, 0] > CUT_W + 0.4)],
        ctr_a[(np.abs(ctr_a[:, 1] - N) < 0.30) & (ctr_a[:, 0] > 0.5)],
        ctr_a[(np.abs(ctr_a[:, 0] - W) < 0.30) & (ctr_a[:, 1] > CUT_H + 0.3)],
        ctr_a[(np.abs(ctr_a[:, 0] - E) < 0.30) & (ctr_a[:, 1] > 0.4)],
    )
    for band in bands:
        if len(band) < 20:
            continue
        centered = band - np.median(band, axis=0)
        _u, _s, vh = np.linalg.svd(centered, full_matrices=False)
        theta = math.atan2(vh[0, 1], vh[0, 0])
        residual = (theta + math.pi / 4) % (math.pi / 2) - math.pi / 4
        extent = float(np.ptp(band @ vh[0]))
        residuals.append((residual, max(extent, 0.5)))
    fine = 0.0
    if residuals:
        fine = -float(sum(a * w for a, w in residuals) / sum(w for _a, w in residuals))
        fine = float(np.clip(fine, math.radians(-3.0), math.radians(3.0)))
        pivot = np.array([ROOM_X / 2.0, ROOM_Y / 2.0])
        map_a = rotate_about(map_a, pivot, fine)
        ctr_a = rotate_about(ctr_a, pivot, fine)
        if traj_a is not None:
            traj_a = rotate_about(traj_a, pivot, fine)
        W, E = wall_face_x(ctr_a, "west"), wall_face_x(ctr_a, "east")
        S = wall_face_y(ctr_a, "south", z=z)
        N = wall_face_y(ctr_a, "north", z=z)
        shift = np.array([0.5 * (W + E - ROOM_X), 0.5 * (S + N - ROOM_Y)])
        map_a -= shift
        ctr_a -= shift
        if traj_a is not None:
            traj_a -= shift
        W, E = wall_face_x(ctr_a, "west"), wall_face_x(ctr_a, "east")
        S = wall_face_y(ctr_a, "south", z=z)
        N = wall_face_y(ctr_a, "north", z=z)
        Ox, Oy = E - W, N - S
        da += math.degrees(fine)
    result = {
        "aligned": map_a,
        "contour": ctr_a,
        "traj": traj_a,
        "W": float(W),
        "E": float(E),
        "S": float(S),
        "N": float(N),
        "Ox": float(Ox),
        "Oy": float(Oy),
        "err_Ox_cm": float((Ox - ROOM_X) * 100),
        "err_Oy_cm": float((Oy - ROOM_Y) * 100),
        "abs_err_cm": float(abs(Ox - ROOM_X) * 100 + abs(Oy - ROOM_Y) * 100),
        "east_err_cm": float((E - ROOM_X) * 100),
        "flip": [fx, fy],
        "delta_deg": float(da),
        "score": float(score),
        "n_cells": int(len(pts)),
        "z": float(z),
        "level_left_deg": 0.0,
        "fine_wall_yaw_deg": float(math.degrees(fine)),
    }
    if level_left:
        result = level_left_about_east(result, z=z)
    return result


def save_overlay(run_name: str, m: dict, out_png: Path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle, Polygon, Rectangle

    z = m["z"]
    cam_label = "up" if z >= 1.0 else "low"
    fig, ax = plt.subplots(figsize=(12.5, 6.4), dpi=130)
    ax.set_aspect("equal")

    for i, (a, b) in enumerate(room_wall_segments(z=z)):
        ax.plot(
            [a[0], b[0]], [a[1], b[1]], color="black", lw=2.0,
            label=f"GT Gazebo/Canvas @{cam_label} {z:.3f}m" if i == 0 else None,
        )
    # Exact window construction from Real Room Floorplan: 32 cm niche,
    # 5 cm glass and 10 cm street wall on the piers (not across openings).
    if z >= WIN_Z0 - 1e-6:
        for i, (x, y, w, d) in enumerate(window_glass_rects()):
            ax.add_patch(Rectangle(
                (x, y), w, d, facecolor="#7ec8e3", edgecolor="#1a6f8a",
                alpha=0.42, lw=0.8, ls="--",
                label="window glass 5 cm" if i == 0 else None,
            ))
        for i, (x, y, w, d) in enumerate(window_street_wall_rects()):
            ax.add_patch(Rectangle(
                (x, y), w, d, facecolor="#b0b4b8", edgecolor="#4a4e52",
                alpha=0.80, lw=0.8,
                label="street wall 10 cm (piers)" if i == 0 else None,
            ))
    ax.add_patch(
        Rectangle((0, 0), CUT_W, CUT_H, facecolor="#eee", edgecolor="gray", ls="--", alpha=0.7)
    )

    for x, y, w, d in invisible_obstacle_rects_at_z(z):
        ax.add_patch(
            Rectangle((x, y), w, d, fill=False, edgecolor="#aaaaaa", ls=":", lw=0.8)
        )
    # desk: visible at low beam
    desk = desk_k1_polygon_m()
    desk_vis = z <= 0.75  # desk H=0.75
    ax.add_patch(
        Polygon(
            desk,
            closed=True,
            fill=desk_vis,
            facecolor="#666666" if desk_vis else "none",
            edgecolor="k" if desk_vis else "#aaaaaa",
            alpha=0.35 if desk_vis else 1.0,
            ls="-" if desk_vis else ":",
            lw=0.6,
        )
    )
    tv_poly = np.asarray(oriented_box_corners(TV_SCREEN), float)
    tv_aabb = (
        float(tv_poly[:, 0].min()), float(tv_poly[:, 1].min()),
        float(np.ptp(tv_poly[:, 0])), float(np.ptp(tv_poly[:, 1])),
    )
    for x, y, w, d in visible_obstacle_rects_at_z(z):
        if max(abs(x-tv_aabb[0]), abs(y-tv_aabb[1]), abs(w-tv_aabb[2]), abs(d-tv_aabb[3])) < 1e-3:
            continue
        ax.add_patch(
            Rectangle((x, y), w, d, facecolor="#666666", edgecolor="k", alpha=0.4, lw=0.5)
        )
    if z >= TV_SCREEN["z0"] - 1e-6 and z <= TV_SCREEN["z0"] + TV_SCREEN["h"] + 1e-6:
        ax.add_patch(Polygon(
            tv_poly, closed=True, facecolor="#7755aa", edgecolor="#3e246b",
            alpha=0.65, lw=1.0, label="TV 135×4 cm, yaw −39°",
        ))
        ax.text(TV_SCREEN["cx"], TV_SCREEN["cy"], "TV", fontsize=7, ha="center", va="center")
    ax.add_patch(
        Circle(
            (PILLAR_SW["cx"], PILLAR_SW["cy"]),
            PILLAR_SW["r"],
            facecolor="#444444",
            edgecolor="k",
            label="structural pillar Ø30 cm",
        )
    )
    ax.text(PILLAR_SW["cx"], PILLAR_SW["cy"], "pillar", fontsize=6, color="white", ha="center", va="center")

    if z >= WIN_Z0 - 1e-6:
        for w in WINDOWS:
            ax.add_patch(
                Rectangle(
                    (w["x"], ROOM_Y),
                    w["w"],
                    WIN_OUT,
                    facecolor="tab:green",
                    alpha=0.18,
                    edgecolor="tab:green",
                    lw=0.9,
                )
            )
    ax.add_patch(
        Rectangle(
            (DOOR["x"], -DOOR["recess"]),
            DOOR["w"],
            DOOR["recess"],
            facecolor="tab:purple",
            alpha=0.18,
            edgecolor="tab:purple",
            lw=0.9,
            label="door recess",
        )
    )

    pts = np.asarray(m["aligned"], float)
    inn = pts[_indoor_mask(pts, z)]
    show = inn if len(inn) >= 80 else pts
    step = max(1, len(show) // 8000)
    ax.scatter(show[::step, 0], show[::step, 1], s=1.2, c="tab:blue", alpha=0.28, label="map")
    rect = np.array(
        [
            [m["W"], m["S"]],
            [m["E"], m["S"]],
            [m["E"], m["N"]],
            [m["W"], m["N"]],
            [m["W"], m["S"]],
        ]
    )
    ax.plot(rect[:, 0], rect[:, 1], "c-", lw=2.2, label="detected wall corners")

    if m.get("traj") is not None and len(m["traj"]):
        ax.plot(m["traj"][:, 0], m["traj"][:, 1], color="orange", lw=1.0, alpha=0.85, label="traj")

    extra = ""
    if abs(float(m.get("level_left_deg", 0.0))) > 0.05:
        extra = f'   left↷{m["level_left_deg"]:+.1f}°'
    fit_label = m.get("fit_label", "rigid fit (no scale)")
    ax.set_title(
        f"{run_name}  ·  {cam_label}@{z:.3f}m  ·  current Gazebo room GT · {fit_label}\n"
        f'Ox {m["Ox"]:.2f}/{ROOM_X:.2f} m ({m["err_Ox_cm"]:+.0f} cm)   '
        f'Oy {m["Oy"]:.2f}/{ROOM_Y:.2f} m ({m["err_Oy_cm"]:+.0f} cm)   '
        f'E {m["east_err_cm"]:+.0f} cm{extra}'
    )
    ax.set_xlim(-1.5, max(ROOM_X, m["E"]) + 1.8)
    ax.set_ylim(-DOOR["recess"] - 0.9, ROOM_Y + (WIN_OUT if z >= WIN_Z0 else 0) + 0.8)
    ax.set_xlabel("X room (m)")
    ax.set_ylabel("Y room (m)")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="lower left", fontsize=8)
    ax.text(
        0.99,
        0.98,
        f"flip={m['flip']} Δ={m['delta_deg']:+.1f}°"
        + (f" left={m.get('level_left_deg', 0):+.1f}°" if m.get("level_left_deg") else "")
        + f"\nsolid=visible @{z:.2f}m; dotted=missed\n"
        f"{'windows niches ON' if z >= WIN_Z0 else 'windows niches OFF (below sill)'}",
        transform=ax.transAxes,
        va="top",
        ha="right",
        fontsize=8,
        bbox=dict(boxstyle="round", fc="white", alpha=0.9),
    )
    fig.tight_layout()
    fig.savefig(out_png, dpi=140, bbox_inches="tight")
    plt.close(fig)


def process_run(results_dir: Path, run_name: str, z: float):
    map_json = results_dir / f"{run_name}.json"
    if not map_json.is_file():
        raise FileNotFoundError(map_json)
    data = json.loads(map_json.read_text(encoding="utf-8"))
    pts = np.asarray(data["map"], float)
    traj = np.asarray([[p["x"], p["y"]] for p in data.get("poses", [])], float)
    if len(traj) == 0:
        traj = None

    m = align_pin_right(pts, traj, z=z)
    inn = np.asarray(m["aligned"], float)
    inn = inn[_indoor_mask(inn, z)]
    aligned_metrics = score_map_real_room(
        inn if len(inn) >= 80 else m["aligned"], align="none"
    )
    cam = "up" if z >= 1.0 else "low"
    overlay = results_dir / f"{run_name}_gt_walls_{cam}.png"
    # also keep a stable name used before
    overlay_alias = results_dir / f"{run_name}_gt_walls_up160.png"
    save_overlay(run_name, m, overlay)
    if overlay != overlay_alias:
        # for low cams don't pretend up160; remove misleading alias if present
        if cam == "low" and overlay_alias.exists() and overlay_alias.resolve() != overlay.resolve():
            overlay_alias.unlink()
        # copy/symlink-like second write for up runs
        if cam == "up":
            save_overlay(run_name, m, overlay_alias)

    gap_left_cm = max(0.0, -float(m["err_Ox_cm"]))
    summary = {
        "run": run_name,
        "results_dir": str(results_dir),
        "camera_z_m": z,
        "camera": cam,
        "align": "rigid four-corner alignment; rotation+translation only; no scale",
        "metrics_aligned": aligned_metrics,
        "metrics_walls_robust": {
            "Ox_m": m["Ox"],
            "Oy_m": m["Oy"],
            "err_Ox_cm": m["err_Ox_cm"],
            "err_Oy_cm": m["err_Oy_cm"],
            "abs_err_cm": m["abs_err_cm"],
            "east_err_cm": m["east_err_cm"],
            "gap_left_cm": gap_left_cm,
            "W": m["W"],
            "E": m["E"],
            "S": m["S"],
            "N": m["N"],
            "n_cells": m["n_cells"],
            "flip": m["flip"],
            "delta_deg": m["delta_deg"],
            "level_left_deg": m.get("level_left_deg", 0.0),
            "ok": True,
            "camera_z_m": z,
            "method": "balanced four-corner fit; no scale; GT 16.465×6.56",
        },
        "overlay_png": str(overlay),
        "gt_room_m": [ROOM_X, ROOM_Y],
        "gt_measured": "2026-08-13 remeasure Ox=1646.5; K1/K2 N/S widths; foam N+S; pallet+body @ win132",
    }
    out_json = results_dir / "accuracy_improved_real.json"
    # merge with existing if present
    if out_json.is_file():
        try:
            prev = json.loads(out_json.read_text(encoding="utf-8"))
            prev.update(
                {
                    "camera_z_m": z,
                    "camera": cam,
                    "align": summary["align"],
                    "metrics_aligned": summary["metrics_aligned"],
                    "metrics_walls_robust": summary["metrics_walls_robust"],
                    "overlay_png": summary["overlay_png"],
                    "gt_room_m": summary["gt_room_m"],
                    "gt_measured": summary["gt_measured"],
                }
            )
            summary = prev
        except Exception:
            pass
    out_json.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def _find_results_dir(run_name: str) -> Path | None:
    matches = sorted(RESULTS_ROOT.glob(f"*/real/{run_name}"))
    matches += sorted(RESULTS_ROOT.glob(f"*/real/room/{run_name}"))
    return matches[-1] if matches else None


def main():
    rows = []
    for run_name, z in RUN_CAMERA_Z.items():
        results_dir = _find_results_dir(run_name)
        if results_dir is None:
            print(f"SKIP {run_name}: no results dir")
            continue
        print("=" * 64)
        print(f"{run_name}  z={z:.2f}m  ({'up' if z>=1 else 'low'})  {results_dir}")
        try:
            s = process_run(results_dir, run_name, z)
            mw = s["metrics_walls_robust"]
            print(
                f'  Ox {mw["err_Ox_cm"]:+.1f}cm  Oy {mw["err_Oy_cm"]:+.1f}cm  '
                f'E {mw["east_err_cm"]:+.1f}cm  |err|={mw["abs_err_cm"]:.1f}  '
                f'flip={mw["flip"]} Δ={mw["delta_deg"]:+.1f}°'
            )
            al = s.get("metrics_aligned") or {}
            rows.append(
                {
                    "run": run_name,
                    "results_dir": str(results_dir),
                    "camera_z_m": z,
                    "camera": "up" if z >= 1 else "low",
                    "err_Ox_cm": round(mw["err_Ox_cm"], 1),
                    "err_Oy_cm": round(mw["err_Oy_cm"], 1),
                    "east_err_cm": round(mw["east_err_cm"], 1),
                    "gap_left_cm": round(mw.get("gap_left_cm", max(0.0, -mw["err_Ox_cm"])), 1),
                    "abs_err_cm": round(mw["abs_err_cm"], 1),
                    "near_gt_median_cm": (
                        None
                        if al.get("near_gt_median_cm") is None
                        else round(al["near_gt_median_cm"], 1)
                    ),
                    "near_gt_p95_cm": (
                        None
                        if al.get("near_gt_p95_cm") is None
                        else round(al["near_gt_p95_cm"], 1)
                    ),
                    "frac_within_50cm": al.get("frac_within_50cm"),
                    "overlay": s["overlay_png"],
                    "ok": True,
                }
            )
        except Exception as exc:
            print(f"  FAILED: {exc}")
            rows.append({"run": run_name, "ok": False, "error": str(exc)})

    out = {
        "note": (
            "Re-eval vs 2026-08-13 GT (Ox=16.465 m). Pin RIGHT (east→ROOM_X); "
            "height-aware objects; robot drives on right. Maps not rebuilt."
        ),
        "metric": "robust wall faces; abs_err=|dOx|+|dOy|; near_gt on indoor aligned cells",
        "gt_room_m": [ROOM_X, ROOM_Y],
        "gt_measured": "2026-08-13 remeasure; K1/K2 N/S widths; foam N+S; pallet+body @ win132",
        "runs": rows,
    }
    path = RESULTS_ROOT / "real_runs_accuracy_summary.json"
    path.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print("\nSUMMARY")
    for r in rows:
        if not r.get("ok"):
            print(f'{r["run"]:12s} FAILED')
            continue
        print(
            f'{r["run"]:12s} {r["camera"]:3s}@{r["camera_z_m"]:.2f}  '
            f'Ox {r["err_Ox_cm"]:+6.1f}  Oy {r["err_Oy_cm"]:+6.1f}  '
            f'E {r["east_err_cm"]:+5.1f}  |err| {r["abs_err_cm"]:6.1f}  '
            f'med {r.get("near_gt_median_cm")}'
        )
    print("wrote", path)

    pandar = [r for r in rows if r.get("ok") and r["run"] in ("run_141753", "run_150507")]
    if pandar:
        ranked = sorted(pandar, key=lambda r: (r["abs_err_cm"], r.get("near_gt_median_cm") or 1e9))
        winner = ranked[0]
        reason = (
            f"smaller wall-span error |ΔOx|+|ΔOy| = {winner['abs_err_cm']:.1f} cm "
            f"(Ox {winner['err_Ox_cm']:+.1f}, Oy {winner['err_Oy_cm']:+.1f})"
        )
        if len(ranked) > 1:
            reason += f"; runner-up {ranked[1]['run']} has {ranked[1]['abs_err_cm']:.1f} cm"
        compare = {
            "note": (
                "Pandar40P horizon maps re-aligned to 2026-08-13 GT (Ox=16.465). "
                "Maps not rebuilt."
            ),
            "gt_room_m": [ROOM_X, ROOM_Y],
            "metric": "robust wall faces; abs_err=|dOx|+|dOy|",
            "verdict": {
                "winner": winner["run"],
                "reason": reason,
                "ranked": [r["run"] for r in ranked],
            },
            "runs": pandar,
        }
        cpath = RESULTS_ROOT / "20260813" / "real" / "room" / "pandar_room_compare.json"
        cpath.parent.mkdir(parents=True, exist_ok=True)
        cpath.write_text(json.dumps(compare, indent=2) + "\n", encoding="utf-8")
        print("wrote", cpath)


if __name__ == "__main__":
    main()
