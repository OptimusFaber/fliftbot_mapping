"""Hand-measured real-room ground truth (meters).

Origin: outer SW corner of the room bounding box. +X right, +Y up.
Tape measurements collected 2026-08; heights for shelves/panels as noted.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

# --- room shell (m) ---
# Top wall chain (cm), x=0 → Ox:
#   191.5, 68.5, 101, 196, 31, 51.5, 11, 100, 76.5,
#   196, 50, 132, 21.5, 54, 132, 132, 72, 30  → 1646.5
ROOM_X = 16.465
ROOM_Y = 6.56
CUT_W = 3.10
CUT_H = 1.22
CELL_M = 0.05

# Columns (west face from top chain; north/south widths differ)
COL_PROTRUDE = 0.26
COL1_X = 5.88  # 191.5+68.5+101+196+31
COL1_W_TOP = 0.515
COL1_W_BOT = 0.525
COL2_X = 12.265  # …+51.5+11+100+76.5+196+50+132+21.5
COL2_W_TOP = 0.54
COL2_W_BOT = 0.53
COL_W = COL2_W_TOP  # legacy single width (K2 north)

# Bottom furniture chain: 310 →1.5→ cab40 →96.5→ door93.5 (table overlaps 3.5)
CAB = dict(x=3.115, w=0.40, y=0.0, d=1.40, h=1.40)
DOOR = dict(x=4.48, w=0.935, recess=0.23)
TABLE_DOOR = dict(x=5.38, w=1.60, y=0.26, d=0.70, h=0.76)
SHELF_BOT = dict(x=6.98, w=1.00, y=0.12, d=0.40, h=2.0)
# Trash / spool on floor — visible to low lidars (~0.135 m)
SPOOL = dict(x=7.98, w=0.80, y=0.0, d=0.20, z0=0.0, h=0.50, kind="dynamic_trash")
# Pipes Ø12; pipe1←32→panel1 65 ←10.5→ K2; K2 ←10.5→ panel2 64 ←103.5→ pipe2
_PIPE_D = 0.12
PIPE1 = dict(x=11.07, d=_PIPE_D, z0=0.69, h=2.0)  # K2 −10.5 −65 −32 −12
# Щитки hang at the same floor height as the pipes (z0=0.69), not from the floor.
PANEL1 = dict(
    x=11.51,
    w=0.65,
    y=0.0,
    d=0.28,
    z0=0.69,
    h=1.80,  # hang @ pipe height; up lidar 1.60 still hits
)
PANEL2 = dict(
    x=12.90,  # K2 south 53 + 10.5
    w=0.64,
    y=0.0,
    d=0.28,
    z0=0.69,
    h=1.80,
)
PIPE2 = dict(x=14.575, d=_PIPE_D, z0=0.69, h=2.0)  # panel2 +64 +103.5

# Real lidar mount heights (m). Horizontal beam does not see objects entirely below z.
UP_LIDAR_Z = 1.60   # top / "up" camera
LOW_LIDAR_Z = 0.135  # left/right dock lidars (centers; stand bottom = 0.10 m)

# Left / SW
TABLE_SW = dict(x=0.0, y=CUT_H, w=1.90, d=0.52, h=0.76)
FRIDGE = dict(x=0.08, y=1.82, w=0.55, d=0.48, h=1.56)
WALL_CAB1 = dict(x=0.18, y=2.30, w=0.27, d=0.80, h=2.0)
WALL_CAB2 = dict(x=0.09, y=5.32, w=0.27, d=0.80, h=2.0)
# Whiteboard: T-legs 80 cm, then panel 180 cm wide × 100 cm tall (top = 180 cm).
# Flush toward шк1 (5 cm gap). Flipped 180°: T-feet against the west wall,
# panel stands ~45 cm into the room (отставлена от стены).
BOARD_LEG_H = 0.80
BOARD_PANEL_H = 1.00
BOARD_W_Y = 1.80  # along the wall
BOARD_T = 0.03
BOARD_FOOT_X = 0.45  # T-foot from wall toward the panel
BOARD_FOOT_Y = 0.08
BOARD_WALL_X = 0.02
BOARD = dict(
    x=BOARD_WALL_X + BOARD_FOOT_X - BOARD_T,  # panel at inner end of T
    y=WALL_CAB1["y"] + WALL_CAB1["d"] + 0.05,  # closer to шк1
    w=BOARD_T,
    d=BOARD_W_Y,
    z0=BOARD_LEG_H,
    h=BOARD_PANEL_H,
    kind="board",
)
_BOARD_LEG_INSET = 0.12
BOARD_STEMS = [
    dict(
        x=BOARD["x"],
        y=BOARD["y"] + _BOARD_LEG_INSET,
        w=0.06,
        d=0.06,
        z0=0.0,
        h=BOARD_LEG_H,
        kind="board_leg",
    ),
    dict(
        x=BOARD["x"],
        y=BOARD["y"] + BOARD["d"] - _BOARD_LEG_INSET - 0.06,
        w=0.06,
        d=0.06,
        z0=0.0,
        h=BOARD_LEG_H,
        kind="board_leg",
    ),
]
BOARD_FEET = [
    dict(
        x=BOARD_WALL_X,
        y=s["y"] - (BOARD_FOOT_Y - s["d"]) / 2.0,
        w=BOARD_FOOT_X,
        d=BOARD_FOOT_Y,
        z0=0.0,
        h=0.04,
        kind="board_leg",
    )
    for s in BOARD_STEMS
]
SINK = dict(x=1.90, y=CUT_H, w=0.48, d=0.50, h=0.85)
DISHWASHER = dict(x=2.38, y=CUT_H, w=0.48, d=0.60, h=0.85)
PILLAR_SW = dict(cx=3.135, cy=CUT_H + 0.20 + 0.15, r=0.15, h=2.0)  # +20 cm Ox from prior pose

SHELF_TOP = dict(x=6.505, w=1.00, y=ROOM_Y - 0.40, d=0.40, h=2.0)  # K1 north + 11 cm
# Right wall crates: south 39.5 cm (two parts) + north 30 cm (top-wall last segment).
# 39.5 along Oy: 18 cm gap from south wall → 80 cm → 13 cm gap → 213 cm.
_SHELF_R_X = ROOM_X - 0.395
SHELF_RIGHT_BOT_A = dict(
    x=_SHELF_R_X, w=0.395, y=0.18, d=0.80, h=2.0, kind="shelf"
)
SHELF_RIGHT_BOT_B = dict(
    x=_SHELF_R_X, w=0.395, y=1.11, d=2.13, h=2.0, kind="shelf"
)
SHELF_RIGHT_BOTS = (SHELF_RIGHT_BOT_A, SHELF_RIGHT_BOT_B)
SHELF_RIGHT_BOT = SHELF_RIGHT_BOT_A  # legacy: first 39.5 part (x/w unchanged)
SHELF_RIGHT_TOP = dict(
    x=ROOM_X - 0.30, w=0.30, y=ROOM_Y / 2.0, d=ROOM_Y / 2.0, h=2.0, kind="shelf"
)
# deepest face (39.5 parts) — used where a single "right shelf" x is needed
SHELF_RIGHT = SHELF_RIGHT_BOT_A

# Foam bumpers: K1 → east wall (39.5 cabinets sit 18 cm off the south wall)
FOAM_BOT = dict(x=COL1_X, w=ROOM_X - COL1_X, y=0.0, d=0.06, h=0.15)
FOAM_TOP = dict(
    x=COL1_X,
    w=SHELF_RIGHT_TOP["x"] - COL1_X,
    y=ROOM_Y - 0.06,
    d=0.06,
    h=0.15,
)
FOAM = FOAM_BOT  # legacy alias (south strip)

# Windows (outside +Y), sill z0=0.80.
# From inside looking out: 32 cm niche, then 5 cm glass (lidar passes through).
# The 10 cm outer wall runs on the piers only — it does not cover the openings.
WIN_DEPTH = 0.32
WIN_GLASS = 0.05
WIN_STREET = 0.10
WIN_OUT = WIN_DEPTH  # niche depth; openings have no back wall
WIN_TOTAL = round(WIN_DEPTH + WIN_GLASS + WIN_STREET, 2)
WIN_Z0 = 0.80
# Inner sill at window height — caps the niche hole and juts into the room.
SILL_IN = 0.06
SILL_T = 0.04
WINDOWS = [
    dict(x=1.915, w=0.685, label="68.5"),
    dict(x=3.61, w=1.96, label="196"),
    dict(x=8.27, w=1.96, label="196"),
    dict(x=10.73, w=1.32, label="132"),
    dict(x=14.125, w=1.32, label="132"),
]
HEATERS = []
for i, w in enumerate(WINDOWS):
    hw = 0.60 if i == 0 else 1.00
    HEATERS.append(
        dict(
            x=w["x"] + (w["w"] - hw) / 2,
            w=hw,
            y=ROOM_Y - 0.10,
            d=0.10,
            z0=0.20,
            h=0.50,
        )
    )

# TV: panel 135×80×4 cm on a 100 cm T-stand. One end at the leftmost window,
# 8 cm from that window's west edge; 39° from the north wall into the room.
# Back of the panel is 4 cm off the window/wall.
TV_LEN = 1.35
TV_H = 0.80
TV_T = 0.04
TV_LEG_H = 1.00
TV_FOOT_L = 1.00
TV_FOOT_W = 0.08
TV_YAW = math.radians(-39.0)
TV_EDGE_GAP = 0.08
TV_WALL_GAP = 0.04


def _rot_xy(lx: float, ly: float, yaw: float) -> tuple[float, float]:
    c, s = math.cos(yaw), math.sin(yaw)
    return lx * c - ly * s, lx * s + ly * c


def _tv_center():
    """Screen centre: west-back corner 8 cm from leftmost window, 4 cm off the wall."""
    w0 = WINDOWS[0]
    x_edge = float(w0["x"]) + TV_EDGE_GAP
    y_edge = ROOM_Y - TV_WALL_GAP
    dx, dy = _rot_xy(-TV_LEN / 2.0, TV_T / 2.0, TV_YAW)
    return x_edge - dx, y_edge - dy


TV_CX, TV_CY = _tv_center()
TV_SCREEN = dict(
    cx=TV_CX,
    cy=TV_CY,
    z0=TV_LEG_H,
    w=TV_LEN,
    d=TV_T,
    h=TV_H,
    yaw=TV_YAW,
    kind="tv",
)
TV_STEM = dict(
    cx=TV_CX,
    cy=TV_CY,
    z0=0.0,
    w=0.08,
    d=0.04,
    h=TV_LEG_H,
    yaw=TV_YAW,
    kind="tv_stand",
)
# T-foot: 100 cm bar on the floor, parallel to the screen.
TV_FOOT = dict(
    cx=TV_CX,
    cy=TV_CY,
    z0=0.0,
    w=TV_FOOT_L,
    d=TV_FOOT_W,
    h=0.04,
    yaw=TV_YAW,
    kind="tv_stand",
)


def window_sill_rects():
    """Inner + niche sill slabs; top = window height WIN_Z0."""
    return [
        dict(
            x=float(w["x"]),
            y=ROOM_Y - SILL_IN,
            w=float(w["w"]),
            d=SILL_IN + WIN_DEPTH,
            z0=WIN_Z0 - SILL_T,
            h=SILL_T,
            kind="sill",
        )
        for w in WINDOWS
    ]


def oriented_box_corners(obj: dict) -> list[tuple[float, float]]:
    hx, hy = obj["w"] / 2.0, obj["d"] / 2.0
    yaw = float(obj.get("yaw", 0.0))
    cx, cy = float(obj["cx"]), float(obj["cy"])
    out = []
    for lx, ly in ((-hx, -hy), (hx, -hy), (hx, hy), (-hx, hy)):
        dx, dy = _rot_xy(lx, ly, yaw)
        out.append((cx + dx, cy + dy))
    return out


def oriented_box_aabb(obj: dict) -> tuple[float, float, float, float]:
    xs, ys = zip(*oriented_box_corners(obj))
    x0, y0 = min(xs), min(ys)
    return x0, y0, max(xs) - x0, max(ys) - y0


def _oriented_as_rect(obj: dict) -> dict:
    x, y, w, d = oriented_box_aabb(obj)
    return {**obj, "x": x, "y": y, "w": w, "d": d}


def oriented_box_segments(obj: dict):
    pts = oriented_box_corners(obj)
    return list(zip(pts, pts[1:] + pts[:1]))


# Oval table by the TV: 180 cm Ox × 300 cm Oy, axis-aligned like the K1 desk,
# 5 cm south of the TV footprint. Height matches the K1 desk.
TV_TABLE_W = 1.80
TV_TABLE_D = 3.00
TV_TABLE_GAP = 0.05
TV_TABLE_H = 0.75
TV_TABLE_SHIFT_X = -0.60  # 60 cm left (−X) from TV centre
TV_TABLE_SHIFT_Y = 0.40  # 40 cm up (+Y, toward the windows)


def _tv_table_center():
    ys = []
    for part in (TV_SCREEN, TV_FOOT, TV_STEM):
        ys.extend(p[1] for p in oriented_box_corners(part))
    return (
        TV_CX + TV_TABLE_SHIFT_X,
        min(ys) - TV_TABLE_GAP - TV_TABLE_D / 2.0 + TV_TABLE_SHIFT_Y,
    )


TV_TABLE_CX, TV_TABLE_CY = _tv_table_center()
TV_TABLE = dict(
    cx=TV_TABLE_CX,
    cy=TV_TABLE_CY,
    rx=TV_TABLE_W / 2.0,
    ry=TV_TABLE_D / 2.0,
    x=TV_TABLE_CX - TV_TABLE_W / 2.0,
    y=TV_TABLE_CY - TV_TABLE_D / 2.0,
    w=TV_TABLE_W,
    d=TV_TABLE_D,
    z0=0.0,
    h=TV_TABLE_H,
    kind="table",
)


def ellipse_segments(cx: float, cy: float, rx: float, ry: float, n: int = 48):
    pts = [
        (
            cx + rx * math.cos(2 * math.pi * i / n),
            cy + ry * math.sin(2 * math.pi * i / n),
        )
        for i in range(n)
    ]
    return list(zip(pts, pts[1:] + pts[:1]))


# Desk from K1 (stepped polygon), H=0.75
DESK_K1_H = 0.75
DESK_K1_SPINE = COL1_X
DESK_K1_Y0 = ROOM_Y


def desk_k1_polygon_m():
    x0, y0 = DESK_K1_SPINE, DESK_K1_Y0
    pts = [(x0, y0)]

    def go(dx, dy):
        x, y = pts[-1]
        pts.append((x + dx, y + dy))

    go(0, -1.15)
    go(0.76, 0)
    go(0, -0.82)
    go(-0.76, 0)
    go(0, -0.80)
    go(0.82, 0)
    go(0, -0.82)
    go(-0.82, 0)
    go(0, -0.20)
    go(-2.40, 0)
    go(0, y0 - pts[-1][1])
    return pts


# Two marked pallet bays (rectangles on the plan). EUR 120×80×15.
# South of K2: loaded euro_pallet (boards + boxes), yaw=π/2 so 80 along X.
PALLET_ZONE_S = dict(
    x=COL2_X - COL2_W_BOT - 1.00,
    w=1.00,
    y=0.38,
    d=1.30,
    kind="pallet_zone",
)
_EUR_W, _EUR_D, _EUR_H = 1.20, 0.80, 0.15  # local X, Y, thickness
CARGO_PALLET = dict(  # footprint after yaw=π/2
    x=PALLET_ZONE_S["x"] + (PALLET_ZONE_S["w"] - _EUR_D) / 2.0,
    w=_EUR_D,
    y=PALLET_ZONE_S["y"] + (PALLET_ZONE_S["d"] - _EUR_W) / 2.0,
    d=_EUR_W,
    z0=0.0,
    h=_EUR_H,
    kind="pallet",
    model="euro_pallet",
    yaw=math.pi / 2.0,
)
CARGO_BOX = dict(  # stacked boxes on euro_pallet (collision 1.00×0.70×0.50)
    x=CARGO_PALLET["x"] + (_EUR_D - 0.70) / 2.0,
    w=0.70,
    y=CARGO_PALLET["y"] + (_EUR_W - 1.00) / 2.0,
    d=1.00,
    z0=0.144,
    h=0.50,
    kind="pallet",
)
CARGO_PALLET_POSE = (
    CARGO_PALLET["x"] + CARGO_PALLET["w"] / 2.0,
    CARGO_PALLET["y"] + CARGO_PALLET["d"] / 2.0,
    0.0,
    0.0,
    0.0,
    CARGO_PALLET["yaw"],
)

# North wall, left of K2: pallet ON ITS SIDE. Wall → far face = 40 cm
# (бортики / foam in the gap, then 15 cm pallet thickness). Height = 80 cm.
PALLET_ZONE_N = dict(
    x=COL2_X - 0.15 - _EUR_W,
    w=_EUR_W,
    y=ROOM_Y - 0.40,
    d=0.40,
    kind="pallet_zone",
)
PALLET_WALL = dict(  # wood only, standing on the 120 cm edge
    x=PALLET_ZONE_N["x"],
    w=_EUR_W,
    y=ROOM_Y - 0.40,
    d=_EUR_H,
    z0=0.0,
    h=_EUR_D,
    kind="pallet",
    model="pallet",
)
# model://pallet origin at floor centre of 1.20×0.80; roll +π/2 → 80 cm up, 15 cm in −Y.
PALLET_WALL_POSE = (
    PALLET_WALL["x"] + PALLET_WALL["w"] / 2.0,
    PALLET_WALL["y"] + PALLET_WALL["d"],  # inner face; thickness extends −Y
    _EUR_D / 2.0,
    math.pi / 2.0,
    0.0,
    0.0,
)

# Floor tape around pallet bays (robot moves pallets; tape stays).
TAPE_W = 0.05  # 5 cm
TAPE_H = 0.015

# Empty second bay (north of K2, room floor): pallet may be moved here.
# From K2 north SE: ←7 cm, ↓28 cm, then 100×130 cm.
PALLET_ZONE_E = dict(
    x=COL2_X + COL2_W_TOP - 0.07 - 1.00,
    w=1.00,
    y=ROOM_Y - COL_PROTRUDE - 0.28 - 1.30,
    d=1.30,
    kind="pallet_zone",
)

PALLET_K2 = PALLET_WALL  # legacy
PALLET_WIN131 = PALLET_WALL
ZONE1 = PALLET_ZONE_S
ZONE2 = PALLET_ZONE_E
BODY_WIN132 = CARGO_BOX  # legacy


def pallet_zone_tape_rects():
    """Hollow 5 cm tape frames around both lying-pallet bays."""
    rects = []
    t = TAPE_W
    for z in (PALLET_ZONE_S, PALLET_ZONE_E):
        x, y, w, d = float(z["x"]), float(z["y"]), float(z["w"]), float(z["d"])
        rects.extend(
            [
                dict(x=x, y=y, w=w, d=t, z0=0.0, h=TAPE_H),
                dict(x=x, y=y + d - t, w=w, d=t, z0=0.0, h=TAPE_H),
                dict(x=x, y=y, w=t, d=d, z0=0.0, h=TAPE_H),
                dict(x=x + w - t, y=y, w=t, d=d, z0=0.0, h=TAPE_H),
            ]
        )
    return rects

# Parked cartbots along the north wall: same SDF as ours minus 3C/3D/lidar_top.
# Origin = rear ICR. Robot +X = head, −X = forks. Wall gap 20 cm.
_CART_FORK_TIP = -0.185
_CART_NOSE = 1.745
_CART_HALF_Y = 0.45
_WIN196 = WINDOWS[2]
FL1_WALL_GAP = 0.20
FL1_YAW = 0.0  # head +X (east), forks west under the north shelf
FL1_ORIGIN_X = (_WIN196["x"] + 0.50) - 1.26
FL1_ORIGIN_Y = ROOM_Y - FL1_WALL_GAP - _CART_HALF_Y
FL2_WALL_GAP = 0.20
FL2_YAW = math.pi  # head west, forks east toward the 30 cm cabinets
FL2_FORK_END = SHELF_RIGHT_TOP["x"] - 0.20
FL2_ORIGIN_X = FL2_FORK_END - 0.185
FL2_ORIGIN_Y = ROOM_Y - FL2_WALL_GAP - _CART_HALF_Y


def _cartbot_aabb(ox: float, oy: float, yaw: float) -> dict:
    """Axis-aligned footprint of cartbot_parked in room XY."""
    c, s = math.cos(yaw), math.sin(yaw)
    corners = [
        (_CART_FORK_TIP, -_CART_HALF_Y),
        (_CART_FORK_TIP, _CART_HALF_Y),
        (_CART_NOSE, -_CART_HALF_Y),
        (_CART_NOSE, _CART_HALF_Y),
    ]
    xs, ys = [], []
    for x, y in corners:
        xs.append(ox + c * x - s * y)
        ys.append(oy + s * x + c * y)
    x0, y0 = min(xs), min(ys)
    return dict(x=x0, y=y0, w=max(xs) - x0, d=max(ys) - y0, h=1.12, kind="forklift")


FL1_BODY = _cartbot_aabb(FL1_ORIGIN_X, FL1_ORIGIN_Y, FL1_YAW)
FL2_BODY = _cartbot_aabb(FL2_ORIGIN_X, FL2_ORIGIN_Y, FL2_YAW)
FL1_POSE = (FL1_ORIGIN_X, FL1_ORIGIN_Y, 0.0, 0.0, 0.0, FL1_YAW)
FL2_POSE = (FL2_ORIGIN_X, FL2_ORIGIN_Y, 0.0, 0.0, 0.0, FL2_YAW)
# Legacy tine fields (plan overlay uses the AABB now)
FORK_TINE = 0.17
FORK_GAP = 0.20
FL1_FORK_TIP = FL1_ORIGIN_X + _CART_FORK_TIP
FL1_FORK_LEN = abs(_CART_NOSE - _CART_FORK_TIP)
FL1_FORK_Y0 = FL1_BODY["y"]
FL2_FORK_LEN = FL1_FORK_LEN
FL2_FORK_Y0 = FL2_BODY["y"]


def room_wall_vertices_simple():
    """Outer L-shape without door/window niches (floor footprint)."""
    return [
        (0.0, CUT_H),
        (0.0, ROOM_Y),
        (ROOM_X, ROOM_Y),
        (ROOM_X, 0.0),
        (CUT_W, 0.0),
        (CUT_W, CUT_H),
        (0.0, CUT_H),
    ]


def room_wall_vertices_up():
    """Outer boundary as seen by up lidar @ UP_LIDAR_Z: + window niches + door recess.

    North: 32 cm window niche; glass is transparent so there is no back wall
    in the opening (beam goes to the street). 10 cm outer wall is on piers only.
    South: door recess indents outward (negative Y) by DOOR['recess'].
    """
    verts: list[tuple[float, float]] = [(0.0, CUT_H), (0.0, ROOM_Y)]
    cursor_x = 0.0
    for w in sorted(WINDOWS, key=lambda w: w["x"]):
        wx0, wx1 = float(w["x"]), float(w["x"] + w["w"])
        if wx0 > cursor_x + 1e-9:
            verts.append((wx0, ROOM_Y))
        verts.extend(
            [
                (wx0, ROOM_Y + WIN_DEPTH),
                (wx1, ROOM_Y + WIN_DEPTH),
                (wx1, ROOM_Y),
            ]
        )
        cursor_x = wx1
    if abs(cursor_x - ROOM_X) > 1e-9:
        verts.append((ROOM_X, ROOM_Y))
    verts.append((ROOM_X, 0.0))
    dx0, dx1 = float(DOOR["x"]), float(DOOR["x"] + DOOR["w"])
    rec = float(DOOR["recess"])
    verts.extend(
        [
            (dx1, 0.0),
            (dx1, -rec),
            (dx0, -rec),
            (dx0, 0.0),
            (CUT_W, 0.0),
            (CUT_W, CUT_H),
            (0.0, CUT_H),
        ]
    )
    return verts


def room_wall_vertices_at_z(z: float = UP_LIDAR_Z):
    """Outer boundary as seen by a horizontal lidar at height z.

    - Door recess (floor niche) visible at low and high beams.
    - Window recesses only if beam is at/above sill (WIN_Z0=0.80); below that
      the north wall is continuous under the windows.
    """
    if z >= WIN_Z0 - 1e-6:
        return room_wall_vertices_up()
    # Low beam: L-shape + door recess only (no window niches)
    verts: list[tuple[float, float]] = [
        (0.0, CUT_H),
        (0.0, ROOM_Y),
        (ROOM_X, ROOM_Y),
        (ROOM_X, 0.0),
    ]
    dx0, dx1 = float(DOOR["x"]), float(DOOR["x"] + DOOR["w"])
    rec = float(DOOR["recess"])
    verts.extend(
        [
            (dx1, 0.0),
            (dx1, -rec),
            (dx0, -rec),
            (dx0, 0.0),
            (CUT_W, 0.0),
            (CUT_W, CUT_H),
            (0.0, CUT_H),
        ]
    )
    return verts


def north_pier_ranges():
    """Inner-x ranges of north-wall piers (between / beside windows)."""
    xs = [0.0]
    for w in sorted(WINDOWS, key=lambda w: w["x"]):
        xs.append(float(w["x"]))
        xs.append(float(w["x"] + w["w"]))
    xs.append(ROOM_X)
    out = []
    for i in range(0, len(xs) - 1, 2):
        x0, x1 = xs[i], xs[i + 1]
        if x1 - x0 > 1e-6:
            out.append((x0, x1))
    return out


def window_glass_rects():
    """Glass strips: 5 cm thick after the 32 cm niche (visual only; lidar passes)."""
    return [
        (float(w["x"]), ROOM_Y + WIN_DEPTH, float(w["w"]), WIN_GLASS)
        for w in WINDOWS
    ]


def window_street_wall_rects():
    """10 cm outer wall on piers only — not across window openings."""
    y0 = ROOM_Y + WIN_DEPTH + WIN_GLASS
    return [(x0, y0, x1 - x0, WIN_STREET) for x0, x1 in north_pier_ranges()]


def _is_window_opening_back(p0, p1) -> bool:
    """True if segment is the (transparent) glass line across a window."""
    y_g = ROOM_Y + WIN_DEPTH
    if abs(p0[1] - y_g) > 1e-4 or abs(p1[1] - y_g) > 1e-4:
        return False
    if abs(p0[1] - p1[1]) > 1e-4:
        return False
    xa, xb = sorted((float(p0[0]), float(p1[0])))
    for w in WINDOWS:
        if abs(xa - float(w["x"])) < 1e-3 and abs(xb - float(w["x"] + w["w"])) < 1e-3:
            return True
    return False


def room_wall_segments(*, lidar_up: bool = False, z: float | None = None):
    """Outer boundary segments. lidar_up/z control window/door niches."""
    if z is None:
        z = UP_LIDAR_Z if lidar_up else 0.0
        if not lidar_up:
            verts = room_wall_vertices_simple()
            return list(zip(verts[:-1], verts[1:]))
    verts = room_wall_vertices_at_z(z)
    segs = list(zip(verts[:-1], verts[1:]))
    if z >= WIN_Z0 - 1e-6:
        segs = [s for s in segs if not _is_window_opening_back(s[0], s[1])]
    return segs


def rect_segments(x, y, w, d):
    return [
        ((x, y), (x + w, y)),
        ((x + w, y), (x + w, y + d)),
        ((x + w, y + d), (x, y + d)),
        ((x, y + d), (x, y)),
    ]


def object_z_span(obj: dict) -> tuple[float, float]:
    z0 = float(obj.get("z0", 0.0))
    h = float(obj.get("h", 0.0))
    return z0, z0 + h


def intersects_beam_plane(obj: dict, z: float = UP_LIDAR_Z) -> bool:
    """True if object occupies the horizontal beam plane at height z."""
    z0, z1 = object_z_span(obj)
    return z0 <= z <= z1


# Back-compat alias
intersects_up_plane = intersects_beam_plane


def static_obstacle_rects():
    """Permanent structure footprints (exclude dynamic trash / forklifts / zones)."""
    rects = []
    # columns: south uses bot width, north uses top width
    for cx, wb, wt in (
        (COL1_X, COL1_W_BOT, COL1_W_TOP),
        (COL2_X, COL2_W_BOT, COL2_W_TOP),
    ):
        rects.append((cx, 0.0, wb, COL_PROTRUDE))
        rects.append((cx, ROOM_Y - COL_PROTRUDE, wt, COL_PROTRUDE))
    for r in (
        CAB,
        TABLE_DOOR,
        SHELF_BOT,
        PANEL1,
        PANEL2,
        FOAM_BOT,
        FOAM_TOP,
        TABLE_SW,
        FRIDGE,
        WALL_CAB1,
        WALL_CAB2,
        BOARD,
        *BOARD_FEET,
        SINK,
        DISHWASHER,
        SHELF_TOP,
        *SHELF_RIGHT_BOTS,
        SHELF_RIGHT_TOP,
    ):
        rects.append((r["x"], r["y"], r["w"], r["d"]))
    for h in HEATERS:
        rects.append((h["x"], h["y"], h["w"], h["d"]))
    for r in window_sill_rects():
        rects.append((r["x"], r["y"], r["w"], r["d"]))
    # pipes as small squares
    for p in (PIPE1, PIPE2):
        rects.append((p["x"], 0.0, p["d"], p["d"]))
    return rects


def visible_obstacle_rects_at_z(z: float = UP_LIDAR_Z):
    """Footprints visible to a horizontal lidar at height z.

    Low lidars (~0.135 m) see floor clutter: pallets, trash/spool, foam,
    tables, desk-height furniture, parked forklifts. Щитки hang at z0=0.69
    (same as pipes) so the low beam misses them. Up (@1.60) sees tall
    cabinets/shelves/columns/pipes/panels; misses short tables/foam if tops < z.
    """
    tall = []
    for cx, wb, wt in (
        (COL1_X, COL1_W_BOT, COL1_W_TOP),
        (COL2_X, COL2_W_BOT, COL2_W_TOP),
    ):
        tall.append(
            dict(x=cx, y=0.0, w=wb, d=COL_PROTRUDE, z0=0.0, h=3.0, kind="column")
        )
        tall.append(
            dict(
                x=cx,
                y=ROOM_Y - COL_PROTRUDE,
                w=wt,
                d=COL_PROTRUDE,
                z0=0.0,
                h=3.0,
                kind="column",
            )
        )
    candidates = [
        {**CAB, "kind": "cabinet"},
        {**SHELF_BOT, "kind": "shelf"},
        {**SHELF_TOP, "kind": "shelf"},
        *[{**r, "kind": "shelf"} for r in SHELF_RIGHT_BOTS],
        {**SHELF_RIGHT_TOP, "kind": "shelf"},
        {**WALL_CAB1, "kind": "cabinet"},
        {**WALL_CAB2, "kind": "cabinet"},
        {**PANEL1, "kind": "panel"},
        {**PANEL2, "kind": "panel"},
        {**PIPE1, "y": 0.0, "w": PIPE1["d"], "kind": "pipe"},
        {**PIPE2, "y": 0.0, "w": PIPE2["d"], "kind": "pipe"},
        {**FRIDGE, "kind": "fridge"},
        {**TABLE_DOOR, "kind": "table"},
        {**TABLE_SW, "kind": "table"},
        {**TV_TABLE, "kind": "table"},
        {**FOAM_BOT, "kind": "foam"},
        {**FOAM_TOP, "kind": "foam"},
        {**BOARD, "kind": "board"},
        *[{**s, "kind": "board_leg"} for s in BOARD_STEMS],
        *[{**f, "kind": "board_leg"} for f in BOARD_FEET],
        {**SINK, "kind": "sink"},
        {**DISHWASHER, "kind": "appliance"},
        # floor clutter — low lidars see these
        {**SPOOL, "kind": "trash"},
        {**CARGO_PALLET, "kind": "pallet"},
        {**CARGO_BOX, "kind": "pallet"},
        {**PALLET_WALL, "kind": "pallet"},
        {**FL1_BODY, "kind": "forklift", "z0": 0.0},
        {**FL2_BODY, "kind": "forklift", "z0": 0.0},
        *[{**h, "kind": "heater"} for h in HEATERS],
        *[{**r, "kind": "sill"} for r in window_sill_rects()],
        *[_oriented_as_rect(p) for p in (TV_SCREEN, TV_STEM, TV_FOOT)],
    ]
    # At up height, force tall furniture even if CAB top is slightly under 1.60.
    force_at_up = {"cabinet", "shelf", "panel", "pipe", "column", "board", "tv"}
    # Explicit low-visible kinds (user): pallets, trash, щиток
    force_at_low = {"pallet", "trash", "foam", "forklift", "board_leg", "tv_stand"}
    out = []
    for o in tall + candidates:
        if intersects_beam_plane(o, z):
            out.append((o["x"], o["y"], o["w"], o["d"]))
        elif z >= UP_LIDAR_Z - 1e-6 and o.get("kind") in force_at_up:
            out.append((o["x"], o["y"], o["w"], o["d"]))
        elif z <= LOW_LIDAR_Z + 0.05 and o.get("kind") in force_at_low:
            out.append((o["x"], o["y"], o["w"], o["d"]))
    return out


def invisible_obstacle_rects_at_z(z: float = UP_LIDAR_Z):
    """Objects present in room but missed by beam at z (for dotted overlay)."""
    all_objs = [
        CAB,
        TABLE_DOOR,
        TABLE_SW,
        FOAM_BOT,
        FOAM_TOP,
        BOARD,
        *BOARD_STEMS,
        *BOARD_FEET,
        SINK,
        DISHWASHER,
        FRIDGE,
        PANEL1,
        PANEL2,
        SPOOL,
        CARGO_PALLET,
        CARGO_BOX,
        PALLET_WALL,
        FL1_BODY,
        FL2_BODY,
        *HEATERS,
        *window_sill_rects(),
        _oriented_as_rect(TV_SCREEN),
        _oriented_as_rect(TV_STEM),
        _oriented_as_rect(TV_FOOT),
    ]
    vis = {(round(x, 4), round(y, 4), round(w, 4), round(d, 4)) for x, y, w, d in visible_obstacle_rects_at_z(z)}
    out = []
    for r in all_objs:
        key = (round(r["x"], 4), round(r["y"], 4), round(r["w"], 4), round(r["d"], 4))
        if key not in vis:
            out.append((r["x"], r["y"], r["w"], r["d"]))
    return out


def up_visible_obstacle_rects(z: float = UP_LIDAR_Z):
    return visible_obstacle_rects_at_z(z)


def up_invisible_obstacle_rects(z: float = UP_LIDAR_Z):
    return invisible_obstacle_rects_at_z(z)


def all_gt_segments(*, include_furniture=True, lidar_up: bool = False):
    segs = list(room_wall_segments(lidar_up=lidar_up))
    if include_furniture:
        rects = (
            up_visible_obstacle_rects()
            if lidar_up
            else static_obstacle_rects()
        )
        for x, y, w, d in rects:
            segs.extend(rect_segments(x, y, w, d))
        tv_parts = (TV_SCREEN,) if lidar_up else (TV_SCREEN, TV_STEM, TV_FOOT)
        for part in tv_parts:
            segs.extend(oriented_box_segments(part))
        if lidar_up:
            for x, y, w, d in window_street_wall_rects():
                segs.extend(rect_segments(x, y, w, d))
        # pillar (tall)
        cx, cy, r = PILLAR_SW["cx"], PILLAR_SW["cy"], PILLAR_SW["r"]
        n = 16
        ring = [
            (
                cx + r * math.cos(2 * math.pi * i / n),
                cy + r * math.sin(2 * math.pi * i / n),
            )
            for i in range(n + 1)
        ]
        segs.extend(zip(ring[:-1], ring[1:]))
        if not lidar_up:
            # desk is below up beam — only in full floorplan GT
            desk = desk_k1_polygon_m()
            segs.extend(zip(desk, desk[1:] + desk[:1]))
            segs.extend(
                ellipse_segments(
                    TV_TABLE["cx"], TV_TABLE["cy"], TV_TABLE["rx"], TV_TABLE["ry"]
                )
            )
    return segs


def floorplan_dict():
    return {
        "name": "real_room",
        "frame": "room",
        "units": "m",
        "origin": [0.0, 0.0, 0.0],
        "room_size_m": [ROOM_X, ROOM_Y],
        "cutout_m": [CUT_W, CUT_H],
        "cell_m": CELL_M,
        "measured": "2026-08-13 remeasure; Ox=1646.5 from top chain; "
        "K1 bot/top 52.5/51.5, K2 53/54; pipes Ø12; shelves h~2m; "
        "щитки hang z0=0.69 (as pipes) → ~2.49; up lidar z=1.60m",
        "up_lidar_z_m": UP_LIDAR_Z,
        "low_lidar_z_m": LOW_LIDAR_Z,
        "heights_m": {
            "shelf_bot": SHELF_BOT["h"],
            "shelf_top": SHELF_TOP["h"],
            "shelf_right_bot_m": [
                {
                    "depth": r["w"],
                    "y0": r["y"],
                    "d": r["d"],
                    "h": r["h"],
                }
                for r in SHELF_RIGHT_BOTS
            ],
            "shelf_right_top_m": {
                "depth": SHELF_RIGHT_TOP["w"],
                "y0": SHELF_RIGHT_TOP["y"],
                "d": SHELF_RIGHT_TOP["d"],
                "h": SHELF_RIGHT_TOP["h"],
            },
            "shelf_right": SHELF_RIGHT_BOT["h"],  # legacy
            "panel1": PANEL1["h"],
            "panel2": PANEL2["h"],
            "wall_cab": 2.0,
            "board_panel": {
                "y0": BOARD["y"],
                "d": BOARD["d"],
                "z0": BOARD["z0"],
                "h": BOARD["h"],
                "top_m": BOARD["z0"] + BOARD["h"],
                "wall_x": BOARD_WALL_X,
                "panel_x": BOARD["x"],
                "offset_from_wall_m": BOARD_FOOT_X - BOARD_T,
            },
            "board_leg_h": BOARD_LEG_H,
            "desk_k1": DESK_K1_H,
            "pillar_sw": PILLAR_SW["h"],
            "windows_z0": WIN_Z0,
            "window_niche_m": WIN_DEPTH,
            "window_glass_m": WIN_GLASS,
            "window_street_wall_m": WIN_STREET,
            "window_street_wall": "piers only; openings have no back wall",
            "window_recess": WIN_OUT,
            "door_recess": DOOR["recess"],
            "heaters_z0": 0.20,
            "heaters_h": 0.50,
            "sill_in_m": SILL_IN,
            "sill_t_m": SILL_T,
            "sill_top_m": WIN_Z0,
            "tv": {
                "len_m": TV_LEN,
                "h_m": TV_H,
                "t_m": TV_T,
                "leg_h_m": TV_LEG_H,
                "foot_l_m": TV_FOOT_L,
                "yaw_deg": math.degrees(TV_YAW),
                "edge_gap_m": TV_EDGE_GAP,
                "wall_gap_m": TV_WALL_GAP,
                "cx": TV_CX,
                "cy": TV_CY,
                "z0": TV_LEG_H,
            },
            "tv_table": {
                "w_ox_m": TV_TABLE_W,
                "d_oy_m": TV_TABLE_D,
                "h_m": TV_TABLE_H,
                "gap_m": TV_TABLE_GAP,
                "cx": TV_TABLE_CX,
                "cy": TV_TABLE_CY,
                "shift_x_m": TV_TABLE_SHIFT_X,
                "shift_y_m": TV_TABLE_SHIFT_Y,
            },
            "pallet_cargo_h": CARGO_BOX["z0"] + CARGO_BOX["h"],
            "up_lidar_z": UP_LIDAR_Z,
        },
        "columns": {
            "col1_x": COL1_X,
            "col2_x": COL2_X,
            "w": COL_W,
            "col1_w_top": COL1_W_TOP,
            "col1_w_bot": COL1_W_BOT,
            "col2_w_top": COL2_W_TOP,
            "col2_w_bot": COL2_W_BOT,
            "protrude": COL_PROTRUDE,
        },
        "walls": [[[a[0], a[1]], [b[0], b[1]]] for a, b in room_wall_segments()],
        "walls_up_lidar": [
            [[a[0], a[1]], [b[0], b[1]]] for a, b in room_wall_segments(lidar_up=True)
        ],
        "segments": [
            [[a[0], a[1]], [b[0], b[1]]] for a, b in all_gt_segments(include_furniture=True)
        ],
        "segments_up_lidar": [
            [[a[0], a[1]], [b[0], b[1]]]
            for a, b in all_gt_segments(include_furniture=True, lidar_up=True)
        ],
        "static_rects_m": [
            {"x": x, "y": y, "w": w, "d": d} for x, y, w, d in static_obstacle_rects()
        ],
        "up_visible_rects_m": [
            {"x": x, "y": y, "w": w, "d": d} for x, y, w, d in up_visible_obstacle_rects()
        ],
        "door": DOOR,
        "windows": [
            {
                **w,
                "niche_m": WIN_DEPTH,
                "glass_m": WIN_GLASS,
                "street_wall_m": WIN_STREET,
                "street_wall": "piers only",
                "lidar_through": True,
                "recess": WIN_OUT,
                "z0": WIN_Z0,
                "sill_in_m": SILL_IN,
                "sill_t_m": SILL_T,
            }
            for w in WINDOWS
        ],
        "up_visibility_note": (
            f"up@{UP_LIDAR_Z}m sees: cabinets/shelves, columns, pipes, panels, "
            "window recesses (open through glass), door recess, TV panel; "
            "not low tables/desk/foam/heaters/sills"
        ),
        "desk_k1_polygon_m": [[x, y] for x, y in desk_k1_polygon_m()],
        "pillar_sw": PILLAR_SW,
        "dynamic": {
            "spool_trash": SPOOL,
            "forklift_k1": {
                "pose": list(FL1_POSE),
                "body": FL1_BODY,
                "model": "cartbot_parked",
            },
            "cargo_pallet": CARGO_PALLET,
            "cargo_box": CARGO_BOX,
            "pallet_wall_on_side": PALLET_WALL,
            "forklift_east": {
                "pose": list(FL2_POSE),
                "body": FL2_BODY,
                "model": "cartbot_parked",
            },
            "note": (
                "two pallet bays: south=euro_pallet+boxes; "
                "north wall=pallet on its side, 40 cm from wall"
            ),
        },
        "pallet_zones": {
            "south_k2": PALLET_ZONE_S,
            "north_k2_empty": PALLET_ZONE_E,
            "tape_w_m": TAPE_W,
        },
        "span_gt_m": {"width_x": ROOM_X, "length_y": ROOM_Y},
    }


def _paint_line(grid, x0, y0, x1, y1, origin_x, origin_y, cell, thick=1):
    h, w = grid.shape
    n = max(int(math.hypot(x1 - x0, y1 - y0) / (cell * 0.5)), 1)
    for i in range(n + 1):
        t = i / n
        x = x0 + t * (x1 - x0)
        y = y0 + t * (y1 - y0)
        c = int((x - origin_x) / cell)
        r = int((y - origin_y) / cell)
        for dr in range(-thick, thick + 1):
            for dc in range(-thick, thick + 1):
                rr, cc = r + dr, c + dc
                if 0 <= rr < h and 0 <= cc < w:
                    grid[rr, cc] = 100


def _paint_rect_fill(grid, x, y, rw, rd, origin_x, origin_y, cell):
    h, w = grid.shape
    c0 = max(0, int((x - origin_x) / cell))
    c1 = min(w - 1, int((x + rw - origin_x) / cell))
    r0 = max(0, int((y - origin_y) / cell))
    r1 = min(h - 1, int((y + rd - origin_y) / cell))
    grid[r0 : r1 + 1, c0 : c1 + 1] = 100


def _paint_oriented_fill(grid, obj, origin_x, origin_y, cell):
    """Fill cells whose centres lie inside a yawed box (cx, cy, w, d, yaw)."""
    yaw = float(obj.get("yaw", 0.0))
    c, s = math.cos(yaw), math.sin(yaw)
    hx, hy = obj["w"] / 2.0, obj["d"] / 2.0
    cx, cy = float(obj["cx"]), float(obj["cy"])
    x0, y0, rw, rd = oriented_box_aabb(obj)
    rows, cols = grid.shape
    c0 = max(0, int((x0 - origin_x) / cell))
    c1 = min(cols - 1, int((x0 + rw - origin_x) / cell))
    r0 = max(0, int((y0 - origin_y) / cell))
    r1 = min(rows - 1, int((y0 + rd - origin_y) / cell))
    for rr in range(r0, r1 + 1):
        wy = origin_y + (rr + 0.5) * cell
        for cc in range(c0, c1 + 1):
            wx = origin_x + (cc + 0.5) * cell
            dx, dy = wx - cx, wy - cy
            lx = dx * c + dy * s
            ly = -dx * s + dy * c
            if abs(lx) <= hx and abs(ly) <= hy:
                grid[rr, cc] = 100


def _paint_ellipse_fill(grid, cx, cy, rx, ry, origin_x, origin_y, cell):
    rows, cols = grid.shape
    c0 = max(0, int((cx - rx - origin_x) / cell))
    c1 = min(cols - 1, int((cx + rx - origin_x) / cell))
    r0 = max(0, int((cy - ry - origin_y) / cell))
    r1 = min(rows - 1, int((cy + ry - origin_y) / cell))
    rx2 = rx * rx if rx else 1.0
    ry2 = ry * ry if ry else 1.0
    for rr in range(r0, r1 + 1):
        wy = origin_y + (rr + 0.5) * cell
        dy2 = (wy - cy) * (wy - cy) / ry2
        for cc in range(c0, c1 + 1):
            wx = origin_x + (cc + 0.5) * cell
            if (wx - cx) * (wx - cx) / rx2 + dy2 <= 1.0:
                grid[rr, cc] = 100


def rasterize_occupancy(cell_m: float = CELL_M, pad_m: float = 0.5):
    """Return (grid[row,col], meta). row0 = south (y min). Occupied=100, free=0."""
    origin_x = -pad_m
    origin_y = -pad_m
    width_m = ROOM_X + 2 * pad_m
    height_m = ROOM_Y + WIN_TOTAL + 2 * pad_m
    cols = int(math.ceil(width_m / cell_m))
    rows = int(math.ceil(height_m / cell_m))
    grid = np.zeros((rows, cols), dtype=np.uint8)

    for (x0, y0), (x1, y1) in room_wall_segments(lidar_up=True):
        _paint_line(grid, x0, y0, x1, y1, origin_x, origin_y, cell_m, thick=1)
    # Piers between windows: solid wall through the 32+5+10 cm envelope
    for x0, x1 in north_pier_ranges():
        _paint_rect_fill(
            grid, x0, ROOM_Y, x1 - x0, WIN_TOTAL, origin_x, origin_y, cell_m
        )
    for x, y, w, d in static_obstacle_rects():
        _paint_rect_fill(grid, x, y, w, d, origin_x, origin_y, cell_m)
    for part in (TV_SCREEN, TV_STEM, TV_FOOT):
        _paint_oriented_fill(grid, part, origin_x, origin_y, cell_m)
    _paint_ellipse_fill(
        grid,
        TV_TABLE["cx"],
        TV_TABLE["cy"],
        TV_TABLE["rx"],
        TV_TABLE["ry"],
        origin_x,
        origin_y,
        cell_m,
    )
    # pillar
    cx, cy, r = PILLAR_SW["cx"], PILLAR_SW["cy"], PILLAR_SW["r"]
    for a, b in zip(
        [
            (cx + r * math.cos(2 * math.pi * i / 24), cy + r * math.sin(2 * math.pi * i / 24))
            for i in range(24)
        ],
        [
            (cx + r * math.cos(2 * math.pi * i / 24), cy + r * math.sin(2 * math.pi * i / 24))
            for i in range(1, 25)
        ],
    ):
        _paint_line(grid, a[0], a[1], b[0], b[1], origin_x, origin_y, cell_m, thick=1)
    _paint_rect_fill(grid, cx - r, cy - r, 2 * r, 2 * r, origin_x, origin_y, cell_m)
    # desk fill via polygon bbox edges only (outline)
    desk = desk_k1_polygon_m()
    for a, b in zip(desk, desk[1:] + desk[:1]):
        _paint_line(grid, a[0], a[1], b[0], b[1], origin_x, origin_y, cell_m, thick=1)
    for r in pallet_zone_tape_rects():
        _paint_rect_fill(
            grid, r["x"], r["y"], r["w"], r["d"], origin_x, origin_y, cell_m
        )
    for r in (FL1_BODY, FL2_BODY, SPOOL):
        _paint_rect_fill(
            grid, r["x"], r["y"], r["w"], r["d"], origin_x, origin_y, cell_m
        )

    meta = {
        "cell_m": cell_m,
        "origin": [origin_x, origin_y, 0.0],
        "width_cells": cols,
        "height_cells": rows,
        "width_m": cols * cell_m,
        "height_m": rows * cell_m,
        "occupied_value": 100,
        "free_value": 0,
        "row0": "south",
    }
    return grid, meta


def grid_to_sparse_cells(grid, meta):
    cell = meta["cell_m"]
    ox, oy = meta["origin"][0], meta["origin"][1]
    rr, cc = np.where(grid == 100)
    xs = ox + (cc + 0.5) * cell
    ys = oy + (rr + 0.5) * cell
    return np.column_stack([xs, ys])


def save_gt_bundle(out_dir: Path, cell_m: float = CELL_M):
    """Write floorplan.json, occupancy PNG+YAML, sparse cells, annotated preview."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    data = floorplan_dict()
    (out_dir / "floorplan.json").write_text(
        json.dumps(data, indent=2) + "\n", encoding="utf-8"
    )

    grid, meta = rasterize_occupancy(cell_m=cell_m)
    cells = grid_to_sparse_cells(grid, meta)
    (out_dir / "floorplan_cells.json").write_text(
        json.dumps(
            {
                "map": cells.tolist(),
                "cell_m": cell_m,
                "origin": meta["origin"],
                "n_cells": int(len(cells)),
                "source": "hand_measured_real_room",
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    # Occupancy PNG (ROS-like): black=occupied, white=free; flip UD so row0 visually top
    # YAML origin = lower-left of map; image saved with north at top for humans.
    try:
        from PIL import Image
    except ImportError:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        img = np.where(grid == 100, 0, 255).astype(np.uint8)
        img = np.flipud(img)
        plt.imsave(out_dir / "floorplan.png", img, cmap="gray", vmin=0, vmax=255)
    else:
        img = np.where(grid == 100, 0, 255).astype(np.uint8)
        img = np.flipud(img)
        Image.fromarray(img, mode="L").save(out_dir / "floorplan.png")

    yaml_text = (
        f"image: floorplan.png\n"
        f"resolution: {cell_m}\n"
        f"origin: [{meta['origin'][0]}, {meta['origin'][1]}, 0.0]\n"
        f"negate: 0\n"
        f"occupied_thresh: 0.65\n"
        f"free_thresh: 0.25\n"
        f"mode: trinary\n"
    )
    (out_dir / "floorplan.yaml").write_text(yaml_text, encoding="utf-8")
    (out_dir / "floorplan_meta.json").write_text(
        json.dumps(meta, indent=2) + "\n", encoding="utf-8"
    )

    # Annotated preview (keeps metric scale annotation)
    _save_annotated_png(out_dir / "floorplan_annotated.png", cells)

    return {
        "dir": str(out_dir),
        "json": str(out_dir / "floorplan.json"),
        "png": str(out_dir / "floorplan.png"),
        "yaml": str(out_dir / "floorplan.yaml"),
        "cells": str(out_dir / "floorplan_cells.json"),
        "annotated": str(out_dir / "floorplan_annotated.png"),
        "n_cells": int(len(cells)),
        "cell_m": cell_m,
        "size_m": [ROOM_X, ROOM_Y],
    }


def _save_annotated_png(path: Path, cells: np.ndarray):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle, Ellipse, Polygon, Rectangle

    fig, ax = plt.subplots(figsize=(ROOM_X * 0.55, ROOM_Y * 0.55 + 1.5), dpi=120)
    ax.set_aspect("equal")
    ax.set_xlim(-0.6, ROOM_X + 0.6)
    ax.set_ylim(-DOOR["recess"] - 0.5, ROOM_Y + WIN_TOTAL + 0.4)
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_title(
        f"Real room GT  {ROOM_X:.2f}×{ROOM_Y:.2f} m  |  up lidar @ {UP_LIDAR_Z:.2f} m\n"
        f"solid grey = visible at beam; dotted = below; "
        f"win = {int(WIN_DEPTH*100)} cm niche + {int(WIN_GLASS*100)} cm glass "
        f"(lidar through); {int(WIN_STREET*100)} cm wall on piers only"
    )

    # simple floor outline (light)
    simple = np.asarray(room_wall_vertices_simple())
    ax.add_patch(
        Polygon(simple, closed=True, fill=False, edgecolor="#aaaaaa", linewidth=1.0, ls=":")
    )
    # up-lidar outline: door recess + window jambs; openings have no back wall
    for (x0, y0), (x1, y1) in room_wall_segments(lidar_up=True):
        ax.plot([x0, x1], [y0, y1], color="black", linewidth=2.2, zorder=4)
    ax.plot([], [], color="black", linewidth=2.2, label=f"walls @ up {UP_LIDAR_Z:.2f}m (+openings)")
    ax.add_patch(
        Rectangle(
            (0, 0), CUT_W, CUT_H, fill=True, facecolor="#dddddd", edgecolor="gray", ls="--"
        )
    )

    # low / invisible at up
    for x, y, w, d in up_invisible_obstacle_rects():
        ax.add_patch(
            Rectangle(
                (x, y), w, d, fill=False, edgecolor="#999999", linestyle=":", linewidth=1.0
            )
        )
    # desk below beam
    ax.add_patch(
        Polygon(
            desk_k1_polygon_m(),
            closed=True,
            fill=False,
            edgecolor="#999999",
            linestyle=":",
            linewidth=1.0,
        )
    )
    ax.add_patch(
        Ellipse(
            (TV_TABLE["cx"], TV_TABLE["cy"]),
            TV_TABLE["w"],
            TV_TABLE["d"],
            fill=False,
            edgecolor="#999999",
            linestyle=":",
            linewidth=1.0,
        )
    )
    # visible at up
    for x, y, w, d in up_visible_obstacle_rects():
        ax.add_patch(
            Rectangle(
                (x, y), w, d, fill=True, facecolor="#666666", edgecolor="black", alpha=0.45
            )
        )
    ax.add_patch(
        Polygon(
            oriented_box_corners(TV_SCREEN),
            closed=True,
            fill=True,
            facecolor="#222222",
            edgecolor="black",
            alpha=0.85,
            zorder=5,
        )
    )
    ax.text(
        TV_CX,
        TV_CY,
        "TV",
        fontsize=7,
        color="white",
        ha="center",
        va="center",
        zorder=6,
    )
    ax.add_patch(
        Circle(
            (PILLAR_SW["cx"], PILLAR_SW["cy"]),
            PILLAR_SW["r"],
            fill=True,
            facecolor="#444444",
            edgecolor="black",
        )
    )

    def _dyn_rect(x, y, w, d, label=None, lx=None, ly=None):
        ax.add_patch(
            Rectangle(
                (x, y),
                w,
                d,
                fill=False,
                edgecolor="red",
                linestyle="--",
                linewidth=2,
            )
        )
        if label:
            ax.text(
                lx if lx is not None else x + w / 2,
                ly if ly is not None else y + d / 2,
                label,
                fontsize=7,
                color="red",
                ha="center",
                va="center",
            )

    # Pallet bays: dashed tape (pallets move). South occupied, north empty.
    for z, name, ls in (
        (PALLET_ZONE_S, "скотч Ю (поддон)", "--"),
        (PALLET_ZONE_E, "скотч С (пусто)", "--"),
    ):
        ax.add_patch(
            Rectangle(
                (z["x"], z["y"]),
                z["w"],
                z["d"],
                fill=False,
                edgecolor="tab:orange",
                linestyle=ls,
                linewidth=2.2,
            )
        )
        ax.text(
            z["x"] + z["w"] / 2,
            z["y"] + z["d"] / 2,
            name,
            fontsize=7,
            color="tab:orange",
            ha="center",
            va="center",
        )

    # dynamic clutter: loaded pallet (south) + wall pallet on side + parked cartbots
    _dyn_rect(SPOOL["x"], SPOOL["y"], SPOOL["w"], SPOOL["d"], "катушка")
    _dyn_rect(FL1_BODY["x"], FL1_BODY["y"], FL1_BODY["w"], FL1_BODY["d"], "FL1")
    _dyn_rect(
        CARGO_PALLET["x"],
        CARGO_PALLET["y"],
        CARGO_PALLET["w"],
        CARGO_PALLET["d"],
        "поддон+коробки",
        ly=CARGO_PALLET["y"] - 0.12,
    )
    _dyn_rect(
        PALLET_WALL["x"],
        PALLET_WALL["y"],
        PALLET_WALL["w"],
        PALLET_WALL["d"],
        "поддон на боку",
        ly=PALLET_WALL["y"] - 0.12,
    )
    _dyn_rect(FL2_BODY["x"], FL2_BODY["y"], FL2_BODY["w"], FL2_BODY["d"], "FL2")

    if len(cells):
        ax.scatter(cells[::3, 0], cells[::3, 1], s=1, c="black", alpha=0.15, zorder=0)

    ax.text(0.05, -DOOR["recess"] - 0.25, "origin (0,0)", fontsize=8, color="tab:red")
    ax.plot(0, 0, "r+", markersize=12)
    ax.text(
        DOOR["x"] + DOOR["w"] / 2,
        -DOOR["recess"] / 2,
        "door recess",
        fontsize=7,
        ha="center",
        color="black",
    )
    for x, y, w, d in window_street_wall_rects():
        ax.add_patch(
            Rectangle(
                (x, y),
                w,
                d,
                facecolor="#b0b4b8",
                edgecolor="#4a4e52",
                linewidth=1.0,
                alpha=0.85,
                zorder=2,
            )
        )
    ax.plot([], [], color="#b0b4b8", linewidth=8, label="street wall 10 cm (piers)")
    for w in WINDOWS:
        ax.add_patch(
            Rectangle(
                (w["x"], ROOM_Y - SILL_IN),
                w["w"],
                SILL_IN + WIN_DEPTH,
                facecolor="#c4b49a",
                edgecolor="#6e5c3e",
                linewidth=0.6,
                alpha=0.7,
                zorder=3,
            )
        )
        ax.add_patch(
            Rectangle(
                (w["x"], ROOM_Y),
                w["w"],
                WIN_TOTAL,
                facecolor="#cfefff",
                edgecolor="#1a6f8a",
                linewidth=0.6,
                linestyle="--",
                alpha=0.25,
                zorder=1,
            )
        )
        ax.add_patch(
            Rectangle(
                (w["x"], ROOM_Y + WIN_DEPTH),
                w["w"],
                WIN_GLASS,
                facecolor="#7ec8e3",
                edgecolor="#1a6f8a",
                linewidth=0.8,
                linestyle="--",
                alpha=0.45,
                zorder=3,
            )
        )
        ax.text(
            w["x"] + w["w"] / 2,
            ROOM_Y + WIN_DEPTH / 2,
            f"win{w['label']}",
            fontsize=6,
            ha="center",
            va="center",
            color="black",
        )
    ax.grid(True, alpha=0.25)
    ax.legend(loc="lower right", fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


DEFAULT_GT_DIR = Path(__file__).resolve().parents[2] / "maps" / "gt" / "real_room"
