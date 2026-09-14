#!/usr/bin/env python3
"""Regenerate docs/handler_bot_*.png.

Origin = midpoint between rear fork wheels (rear ICR).
Body / boards / up-lidar frame: measured layout (not the old SDF mast).
"""
from __future__ import annotations

import math
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Polygon, Rectangle, Wedge
from matplotlib.transforms import Affine2D

DOC = Path(__file__).resolve().parent

# --- forks / wheels ---
FORK_LEN = 1.14
FORK_WIDTH = 0.17          # 17 cm each
FORK_INNER = 0.20          # 20 cm between forks
FORK_Y = FORK_INNER / 2 + FORK_WIDTH / 2
FORK = dict(x=-0.83, y=FORK_Y, z=0.08, size=(FORK_LEN, FORK_WIDTH, 0.06))
FORK_R, FORK_WW = 0.04, 0.05  # Ø8 cm
WHEEL_GAP = 0.01             # 1 cm between F/B wheels (rim to rim)
FORK_W_CTC = 2 * FORK_R + WHEEL_GAP
REAR_AXLE = -1.235
OX = REAR_AXLE
# Load-wheel pocket in the tines (model x = ±POCKET_HALF around origin).
POCKET_HALF = 0.10
SLOT_W = 0.08
SLOT_WALL = (FORK_WIDTH - SLOT_W) / 2
FORK_BOTTOM = FORK["z"] - FORK["size"][2] / 2  # 5 cm off the floor
_xw_b = REAR_AXLE - FORK_W_CTC / 2
_xw_f = REAR_AXLE + FORK_W_CTC / 2
FORK_W = [
    (_xw_b, FORK_Y, FORK_R),
    (_xw_f, FORK_Y, FORK_R),
    (_xw_b, -FORK_Y, FORK_R),
    (_xw_f, -FORK_Y, FORK_R),
]
# Pallet-jack lift: forks rest 5 cm off the floor, rise 10–20 cm (use 15).
# Load wheels sit in tine slots; a chassis (основа) links them to corpus 6.
FORK_LIFT_H = 0.15
ORIGIN_TO_BODY_DOWN = 0.975  # m, measured forks down
ORIGIN_TO_BODY_UP = 0.945    # m, measured forks up (15 cm lift)
C_BOGIE = "#37474f"

# --- body (user measurements) ---
BODY_LEN = 0.64
BODY_W = 0.73
BODY_H = 0.80
CLEAR = 0.025
SIDE_W = 0.085
FRONT_L = 0.18
# 7A/7B / 8: top at 10 cm so dock lidars sit on the corpus, not inside it.
BOARD_TOP = 0.10
SIDE_H = BOARD_TOP - CLEAR  # 7.5 cm (was 10 cm from CLEAR → top 12.5 cm)

# fork tip → body rear = 114 cm; body rear from measured origin→body (forks down)
FORK_TIP = FORK["x"] - FORK["size"][0] / 2
BODY_REAR = OX + ORIGIN_TO_BODY_DOWN
BODY_FRONT = BODY_REAR + BODY_LEN
BOGIE_SLIDE = ORIGIN_TO_BODY_DOWN - ORIGIN_TO_BODY_UP  # 3 cm toward body when up
NOSE_FRONT = BODY_FRONT + FRONT_L
OUTER_W = BODY_W + 2 * SIDE_W
OUTER_HALF = OUTER_W / 2
DECK_C = ((BODY_REAR + BODY_FRONT) / 2, 0.0, CLEAR + BODY_H / 2)
DECK_SIZE = (BODY_LEN, BODY_W, BODY_H)
SIDE_LEN = BODY_LEN
SIDE_XC = (BODY_REAR + BODY_FRONT) / 2
SIDE_YC = BODY_W / 2 + SIDE_W / 2
FRONT_C = ((BODY_FRONT + NOSE_FRONT) / 2, 0.0, CLEAR + SIDE_H / 2)
FRONT_SIZE = (FRONT_L, OUTER_W, SIDE_H)

# --- up frame ---
# 1A/1B: 70 cm posts on the OUTSIDE of the 73 cm body.
# From the left (rear) face of the body they sit 33.5 cm to the right (+X / head).
# 2A/2B: 30 cm arms toward forks (−X); 3: crossbar; 3C: box on 3.
TUBE_W = 0.04
TUBE_Z0 = 0.415
TUBE_H = 0.70
TUBE_ARM = 0.30
POST_DX = 0.335
TUBE_Y = BODY_W / 2 + TUBE_W / 2
TUBE_X = BODY_REAR + POST_DX
ARM_END_X = TUBE_X - TUBE_ARM
TUBE_Z1 = TUBE_Z0 + TUBE_H
# 3C on 2A/2B: bottom-left corner = top-left corner of 2A/2B, then 19 cm to the right
BOX3_SX = 0.19
BOX3_SY = 0.71
BOX3_SZ = 0.15
BOX3_X0 = ARM_END_X - TUBE_W / 2
BOX3_X1 = BOX3_X0 + BOX3_SX
BOX3_Z0 = TUBE_Z1
BOX3_Z1 = TUBE_Z1 + BOX3_SZ
# 3D: hoop left of 3C. Its top surface is z=159.5 cm.
FRAME3_H = 0.48
FRAME3_Z0 = BOX3_Z0
FRAME3_Z1 = BOX3_Z0 + FRAME3_H
FRAME3_X1 = BOX3_X0
FRAME3_X0 = BOX3_X0 - TUBE_W
C_TAG = "#1a1a1a"
C_BOX = "#5d4037"
C_FRAME = "#455a64"

DRIVE_R, DRIVE_W = 0.10, 0.08
DRIVE_FROM_ORIGIN = 1.41   # measured rear-axle origin -> driven-wheel centre
CENTER_R, CENTER_W = 0.05, 0.09
CENTER_Y = 0.34
LIDAR_S = 0.05              # 5×5 cm footprint
LIDAR_R = LIDAR_S / 2       # Ø 5 cm cylinder
LIDAR_H = 0.03              # 3 cm tall
LIDAR_STAND_H = 0.02        # 5×5×2 cm stand under each lidar
PANDAR_D = 0.118             # Pandar40P outer diameter (nominal Ø118 mm)
PANDAR_R = PANDAR_D / 2
PANDAR_H = 0.1167            # body height
PANDAR_STAND_S = 0.08        # simulated adapter plate 8×8 cm
PANDAR_STAND_H = 0.02

# Source geometry above is expressed in the historical SDF coordinate system;
# OX is the rear-wheel midpoint in that system.  Keep the measured wheelbase
# explicit instead of deriving it from a body-edge clearance.
DRIVE = (OX + DRIVE_FROM_ORIGIN, 0.0, DRIVE_R)
DRIVE_GAP = BODY_FRONT - DRIVE[0]
CENTER = [
    (DRIVE[0], CENTER_Y, CENTER_R),
    (DRIVE[0], -CENTER_Y, CENTER_R),
]

# 1C tiller (pallet-jack architecture): stem + U-handle toward the head.
# Bottom sits on top of corpus 6. Dark gray, not darker than the side boards.
BODY_TOP = CLEAR + BODY_H
HANDLE_SZ = 0.50
STEM_W = 0.08
TILLER_HALF_Y = 0.10
TILLER_DX = 0.12
TILLER_T = 0.06
TILLER_PITCH = 0.26          # ~15°, top leans toward +X / operator
JOY_BASE_R = 0.08
HANDLE_X_STEM = TUBE_X
HANDLE_Y = 0.0
HANDLE_Z0 = BODY_TOP
HANDLE_Z1 = HANDLE_Z0 + HANDLE_SZ
C_HANDLE = "#6e747a"

# Cameras inside 3C; Ø6 cm holes in the +X face (two) and +Z face (one).
# Ø 5.5 cm bodies; 2.5 cm from the Y-edges of 3C to the camera rims.
# Left + right (+X) are the stereo pair; view_cameras.sh stereo stitches them.
CAM_D = 0.055
CAM_R = CAM_D / 2
CAM_EDGE = 0.025
CAM_BODY = 0.03              # length along view axis, flush with the hole
CAM_HOLE = 0.06              # aperture diameter
CAM_HOLE_R = CAM_HOLE / 2
CAM_X = BOX3_X1 - CAM_BODY / 2
CAM_Y = BOX3_SY / 2 - CAM_EDGE - CAM_R
CAM_Z = (BOX3_Z0 + BOX3_Z1) / 2
CAMS = (
    (CAM_X, CAM_Y, CAM_Z),
    (CAM_X, -CAM_Y, CAM_Z),
)
# Third camera inside 3C, looking +Z. Center from 3C XY bottom-left
# (min X, min Y) = +12 cm Ox, +7 cm Oy.
CAM_UP_X = BOX3_X0 + 0.12
CAM_UP_Y = -BOX3_SY / 2 + 0.07
CAM_UP_Z = BOX3_Z1 - CAM_BODY / 2
C_CAM = "#0d47a1"
C_HOLE = "#f4f1ee"

# dock on front-board corners; stand sits on corpus/board top (floor → bottom = 10 cm)
DOCK_FROM_ORIGIN = 1.71
BODY_CARRIAGE_GAP = 0.055   # m, rear corpus face → fork carriage
# SDF fork tip x; tip→4A Ox = 97.5 cm with dock lidars at 1.71 m
FORK_TIP_SDF = DOCK_FROM_ORIGIN - ORIGIN_TO_BODY_DOWN
L_Z = BOARD_TOP + LIDAR_STAND_H + LIDAR_H / 2  # cylinder center
L_LEFT = (OX + DOCK_FROM_ORIGIN, OUTER_HALF - 0.022, L_Z)
L_RIGHT = (OX + DOCK_FROM_ORIGIN, -(OUTER_HALF - 0.022), L_Z)
L_UP = (
    (FRAME3_X0 + FRAME3_X1) / 2,
    0.0,
    FRAME3_Z1 + PANDAR_STAND_H + PANDAR_H / 2,
)
L_UP_BOTTOM = FRAME3_Z1 + PANDAR_STAND_H
L_UP_TOP = L_UP_BOTTOM + PANDAR_H

C_BODY = "#c5c8cc"
C_EDGE = "#4a4e52"
C_FORK = "#d4a017"
C_BUMP = "#5c6166"
C_MAST = "#8a9096"
C_UP = "#2e7d32"
C_LEFT = "#1565c0"
C_RIGHT = "#6a1b9a"
C_DRIVE = "#c62828"
C_CENTER = "#424242"
C_FW = "#111111"
C_DIM = "#333333"


def X(x):
    """SDF / base_link x → plot x, origin between rear wheels."""
    return x - OX


def cm(m):
    v = m * 100.0
    if abs(v - round(v)) < 0.05:
        return f"{int(round(v))}"
    return f"{v:.1f}".replace(".", ",")


def plot_x(sdf_x: float) -> float:
    """SDF / base_link x → plot x (origin at rear wheels)."""
    return sdf_x


def fork_carriage_x() -> float:
    return X(BODY_REAR) - BODY_CARRIAGE_GAP


def tip_to_dock_x() -> float:
    return DOCK_FROM_ORIGIN - FORK_TIP_SDF


def camera_sdf_positions():
    """Camera centres in SDF base_link frame (matches model.sdf)."""
    return {
        "left": (1.165, CAM_Y, CAM_Z),
        "right": (1.165, -CAM_Y, CAM_Z),
        "up": (1.110, CAM_UP_Y, CAM_UP_Z),
    }


def draw_camera_coord_dims_xy(ax):
    """Camera XY offsets from origin and 3C corner."""
    for name, (sx, sy, _sz) in camera_sdf_positions().items():
        px, py = sx, sy
        ax.plot([0, px], [py, py], ":", color=C_CAM, lw=0.45, zorder=5)
        ax.plot([px, px], [0, py], ":", color=C_CAM, lw=0.45, zorder=5)
        ax.text(
            px / 2, py + (0.04 if sy > 0 else -0.06),
            f"cam {name} x={cm(sx)} y={cm(abs(sy))}",
            fontsize=6, color=C_CAM, ha="center", zorder=14,
            bbox=dict(boxstyle="round,pad=0.1", facecolor="white", alpha=0.85),
        )
    dim(
        ax, X(BOX3_X0), X(BOX3_X0 + 0.12),
        f"cam↑ Ox от 3C {cm(0.12)}",
        offset=BOX3_SY / 2 + 0.22, fs=6, color=C_CAM,
    )
    dim(
        ax, -BOX3_SY / 2, CAM_UP_Y,
        f"cam↑ Oy {cm(CAM_UP_Y + BOX3_SY / 2)}",
        horizontal=False, offset=X(BOX3_X1) + 0.28, fs=6, color=C_CAM,
    )


def draw_camera_coord_dims_xz(ax):
    for name, (sx, _sy, sz) in camera_sdf_positions().items():
        px = sx
        dim(
            ax, 0.0, sz, f"cam {name} z={cm(sz)}",
            horizontal=False, offset=px + (0.10 if name != "up" else -0.12),
            fs=6, color=C_CAM,
        )
    dim(
        ax, X(BOX3_X0), X(CAM_X),
        f"cam L/R x={cm(CAM_X - BOX3_X0)} от 3C",
        offset=BOX3_Z1 + 0.24, fs=6, color=C_CAM,
    )
    dim(
        ax, BOX3_Z0, BOX3_Z1, f"3C h {cm(BOX3_SZ)}",
        horizontal=False, offset=X(BOX3_X1) + 0.10, fs=6, color=C_BOX,
    )


def draw_lidar_coord_dims_xy(ax):
    dim(
        ax, FORK_TIP_SDF, DOCK_FROM_ORIGIN,
        f"{cm(tip_to_dock_x())} tip→4A/4B Ox",
        offset=0.58, fs=7, color=C_LEFT,
    )
    dim(
        ax, X(BODY_REAR), fork_carriage_x(),
        f"{cm(BODY_CARRIAGE_GAP)} корпус→carriage",
        offset=-0.42, fs=7, color=C_BOGIE,
    )
    dim(
        ax, 0.0, L_LEFT[1], f"4A y={cm(L_LEFT[1])}",
        horizontal=False, offset=X(OX + DOCK_FROM_ORIGIN) + 0.14,
        fs=7, color=C_LEFT,
    )
    dim(
        ax, 0.0, X(OX + DOCK_FROM_ORIGIN),
        f"4A x={cm(DOCK_FROM_ORIGIN)}",
        offset=0.48, fs=7, color=C_LEFT,
    )
    dim(
        ax, 0.0, X(L_UP[0]),
        f"5 x={cm(X(L_UP[0]))}",
        offset=0.34, fs=7, color=C_UP,
    )


def draw_lidar_coord_dims_xz(ax):
    dim(
        ax, 0.0, L_LEFT[2], f"4A z={cm(L_LEFT[2])}",
        horizontal=False, offset=X(OX + DOCK_FROM_ORIGIN) + 0.10,
        fs=7, color=C_LEFT,
    )
    dim(
        ax, 0.0, BOARD_TOP, f"низ 4A z={cm(BOARD_TOP)}",
        horizontal=False, offset=X(OX + DOCK_FROM_ORIGIN) - 0.10,
        fs=6, color=C_LEFT,
    )
    dim(
        ax, 0.0, L_UP[2], f"5 z={cm(L_UP[2])}",
        horizontal=False, offset=X(L_UP[0]) + 0.14, fs=7, color=C_UP,
    )
    dim(
        ax, 0.0, X(L_UP[0]),
        f"5 x={cm(X(L_UP[0]))}",
        offset=L_UP[2] + 0.12, fs=7, color=C_UP,
    )


def draw_board_body_dims_xy(ax):
    dim(
        ax, X(BODY_REAR), X(BODY_FRONT),
        f"{cm(BODY_LEN)} корпус 6",
        offset=-0.54, fs=8,
    )
    dim(
        ax, X(BODY_FRONT), X(NOSE_FRONT),
        f"{cm(FRONT_L)} борт 8",
        offset=-0.38, fs=7, color=C_BUMP,
    )
    dim(
        ax, BODY_W / 2, OUTER_HALF,
        f"{cm(SIDE_W)} борт 7",
        horizontal=False, offset=X(NOSE_FRONT) + 0.20, fs=7, color=C_BUMP,
    )
    dim(
        ax, X(FORK_TIP), X(BODY_REAR),
        f"{cm(FORK_LEN)} вилы",
        offset=-0.68, fs=8, color=C_FORK,
    )
def fork_tip():
    return FORK_TIP


def deck_head():
    return BODY_FRONT


def bump_head():
    return NOSE_FRONT


def box_xy(c, size):
    sx, sy, _sz = size
    return X(c[0]) - sx / 2, c[1] - sy / 2, sx, sy


def box_xz(c, size):
    sx, _sy, sz = size
    return X(c[0]) - sx / 2, c[2] - sz / 2, sx, sz


def box_yz(c, size):
    _sx, sy, sz = size
    return c[1] - sy / 2, c[2] - sz / 2, sy, sz


def body_silhouette_xy():
    """Deck + side boards + front board (blocks dock lidars)."""
    x0, x1 = X(BODY_REAR), X(NOSE_FRONT)
    y0, y1 = -OUTER_HALF, OUTER_HALF
    return [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]


def draw_dock_fov(ax, lidar, color, radius=2.0):
    """Corner dock lidar: corpus+boards occupy the inner 90°, visible 270° out.

    Lidars sit on the front-board corners; do not treat them as inside the
    silhouette (that used to collapse the wedge).
    """
    _lx, ly = X(lidar[0]), lidar[1]
    if ly >= 0:
        # left (+Y): open from front (−Y edge) CCW through +X/+Y to the side (−X)
        theta1, theta2 = -90.0, 180.0
    else:
        # right (−Y): open from the side (−X) CCW through −Y/+X to the front (+Y)
        theta1, theta2 = 180.0, 90.0 + 360.0
    ax.add_patch(Wedge(
        (X(lidar[0]), ly), radius, theta1, theta2,
        facecolor=color, alpha=0.10, edgecolor=color, lw=0.4, zorder=1,
    ))


def draw_lidar_side(ax, u, z_c, color):
    """Cylinder Ø5×3 cm on a 5×5×2 stand. u is x or y."""
    stand_z0 = z_c - LIDAR_H / 2 - LIDAR_STAND_H
    ax.add_patch(Rectangle(
        (u - LIDAR_S / 2, stand_z0), LIDAR_S, LIDAR_STAND_H,
        facecolor="#3a3d40", edgecolor=C_EDGE, lw=0.4, zorder=6,
    ))
    ax.add_patch(Circle(
        (u, z_c), LIDAR_R, facecolor=color, edgecolor="white", lw=0.5, zorder=7,
    ))


def draw_pandar_side(ax, u, z_c):
    """Pandar40P body on its adapter plate, standing on frame 3D."""
    stand_z0 = z_c - PANDAR_H / 2 - PANDAR_STAND_H
    ax.add_patch(Rectangle(
        (u - PANDAR_STAND_S / 2, stand_z0),
        PANDAR_STAND_S, PANDAR_STAND_H,
        facecolor="#3a3d40", edgecolor=C_EDGE, lw=0.5, zorder=8,
    ))
    ax.add_patch(Rectangle(
        (u - PANDAR_R, z_c - PANDAR_H / 2), PANDAR_D, PANDAR_H,
        facecolor=C_UP, edgecolor="#163d1a", lw=0.7, zorder=9,
    ))
    ax.add_patch(Rectangle(
        (u - PANDAR_R, z_c - 0.020), PANDAR_D, 0.040,
        facecolor="#d29a28", edgecolor="#6b4d12", lw=0.4, zorder=10,
    ))


def dim(ax, a, b, text, horizontal=True, offset=0.06, color=C_DIM, fs=8):
    bbox = dict(boxstyle="round,pad=0.12", facecolor="white", edgecolor="none", alpha=0.9)
    if horizontal:
        y = offset
        ax.annotate(
            "", xy=(b, y), xytext=(a, y),
            arrowprops=dict(arrowstyle="<->", color=color, lw=0.9),
        )
        ax.plot([a, a], [y - 0.02, y + 0.02], color=color, lw=0.7, clip_on=False)
        ax.plot([b, b], [y - 0.02, y + 0.02], color=color, lw=0.7, clip_on=False)
        ax.text(
            (a + b) / 2, y + 0.012, text, ha="center", va="bottom",
            fontsize=fs, color=color, clip_on=False, zorder=14, bbox=bbox,
        )
    else:
        x = offset
        ax.annotate(
            "", xy=(x, b), xytext=(x, a),
            arrowprops=dict(arrowstyle="<->", color=color, lw=0.9),
        )
        ax.plot([x - 0.02, x + 0.02], [a, a], color=color, lw=0.7, clip_on=False)
        ax.plot([x - 0.02, x + 0.02], [b, b], color=color, lw=0.7, clip_on=False)
        ax.text(
            x + 0.025, (a + b) / 2, text, ha="left", va="center",
            fontsize=fs, color=color, rotation=90, clip_on=False, zorder=14, bbox=bbox,
        )


def dim_drive_to_body(ax, y_dim=-0.56):
    """14 cm / 4 cm to corpus 6, below 1C so the handle does not cover the dim."""
    x0 = X(DRIVE[0])
    x1 = X(BODY_FRONT)
    xr = x0 + DRIVE_R
    y_obj = -BODY_W / 2 - 0.04
    dim(ax, x0, x1, f"{cm(DRIVE_GAP)} центр колеса → корпус 6",
        offset=y_dim, fs=8, color=C_DRIVE)
    dim(ax, xr, x1, f"{cm(BODY_FRONT - (DRIVE[0] + DRIVE_R))} обод → 6",
        offset=y_dim - 0.16, fs=7, color=C_DRIVE)
    for x in (x0, xr, x1):
        ax.plot(
            [x, x], [y_obj, y_dim - 0.16],
            color=C_DRIVE, lw=0.45, ls=":", zorder=6, clip_on=False,
        )


def draw_boards_xy(ax):
    for sgn in (1, -1):
        ax.add_patch(Rectangle(
            (X(SIDE_XC) - SIDE_LEN / 2, sgn * SIDE_YC - SIDE_W / 2),
            SIDE_LEN, SIDE_W,
            facecolor=C_BUMP, edgecolor=C_EDGE, lw=0.5, zorder=3,
        ))
    bx, by, bsx, bsy = box_xy(FRONT_C, FRONT_SIZE)
    ax.add_patch(Rectangle(
        (bx, by), bsx, bsy, facecolor=C_BUMP, edgecolor=C_EDGE, lw=0.6, zorder=3,
    ))


def draw_handle_xy(ax):
    """Pallet-jack tiller: thick stem + U-loop, yoke joins stem to the arms."""
    sx = X(HANDLE_X_STEM)
    lean = HANDLE_SZ * math.sin(TILLER_PITCH)
    hx = sx + lean
    hy = TILLER_HALF_Y
    t = TILLER_T
    ax.add_patch(Circle(
        (sx, HANDLE_Y), JOY_BASE_R * 0.7, facecolor=C_HANDLE,
        edgecolor=C_EDGE, lw=0.5, zorder=8,
    ))
    ax.add_patch(Rectangle(
        (hx - t / 2, -hy - t / 2), t, 2 * hy + t,
        facecolor=C_HANDLE, edgecolor=C_EDGE, lw=0.5, zorder=8,
    ))
    ax.add_patch(Rectangle(
        (hx, -hy - t / 2), TILLER_DX, t,
        facecolor=C_HANDLE, edgecolor=C_EDGE, lw=0.5, zorder=8,
    ))
    ax.add_patch(Rectangle(
        (hx, hy - t / 2), TILLER_DX, t,
        facecolor=C_HANDLE, edgecolor=C_EDGE, lw=0.5, zorder=8,
    ))
    ax.add_patch(Rectangle(
        (hx + TILLER_DX - t / 2, -hy - t / 2), t, 2 * hy + t,
        facecolor=C_HANDLE, edgecolor=C_EDGE, lw=0.5, zorder=9,
    ))


def draw_handle_xz(ax):
    sx = X(HANDLE_X_STEM)
    tr = Affine2D().rotate_around(sx, HANDLE_Z0, -TILLER_PITCH) + ax.transData
    ax.add_patch(Rectangle(
        (sx - 0.08, HANDLE_Z0), 0.16, 0.10,
        facecolor=C_HANDLE, edgecolor=C_EDGE, lw=0.5, zorder=8,
    ))
    ax.add_patch(Rectangle(
        (sx - STEM_W / 2, HANDLE_Z0), STEM_W, HANDLE_SZ,
        facecolor=C_HANDLE, edgecolor=C_EDGE, lw=0.6, zorder=8,
        transform=tr,
    ))
    loop_z = HANDLE_Z1 - TILLER_T / 2
    ax.add_patch(Rectangle(
        (sx, loop_z - TILLER_T / 2), TILLER_DX, TILLER_T,
        facecolor=C_HANDLE, edgecolor=C_EDGE, lw=0.6, zorder=9,
        transform=tr,
    ))


def draw_handle_yz(ax):
    hy = TILLER_HALF_Y
    loop_z = HANDLE_Z1 - TILLER_T / 2
    ax.add_patch(Rectangle(
        (HANDLE_Y - STEM_W / 2, HANDLE_Z0), STEM_W, HANDLE_SZ * math.cos(TILLER_PITCH),
        facecolor=C_HANDLE, edgecolor=C_EDGE, lw=0.6, zorder=9,
    ))
    ax.add_patch(Rectangle(
        (-hy - TILLER_T / 2, loop_z - TILLER_T / 2), 2 * hy + TILLER_T, TILLER_T,
        facecolor=C_HANDLE, edgecolor=C_EDGE, lw=0.6, zorder=10,
    ))


def draw_cams_xy(ax, wedges=True):
    """Cameras inside 3C, looking out through holes in the +X / +Z faces."""
    for _x, y, _z in CAMS:
        ax.add_patch(Circle(
            (X(BOX3_X1), y), CAM_HOLE_R, facecolor=C_HOLE,
            edgecolor=C_EDGE, lw=0.5, zorder=7,
        ))
        if wedges:
            ax.add_patch(Wedge(
                (X(BOX3_X1), y), 0.16, -35, 35,
                facecolor=C_CAM, alpha=0.10, edgecolor=C_CAM, lw=0.35, zorder=2,
            ))
        ax.add_patch(Circle(
            (X(CAM_X), y), CAM_R, facecolor=C_CAM,
            edgecolor="white", lw=0.6, zorder=8,
        ))
    ax.add_patch(Circle(
        (X(CAM_UP_X), CAM_UP_Y), CAM_HOLE_R, facecolor=C_HOLE,
        edgecolor=C_EDGE, lw=0.5, zorder=7,
    ))
    ax.add_patch(Circle(
        (X(CAM_UP_X), CAM_UP_Y), CAM_R, facecolor=C_CAM,
        edgecolor="white", lw=0.6, zorder=8,
    ))


def draw_cams_xz(ax, wedges=True):
    ax.add_patch(Circle(
        (X(BOX3_X1), CAM_Z), CAM_HOLE_R, facecolor=C_HOLE,
        edgecolor=C_EDGE, lw=0.5, zorder=7,
    ))
    if wedges:
        ax.add_patch(Wedge(
            (X(BOX3_X1), CAM_Z), 0.28, -35, 35,
            facecolor=C_CAM, alpha=0.12, edgecolor=C_CAM, lw=0.4, zorder=2,
        ))
    ax.add_patch(Circle(
        (X(CAM_X), CAM_Z), CAM_R, facecolor=C_CAM,
        edgecolor="white", lw=0.6, zorder=8,
    ))
    ax.add_patch(Circle(
        (X(CAM_UP_X), BOX3_Z1), CAM_HOLE_R, facecolor=C_HOLE,
        edgecolor=C_EDGE, lw=0.5, zorder=7,
    ))
    if wedges:
        ax.add_patch(Wedge(
            (X(CAM_UP_X), BOX3_Z1), 0.22, 55, 125,
            facecolor=C_CAM, alpha=0.12, edgecolor=C_CAM, lw=0.4, zorder=2,
        ))
    ax.add_patch(Circle(
        (X(CAM_UP_X), CAM_UP_Z), CAM_R, facecolor=C_CAM,
        edgecolor="white", lw=0.6, zorder=8,
    ))


def draw_cams_yz(ax):
    for _x, y, _z in CAMS:
        ax.add_patch(Circle(
            (y, CAM_Z), CAM_HOLE_R, facecolor=C_HOLE,
            edgecolor=C_EDGE, lw=0.5, zorder=8,
        ))
        ax.add_patch(Circle(
            (y, CAM_Z), CAM_R, facecolor=C_CAM,
            edgecolor="white", lw=0.6, zorder=9,
        ))
    ax.add_patch(Circle(
        (CAM_UP_Y, BOX3_Z1), CAM_HOLE_R, facecolor=C_HOLE,
        edgecolor=C_EDGE, lw=0.5, zorder=8,
    ))
    ax.add_patch(Circle(
        (CAM_UP_Y, CAM_UP_Z), CAM_R, facecolor=C_CAM,
        edgecolor="white", lw=0.6, zorder=9,
    ))


def draw_up_frame_xy(ax, wedges=True):
    w = TUBE_W
    x0, x1 = X(ARM_END_X), X(TUBE_X)
    for sgn in (1, -1):
        y = sgn * TUBE_Y
        ax.add_patch(Rectangle(
            (x1 - w / 2, y - w / 2), w, w,
            facecolor=C_MAST, edgecolor=C_EDGE, lw=0.7, zorder=4,
        ))
        ax.add_patch(Rectangle(
            (min(x0, x1) - w / 2, y - w / 2),
            abs(x1 - x0) + w, w,
            facecolor=C_MAST, edgecolor=C_EDGE, lw=0.55, zorder=4,
        ))
    ax.add_patch(Rectangle(
        (x0 - w / 2, -TUBE_Y - w / 2), w, 2 * TUBE_Y + w,
        facecolor=C_MAST, edgecolor=C_EDGE, lw=0.7, zorder=5,
    ))
    ax.add_patch(Rectangle(
        (X(BOX3_X0), -BOX3_SY / 2), BOX3_SX, BOX3_SY,
        facecolor=C_BOX, edgecolor=C_EDGE, lw=0.7, zorder=6, alpha=0.55,
    ))
    w = TUBE_W
    ax.add_patch(Rectangle(
        (X(FRAME3_X0), -BOX3_SY / 2 - w / 2), w, BOX3_SY + w,
        facecolor=C_FRAME, edgecolor=C_EDGE, lw=0.7, zorder=7,
    ))
    draw_cams_xy(ax, wedges=wedges)
    draw_handle_xy(ax)


def draw_up_frame_xz(ax, wedges=True):
    w = TUBE_W
    x0, x1 = X(ARM_END_X), X(TUBE_X)
    ax.add_patch(Rectangle(
        (x1 - w / 2, TUBE_Z0), w, TUBE_H,
        facecolor=C_MAST, edgecolor=C_EDGE, lw=0.7, zorder=4,
    ))
    ax.add_patch(Rectangle(
        (min(x0, x1) - w / 2, TUBE_Z1 - w),
        abs(x1 - x0) + w, w,
        facecolor=C_MAST, edgecolor=C_EDGE, lw=0.7, zorder=5,
    ))
    ax.add_patch(Rectangle(
        (X(BOX3_X0), BOX3_Z0), BOX3_SX, BOX3_SZ,
        facecolor=C_BOX, edgecolor=C_EDGE, lw=0.7, zorder=6, alpha=0.55,
    ))
    w = TUBE_W
    ax.add_patch(Rectangle(
        (X(FRAME3_X0), FRAME3_Z0), w, FRAME3_H,
        facecolor=C_FRAME, edgecolor=C_EDGE, lw=0.7, zorder=7,
    ))
    draw_cams_xz(ax, wedges=wedges)
    draw_handle_xz(ax)


def draw_up_frame_yz(ax):
    w = TUBE_W
    for sgn in (1, -1):
        ax.add_patch(Rectangle(
            (sgn * TUBE_Y - w / 2, TUBE_Z0), w, TUBE_H,
            facecolor=C_MAST, edgecolor=C_EDGE, lw=0.7, zorder=4,
        ))
    ax.add_patch(Rectangle(
        (-TUBE_Y - w / 2, TUBE_Z1 - w), 2 * TUBE_Y + w, w,
        facecolor=C_MAST, edgecolor=C_EDGE, lw=0.7, zorder=6,
    ))
    ax.add_patch(Rectangle(
        (-BOX3_SY / 2, BOX3_Z0), BOX3_SY, BOX3_SZ,
        facecolor=C_BOX, edgecolor=C_EDGE, lw=0.7, zorder=7, alpha=0.55,
    ))
    w = TUBE_W
    y_span = BOX3_SY
    ax.add_patch(Rectangle(
        (-y_span / 2 - w / 2, FRAME3_Z0), y_span + w, w,
        facecolor=C_FRAME, edgecolor=C_EDGE, lw=0.7, zorder=8,
    ))
    for sgn in (1, -1):
        ax.add_patch(Rectangle(
            (sgn * y_span / 2 - w / 2, FRAME3_Z0), w, FRAME3_H,
            facecolor=C_FRAME, edgecolor=C_EDGE, lw=0.7, zorder=8,
        ))
    ax.add_patch(Rectangle(
        (-y_span / 2 - w / 2, FRAME3_Z1 - w), y_span + w, w,
        facecolor=C_FRAME, edgecolor=C_EDGE, lw=0.7, zorder=8,
    ))
    draw_cams_yz(ax)
    draw_handle_yz(ax)


def tag(ax, x, y, code, color=C_TAG, fs=7):
    ax.text(
        x, y, code, fontsize=fs, fontweight="bold", color="white",
        ha="center", va="center", zorder=12, clip_on=False,
        bbox=dict(boxstyle="round,pad=0.18", facecolor=color, edgecolor="white", lw=0.6),
    )


def zoom_box(ax, x0, x1, y0, y1, label, color="#1565c0"):
    ax.add_patch(Rectangle(
        (x0, y0), x1 - x0, y1 - y0,
        fill=False, ls="--", lw=1.15, edgecolor=color, zorder=13, clip_on=False,
    ))
    ax.text(
        x0 + 0.02, y1 - 0.015, label, fontsize=7, color=color, va="top",
        fontweight="bold", zorder=14, clip_on=False,
        bbox=dict(boxstyle="round,pad=0.15", facecolor="white", edgecolor=color, lw=0.6),
    )


def head_zoom_lim():
    return (X(BODY_FRONT) - 0.10, X(NOSE_FRONT) + 0.32, -OUTER_HALF - 0.16, OUTER_HALF + 0.16)


def cams_zoom_lim():
    return (X(FRAME3_X0) - 0.10, X(BOX3_X1) + 0.14, -BOX3_SY / 2 - 0.14, BOX3_SY / 2 + 0.14)


def upper_zoom_lim():
    return (X(FRAME3_X0) - 0.14, X(BOX3_X1) + 0.22, BOX3_Z0 - 0.10, L_UP_TOP + 0.14)


def draw_part_labels_xy(ax, region="overview"):
    if region in ("overview", "wheels"):
        tag(ax, X(DECK_C[0]), 0.18, "6", C_EDGE)
        tag(ax, X(FORK["x"]), FORK["y"] + 0.12, "9A", "#8a6a10")
        tag(ax, X(FORK["x"]), -FORK["y"] - 0.12, "9B", "#8a6a10")
        tag(ax, 0.08, FORK_Y + 0.16, "основа", C_BOGIE, fs=6)
        tag(ax, X(L_UP[0]), L_UP[1] + 0.18, "5", C_UP)
        tag(ax, X(FRONT_C[0]), 0.0, "8", C_BUMP)
        tag(ax, X(L_LEFT[0]) + 0.10, L_LEFT[1] + 0.08, "4A", C_LEFT)
        tag(ax, X(L_RIGHT[0]) + 0.10, L_RIGHT[1] - 0.08, "4B", C_RIGHT)
        tag(ax, X(DRIVE[0]), -0.13, "W", C_DRIVE)
        tag(ax, 0.0, 0.0, "R", C_BOGIE)
    elif region == "head":
        tag(ax, X(FRONT_C[0]), 0.0, "8", C_BUMP)
        tag(ax, X(L_LEFT[0]) + 0.10, L_LEFT[1] + 0.08, "4A", C_LEFT)
        tag(ax, X(L_RIGHT[0]) + 0.10, L_RIGHT[1] - 0.08, "4B", C_RIGHT)
        tag(ax, X(DECK_C[0]) + 0.12, 0.16, "6", C_EDGE)
    elif region == "cams":
        tag(ax, X(TUBE_X), TUBE_Y + 0.07, "1A", C_MAST)
        tag(ax, X(TUBE_X), -TUBE_Y - 0.07, "1B", C_MAST)
        tag(ax, X((TUBE_X + ARM_END_X) / 2), TUBE_Y + 0.07, "2A", C_MAST)
        tag(ax, X((TUBE_X + ARM_END_X) / 2), -TUBE_Y - 0.07, "2B", C_MAST)
        tag(ax, X(ARM_END_X) - 0.07, 0.16, "3", C_MAST)
        tag(ax, X(BOX3_X0) + BOX3_SX / 2, BOX3_SY / 2 + 0.09, "3C", C_BOX)
        tag(ax, X(FRAME3_X0) - 0.07, 0.14, "3D", C_FRAME)
        tag(ax, X(CAM_X) + 0.10, CAM_Y + 0.07, "cam", C_CAM, fs=6)
        tag(ax, X(CAM_X) + 0.10, -CAM_Y - 0.07, "cam", C_CAM, fs=6)
        tag(ax, X(CAM_UP_X), CAM_UP_Y - 0.09, "cam↑", C_CAM, fs=6)
        tag(ax, X(L_UP[0]), L_UP[1] + 0.14, "5", C_UP)


def draw_part_labels_xz(ax, region="overview"):
    if region == "overview":
        tag(ax, X(DECK_C[0]), DECK_C[2], "6", C_EDGE)
        tag(ax, X(FORK["x"]), FORK["z"] + 0.10, "9A", "#8a6a10")
        tag(ax, X(TUBE_X) + 0.08, TUBE_Z0 + TUBE_H / 2, "1A", C_MAST)
        tag(ax, X(HANDLE_X_STEM) + 0.12, HANDLE_Z1 + 0.06, "1C", C_HANDLE)
        tag(ax, X(L_UP[0]), L_UP[2] + 0.12, "5", C_UP)
        tag(ax, X(FRONT_C[0]), CLEAR + SIDE_H + 0.10, "8", C_BUMP)
        tag(ax, X(L_LEFT[0]), L_LEFT[2] + 0.12, "4A", C_LEFT)
        tag(ax, X(DRIVE[0]), DRIVE[2] + 0.16, "W", C_DRIVE)
        tag(ax, 0.0, 0.12, "R", C_BOGIE)
    elif region == "upper":
        tag(ax, X((TUBE_X + ARM_END_X) / 2), TUBE_Z1 + 0.07, "2A", C_MAST)
        tag(ax, X(ARM_END_X) - 0.07, TUBE_Z1 + 0.07, "3", C_MAST)
        tag(ax, X((BOX3_X0 + BOX3_X1) / 2), BOX3_Z1 + 0.08, "3C", C_BOX)
        tag(ax, X(CAM_X) + 0.10, CAM_Z, "cam", C_CAM, fs=6)
        tag(ax, X(CAM_UP_X) - 0.09, CAM_UP_Z + 0.08, "cam↑", C_CAM, fs=6)
        tag(ax, X(FRAME3_X0) - 0.08, FRAME3_Z0 + FRAME3_H / 2, "3D", C_FRAME)
        tag(ax, X(L_UP[0]), L_UP[2] + 0.10, "5", C_UP)
        tag(ax, X(HANDLE_X_STEM) + 0.10, HANDLE_Z1 + 0.05, "1C", C_HANDLE)
    elif region == "wheels":
        tag(ax, X(DECK_C[0]), DECK_C[2], "6", C_EDGE)
        tag(ax, X(FRONT_C[0]), CLEAR + SIDE_H + 0.10, "8", C_BUMP)
        tag(ax, X(L_LEFT[0]), L_LEFT[2] + 0.12, "4A", C_LEFT)


def draw_part_labels_yz(ax, region="overview"):
    tag(ax, 0.0, DECK_C[2], "6", C_EDGE)
    tag(ax, SIDE_YC + 0.10, CLEAR + SIDE_H + 0.10, "7A", C_BUMP)
    tag(ax, -SIDE_YC - 0.10, CLEAR + SIDE_H + 0.10, "7B", C_BUMP)
    tag(ax, TUBE_Y + 0.10, TUBE_Z0 + TUBE_H / 2, "1A", C_MAST)
    tag(ax, -TUBE_Y - 0.10, TUBE_Z0 + TUBE_H / 2, "1B", C_MAST)
    tag(ax, L_LEFT[1], L_LEFT[2] + 0.14, "4A", C_LEFT)
    tag(ax, L_RIGHT[1], L_RIGHT[2] + 0.14, "4B", C_RIGHT)
    tag(ax, 0.0, L_UP[2] + 0.14, "5", C_UP)
    if region == "cams":
        tag(ax, 0.0, BOX3_Z1 + 0.08, "3C", C_BOX)
        tag(ax, CAM_Y, CAM_Z + 0.09, "cam", C_CAM, fs=6)
        tag(ax, -CAM_Y, CAM_Z + 0.09, "cam", C_CAM, fs=6)
        tag(ax, CAM_UP_Y, CAM_UP_Z + 0.09, "cam↑", C_CAM, fs=6)
        tag(ax, 0.0, FRAME3_Z1 + 0.08, "3D", C_FRAME)


def draw_robot_xy(ax, fov=True, region="overview"):
    x0, y0, sx, sy = box_xy(DECK_C, DECK_SIZE)
    ax.add_patch(Rectangle((x0, y0), sx, sy, facecolor=C_BODY,
                           edgecolor=C_EDGE, lw=0.8, zorder=2))
    draw_boards_xy(ax)
    tip_x = X(FORK_TIP)
    heel_x = X(BODY_REAR)
    draw_up_frame_xy(ax, wedges=(region == "cams"))

    pocket0, pocket1 = -POCKET_HALF, POCKET_HALF
    for sgn in (1, -1):
        y = sgn * FORK_Y
        hw = FORK_WIDTH / 2
        tw = hw * 0.55
        ax.add_patch(Polygon(
            [
                (pocket0, y - hw),
                (pocket0, y + hw),
                (tip_x, y + tw),
                (tip_x, y - tw),
            ],
            facecolor=C_FORK, edgecolor="#8a6a10", lw=0.5, zorder=2,
        ))
        ax.add_patch(Rectangle(
            (pocket1, y - hw), heel_x - pocket1, FORK_WIDTH,
            facecolor=C_FORK, edgecolor="#8a6a10", lw=0.5, zorder=2,
        ))
        ax.add_patch(Rectangle(
            (pocket0, y - hw), pocket1 - pocket0, SLOT_WALL,
            facecolor=C_FORK, edgecolor="#8a6a10", lw=0.4, zorder=3,
        ))
        ax.add_patch(Rectangle(
            (pocket0, y + hw - SLOT_WALL), pocket1 - pocket0, SLOT_WALL,
            facecolor=C_FORK, edgecolor="#8a6a10", lw=0.4, zorder=3,
        ))

    # Wheel chassis: cheeks at the load wheels, rails to corpus 6.
    for sgn in (1, -1):
        y = sgn * FORK_Y
        ax.add_patch(Rectangle(
            (pocket1, y - 0.035), heel_x - pocket1, 0.07,
            facecolor=C_BOGIE, edgecolor=C_EDGE, lw=0.4, zorder=1, alpha=0.85,
        ))
    ax.add_patch(Rectangle(
        (-0.03, -0.16), 0.06, 0.32,
        facecolor=C_BOGIE, edgecolor=C_EDGE, lw=0.4, zorder=3, alpha=0.75,
    ))
    ax.add_patch(Rectangle(
        (X(BODY_REAR) - 0.10, -0.185), 0.08, 0.37,
        facecolor=C_BOGIE, edgecolor=C_EDGE, lw=0.4, zorder=3, alpha=0.75,
    ))

    ax.add_patch(Circle((X(DRIVE[0]), DRIVE[1]), DRIVE_R, facecolor=C_DRIVE,
                        edgecolor="white", lw=0.6, zorder=5))
    for p in CENTER:
        ax.add_patch(Circle((X(p[0]), p[1]), CENTER_R, facecolor=C_CENTER,
                            edgecolor="white", lw=0.4, zorder=4))
    for p in FORK_W:
        ax.add_patch(Circle((X(p[0]), p[1]), p[2], facecolor=C_FW, zorder=4))

    if fov:
        r = 2.4 if region == "head" else 1.7
        draw_dock_fov(ax, L_LEFT, C_LEFT, radius=r)
        draw_dock_fov(ax, L_RIGHT, C_RIGHT, radius=r)
        if region == "overview":
            ax.add_patch(Circle(
                (X(L_UP[0]), L_UP[1]), 0.55, fill=False,
                ls="--", color=C_UP, lw=0.7, zorder=1,
            ))

    ax.add_patch(Circle((X(L_LEFT[0]), L_LEFT[1]), LIDAR_R, facecolor=C_LEFT,
                        edgecolor="white", lw=0.5, zorder=6))
    ax.add_patch(Circle((X(L_RIGHT[0]), L_RIGHT[1]), LIDAR_R, facecolor=C_RIGHT,
                        edgecolor="white", lw=0.5, zorder=6))
    ax.add_patch(Circle((X(L_UP[0]), L_UP[1]), PANDAR_R, facecolor=C_UP,
                        edgecolor="white", lw=0.5, zorder=7))
    ax.plot(0, 0, "k+", ms=11, mew=1.5, zorder=7)
    draw_part_labels_xy(ax, region=region)

    if region == "overview":
        ax.annotate("", xy=(0.50, 0), xytext=(0, 0),
                    arrowprops=dict(arrowstyle="-|>", color="#222", lw=1.1))
        ax.annotate("", xy=(0, 0.50), xytext=(0, 0),
                    arrowprops=dict(arrowstyle="-|>", color="#222", lw=1.1))
        ax.text(0.53, 0.02, "+X head", fontsize=8, color="#222")
        ax.text(0.04, 0.53, "+Y left", fontsize=8, color="#222")
        ax.text(0.04, -0.12, "origin", fontsize=7, color="#222")


def draw_robot_xz(ax, region="overview"):
    ax.axhline(0, color="#bdbdbd", lw=1.2, zorder=0)
    ax.add_patch(Circle((X(DRIVE[0]), DRIVE[2]), DRIVE_R, facecolor=C_DRIVE,
                        edgecolor="#7f1d1d", zorder=4))
    ax.add_patch(Circle((X(CENTER[0][0]), CENTER[0][2]), CENTER_R,
                        facecolor=C_CENTER, zorder=3))
    ax.add_patch(Circle((X(FORK_W[0][0]), FORK_W[0][2]), FORK_R,
                        facecolor=C_FW, zorder=4))
    x0, z0, sx, sz = box_xz(DECK_C, DECK_SIZE)
    ax.add_patch(Rectangle((x0, z0), sx, sz, facecolor=C_BODY,
                           edgecolor=C_EDGE, lw=0.8, zorder=2))
    ax.add_patch(Rectangle(
        (X(SIDE_XC) - SIDE_LEN / 2, CLEAR), SIDE_LEN, SIDE_H,
        facecolor=C_BUMP, edgecolor=C_EDGE, lw=0.5, zorder=3, alpha=0.85,
    ))
    fx0, fz0, fsx, fsz = box_xz(FRONT_C, FRONT_SIZE)
    ax.add_patch(Rectangle(
        (fx0, fz0), fsx, fsz,
        facecolor=C_BUMP, edgecolor=C_EDGE, lw=0.5, zorder=3, alpha=0.9,
    ))
    fx = X(FORK_TIP)
    heel = X(BODY_REAR)
    fz = FORK_BOTTOM
    fh = FORK["size"][2]
    pocket0, pocket1 = -POCKET_HALF, POCKET_HALF
    ax.add_patch(Polygon(
        [
            (pocket0, fz),
            (pocket0, fz + fh),
            (fx, fz + 0.03),
            (fx, fz),
        ],
        facecolor=C_FORK, edgecolor="#8a6a10", lw=0.5, zorder=2,
    ))
    ax.add_patch(Rectangle(
        (pocket1, fz), heel - pocket1, fh,
        facecolor=C_FORK, edgecolor="#8a6a10", lw=0.5, zorder=2,
    ))
    # Outer slot wall (side view): wheel sits in the cutout.
    ax.add_patch(Rectangle(
        (pocket0, fz), pocket1 - pocket0, fh,
        facecolor=C_FORK, edgecolor="#8a6a10", lw=0.4, zorder=2, alpha=0.45,
    ))
    # Chassis rail under the tine (stays down when forks lift).
    ax.add_patch(Rectangle(
        (pocket1, 0.020), heel - pocket1, 0.025,
        facecolor=C_BOGIE, edgecolor=C_EDGE, lw=0.4, zorder=3,
    ))
    # Lifted forks (10–20 cm): dashed, chassis stays on the floor.
    ax.add_patch(Rectangle(
        (pocket1, fz + FORK_LIFT_H), heel - pocket1, fh,
        fill=False, edgecolor=C_FORK, lw=0.9, ls="--", zorder=3,
    ))
    draw_up_frame_xz(ax, wedges=(region == "upper"))
    draw_lidar_side(ax, X(L_LEFT[0]), L_LEFT[2], C_LEFT)
    draw_pandar_side(ax, X(L_UP[0]), L_UP[2])
    ax.plot(0, 0, "k+", ms=10, mew=1.4, zorder=7)
    draw_part_labels_xz(ax, region=region)


def draw_robot_yz(ax, region="overview"):
    ax.axhline(0, color="#bdbdbd", lw=1.2, zorder=0)
    y0, z0, sy, sz = box_yz(DECK_C, DECK_SIZE)
    ax.add_patch(Rectangle((y0, z0), sy, sz, facecolor=C_BODY,
                           edgecolor=C_EDGE, lw=0.8, zorder=2))
    for sgn in (1, -1):
        ax.add_patch(Rectangle(
            (sgn * SIDE_YC - SIDE_W / 2, CLEAR), SIDE_W, SIDE_H,
            facecolor=C_BUMP, edgecolor=C_EDGE, lw=0.5, zorder=3,
        ))
    by, bz, bsy, bsz = box_yz(FRONT_C, FRONT_SIZE)
    ax.add_patch(Rectangle(
        (by, bz), bsy, bsz, facecolor=C_BUMP,
        edgecolor=C_EDGE, lw=0.5, zorder=3, alpha=0.55,
    ))
    for sgn in (1, -1):
        ax.add_patch(Rectangle(
            (sgn * FORK["y"] - FORK["size"][1] / 2,
             FORK["z"] - FORK["size"][2] / 2),
            FORK["size"][1], FORK["size"][2],
            facecolor=C_FORK, edgecolor="#8a6a10", lw=0.5, zorder=4,
        ))
    draw_up_frame_yz(ax)
    ax.add_patch(Circle((DRIVE[1], DRIVE[2]), DRIVE_R, facecolor=C_DRIVE,
                        edgecolor="#7f1d1d", zorder=5))
    for p in CENTER:
        ax.add_patch(Circle((p[1], p[2]), CENTER_R, facecolor=C_CENTER, zorder=4))
    for p in FORK_W:
        ax.add_patch(Circle((p[1], p[2]), p[2], facecolor=C_FW, zorder=5))
    draw_lidar_side(ax, L_LEFT[1], L_LEFT[2], C_LEFT)
    draw_lidar_side(ax, L_RIGHT[1], L_RIGHT[2], C_RIGHT)
    draw_pandar_side(ax, L_UP[1], L_UP[2])
    ax.plot(0, 0, "k+", ms=8, mew=1.2, zorder=7)
    draw_part_labels_yz(ax, region=region)


def style_ax(ax, xlabel, ylabel):
    ax.set_aspect("equal")
    ax.grid(True, which="major", alpha=0.35)
    ax.grid(True, which="minor", alpha=0.15)
    ax.minorticks_on()
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)


def make_lidar_sheet():
    fig = plt.figure(figsize=(16.4, 10.6), dpi=140)
    fig.suptitle(
        f"handler_bot · корпус {cm(BODY_W)} см · борты {cm(SIDE_W)} см · "
        f"габарит {cm(OUTER_W)} см · длина {cm(NOSE_FRONT - FORK_TIP)} см\n"
        "R/origin = середина задней колёсной опоры · W = ведущее/рулевое колесо · пол z=0 · размеры в см\n"
        "up = Hesai Pandar40P на верхней перекладине рамы 3D · общие виды + детальные зумы",
        fontsize=10.5, fontweight="bold", y=0.992,
    )
    gs = fig.add_gridspec(
        2, 3, hspace=0.30, wspace=0.20,
        left=0.04, right=0.99, top=0.88, bottom=0.105,
    )
    tip, head, nose = X(FORK_TIP), X(BODY_FRONT), X(NOSE_FRONT)
    rear = X(BODY_REAR)
    hx0, hx1, hy0, hy1 = head_zoom_lim()
    cx0, cx1, cy0, cy1 = cams_zoom_lim()
    ux0, ux1, uz0, uz1 = upper_zoom_lim()

    ax = fig.add_subplot(gs[0, 0])
    ax.set_title("План XY — общее", loc="left", fontsize=10)
    draw_robot_xy(ax, fov=False, region="overview")
    zoom_box(ax, hx0, hx1, hy0, hy1, "зум A: голова", C_LEFT)
    zoom_box(ax, cx0, cx1, cy0, cy1, "зум B: 3C", C_BOX)
    draw_board_body_dims_xy(ax)
    draw_lidar_coord_dims_xy(ax)
    dim(ax, 0.0, rear, f"{cm(ORIGIN_TO_BODY_DOWN)} опущ. / {cm(ORIGIN_TO_BODY_UP)} подн.",
        offset=-1.08, fs=7, color=C_BOGIE)
    dim(ax, 0.0, X(DRIVE[0]), f"{cm(DRIVE_FROM_ORIGIN)} база R→W",
        offset=0.91, fs=8, color=C_DRIVE)
    dim(ax, -OUTER_HALF, OUTER_HALF, f"{cm(OUTER_W)} габарит",
        horizontal=False, offset=X(NOSE_FRONT) + 0.18, fs=7)
    dim(ax, -BODY_W / 2, BODY_W / 2, f"{cm(BODY_W)} корпус",
        horizontal=False, offset=X(NOSE_FRONT) + 0.36, fs=7)
    dim(ax, -FORK["y"] + FORK["size"][1] / 2, FORK["y"] - FORK["size"][1] / 2,
        f"{cm(FORK_INNER)} между вилами", horizontal=False, offset=-0.32, fs=7, color="#8a6a10")
    style_ax(ax, "x (м)   − вилы          + голова", "y (м)")
    ax.set_xlim(-0.55, 2.35)
    ax.set_ylim(-1.28, 1.05)

    ax = fig.add_subplot(gs[0, 1])
    ax.set_title("Бок XZ — общее", loc="left", fontsize=10)
    draw_robot_xz(ax, region="overview")
    zoom_box(ax, ux0, ux1, uz0, uz1, "зум C: 3C/3D/5", C_FRAME)
    draw_lidar_coord_dims_xz(ax)
    dim(ax, tip, rear, f"{cm(FORK_LEN)} вилы", offset=-0.22, fs=8)
    dim(ax, 0.0, FORK_BOTTOM, f"{cm(FORK_BOTTOM)} вилы от пола",
        horizontal=False, offset=X(FORK_TIP) + 0.18, fs=6, color=C_FORK)
    dim(ax, FORK_BOTTOM, FORK_BOTTOM + FORK_LIFT_H, f"подъём {cm(FORK_LIFT_H)}",
        horizontal=False, offset=X(FORK_TIP) - 0.12, fs=7, color=C_FORK)
    dim(ax, 0.0, rear, f"{cm(ORIGIN_TO_BODY_UP)} подн. origin→6",
        offset=-0.54, fs=7, color=C_BOGIE)
    dim(ax, rear, head, f"{cm(BODY_LEN)} корпус", offset=-0.38, fs=7)
    dim(ax, head, nose, f"{cm(FRONT_L)} пер.", offset=-0.22, fs=7, color=C_BUMP)
    dim(ax, 0.0, TUBE_Z0, f"{cm(TUBE_Z0)} до 1A", horizontal=False, offset=-0.38, fs=7, color=C_MAST)
    dim(ax, TUBE_Z0, TUBE_Z1, f"{cm(TUBE_H)} 1A", horizontal=False, offset=-0.38, fs=7, color=C_MAST)
    dim(ax, CLEAR, CLEAR + BODY_H, f"{cm(BODY_H)} корпус", horizontal=False, offset=X(0.55), fs=7)
    dim(ax, SIDE_H, BOARD_TOP, f"{cm(SIDE_H)} борт 7", horizontal=False, offset=X(BODY_FRONT) + 0.12, fs=6, color=C_BUMP)
    dim(ax, 0.0, L_UP_TOP, f"{cm(L_UP_TOP)} верх 5", horizontal=False, offset=X(L_UP[0]) + 0.20, fs=7, color=C_UP)
    style_ax(ax, "x (м)", "z (м)")
    ax.set_xlim(-0.50, 2.15)
    ax.set_ylim(-0.55, 2.00)

    ax = fig.add_subplot(gs[0, 2])
    ax.set_title("Фасад YZ — общее", loc="left", fontsize=10)
    draw_robot_yz(ax, region="overview")
    dim(ax, -OUTER_HALF, OUTER_HALF, f"{cm(OUTER_W)} габарит", offset=-0.28, fs=7)
    dim(ax, -BODY_W / 2, BODY_W / 2, f"{cm(BODY_W)} корпус 6", offset=0.18, fs=7)
    dim(ax, BODY_W / 2, OUTER_HALF, f"{cm(SIDE_W)} борт 7", offset=0.34, fs=7, color=C_BUMP)
    dim(ax, -TUBE_Y, TUBE_Y, f"1A/1B ±{cm(TUBE_Y)}", offset=-0.12, fs=7, color=C_MAST)
    dim(ax, L_RIGHT[1], L_LEFT[1], f"4A/4B y=±{cm(L_LEFT[1])}", offset=0.42, fs=7, color=C_LEFT)
    dim(ax, 0.0, L_LEFT[2], f"4A z={cm(L_LEFT[2])}", horizontal=False, offset=L_LEFT[1] + 0.10, fs=7, color=C_LEFT)
    dim(ax, 0.0, L_UP[2], f"5 z={cm(L_UP[2])}", horizontal=False, offset=0.42, fs=7, color=C_UP)
    dim(ax, 0.0, L_UP_TOP, f"{cm(L_UP_TOP)} верх 5", horizontal=False, offset=0.55, fs=7, color=C_UP)
    dim(ax, -PANDAR_R, PANDAR_R, f"Ø{cm(PANDAR_D)} Pandar40P", offset=1.84,
        fs=7, color=C_UP)
    dim(ax, 0.0, BOARD_TOP, f"{cm(BOARD_TOP)} низ 4A", horizontal=False,
        offset=L_LEFT[1] + 0.22, fs=6, color=C_LEFT)
    style_ax(ax, "y (м)   право −     + лево", "z (м)")
    ax.set_xlim(-0.68, 0.68)
    ax.set_ylim(-0.42, 2.05)
    ax.invert_xaxis()

    ax = fig.add_subplot(gs[1, 0])
    ax.set_title("Зум A — голова: борт 8, 4A/4B, колесо", loc="left", fontsize=10)
    draw_robot_xy(ax, fov=True, region="head")
    dim(ax, 0.0, L_LEFT[1], f"{cm(L_LEFT[1])} left", horizontal=False, offset=X(0.55), fs=8, color=C_LEFT)
    dim(ax, L_RIGHT[1], 0.0, f"{cm(-L_RIGHT[1])} right", horizontal=False, offset=X(0.72), fs=8, color=C_RIGHT)
    dim(ax, head, nose, f"{cm(FRONT_L)} борт 8", offset=-0.48, fs=8, color=C_BUMP)
    dim_drive_to_body(ax, y_dim=-0.28)
    dim(ax, BODY_W / 2, OUTER_HALF, f"{cm(SIDE_W)} борт 7",
        horizontal=False, offset=X(NOSE_FRONT) + 0.12, fs=7, color=C_BUMP)
    ax.text(X(L_LEFT[0]) + 0.08, 0.52, "270° видно", color=C_LEFT, fontsize=8)
    ax.text(X(L_RIGHT[0]) + 0.08, -0.55, "270° видно", color=C_RIGHT, fontsize=8)
    style_ax(ax, "x (м)", "y (м)")
    ax.set_xlim(hx0, hx1)
    ax.set_ylim(hy0 - 0.28, hy1 + 0.08)

    ax = fig.add_subplot(gs[1, 1])
    ax.set_title("Зум B — 3C: камеры внутри, дырки Ø6", loc="left", fontsize=10)
    draw_robot_xy(ax, fov=False, region="cams")
    draw_camera_coord_dims_xy(ax)
    dim(ax, X(BOX3_X0), X(BOX3_X1), f"{cm(BOX3_SX)} 3C", offset=0.40, fs=8, color=C_BOX)
    dim(ax, X(ARM_END_X), X(TUBE_X), f"{cm(TUBE_ARM)} 2A", offset=0.52, fs=7, color=C_MAST)
    dim(ax, X(FRAME3_X0), X(FRAME3_X1), f"{cm(TUBE_W)} 3D", offset=-0.40, fs=7, color=C_FRAME)
    dim(ax, CAM_Y + CAM_R, BOX3_SY / 2, f"{cm(CAM_EDGE)} до cam",
        horizontal=False, offset=X(BOX3_X1) + 0.08, fs=7, color=C_CAM)
    dim(ax, -BOX3_SY / 2, -CAM_Y - CAM_R, f"{cm(CAM_EDGE)}",
        horizontal=False, offset=X(BOX3_X1) + 0.08, fs=7, color=C_CAM)
    dim(ax, -CAM_Y, CAM_Y, f"стерео {cm(2 * CAM_Y)}",
        horizontal=False, offset=X(BOX3_X1) + 0.22, fs=6, color=C_CAM)
    ax.text(X(CAM_X) + 0.04, CAM_Y + 0.02, "cam L", color=C_CAM, fontsize=6)
    ax.text(X(CAM_X) + 0.04, -CAM_Y - 0.02, "cam R", color=C_CAM, fontsize=6)
    ax.text(X(CAM_UP_X), CAM_UP_Y + 0.08, "дыра +Z", color=C_CAM, fontsize=7, ha="center")
    style_ax(ax, "x (м)", "y (м)")
    ax.set_xlim(cx0, cx1 + 0.10)
    ax.set_ylim(cy0 - 0.08, cy1 + 0.12)

    ax = fig.add_subplot(gs[1, 2])
    ax.set_title("Зум C — верх: 3C / 3D / камеры / up Pandar40P", loc="left", fontsize=10)
    draw_robot_xz(ax, region="upper")
    draw_camera_coord_dims_xz(ax)
    dim(ax, X(BOX3_X0), X(BOX3_X1), f"{cm(BOX3_SX)} 3C", offset=BOX3_Z1 + 0.18, fs=8, color=C_BOX)
    dim(ax, FRAME3_Z0, FRAME3_Z1, f"{cm(FRAME3_H)} 3D", horizontal=False,
        offset=X(FRAME3_X0) - 0.08, fs=7, color=C_FRAME)
    dim(ax, BOX3_Z0, BOX3_Z1, f"{cm(BOX3_SZ)}", horizontal=False, offset=X(BOX3_X1) + 0.08, fs=7, color=C_BOX)
    dim(ax, CAM_Z - CAM_R, CAM_Z + CAM_R, f"Ø{cm(CAM_D)}", horizontal=False,
        offset=X(CAM_X) + 0.12, fs=6, color=C_CAM)
    dim(ax, L_UP_BOTTOM, L_UP_TOP, f"{cm(PANDAR_H)} Pandar", horizontal=False,
        offset=X(L_UP[0]) + 0.10, fs=7, color=C_UP)
    dim(ax, X(L_UP[0]) - PANDAR_R, X(L_UP[0]) + PANDAR_R,
        f"Ø{cm(PANDAR_D)}", offset=L_UP_TOP + 0.035, fs=7, color=C_UP)
    dim(ax, FRAME3_Z1, L_UP_BOTTOM, f"{cm(PANDAR_STAND_H)} адаптер",
        horizontal=False, offset=X(L_UP[0]) - 0.10, fs=6, color=C_UP)
    ax.text(X(L_UP[0]), L_UP_TOP + 0.075, "5 — up: Hesai Pandar40P на раме 3D",
            color=C_UP, fontsize=7, ha="center")
    style_ax(ax, "x (м)", "z (м)")
    ax.set_xlim(ux0, ux1)
    ax.set_ylim(uz0, uz1)

    fig.text(
        0.50, 0.012,
        "1A/1B — стойки 70 см · 1C — рулевая рукоять · 2A/2B — верхние трубы 30 см · 3 — поперечина\n"
        "3C — короб камер · 3D — несущая рама up · 4A/4B — left/right dock-лидары · 5 — up Hesai Pandar40P\n"
        "6 — корпус · 7A/7B — боковые борта · 8 — передний борт · 9A/9B — вилы\n"
        "W — центральное ведущее/рулевое колесо · R/origin — середина задней колёсной опоры",
        ha="center", va="bottom", fontsize=7.2, color="#222", linespacing=1.15,
    )

    out = DOC / "handler_bot_lidar_dimensions.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print("wrote", out)


def make_plan_top():
    fig = plt.figure(figsize=(12.4, 10.8), dpi=140)
    fig.suptitle(
        "handler_bot  ·  план сверху (XY)  ·  origin = между задними колёсами\n"
        "+X голова / dock    −X вилы    +Y влево    размеры в см",
        fontsize=12, fontweight="bold", y=0.98,
    )
    gs = fig.add_gridspec(
        2, 2, height_ratios=[1.35, 1.0], hspace=0.28, wspace=0.18,
        left=0.06, right=0.98, top=0.90, bottom=0.07,
    )
    tip, rear, head, nose = X(FORK_TIP), X(BODY_REAR), X(BODY_FRONT), X(NOSE_FRONT)
    hx0, hx1, hy0, hy1 = head_zoom_lim()
    cx0, cx1, cy0, cy1 = cams_zoom_lim()

    ax = fig.add_subplot(gs[0, :])
    ax.set_title("Общий план — габариты (детали на зумах внизу)", loc="left", fontsize=10)
    draw_robot_xy(ax, fov=False, region="overview")
    zoom_box(ax, hx0, hx1, hy0, hy1, "зум: голова", C_LEFT)
    zoom_box(ax, cx0, cx1, cy0, cy1, "зум: 3C / cam", C_BOX)
    draw_board_body_dims_xy(ax)
    draw_lidar_coord_dims_xy(ax)
    dim(ax, tip, nose, f"{cm(NOSE_FRONT - FORK_TIP)} вилы+корпус+борт", offset=-0.90, fs=9)
    dim(ax, 0.0, rear, f"{cm(ORIGIN_TO_BODY_DOWN)} опущ. / {cm(ORIGIN_TO_BODY_UP)} подн.",
        offset=-1.26, fs=7, color=C_BOGIE)
    dim(ax, 0.0, X(DRIVE[0]),
        f"{cm(DRIVE_FROM_ORIGIN)} origin→ведущее и боковые колёса",
        offset=0.64, fs=8, color=C_DRIVE)
    dim(ax, 0.0, CENTER_Y, f"{cm(CENTER_Y)} origin→боковое по Y",
        horizontal=False, offset=X(DRIVE[0]) - 0.18, fs=7, color=C_CENTER)
    dim(ax, -OUTER_HALF, OUTER_HALF, f"{cm(OUTER_W)} габарит",
        horizontal=False, offset=X(NOSE_FRONT) + 0.16, fs=8)
    dim(ax, -BODY_W / 2, BODY_W / 2, f"{cm(BODY_W)} корпус",
        horizontal=False, offset=X(NOSE_FRONT) + 0.34, fs=8)
    dim(ax, -FORK["y"] + FORK["size"][1] / 2, FORK["y"] - FORK["size"][1] / 2,
        f"{cm(FORK_INNER)} между вилами", horizontal=False, offset=-0.36, fs=8, color="#8a6a10")
    dim(ax, FORK_Y - FORK_WIDTH / 2, FORK_Y + FORK_WIDTH / 2,
        f"{cm(FORK_WIDTH)} ширина вилы", horizontal=False, offset=-0.54, fs=8, color="#8a6a10")
    style_ax(ax, "x (м)", "y (м)")
    ax.set_xlim(-0.70, 2.50)
    ax.set_ylim(-1.35, 1.10)

    ax = fig.add_subplot(gs[1, 0])
    ax.set_title("Зум — голова: 4A/4B 270°, борт 8, привод", loc="left", fontsize=10)
    draw_robot_xy(ax, fov=True, region="head")
    dim(ax, 0.0, L_LEFT[1], f"{cm(L_LEFT[1])} left", horizontal=False, offset=X(0.55), fs=8, color=C_LEFT)
    dim(ax, L_RIGHT[1], 0.0, f"{cm(-L_RIGHT[1])} right", horizontal=False, offset=X(0.72), fs=8, color=C_RIGHT)
    dim(ax, head, nose, f"{cm(FRONT_L)} борт 8", offset=-0.48, fs=8, color=C_BUMP)
    dim_drive_to_body(ax, y_dim=-0.28)
    ax.text(X(L_LEFT[0]) + 0.08, 0.50, "270° видно", color=C_LEFT, fontsize=8)
    ax.text(X(L_RIGHT[0]) + 0.08, -0.52, "270° видно", color=C_RIGHT, fontsize=8)
    style_ax(ax, "x (м)", "y (м)")
    ax.set_xlim(hx0, hx1)
    ax.set_ylim(hy0 - 0.28, hy1 + 0.06)

    ax = fig.add_subplot(gs[1, 1])
    ax.set_title("Зум — 3C: камеры в коробе, дырки Ø6", loc="left", fontsize=10)
    draw_robot_xy(ax, fov=False, region="cams")
    draw_camera_coord_dims_xy(ax)
    dim(ax, X(BOX3_X0), X(BOX3_X1), f"{cm(BOX3_SX)} 3C", offset=0.42, fs=8, color=C_BOX)
    dim(ax, X(ARM_END_X), X(TUBE_X), f"{cm(TUBE_ARM)} 2A", offset=0.54, fs=7, color=C_MAST)
    dim(ax, CAM_Y + CAM_R, BOX3_SY / 2, f"{cm(CAM_EDGE)} до cam",
        horizontal=False, offset=X(BOX3_X1) + 0.08, fs=7, color=C_CAM)
    dim(ax, -CAM_Y, CAM_Y, f"стерео {cm(2 * CAM_Y)}",
        horizontal=False, offset=X(BOX3_X1) + 0.20, fs=6, color=C_CAM)
    dim(ax, -BOX3_SY / 2, BOX3_SY / 2, f"{cm(BOX3_SY)} 3C",
        horizontal=False, offset=X(FRAME3_X0) - 0.10, fs=7, color=C_BOX)
    ax.text(X(CAM_X) + 0.04, CAM_Y + 0.02, "L", color=C_CAM, fontsize=6)
    ax.text(X(CAM_X) + 0.04, -CAM_Y - 0.02, "R", color=C_CAM, fontsize=6)
    ax.text(X(CAM_UP_X), CAM_UP_Y + 0.08, "cam↑ дыра +Z", color=C_CAM, fontsize=7, ha="center")
    style_ax(ax, "x (м)", "y (м)")
    ax.set_xlim(cx0, cx1 + 0.12)
    ax.set_ylim(cy0 - 0.12, cy1 + 0.14)

    fig.text(
        0.50, 0.012,
        "1A/1B стойки   1C tiller   2A/2B   3 / 3C+cam / 3D   4A/4B dock   5 up   "
        "6 корпус   7A/7B борты   8 передний   9A/9B вилы",
        ha="center", va="bottom", fontsize=8, color="#222",
    )
    out = DOC / "handler_bot_plan_top.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print("wrote", out)


def make_parts_sheet():
    fig = plt.figure(figsize=(13.4, 10.4), dpi=140)
    fig.suptitle(
        "handler_bot  ·  колёса и остальные части  ·  корпус/рамка по обмеру",
        fontsize=13, fontweight="bold", y=0.98,
    )
    gs = fig.add_gridspec(
        2, 2, height_ratios=[1.05, 1.15], hspace=0.28, wspace=0.18,
        left=0.05, right=0.98, top=0.90, bottom=0.06,
    )

    ax = fig.add_subplot(gs[0, 0])
    ax.set_title("Колёса на плане XY (отдельно)", loc="left", fontsize=10)
    draw_robot_xy(ax, fov=False, region="wheels")
    dim(ax, 0.0, X(DRIVE[0]), f"{cm(DRIVE_FROM_ORIGIN)} drive/side X", offset=0.62, fs=8, color=C_DRIVE)
    dim_drive_to_body(ax, y_dim=-0.52)
    dim(ax, -CENTER_Y, CENTER_Y, f"{cm(2 * CENTER_Y)} боковые колея",
        horizontal=False, offset=X(0.50), fs=7)
    dim(ax, X(FORK_W[0][0]) + FORK_R, X(FORK_W[1][0]) - FORK_R, f"{cm(WHEEL_GAP)} между колёсами",
        offset=-0.78, fs=7)
    ax.text(X(DRIVE[0]), 0.18, f"drive Ø{cm(DRIVE_R * 2)}", color=C_DRIVE, fontsize=7, ha="center")
    ax.text(X(CENTER[0][0]), CENTER[0][1] + 0.10, f"Ø{cm(CENTER_R * 2)}", color=C_CENTER, fontsize=7, ha="center")
    ax.text(X(FORK_W[0][0]), FORK_W[0][1] + 0.12, f"Ø{cm(FORK_R * 2)}", color=C_FW, fontsize=7, ha="center")
    style_ax(ax, "x (м)", "y (м)")
    ax.set_xlim(-0.50, 2.20)
    ax.set_ylim(-0.95, 0.90)

    ax = fig.add_subplot(gs[0, 1])
    ax.set_title("Колёса сбоку XZ (диаметры / клиренс)", loc="left", fontsize=10)
    draw_robot_xz(ax, region="wheels")
    dim(ax, 0.0, DRIVE_R * 2, f"drive Ø{cm(DRIVE_R * 2)}", horizontal=False,
        offset=X(0.42), fs=8, color=C_DRIVE)
    dim(ax, X(DRIVE[0]), X(BODY_FRONT), f"{cm(DRIVE_GAP)} центр→корпус 6", offset=-0.18, fs=8, color=C_DRIVE)
    dim(ax, X(DRIVE[0]) + DRIVE_R, X(BODY_FRONT),
        f"{cm(BODY_FRONT - (DRIVE[0] + DRIVE_R))} обод→6", offset=-0.34, fs=8, color=C_DRIVE)
    dim(ax, 0.0, FORK_R * 2, f"задние Ø{cm(FORK_R * 2)}", horizontal=False,
        offset=-0.38, fs=8)
    dim(ax, 0.0, CENTER_R * 2, f"бок Ø{cm(CENTER_R * 2)}", horizontal=False,
        offset=X(DRIVE[0]) + 0.28, fs=8)
    style_ax(ax, "x (м)", "z (м)")
    ax.set_xlim(-0.50, 2.20)
    ax.set_ylim(-0.50, 1.90)

    ax = fig.add_subplot(gs[1, :])
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title(
        "Таблица  (см)  ·  origin = между задними колёсами;  "
        f"SDF base_link в этой СК x=+{cm(-OX)}",
        loc="left", fontsize=10,
    )

    col1 = (
        "КОЛЁСА  (x от задней оси)\n"
        f"  drive        1 шт   Ø{cm(DRIVE_R * 2)}  ширина {cm(DRIVE_W)}   "
        f"центр ({cm(X(DRIVE[0]))}, 0, {cm(DRIVE[2])})  {cm(DRIVE_GAP)} центр→корпус 6  "
        f"{cm(BODY_FRONT - (DRIVE[0] + DRIVE_R))} обод→6\n"
        f"  side         2 шт   Ø{cm(CENTER_R * 2)}  ширина {cm(CENTER_W)}   "
        f"та же ось x={cm(X(DRIVE[0]))}   y=±{cm(CENTER_Y)}   z={cm(CENTER_R)}\n"
        f"  fork F/B     4 шт   Ø{cm(FORK_R * 2)}   ширина {cm(FORK_WW)}   "
        f"в прорези вил  x={cm(X(_xw_b))} / {cm(X(_xw_f))}   "
        f"y=±{cm(FORK_Y)}   зазор F/B {cm(WHEEL_GAP)}\n"
        f"  origin       x=0  между задними колёсами   "
        f"drive и side от origin по X {cm(DRIVE_FROM_ORIGIN)}; "
        f"side от origin по Y ±{cm(CENTER_Y)}\n"
        f"  подъём вил   {cm(FORK_LIFT_H)} (10–20)  ·  низ вил {cm(FORK_BOTTOM)}  ·  "
        f"основа (рельсы) к корпусу 6  ·  origin→корпус {cm(ORIGIN_TO_BODY_DOWN)} опущ. → "
        f"{cm(ORIGIN_TO_BODY_UP)} подн.\n"
        "\n"
        "ЛИДАРЫ  (x от задней оси)\n"
        f"  left   ({cm(X(L_LEFT[0]))}, +{cm(L_LEFT[1])}, {cm(L_LEFT[2])})  "
        f"на корпусе, низ {cm(BOARD_TOP)}  FOV 270° наружу\n"
        f"  right  ({cm(X(L_RIGHT[0]))}, {cm(L_RIGHT[1])}, {cm(L_RIGHT[2])})  "
        f"на корпусе, низ {cm(BOARD_TOP)}  FOV 270° наружу\n"
        f"  up Pandar40P ({cm(X(L_UP[0]))}, 0, {cm(L_UP[2])})  Ø{cm(PANDAR_D)}×{cm(PANDAR_H)} см "
        f"на адаптере {cm(PANDAR_STAND_S)}×{cm(PANDAR_STAND_S)}×{cm(PANDAR_STAND_H)}; "
        f"низ корпуса {cm(L_UP_BOTTOM)}, верх {cm(L_UP_TOP)}, FOV 360°\n"
        f"  dock CTC {cm(L_LEFT[1] - L_RIGHT[1])}   "
        f"origin→dock X {cm(DOCK_FROM_ORIGIN)}   tip→4A Ox {cm(tip_to_dock_x())}\n"
        f"  корпус→carriage {cm(BODY_CARRIAGE_GAP)}\n"
        "\n"
        "КАМЕРЫ  (3 шт внутри 3C, дырки Ø6 см)\n"
        f"  2× смотрят +X  Ø{cm(CAM_D)}   {cm(CAM_EDGE)} от Y-краёв 3C  "
        f"стерео CTC {cm(2 * CAM_Y)}  центры SDF x={cm(1.165)} y=±{cm(CAM_Y)} z={cm(CAM_Z)}\n"
        f"  1× смотрит +Z  центр SDF ({cm(1.110)}, {cm(CAM_UP_Y)}, {cm(CAM_UP_Z)}); "
        f"от левого нижнего XY 3C: Ox {cm(0.12)}  Oy {cm(0.07)}\n"
        "  MJPG  1280×960 @ 30 fps  (также 1280×720, 1024×768, 800×600,\n"
        "        640×480, 640×360, 320×240; 5–30 fps)\n"
        "  YUYV  640×480 @ 30 fps;  1280×960 / 720 / 1024×768 только 5 fps;\n"
        "        800×600 @ 20 fps;  320×240 @ 30 fps"
    )
    col2 = (
        "КОРПУС И БОРТЫ\n"
        f"  корпус       {cm(BODY_LEN)} × {cm(BODY_W)} × {cm(BODY_H)}   "
        f"клиренс {cm(CLEAR)}\n"
        f"  борт боковой {cm(SIDE_W)} с каждой стороны   "
        f"высота {cm(SIDE_H)} (клиренс {cm(CLEAR)} → верх {cm(BOARD_TOP)})   "
        f"габарит {cm(OUTER_W)}\n"
        f"  борт передний вынос {cm(FRONT_L)}   "
        f"длина вилы→нос {cm(NOSE_FRONT - FORK_TIP)}\n"
        f"  4A/4B стоят на корпусе/борте 8, пол → низ подставки {cm(BOARD_TOP)}\n"
        f"  вилы до корпуса {cm(FORK_LEN)}   зазор корпус→carriage {cm(BODY_CARRIAGE_GAP)}\n"
        "\n"
        "ВИЛЫ\n"
        f"  длина {cm(FORK_LEN)}   ширина одной {cm(FORK_WIDTH)}   толщина {cm(0.06)}\n"
        f"  низ {cm(FORK_BOTTOM)} от пола   центры y=±{cm(FORK_Y)}   CTC {cm(2 * FORK_Y)}\n"
        f"  просвет (между вилами) {cm(FORK_INNER)}   наруж. {cm(2 * FORK_Y + FORK_WIDTH)}\n"
        f"  прорезь под колёса Ø{cm(FORK_R * 2)}  ·  основа (рельсы) к корпусу 6\n"
        f"  подъём {cm(FORK_LIFT_H)} (10–20 см)  ·  колёса остаются на полу\n"
        f"  origin→корпус 6: {cm(ORIGIN_TO_BODY_DOWN)} опущено, "
        f"{cm(ORIGIN_TO_BODY_UP)} после подъёма\n"
        "\n"
        "РАМКА ВЕРХНЕГО ЛИДАРА\n"
        f"  1A/1B стойки 4×4  снаружи корпуса  y=±{cm(TUBE_Y)}\n"
        f"  от левого края корпуса вправо {cm(POST_DX)}\n"
        f"  пол → низ 1A {cm(TUBE_Z0)}   высота {cm(TUBE_H)}   верх {cm(TUBE_Z1)}\n"
        f"  2A/2B от стойки влево {cm(TUBE_ARM)}   3 перемычка по концам\n"
        f"  3C короб: левый низ на левом верху 2A/2B; {cm(BOX3_SX)} вправо × {cm(BOX3_SY)} вдоль × {cm(BOX3_SZ)} вверх\n"
        f"  камеры внутри 3C (дыры Ø{cm(CAM_HOLE)}): 2 шт Ø{cm(CAM_D)} в +X, "
        f"{cm(CAM_EDGE)} от Y-краёв, стерео CTC {cm(2 * CAM_Y)}; "
        f"1 шт в крышке смотрит +Z, центр от левого нижнего XY: Ox {cm(0.12)} Oy {cm(0.07)}\n"
        f"    MJPG 1280×960 @ 30 fps;  YUYV 640×480 @ 30 fps\n"
        f"  3D рамка слева от 3C вплотную: низ вдоль левого ребра 3C, стойки {cm(FRAME3_H)}, верхняя перекладина\n"
        f"  1C tiller серый (не темнее бортов) на оси y=0, низ на верхе 6 (z={cm(HANDLE_Z0)}); "
        f"высота {cm(HANDLE_SZ)}, U-ручка {cm(2 * TILLER_HALF_Y)} к голове\n"
        f"  5 up Pandar40P Ø{cm(PANDAR_D)}×{cm(PANDAR_H)} на адаптере "
        f"{cm(PANDAR_STAND_S)}×{cm(PANDAR_STAND_S)}×{cm(PANDAR_STAND_H)} на верху 3D, верхушка {cm(L_UP_TOP)}  "
        f"центр ({cm(X(L_UP[0]))}, 0, {cm(L_UP[2])})"
    )
    ax.text(
        0.01, 0.97, col1, va="top", ha="left", fontsize=7.6,
        family="DejaVu Sans", linespacing=1.32,
        bbox=dict(boxstyle="square,pad=0.4", facecolor="#f4f4f5", edgecolor="#d4d4d8"),
    )
    ax.text(
        0.51, 0.97, col2, va="top", ha="left", fontsize=7.6,
        family="DejaVu Sans", linespacing=1.32,
        bbox=dict(boxstyle="square,pad=0.4", facecolor="#f4f4f5", edgecolor="#d4d4d8"),
    )

    out = DOC / "handler_bot_wheels_and_parts.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print("wrote", out)


def main():
    make_lidar_sheet()
    make_plan_top()
    make_parts_sheet()


if __name__ == "__main__":
    main()
