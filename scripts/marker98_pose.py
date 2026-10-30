"""ROS-independent marker-98 pose conversion helpers."""

from __future__ import annotations

import math


def wrap_angle(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def pose_in_marker98_frame(
    x: float, y: float, alpha: float,
    marker_x: float, marker_y: float, marker_alpha: float = 0.0,
) -> tuple[float, float, float]:
    """Express map pose in a frame at marker 98 with its heading as X axis."""
    dx, dy = x - marker_x, y - marker_y
    c, s = math.cos(marker_alpha), math.sin(marker_alpha)
    return c * dx + s * dy, -s * dx + c * dy, wrap_angle(alpha - marker_alpha)
