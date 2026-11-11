"""ROS-independent marker-98 pose conversion helpers."""

from __future__ import annotations

import math


def wrap_angle(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def pose_in_marker98_frame(
    x: float, y: float, alpha: float,
    marker_x: float, marker_y: float, marker_alpha: float = 0.0,
    *, y_axis_sign: float = 1.0,
) -> tuple[float, float, float]:
    """Express map pose relative to the reference X axis.

    y_axis_sign=-1 reflects published Y and heading for an X-up/Y-right
    navigation frame. Internal ROS transforms retain their original handedness.
    """
    dx, dy = x - marker_x, y - marker_y
    c, s = math.cos(marker_alpha), math.sin(marker_alpha)
    if y_axis_sign not in (-1, 1):
        raise ValueError("y_axis_sign must be -1 or +1")
    return (c * dx + s * dy, y_axis_sign * (-s * dx + c * dy),
            wrap_angle(y_axis_sign * (alpha - marker_alpha)))
