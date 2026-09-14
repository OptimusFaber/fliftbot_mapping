"""Utilities for making a stable, JSON-serialisable Cartographer trajectory.

The ROS service returns optimized poses as ``PoseStamped`` objects.  Keeping the
normalisation here (instead of in the ROS client) makes the JSON contract easy
to test without a running ROS graph and preserves both map-frame and local
coordinates in the output.
"""

from __future__ import annotations

import math
from typing import Any, Iterable, Mapping


def _wrap_pi(value: float) -> float:
    return (value + math.pi) % (2.0 * math.pi) - math.pi


def quaternion_to_euler(x: float, y: float, z: float, w: float) -> tuple[float, float, float]:
    """Return roll, pitch, yaw in radians, tolerating a zero quaternion."""
    norm = math.sqrt(x * x + y * y + z * z + w * w)
    if norm < 1.0e-12:
        x, y, z, w = 0.0, 0.0, 0.0, 1.0
    else:
        x, y, z, w = x / norm, y / norm, z / norm, w / norm
    roll = math.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    sin_pitch = max(-1.0, min(1.0, 2.0 * (w * y - z * x)))
    pitch = math.asin(sin_pitch)
    yaw = math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    return roll, pitch, yaw


def _pose_values(pose: Any) -> tuple[float, float, float, float, float, float, float]:
    """Extract x/y/z and quaternion from a ROS or mapping-like pose."""
    position = getattr(pose, "position", None) or pose.get("position", {})
    orientation = getattr(pose, "orientation", None) or pose.get("orientation", {})

    def value(obj: Any, name: str, default: float = 0.0) -> float:
        if isinstance(obj, Mapping):
            return float(obj.get(name, default))
        return float(getattr(obj, name, default))

    return (
        value(position, "x"), value(position, "y"), value(position, "z"),
        value(orientation, "x"), value(orientation, "y"),
        value(orientation, "z"), value(orientation, "w", 1.0),
    )


def _stamp_ns(item: Any) -> int:
    header = getattr(item, "header", None)
    if header is None and isinstance(item, Mapping):
        header = item.get("header", {})
    stamp = getattr(header, "stamp", None) if header is not None else None
    if stamp is None and isinstance(header, Mapping):
        stamp = header.get("stamp", {})
    if stamp is None:
        return 0
    if isinstance(stamp, Mapping):
        return int(stamp.get("sec", 0)) * 1_000_000_000 + int(stamp.get("nanosec", 0))
    return int(getattr(stamp, "sec", 0)) * 1_000_000_000 + int(getattr(stamp, "nanosec", 0))


def build_trajectory_json(
    poses: Iterable[Any],
    *,
    source_pbstream: str,
    trajectory_id: int,
    frame_id: str = "map",
) -> dict[str, Any]:
    """Build the versioned trajectory document from optimized poses.

    The first valid pose defines a local frame.  Local coordinates are obtained
    with the full SE(2) transform ``R(-yaw0) * (p - p0)``.  ``global`` retains
    the original map-frame coordinates and all quaternion/tilt information.
    """
    records = []
    for index, item in enumerate(poses):
        pose = item.get("pose", item) if isinstance(item, Mapping) else getattr(item, "pose", item)
        x, y, z, qx, qy, qz, qw = _pose_values(pose)
        if not all(math.isfinite(v) for v in (x, y, z, qx, qy, qz, qw)):
            continue
        roll, pitch, yaw = quaternion_to_euler(qx, qy, qz, qw)
        records.append({
            "index": index,
            "timestamp_ns": _stamp_ns(item),
            "global": {
                "x": x, "y": y, "z": z,
                "roll": roll, "pitch": pitch, "yaw": yaw,
                "quaternion": {"x": qx, "y": qy, "z": qz, "w": qw},
            },
        })
    if not records:
        raise ValueError("Cartographer returned no finite trajectory poses")

    origin = records[0]
    x0, y0 = origin["global"]["x"], origin["global"]["y"]
    z0, yaw0 = origin["global"]["z"], origin["global"]["yaw"]
    c, s = math.cos(yaw0), math.sin(yaw0)
    for record in records:
        gp = record["global"]
        dx, dy = gp["x"] - x0, gp["y"] - y0
        record["pose"] = {
            "x": c * dx + s * dy,
            "y": -s * dx + c * dy,
            "z": gp["z"] - z0,
            "roll": gp["roll"],
            "pitch": gp["pitch"],
            "yaw": _wrap_pi(gp["yaw"] - yaw0),
        }

    return {
        "schema": "fliftbot.cartographer_trajectory.v1",
        "frame_id": frame_id,
        "trajectory_id": int(trajectory_id),
        "source": {"pbstream": source_pbstream, "pose_source": "cartographer_trajectory_query"},
        "origin": {
            "timestamp_ns": origin["timestamp_ns"],
            "frame_id": frame_id,
            "global": origin["global"],
        },
        "normalization": {
            "description": "local = R(-origin.yaw) * (global_xy - origin_xy)",
            "origin_global": {"x": x0, "y": y0, "z": z0, "yaw": yaw0},
            "rotation_global_to_local_yaw": -yaw0,
            "translation_global_to_local": {"x": -x0, "y": -y0, "z": -z0},
        },
        "poses": records,
    }
