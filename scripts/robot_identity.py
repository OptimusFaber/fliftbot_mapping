"""Resolve the Fliftbot ROS namespace from the machine hostname."""

from __future__ import annotations

import re
import socket


def robot_number(hostname: str | None = None) -> str:
    name = hostname or socket.gethostname()
    match = re.search(r"(\d+)$", name)
    if not match:
        raise ValueError(
            f"hostname {name!r} does not end with a robot number "
            "(expected e.g. flbot204)")
    return match.group(1)


def robot_namespace(hostname: str | None = None) -> str:
    return f"fliftbot{robot_number(hostname)}"


def mc_topic(hostname: str | None = None) -> str:
    return f"/{robot_namespace(hostname)}/mc_telemetry"
