#!/usr/bin/env python3
"""Publish LaserScan from a growing RPLRAW1 file written by record_with_video.sh."""

import argparse
import math
from pathlib import Path
import struct
import time

import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan

from direct_udp_scan_publisher import points_to_scan

MAGIC = b"RPLRAW1\0"
RECORD = struct.Struct("<ddddB7x")


def completed_scans(path):
    """Follow complete records; start at the next revolution after opening."""
    path = Path(path)
    while not path.is_file():
        time.sleep(0.2)
    with path.open("rb") as stream:
        if stream.read(len(MAGIC)) != MAGIC:
            raise ValueError(f"{path} is not an RPLRAW1 recording")
        points, previous_angle, previous_device_t = [], None, None
        while True:
            blob = stream.read(RECORD.size)
            if len(blob) < RECORD.size:
                stream.seek(-len(blob), 1)
                time.sleep(0.02)
                continue
            _, x, y, device_t, _ = RECORD.unpack(blob)
            if not (math.isfinite(x) and math.isfinite(y) and math.isfinite(device_t)):
                continue
            angle = math.atan2(y, x)
            # The recorded robot data uses monotonic device_t, so the angle
            # wrap marks a full turn. A reset is also accepted for other units.
            boundary = (previous_device_t is not None and device_t + 1e-9 < previous_device_t)
            boundary |= previous_angle is not None and abs(angle - previous_angle) > math.pi
            if boundary:
                if len(points) >= 100:
                    yield points
                points = []
            points.append((x, y))
            previous_angle, previous_device_t = angle, device_t


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recording", required=True)
    parser.add_argument("--scan-topic", default="/scan")
    parser.add_argument("--scan-rate-hz", type=float, default=10.0)
    parser.add_argument("--range-max", type=float, default=20.0)
    parser.add_argument("--sensor-x", type=float, required=True)
    parser.add_argument("--sensor-y", type=float, required=True)
    parser.add_argument("--sensor-yaw-deg", type=float, required=True)
    args = parser.parse_args()
    if args.scan_rate_hz <= 0:
        parser.error("--scan-rate-hz must be positive")
    rclpy.init()
    node = Node("rplidar_recording_scan_publisher")
    publisher = node.create_publisher(LaserScan, args.scan_topic, 10)
    yaw = math.radians(args.sensor_yaw_deg)
    c, s = math.cos(yaw), math.sin(yaw)
    node.get_logger().info(f"Following {args.recording}; publishing {args.scan_topic}")
    try:
        for points in completed_scans(args.recording):
            array = np.asarray(points, dtype=float)
            base = array @ np.asarray(((c, s), (-s, c)))
            base += (args.sensor_x, args.sensor_y)
            publisher.publish(points_to_scan(base, node.get_clock().now().to_msg(),
                                             scan_time=1.0 / args.scan_rate_hz,
                                             range_max=args.range_max))
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
