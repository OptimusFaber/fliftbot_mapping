#!/usr/bin/env python3
"""Publish DePHAN scans from a growing M360RAW1 recording."""
import argparse
import math
from pathlib import Path
import time

import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan

import receive_scan as dephan
from direct_udp_scan_publisher import points_to_scan


def completed_scans(path):
    path = Path(path)
    while not path.is_file():
        time.sleep(.1)
    with path.open('rb') as stream:
        if stream.read(len(dephan.RAW_MAGIC)) != dephan.RAW_MAGIC:
            raise ValueError(f'{path} is not M360RAW1')
        current, scan = None, [None] * dephan.POINTS_PER_SCAN
        first_revolution = True
        while True:
            blob = stream.read(dephan.RAW_RECORD.size)
            if len(blob) < dephan.RAW_RECORD.size:
                stream.seek(-len(blob), 1)
                time.sleep(.02)
                continue
            host_t, raw = dephan.RAW_RECORD.unpack(blob)
            packet = dephan.decode_packet(raw, host_t)
            if current is None:
                current = packet.revolution
            elif packet.revolution != current:
                rows = list(dephan.scan_rows(scan))
                if not first_revolution and len(rows) >= 100:
                    yield [(row[6], row[7]) for row in rows]
                first_revolution = False
                scan = [None] * dephan.POINTS_PER_SCAN
                current = packet.revolution
            start = packet.encoder
            end = min(start + dephan.POINTS_PER_PACKET, dephan.POINTS_PER_SCAN)
            if start < dephan.POINTS_PER_SCAN:
                scan[start:end] = [(d, i, packet.timestamp, host_t)
                                   for d, i in packet.points[:end-start]]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--recording', required=True)
    parser.add_argument('--scan-topic', default='/scan')
    parser.add_argument('--scan-rate-hz', type=float, default=20.)
    parser.add_argument('--range-max', type=float, default=80.)
    parser.add_argument('--sensor-x', type=float, required=True)
    parser.add_argument('--sensor-y', type=float, required=True)
    parser.add_argument('--sensor-yaw-deg', type=float, required=True)
    args = parser.parse_args()
    if args.scan_rate_hz <= 0:
        parser.error('--scan-rate-hz must be positive')
    from rclpy.signals import SignalHandlerOptions
    rclpy.init(signal_handler_options=SignalHandlerOptions.NO)
    node = Node('m360_recording_scan_publisher')
    pub = node.create_publisher(LaserScan, args.scan_topic, 10)
    yaw = math.radians(args.sensor_yaw_deg)
    c, s = math.cos(yaw), math.sin(yaw)
    try:
        for points in completed_scans(args.recording):
            base = np.asarray(points, float) @ np.array(((c, s), (-s, c)))
            base += (args.sensor_x, args.sensor_y)
            pub.publish(points_to_scan(base, node.get_clock().now().to_msg(),
                                       scan_time=1./args.scan_rate_hz, range_max=args.range_max))
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
