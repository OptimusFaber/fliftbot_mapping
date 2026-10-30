#!/usr/bin/env python3
"""Receive Pandar40P/DePHAN UDP and publish a base_link LaserScan."""

from __future__ import annotations

import argparse
import math
import socket
import threading
import time

import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan

import receive_pandar40p as pandar
import receive_scan as dephan
from lidar_profiles import load_profile


def points_to_scan(points, stamp, *, bins=720, range_min=0.15, range_max=30.0,
                   frame_id="base_link", scan_time=0.05):
    """Bin base_link XY points into LaserScan; exported separately for tests."""
    msg = LaserScan()
    msg.header.stamp = stamp
    msg.header.frame_id = frame_id
    msg.angle_min = -math.pi
    msg.angle_increment = 2.0 * math.pi / bins
    msg.angle_max = math.pi - msg.angle_increment
    msg.range_min, msg.range_max = range_min, range_max
    msg.scan_time = scan_time
    # Re-binning after a mount transform changes bearing order. There is no
    # valid beam time in that order; don't invent future acquisition times.
    msg.time_increment = 0.0
    points_array = np.asarray(points, dtype=float).reshape((-1, 2))
    ranges = np.full(bins, np.inf, dtype=float)
    if len(points_array):
        radii = np.hypot(points_array[:, 0], points_array[:, 1])
        valid = (radii >= range_min) & (radii <= range_max)
        angles = np.arctan2(points_array[valid, 1], points_array[valid, 0])
        indices = (((angles + math.pi) / msg.angle_increment).astype(np.int64)
                   % bins)
        np.minimum.at(ranges, indices, radii[valid])
    msg.ranges = ranges.tolist()
    msg.intensities = []
    return msg


def receive_scans(sock, sensor, source):
    if sensor == "dephan":
        current, scan = None, [None] * dephan.POINTS_PER_SCAN
        while True:
            try:
                data, address = sock.recvfrom(2048)
            except socket.timeout:
                continue
            if (source and address[0] != source) or len(data) != dephan.PACKET_SIZE:
                continue
            host_t = time.time()
            packet = dephan.decode_packet(data, host_t)
            if current is None:
                current = packet.revolution
            elif packet.revolution != current:
                rows = list(dephan.scan_rows(scan))
                if len(rows) >= 100:
                    yield max(row[1] for row in rows), [(row[6], row[7]) for row in rows]
                scan = [None] * dephan.POINTS_PER_SCAN
                current = packet.revolution
            start = packet.encoder
            end = min(start + dephan.POINTS_PER_PACKET, dephan.POINTS_PER_SCAN)
            if start < dephan.POINTS_PER_SCAN:
                scan[start:end] = [(d, i, packet.timestamp, host_t)
                                   for d, i in packet.points[:end - start]]
    else:
        last_az, points, stamp = None, [], 0.0
        while True:
            try:
                data, address = sock.recvfrom(2048)
            except socket.timeout:
                continue
            if (source and address[0] != source) or len(data) not in pandar.VALID_SIZES:
                continue
            host_t = time.time()
            try:
                packet = pandar.decode_packet(data, host_t, {pandar.HORIZON_CHANNEL})
            except ValueError:
                continue
            if last_az is not None and last_az - packet.azimuth0_deg > 180.0:
                if len(points) >= 100:
                    yield stamp, points
                points, stamp = [], 0.0
            for _, channel, azimuth, distance, _, t_host, _ in packet.points:
                if 0.15 <= distance:
                    x, y, _ = pandar.xyz(distance, azimuth, channel)
                    points.append((x, y))
                    stamp = max(stamp, t_host)
            last_az = packet.azimuth0_deg


class DirectUDPScanPublisher(Node):
    def __init__(self, args):
        super().__init__("direct_udp_scan_publisher")
        profile = load_profile(args.sensor)
        port = args.port if args.port is not None else profile["udp_port"]
        source = profile["source_ip"] if args.source is None else args.source
        mount_x = profile["mount"][0] if args.sensor_x is None else args.sensor_x
        mount_y = profile["mount"][1] if args.sensor_y is None else args.sensor_y
        mount_yaw_deg = (profile["sensor_yaw_deg"] if args.sensor_yaw_deg is None
                         else args.sensor_yaw_deg)
        mount_yaw = math.radians(mount_yaw_deg)
        self.pub = self.create_publisher(LaserScan, args.scan_topic, 10)
        self.scan_time = 1.0 / args.scan_rate_hz
        self.range_max = args.range_max
        self.stop = threading.Event()
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 16 << 20)
        self.sock.settimeout(0.5)
        self.sock.bind((args.bind, port))
        self.get_logger().info(
            f"{args.sensor} UDP {args.bind}:{port}, source={source}; publishing "
            f"{args.scan_topic} in base_link at {args.scan_rate_hz:g} Hz "
            f"(mount=({mount_x:.3f},{mount_y:.3f},"
            f"{mount_yaw_deg:.3f} deg)"
        )
        self.thread = threading.Thread(
            target=self.receive_loop,
            args=(args.sensor, source, mount_x, mount_y, mount_yaw), daemon=True)
        self.thread.start()

    def receive_loop(self, sensor, source, mount_x, mount_y, mount_yaw):
        c, s = math.cos(mount_yaw), math.sin(mount_yaw)
        scans = receive_scans(self.sock, sensor, source)
        while not self.stop.is_set():
            try:
                host_stamp, sensor_points = next(scans)
            except socket.timeout:
                continue
            except OSError:
                return
            sensor_array = np.asarray(sensor_points, dtype=float).reshape((-1, 2))
            base_points = sensor_array @ np.asarray(((c, s), (-s, c)))
            base_points += np.asarray((mount_x, mount_y))
            # Publish in ROS host-clock domain; this also avoids putting the
            # sensor's unsynchronised clock into Cartographer's TF lookup.
            stamp = self.get_clock().now().to_msg()
            self.pub.publish(points_to_scan(
                base_points, stamp, frame_id="base_link",
                scan_time=self.scan_time, range_max=self.range_max))

    def close(self):
        self.stop.set()
        self.sock.close()
        self.thread.join(timeout=1.0)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sensor", choices=("pandar", "dephan"), required=True)
    p.add_argument("--bind", default="0.0.0.0")
    p.add_argument("--port", type=int)
    p.add_argument("--source")
    p.add_argument("--scan-topic", default="/scan")
    p.add_argument("--scan-rate-hz", type=float, default=20.0)
    p.add_argument("--range-max", type=float, default=30.0)
    p.add_argument("--sensor-x", type=float)
    p.add_argument("--sensor-y", type=float)
    p.add_argument("--sensor-yaw-deg", type=float,
                   help="override the shared sensor profile")
    args = p.parse_args()
    if args.scan_rate_hz <= 0:
        p.error("--scan-rate-hz must be positive")
    from rclpy.signals import SignalHandlerOptions
    rclpy.init(signal_handler_options=SignalHandlerOptions.NO)
    node = DirectUDPScanPublisher(args)
    from runtime_signals import spin_node
    spin_node(node, before_destroy=node.close)


if __name__ == "__main__":
    main()
