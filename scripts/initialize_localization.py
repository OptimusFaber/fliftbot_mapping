#!/usr/bin/env python3
"""Start Cartographer only after a LiDAR scan has matched the saved occupancy map."""
import argparse
import math
from pathlib import Path
import sys
import threading

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.utilities import remove_ros_args
from sensor_msgs.msg import LaserScan
from cartographer_ros_msgs.srv import StartTrajectory
from tf2_ros import Buffer, TransformListener
from scipy.spatial import cKDTree

import direct_udp_mc_localizer as matcher


class Initializer(Node):
    def __init__(self, args):
        super().__init__("lidar_nav_initializer")
        self.args = args
        points, self.field, self.resolution, self.origin, self.shape = matcher.load_map(args.map_dir / "map.yaml")
        self.tree = cKDTree(points)
        self.buffer = Buffer()
        self.listener = TransformListener(self.buffer, self)
        self.client = self.create_client(StartTrajectory, "/start_trajectory")
        self.started = False
        self.running = False
        self.candidate = None
        self.future = None
        self.subscription = self.create_subscription(LaserScan, args.scan_topic, self.scan, qos_profile_sensor_data)
        self.create_timer(.2, self.start)

    def scan(self, message):
        if self.started or self.running or self.candidate is not None:
            return
        ranges = np.asarray(message.ranges)
        valid = np.isfinite(ranges) & (ranges >= message.range_min) & (ranges <= message.range_max)
        angles = message.angle_min + np.arange(len(ranges)) * message.angle_increment
        points = np.column_stack((ranges[valid]*np.cos(angles[valid]), ranges[valid]*np.sin(angles[valid])))
        if len(points) < 100:
            return
        if message.header.frame_id != self.args.base_frame:
            try:
                tf = self.buffer.lookup_transform(self.args.base_frame, message.header.frame_id, rclpy.time.Time())
            except Exception:
                return
            q = tf.transform.rotation
            yaw = math.atan2(2*q.w*q.z, 1-2*q.z*q.z)
            points = matcher.transform(points, (tf.transform.translation.x, tf.transform.translation.y, yaw))
        self.running = True
        threading.Thread(target=self.match, args=(points,), daemon=True).start()

    def match(self, points):
        try:
            seed = (tuple(map(float, self.args.initial_pose.split(','))) if self.args.initial_pose else
                    matcher.global_pose(points, self.field, self.resolution, self.origin, self.shape))
            if len(seed) != 3:
                raise ValueError("initial_pose must be X_MAP,Y_MAP,YAW_RAD")
            candidate, rmse, overlap = matcher.icp(points, self.tree, seed)
            if math.isfinite(rmse) and rmse <= .20 and overlap >= .50:
                self.candidate = candidate
                self.get_logger().info(f"map acquisition accepted: pose={candidate}, rmse={rmse:.3f}, overlap={overlap:.3f}")
            else:
                self.get_logger().warn(f"map acquisition rejected: rmse={rmse:.3f}, overlap={overlap:.3f}")
        except Exception as exc:
            self.get_logger().error(f"map acquisition failed: {exc}")
        finally:
            self.running = False

    def start(self):
        if self.started or self.candidate is None or not self.client.service_is_ready():
            return
        if self.future is None:
            x, y, yaw = self.candidate
            request = StartTrajectory.Request()
            request.configuration_directory = str(self.args.configuration_directory)
            request.configuration_basename = self.args.configuration_basename
            request.use_initial_pose = True
            request.relative_to_trajectory_id = 0
            request.initial_pose.position.x, request.initial_pose.position.y = x, y
            request.initial_pose.orientation.z, request.initial_pose.orientation.w = math.sin(yaw/2), math.cos(yaw/2)
            self.future = self.client.call_async(request)
        elif self.future.done():
            response = self.future.result()
            if response.status.code != 0:
                self.get_logger().error(f"Cartographer start refused: {response.status.message}")
                self.future = None
                return
            self.started = True
            self.get_logger().info(f"localized trajectory {response.trajectory_id} started on frozen map")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--map-dir", type=Path, required=True)
    parser.add_argument("--configuration-directory", type=Path, required=True)
    parser.add_argument("--configuration-basename", default="fliftbot_2d_pure_localization.lua")
    parser.add_argument("--scan-topic", default="/scan")
    parser.add_argument("--base-frame", default="base_link")
    parser.add_argument("--initial-pose", default="", help="optional X_MAP,Y_MAP,YAW_RAD; still scan-validated")
    args = parser.parse_args(remove_ros_args(sys.argv)[1:])
    if args.initial_pose:
        try:
            values = tuple(map(float, args.initial_pose.split(',')))
            if len(values) != 3 or not all(math.isfinite(v) for v in values):
                raise ValueError()
        except ValueError:
            parser.error("initial-pose must be finite X_MAP,Y_MAP,YAW_RAD")
    from rclpy.signals import SignalHandlerOptions
    rclpy.init(signal_handler_options=SignalHandlerOptions.NO)
    node = Initializer(args)
    from runtime_signals import spin_node
    spin_node(node)


if __name__ == "__main__":
    main()
