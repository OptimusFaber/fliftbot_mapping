#!/usr/bin/env python3
"""Start Cartographer only after a LiDAR scan has matched the saved occupancy map."""
import argparse
import math
from pathlib import Path
import sys
import threading
import time

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.utilities import remove_ros_args
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from cartographer_ros_msgs.srv import StartTrajectory
from tf2_ros import Buffer, TransformListener
from scipy.spatial import cKDTree

import direct_udp_mc_localizer as matcher
from robot_identity import labels_topic


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
        self.matches = []
        self.marker_seed = None
        self.marker_time = 0.0
        if args.initial_localization == 'markers':
            self.local_to_pb = np.linalg.inv(np.load(args.map_dir / 'pbstream_to_local_matrix.npy'))
            self.marker_topic = args.marker_pose_topic or labels_topic()
            self.marker_frame = self.marker_topic.strip('/').split('/')[0] + '/odom'
            self.marker_subscription = self.create_subscription(Odometry, self.marker_topic, self.marker, 10)
            self.get_logger().info(f'initial marker odometry: {self.marker_topic}, frame={self.marker_frame}')
        self.subscription = self.create_subscription(LaserScan, args.scan_topic, self.scan, qos_profile_sensor_data)
        self.create_timer(.2, self.start)

    def marker(self, message):
        try:
            if message.header.frame_id.lstrip('/') != self.marker_frame:
                raise ValueError(f'unexpected frame_id {message.header.frame_id!r}')
            if message.child_frame_id.lstrip('/') != self.marker_frame.rsplit('/', 1)[0] + '/base_footprint':
                raise ValueError(f'unexpected child_frame_id {message.child_frame_id!r}')
            stamp = message.header.stamp.sec + message.header.stamp.nanosec * 1e-9
            x, y = message.pose.pose.position.x, message.pose.pose.position.y
            q = message.pose.pose.orientation
            norm = math.sqrt(q.x*q.x + q.y*q.y + q.z*q.z + q.w*q.w)
            if abs(norm-1.) > .01 or abs(q.x) > .01 or abs(q.y) > .01:
                raise ValueError('invalid planar orientation quaternion')
            yaw = 2*math.atan2(q.z, q.w)
            if not all(math.isfinite(v) for v in (stamp, x, y, yaw)) or not 0 <= time.time()-stamp <= self.args.marker_max_age:
                return
            point = self.local_to_pb @ np.array([x, y, 1.0])
            direction = self.local_to_pb[:2, :2] @ np.array([math.cos(yaw), math.sin(yaw)])
            self.marker_seed = (float(point[0]), float(point[1]), math.atan2(direction[1], direction[0]))
            self.marker_time = stamp
        except (ValueError, TypeError) as exc:
            self.get_logger().warn(f'invalid marker pose: {exc}')

    def scan(self, message):
        if self.started or self.running or self.candidate is not None:
            return
        if self.args.initial_localization == 'markers' and (self.marker_seed is None or time.time()-self.marker_time > self.args.marker_max_age):
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
            seed = (self.marker_seed if self.args.initial_localization == 'markers' else
                    tuple(map(float, self.args.initial_pose.split(','))) if self.args.initial_pose else
                    matcher.global_pose(points, self.field, self.resolution, self.origin, self.shape))
            if len(seed) != 3:
                raise ValueError("initial_pose must be X_MAP,Y_MAP,YAW_RAD")
            candidate, rmse, overlap = matcher.icp(points, self.tree, seed)
            if math.isfinite(rmse) and rmse <= .20 and overlap >= .50:
                if self.matches:
                    previous = self.matches[-1]
                    distance = math.hypot(candidate[0]-previous[0], candidate[1]-previous[1])
                    angle = abs(math.atan2(math.sin(candidate[2]-previous[2]), math.cos(candidate[2]-previous[2])))
                    if distance > self.args.stability_distance or angle > math.radians(self.args.stability_yaw_deg):
                        self.matches.clear()
                self.matches.append(candidate)
                if len(self.matches) >= self.args.stable_scans:
                    self.candidate = candidate
                    self.get_logger().warn(f"map acquisition tentative after {len(self.matches)} scans: pose={candidate}, rmse={rmse:.3f}, overlap={overlap:.3f}; spatial correctness not verified")
            else:
                self.matches.clear()
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
    parser.add_argument('--initial-localization', choices=('lidar', 'markers'), default='lidar')
    parser.add_argument('--marker-pose-topic', default='')
    parser.add_argument('--marker-max-age', type=float, default=0.5)
    parser.add_argument('--stable-scans', type=int, default=5)
    parser.add_argument('--stability-distance', type=float, default=0.3)
    parser.add_argument('--stability-yaw-deg', type=float, default=10.0)
    args = parser.parse_args(remove_ros_args(sys.argv)[1:])
    if args.stable_scans < 2 or args.marker_max_age <= 0 or args.stability_distance <= 0 or args.stability_yaw_deg <= 0:
        parser.error('stability and age parameters must be positive; stable-scans >= 2')
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
