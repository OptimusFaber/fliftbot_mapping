#!/usr/bin/env python3
"""Copy simulator scan/measured-wheel odometry into an isolated localization TF tree."""
import argparse
import copy
import json
import math
from pathlib import Path
import sys

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.utilities import remove_ros_args
from geometry_msgs.msg import TransformStamped
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from tf2_ros import StaticTransformBroadcaster, TransformBroadcaster


class SimInputs(Node):
    def __init__(self, config):
        super().__init__("lidar_nav_sim_inputs")
        self.scan_pub = self.create_publisher(LaserScan, "/lidar_nav/scan", 10)
        self.odom_pub = self.create_publisher(Odometry, "/lidar_nav/wheel_odom", 10)
        self.tf = TransformBroadcaster(self)
        self.static_tf = StaticTransformBroadcaster(self)
        mount = config["scan_to_base"]
        tf = TransformStamped()
        tf.header.stamp = self.get_clock().now().to_msg()
        tf.header.frame_id = "lidar_nav_base"
        tf.child_frame_id = "lidar_nav_sensor"
        tf.transform.translation.x = float(mount["x_m"])
        tf.transform.translation.y = float(mount["y_m"])
        yaw = math.radians(mount["yaw_deg"])
        tf.transform.rotation.z, tf.transform.rotation.w = math.sin(yaw/2), math.cos(yaw/2)
        self.static_tf.sendTransform(tf)
        self.create_subscription(LaserScan, "/lidar_top/scan", self.scan, qos_profile_sensor_data)
        self.create_subscription(Odometry, "/wheel_encoder_odom", self.odom, qos_profile_sensor_data)

    def scan(self, message):
        message = copy.deepcopy(message)
        message.header.frame_id = "lidar_nav_sensor"
        self.scan_pub.publish(message)

    def odom(self, message):
        message = copy.deepcopy(message)
        message.header.frame_id = "lidar_nav_odom"
        message.child_frame_id = "lidar_nav_base"
        self.odom_pub.publish(message)
        tf = TransformStamped()
        tf.header, tf.child_frame_id = message.header, message.child_frame_id
        position = message.pose.pose.position
        tf.transform.translation.x, tf.transform.translation.y = position.x, position.y
        tf.transform.rotation = message.pose.pose.orientation
        self.tf.sendTransform(tf)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--map-dir", type=Path, required=True)
    args = parser.parse_args(remove_ros_args(sys.argv)[1:])
    config = json.loads((args.map_dir / "navigation_config.json").read_text())
    from rclpy.signals import SignalHandlerOptions
    rclpy.init(signal_handler_options=SignalHandlerOptions.NO)
    node = SimInputs(config)
    from runtime_signals import spin_node
    spin_node(node)


if __name__ == "__main__":
    main()
