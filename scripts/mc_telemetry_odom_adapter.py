#!/usr/bin/env python3
"""ROS 2 adapter: hostname-selected ``mc_telemetry`` -> ``nav_msgs/Odometry``.

This is intentionally a small, live-only starting point.  It does not open a
serial port and it does not read a bag by itself: when started, it subscribes
to the known MC topic and publishes wheel odometry for Cartographer.  The
speed sign is configurable because the firmware currently publishes
``-enc2mps(...)`` and must be verified on the particular robot.

Example (inside the ROS/Docker environment):
    python3 mc_telemetry_odom_adapter.py --speed-sign -1

The output frame contract is ``odom -> base_link``.  Replace the bicycle
model or wheelbase after validating the vehicle calibration.
"""

from __future__ import annotations

import argparse
import math
from typing import Optional

from builtin_interfaces.msg import Time
import rclpy
from geometry_msgs.msg import Quaternion
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from fliftbot_interfaces.msg import MC
from geometry_msgs.msg import TransformStamped
from tf2_ros import TransformBroadcaster
from robot_identity import mc_topic
from wheel_motion import telemetry_twist


def yaw_quaternion(yaw: float) -> Quaternion:
    q = Quaternion()
    q.z = math.sin(yaw * 0.5)
    q.w = math.cos(yaw * 0.5)
    return q


class MCTelemetryOdomAdapter(Node):
    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__("mc_telemetry_odom_adapter")
        self.topic = args.topic
        self.frame_id = args.frame_id
        self.child_frame_id = args.child_frame_id
        self.wheelbase = args.wheelbase
        self.speed_sign = args.speed_sign
        self.steering_zero_deg = args.steering_zero_deg
        self.model = args.model
        self.steering_sign = args.steering_sign
        self.publish_odom = not args.no_publish
        self.x = 0.0
        self.y = 0.0
        self.yaw = 0.0
        self.last_t: Optional[float] = None
        self.last_msg_t: Optional[float] = None
        self.count = 0
        qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
        )
        self.odom_pub = self.create_publisher(Odometry, args.odom_topic, 10)
        # slam_toolbox/Cartographer need this edge to connect the scan frame
        # to the wheel odometry frame.  Publishing the same pose in Odometry
        # and TF also keeps RViz and downstream consumers consistent.
        self.tf_broadcaster = TransformBroadcaster(self)
        self.sub = self.create_subscription(MC, self.topic, self.callback, qos)
        self.get_logger().info(
            f"Listening to {self.topic} (BEST_EFFORT); output {args.odom_topic}"
        )

    def callback(self, msg: MC) -> None:
        # feedback_timestamp is a wall-clock Unix timestamp in the current
        # firmware.  Fall back to ROS reception time if it is absent/invalid.
        msg_t = float(msg.feedback_timestamp)
        timestamp_is_valid = math.isfinite(msg_t) and msg_t > 0.0
        if not timestamp_is_valid:
            now = self.get_clock().now()
            t = now.nanoseconds * 1e-9
        else:
            t = msg_t
        if self.last_t is None:
            self.last_t = t
            self.last_msg_t = msg_t
            return
        dt = t - self.last_t
        self.last_t = t
        self.last_msg_t = msg_t
        if not (0.0 < dt <= 1.0):
            self.get_logger().warn(f"Ignoring telemetry dt={dt:.3f}s")
            return

        v, wz = telemetry_twist(
            float(msg.enc_lin_vel_mps), float(msg.enc_whl_yaw_dgr),
            speed_sign=self.speed_sign, steering_zero_deg=self.steering_zero_deg,
            wheelbase=self.wheelbase, model=self.model, steering_sign=self.steering_sign)
        # Exact constant-twist arc, matching the offline map builder.
        theta = wz * dt
        dx = v * math.sin(theta) / wz if abs(wz) > 1e-9 else v * dt
        dy = v * (1 - math.cos(theta)) / wz if abs(wz) > 1e-9 else 0.0
        self.x += math.cos(self.yaw) * dx - math.sin(self.yaw) * dy
        self.y += math.sin(self.yaw) * dx + math.cos(self.yaw) * dy
        self.yaw = math.atan2(math.sin(self.yaw + wz * dt), math.cos(self.yaw + wz * dt))
        self.count += 1
        if not self.publish_odom:
            return
        if timestamp_is_valid:
            sec = int(t)
            stamp = Time()
            stamp.sec = sec
            stamp.nanosec = int((t - sec) * 1e9)
        else:
            stamp = self.get_clock().now().to_msg()
        out = Odometry()
        out.header.stamp = stamp
        out.header.frame_id = self.frame_id
        out.child_frame_id = self.child_frame_id
        out.pose.pose.position.x = self.x
        out.pose.pose.position.y = self.y
        out.pose.pose.orientation = yaw_quaternion(self.yaw)
        out.twist.twist.linear.x = v
        out.twist.twist.angular.z = wz
        self.odom_pub.publish(out)
        tf = TransformStamped()
        tf.header = out.header
        tf.child_frame_id = out.child_frame_id
        tf.transform.translation.x = self.x
        tf.transform.translation.y = self.y
        tf.transform.rotation = out.pose.pose.orientation
        self.tf_broadcaster.sendTransform(tf)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--topic",
                   help="MC topic; default is derived from hostname suffix")
    p.add_argument("--odom-topic", default="/wheel_odom")
    p.add_argument("--frame-id", default="odom")
    p.add_argument("--child-frame-id", default="base_link")
    p.add_argument("--wheelbase", type=float, default=1.410)
    p.add_argument("--steering-zero-deg", type=float, default=0.0,
                   help="raw wheel angle corresponding to straight travel")
    # navigation_controller_utils.mc_dict_from_msg negates enc_lin_vel_mps;
    # keep that project convention as the safe default.  Override after a
    # short forward-motion test if the particular firmware is different.
    p.add_argument("--speed-sign", type=float, choices=(-1.0, 1.0), default=-1.0)
    p.add_argument("--steering-sign", type=float, choices=(-1., 1.), default=1.)
    p.add_argument("--model", choices=("drive_wheel", "legacy_sin", "bicycle"), default="drive_wheel")
    p.add_argument("--no-publish", action="store_true", help="Subscribe/check only")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.topic = args.topic or mc_topic()
    from rclpy.signals import SignalHandlerOptions
    rclpy.init(signal_handler_options=SignalHandlerOptions.NO)
    node = MCTelemetryOdomAdapter(args)
    from runtime_signals import spin_node
    spin_node(node)


if __name__ == "__main__":
    main()
