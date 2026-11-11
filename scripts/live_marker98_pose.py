#!/usr/bin/env python3
"""Export Cartographer ``map -> base_link`` as marker-98 x,y,alpha,ts.

The node publishes one JSON record per TF sample on a String topic and appends
the same records to CSV.  The marker coordinates are supplied explicitly so
the physical marker is not confused with a trajectory node index.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import sys
import time

import rclpy
from geometry_msgs.msg import TransformStamped
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from std_msgs.msg import String
from tf2_ros import Buffer, TransformListener

from marker98_pose import pose_in_marker98_frame


from navigation_output import LidarSensorSender, LIDAR_VARIANCE
from localization_journal import Journal, load_scenario
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry


class Marker98PoseNode(Node):
    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__("marker98_pose_export")
        reference = json.loads(Path(args.marker_reference).read_text(encoding="utf-8"))
        origin_xy = reference.get("origin_map_xy_m", reference.get("marker98_map_xy_m"))
        if origin_xy is None:
            raise ValueError(
                "reference must contain origin_map_xy_m or marker98_map_xy_m")
        self.marker_x = float(origin_xy[0])
        self.marker_y = float(origin_xy[1])
        self.marker_alpha = float(reference.get(
            "origin_map_yaw_rad", reference.get("marker98_map_yaw_rad", 0.0)))
        from map_bundle import read_y_axis_sign
        self.y_axis_sign = read_y_axis_sign(Path(args.marker_reference))
        self.scenario = load_scenario(args.scenario_config or None)
        journal_path = args.journal or (str(args.csv) + ".journal.jsonl" if args.csv else "")
        self.journal = Journal(journal_path, self.scenario, {
            "algorithm": "cartographer", "reference_file": str(Path(args.marker_reference).resolve()),
            "reference": reference, "map_frame": args.map_frame, "base_frame": args.base_frame,
            "pose_offset_m": [args.pose_offset_x_m, args.pose_offset_y_m],
            "clock": "ros", "units": "metres/radians",
            "runtime_settings": vars(args),
            "navigation_config": json.loads((Path(args.marker_reference).parent / "navigation_config.json").read_text())
                if (Path(args.marker_reference).parent / "navigation_config.json").is_file() else None,
        }) if journal_path else None
        self.last_scan = None
        self.last_odom = None
        self.last_journal_time = None
        self.scan_sub = self.create_subscription(LaserScan, args.scan_topic, self.on_scan, qos_profile_sensor_data)
        self.odom_sub = self.create_subscription(Odometry, args.odom_topic, self.on_odom, qos_profile_sensor_data)
        self.map_frame = args.map_frame
        self.base_frame = args.base_frame
        self.pose_offset_x = float(args.pose_offset_x_m)
        self.pose_offset_y = float(args.pose_offset_y_m)
        self.buffer = Buffer()
        self.listener = TransformListener(self.buffer, self)
        self.pub = self.create_publisher(String, args.topic, 10)
        self.sensor_sender = (
            None if args.disable_sensor_sender or not args.send_to_navigation
            else LidarSensorSender(args.sensor_socket)
        )
        self.pose_period = 1.0 / args.rate
        self.sensor_send_period = 1.0 / args.sensor_send_rate
        self.next_pose_monotonic = 0.0
        self.next_sensor_send_monotonic = 0.0
        if self.sensor_sender:
            self.get_logger().info(
                f"sending top_navigation [x_m,y_m,yaw_rad] to "
                f"{args.sensor_socket} at "
                f"{args.sensor_send_rate:g} Hz; variance diagonal={LIDAR_VARIANCE}"
            )
        else:
            self.get_logger().info("ContinuousNav output is disabled")
        self.get_logger().info(
            f"pose offset in base_link: x={self.pose_offset_x:+.3f} m, "
            f"y={self.pose_offset_y:+.3f} m"
        )
        self.last_send_warning_monotonic = -math.inf
        self.sender_first_message_logged = False
        self.last_sensor_timestamp = None
        self.skipped_duplicate_timestamps = 0
        self.max_sensor_age = args.max_sensor_age
        self.skipped_stale_measurements = 0
        self.csv_path = Path(args.csv) if args.csv else None
        self.csv_file = None
        self.csv_writer = None
        self.last_flush = time.monotonic()
        if self.csv_path:
            self.csv_path.parent.mkdir(parents=True, exist_ok=True)
            is_new = not self.csv_path.exists() or self.csv_path.stat().st_size == 0
            self.csv_file = self.csv_path.open("a", newline="", encoding="utf-8")
            self.csv_writer = csv.writer(self.csv_file)
            if is_new:
                self.csv_writer.writerow(["x", "y", "alpha_deg", "ts"])
                self.csv_file.flush()
        # Poll TF fast enough for both independently rate-limited outputs.
        self.timer = self.create_timer(
            1.0 / max(args.rate, args.sensor_send_rate), self.tick)

    def on_scan(self, message):
        now = self.get_clock().now().nanoseconds * 1e-9
        stamp = message.header.stamp.sec + message.header.stamp.nanosec * 1e-9
        count = sum(math.isfinite(v) and message.range_min <= v <= message.range_max for v in message.ranges)
        self.last_scan = {"received_s": now, "stamp_s": stamp, "valid_points": count,
                          "total_points": len(message.ranges),
                          "receive_interval_s": now-self.last_scan["received_s"] if self.last_scan else None}

    def on_odom(self, message):
        self.last_odom = {"received_s": self.get_clock().now().nanoseconds * 1e-9,
                          "stamp_s": message.header.stamp.sec + message.header.stamp.nanosec * 1e-9,
                          "linear_speed_m_s": message.twist.twist.linear.x,
                          "angular_speed_rad_s": message.twist.twist.angular.z}

    def record_state(self, stamp=None, pose=None, reason=None):
        if self.journal is None:
            return
        now = self.get_clock().now().nanoseconds * 1e-9
        if self.last_journal_time is not None and now-self.last_journal_time < self.pose_period and now >= self.last_journal_time:
            return
        self.last_journal_time = now
        fields = {}
        for name, value in (("scan", self.last_scan), ("odom", self.last_odom)):
            fields[name] = value
            fields[name+"_available"] = value is not None and 0 <= now-value["received_s"] <= self.scenario["max_sensor_gap_s"]
        self.journal.observe(now, stamp, pose, reason, **fields)

    @staticmethod
    def _advance_deadline(deadline: float, period: float, now: float) -> float:
        """Advance a periodic deadline without accumulating timer drift."""
        if deadline <= 0.0:
            return now + period
        while deadline <= now:
            deadline += period
        return deadline

    def tick(self) -> None:
        try:
            tf: TransformStamped = self.buffer.lookup_transform(
                self.map_frame, self.base_frame, rclpy.time.Time())
        except Exception:
            self.record_state(reason="tf_unavailable")
            return
        q = tf.transform.rotation
        alpha = math.atan2(2.0 * (q.w * q.z), 1.0 - 2.0 * q.z * q.z)
        c, s = math.cos(alpha), math.sin(alpha)
        corrected_x = tf.transform.translation.x + c * self.pose_offset_x - s * self.pose_offset_y
        corrected_y = tf.transform.translation.y + s * self.pose_offset_x + c * self.pose_offset_y
        x, y, alpha_rad = pose_in_marker98_frame(
            corrected_x, corrected_y, alpha,
            self.marker_x, self.marker_y, self.marker_alpha, y_axis_sign=self.y_axis_sign)
        alpha_deg = math.degrees(alpha_rad)
        journal_stamp = tf.header.stamp.sec + tf.header.stamp.nanosec * 1e-9
        self.record_state(journal_stamp, (x, y, alpha_rad))
        ts = tf.header.stamp.sec + tf.header.stamp.nanosec * 1e-9
        now_s = self.get_clock().now().nanoseconds * 1e-9
        sensor_age = now_s - ts
        now_monotonic = time.monotonic()
        if now_monotonic >= self.next_pose_monotonic:
            record = {"x": x, "y": y, "alpha": alpha_deg,
                      "alpha_unit": "deg", "ts": ts}
            self.pub.publish(String(data=json.dumps(record, separators=(",", ":"))))
            if self.csv_writer:
                self.csv_writer.writerow((x, y, alpha_deg, ts))
                if now_monotonic - self.last_flush >= 1.0:
                    self.csv_file.flush()
                    self.last_flush = now_monotonic
            self.next_pose_monotonic = self._advance_deadline(
                self.next_pose_monotonic, self.pose_period, now_monotonic)

        sensor_send_due = now_monotonic >= self.next_sensor_send_monotonic
        is_new_sensor_sample = (
            math.isfinite(ts)
            and ts > 0.0
            and sensor_age <= self.max_sensor_age
            and sensor_age >= -0.25
            and (self.last_sensor_timestamp is None
                 or ts > self.last_sensor_timestamp + 1e-6)
        )
        if self.sensor_sender and sensor_send_due and is_new_sensor_sample:
            try:
                self.sensor_sender.send(ts, x, y, alpha_deg)
                self.last_sensor_timestamp = ts
                self.next_sensor_send_monotonic = self._advance_deadline(
                    self.next_sensor_send_monotonic,
                    self.sensor_send_period,
                    now_monotonic,
                )
                if not self.sender_first_message_logged:
                    self.get_logger().info(
                        "sent first top_navigation measurement: "
                        f"ts={ts:.6f}, value=[{x:.3f},{y:.3f},"
                        f"{math.radians(alpha_deg):.6f}]"
                    )
                    self.sender_first_message_logged = True
            except self.sensor_sender.zmq.Again:
                if now_monotonic - self.last_send_warning_monotonic >= 5.0:
                    self.get_logger().warning("ContinuousNav lidar send would block")
                    self.last_send_warning_monotonic = now_monotonic
        elif self.sensor_sender and sensor_send_due:
            if sensor_age > self.max_sensor_age or sensor_age < -0.25:
                self.skipped_stale_measurements += 1
                if self.skipped_stale_measurements % 100 == 1:
                    self.get_logger().warning(
                        "not sending stale localization to ContinuousNav: "
                        f"age={sensor_age:.3f}s, max={self.max_sensor_age:.3f}s"
                    )
            else:
                self.skipped_duplicate_timestamps += 1
            if (sensor_age <= self.max_sensor_age and sensor_age >= -0.25
                    and self.skipped_duplicate_timestamps % 100 == 1):
                self.get_logger().warning(
                    "not sending stale/duplicate localization timestamp: "
                    f"current={ts:.6f}, last={self.last_sensor_timestamp}"
                )

    def destroy_node(self):
        if self.journal:
            self.journal.close(self.get_clock().now().nanoseconds * 1e-9)
        if self.sensor_sender:
            self.sensor_sender.close()
            self.sensor_sender = None
        if self.csv_file:
            self.csv_file.flush()
            self.csv_file.close()
            self.csv_file = None
        return super().destroy_node()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--marker-reference", required=True,
                   help="JSON origin/reference belonging to the selected pbstream")
    p.add_argument("--topic", default="/localization/marker98_pose")
    p.add_argument("--map-frame", default="map")
    p.add_argument("--base-frame", default="base_link")
    p.add_argument("--csv", default="")
    p.add_argument("--journal", default="", help="new JSONL file; default CSV path + .journal.jsonl")
    p.add_argument("--scenario-config", default="", help="scenario JSON with acceptance thresholds")
    p.add_argument("--scan-topic", default="/scan")
    p.add_argument("--odom-topic", default="/wheel_odom")
    p.add_argument("--rate", type=float, default=10.0)
    p.add_argument(
        "--sensor-send-rate", type=float, default=10.0,
        help="fixed maximum ContinuousNav transmission rate in Hz",
    )
    p.add_argument(
        "--send-to-navigation", action=argparse.BooleanOptionalAction,
        default=True,
        help="enable/disable sending top_navigation to ContinuousNav",
    )
    p.add_argument(
        "--sensor-socket",
        default="ipc:///tmp/fliftbot_continuous_nav_sensor.ipc",
        help="ContinuousNav SensorSender-compatible ZMQ PUB endpoint",
    )
    p.add_argument(
        "--disable-sensor-sender", action="store_true",
        help="publish ROS/CSV only",
    )
    p.add_argument(
        "--max-sensor-age", type=float, default=0.5,
        help="do not send poses older than this many seconds to ContinuousNav",
    )
    p.add_argument(
        "--pose-offset-x-m", type=float, default=0.0,
        help="experimental output-point offset in base_link X (forward), metres",
    )
    p.add_argument(
        "--pose-offset-y-m", type=float, default=0.0,
        help="experimental output-point offset in base_link Y (left), metres",
    )
    from rclpy.utilities import remove_ros_args
    args = p.parse_args(remove_ros_args(sys.argv)[1:])
    if args.rate <= 0:
        p.error("--rate must be positive")
    if args.sensor_send_rate <= 0:
        p.error("--sensor-send-rate must be positive")
    if args.max_sensor_age <= 0:
        p.error("--max-sensor-age must be positive")
    from rclpy.signals import SignalHandlerOptions
    rclpy.init(signal_handler_options=SignalHandlerOptions.NO)
    node = Marker98PoseNode(args)
    from runtime_signals import spin_node
    spin_node(node)


if __name__ == "__main__":
    main()
