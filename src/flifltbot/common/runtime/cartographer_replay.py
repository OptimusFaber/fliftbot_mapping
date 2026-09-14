"""Replay a DePHAN horizon NPZ run directly into Cartographer ROS 2.

The recorder format is deliberately kept independent of rosbag2.  This module
is the small ROS adapter at the boundary: it reads ``horizon/*.npz`` and
``encoder.jsonl``, rebases the recording clock (by default), and publishes the
same messages Cartographer would receive from a live robot.

The non-ROS helpers are kept importable so the input contract can be tested on
machines where the ROS Python extension is not available for the system
interpreter.
"""

from __future__ import annotations

import argparse
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

from flifltbot.mapping.config import (
    MAX_RANGE,
    MIN_RANGE,
    UP_LIDAR_X,
    UP_LIDAR_Z,
    WHEELBASE_M,
    WHEEL_DIRECTION_OFFSET,
    WHEEL_DIRECTION_SIGN,
    WHEEL_SPEED_SCALE,
)
from flifltbot.mapping.encoder_sync import motion_from_encoders
from flifltbot.mapping.geometry_icp import compose
from flifltbot.mapping.io_frames import list_run_frame_paths
from flifltbot.mapping.source_io import collect_run_encoder_samples


@dataclass(frozen=True)
class ReplayFrame:
    """One valid horizon scan, retaining its original point timestamps."""

    path: Path
    xy: np.ndarray
    point_times: np.ndarray
    stamp: float
    frame: int


def load_replay_frame(path: Path, max_range: float = 30.0) -> ReplayFrame | None:
    """Read a horizon NPZ without the mapping loader's 12 m map cutoff.

    Cartographer's Lua profile accepts 30 m.  The regular mapping loader uses
    a tighter 12 m cutoff for its occupancy-map baseline, which would silently
    discard valid DePHAN returns in a large room, so the replay path has its
    own explicit sensor limits.
    """
    path = Path(path)
    try:
        with np.load(path, allow_pickle=False) as data:
            xy = np.asarray(data['xy'], dtype=np.float64)
            if xy.ndim != 2 or xy.shape[1] != 2:
                return None
            point_times = np.asarray(
                data['t'], dtype=np.float64
            ) if 't' in data.files else np.full(len(xy), np.nan)
            if point_times.shape != (len(xy),):
                point_times = np.full(len(xy), np.nan, dtype=np.float64)
            stamp = float(data['stamp']) if 'stamp' in data.files else float('nan')
            if not np.isfinite(stamp):
                finite = point_times[np.isfinite(point_times)]
                stamp = float(np.median(finite)) if len(finite) else float('nan')
            frame = int(data['frame']) if 'frame' in data.files else int(path.name[:5])
    except (OSError, ValueError, KeyError, IndexError):
        return None
    if not np.isfinite(stamp) or len(xy) < 2:
        return None
    ranges = np.hypot(xy[:, 0], xy[:, 1])
    keep = (
        np.isfinite(xy).all(axis=1)
        & np.isfinite(ranges)
        & (ranges >= float(MIN_RANGE))
        & (ranges <= float(max_range))
    )
    xy = xy[keep]
    point_times = point_times[keep]
    if len(xy) < 2:
        return None
    # LaserScan requires increasing, approximately uniform bearings.  DePHAN
    # horizon files are azimuth-ordered; sorting makes this robust to a packet
    # boundary or a future recorder that starts at a different azimuth.
    angles = np.arctan2(xy[:, 1], xy[:, 0])
    order = np.argsort(angles)
    return ReplayFrame(path, xy[order], point_times[order], stamp, frame)


def load_replay_frames(run_dir: Path, max_range: float = 30.0) -> list[ReplayFrame]:
    """Load valid frames in chronological order."""
    frames = [
        frame
        for path in list_run_frame_paths(run_dir)
        if (frame := load_replay_frame(path, max_range=max_range)) is not None
    ]
    return sorted(frames, key=lambda item: (item.stamp, item.frame))


def interpolate_encoder_series(samples: list[dict], timestamps: Iterable[float]) -> list[dict | None]:
    """Interpolate steering/speed at each scan timestamp.

    Steering is unwrapped before interpolation, preventing a -90°/90° wrap
    from producing a false straight-ahead command.  Samples outside the
    encoder interval are returned as ``None`` instead of edge extrapolation.
    """
    if len(samples) < 2:
        return [None for _ in timestamps]
    samples = sorted(samples, key=lambda item: float(item['enc_t']))
    ts = np.asarray([float(item['enc_t']) for item in samples], dtype=np.float64)
    vel = np.asarray([float(item.get('enc_v', 0.0)) for item in samples], dtype=np.float64)
    steer = np.unwrap(np.radians([
        float(item.get('enc_dgr', 0.0)) for item in samples
    ]))
    result = []
    for stamp in timestamps:
        t = float(stamp)
        if t < ts[0] or t > ts[-1]:
            result.append(None)
            continue
        v = float(np.interp(t, ts, vel))
        angle = float(np.interp(t, ts, steer))
        result.append({
            'frame_t': t,
            'enc_t': t,
            'enc_v': v,
            'enc_dgr': math.degrees(angle),
            'enc_angle': angle,
            'enc_available': True,
        })
    return result


def build_wheel_poses(frames: list[ReplayFrame], samples: list[dict]):
    """Integrate the project's tricycle model at each LiDAR timestamp."""
    synced = interpolate_encoder_series(samples, [f.stamp for f in frames])
    poses: list[list[float]] = []
    if not frames:
        return poses, synced
    poses.append([0.0, 0.0, 0.0])
    for index in range(1, len(frames)):
        previous, current, pose = synced[index - 1], synced[index], poses[-1]
        delta = motion_from_encoders(previous, current) if previous and current else None
        poses.append(compose(pose, delta) if delta is not None else list(pose))
    return poses, synced


def scan_geometry(xy: np.ndarray):
    """Return a uniform LaserScan grid and bearing quantization diagnostics.

    A DePHAN ring can contain invalid/missing returns.  Compacting the valid
    points into ``ranges`` would move every point after a gap to the wrong
    bearing.  Instead, retain the native angular increment and put missing
    returns into ``inf`` bins; this is lossless for valid points and preserves
    their geometry for Cartographer.
    """
    xy = np.asarray(xy, dtype=np.float64)
    angles = np.unwrap(np.arctan2(xy[:, 1], xy[:, 0]))
    raw_ranges = np.hypot(xy[:, 0], xy[:, 1])
    if len(angles) > 1:
        increments = np.diff(angles)
        increment = float(np.median(increments[increments > 1e-7]))
        n_bins = max(2, int(round((angles[-1] - angles[0]) / increment)) + 1)
    else:
        increment, n_bins = 0.0, 1
    angle_min = float(angles[0])
    ranges = np.full(n_bins, np.inf, dtype=np.float64)
    quantization_error = 0.0
    if increment > 0.0:
        indices = np.rint((angles - angle_min) / increment).astype(np.int64)
        valid = (indices >= 0) & (indices < n_bins)
        for index, distance, angle in zip(
            indices[valid], raw_ranges[valid], angles[valid]
        ):
            quantization_error = max(
                quantization_error,
                abs(float(angle) - (angle_min + float(index) * increment)),
            )
            if distance < ranges[index]:
                ranges[index] = distance
    elif n_bins:
        ranges[0] = raw_ranges[0]
    return angle_min, increment, ranges, quantization_error


def _ros_time(seconds: float):
    """Create builtin_interfaces/Time without importing ROS at module load."""
    from builtin_interfaces.msg import Time
    seconds = max(0.0, float(seconds))
    sec = int(seconds)
    return Time(sec=sec, nanosec=int(round((seconds - sec) * 1e9)))


class NpzCartographerReplay:
    """ROS 2 publisher for one NPZ recording."""

    def __init__(self, args):
        import rclpy
        from geometry_msgs.msg import TransformStamped
        from nav_msgs.msg import Odometry
        from rosgraph_msgs.msg import Clock
        from sensor_msgs.msg import LaserScan
        from tf2_ros import StaticTransformBroadcaster, TransformBroadcaster
        from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy

        self.rclpy = rclpy
        self.node = rclpy.create_node('npz_cartographer_replay')
        self.args = args
        self.frames = load_replay_frames(Path(args.run_dir), args.max_range)
        self.encoder = collect_run_encoder_samples([], local_dir=Path(args.run_dir))
        self.poses, self.synced = build_wheel_poses(self.frames, self.encoder)
        if not self.frames:
            raise RuntimeError(f'no usable horizon NPZ frames in {args.run_dir}')
        if not self.encoder:
            raise RuntimeError(f'no encoder.jsonl samples in {args.run_dir}')
        valid = sum(item is not None for item in self.synced)
        if valid < len(self.frames):
            self.node.get_logger().warning(
                f'encoder brackets {valid}/{len(self.frames)} scans; '
                'unbracketed poses are held'
            )
        self.t0 = self.frames[0].stamp
        self.clock_start = float(args.clock_start)
        self.offset = self.clock_start - self.t0 if args.time_mode == 'sim' else 0.0
        qos = QoSProfile(depth=100)
        qos.reliability = ReliabilityPolicy.RELIABLE
        self.scan_pub = self.node.create_publisher(LaserScan, args.scan_topic, qos)
        self.odom_pub = self.node.create_publisher(Odometry, args.odom_topic, qos)
        clock_qos = QoSProfile(depth=10)
        clock_qos.reliability = ReliabilityPolicy.BEST_EFFORT
        self.clock_pub = self.node.create_publisher(Clock, '/clock', clock_qos)
        self.tf_pub = TransformBroadcaster(self.node)
        self.static_tf_pub = StaticTransformBroadcaster(self.node)
        self._publish_static_tf(TransformStamped)
        self._types = (LaserScan, Odometry, Clock, TransformStamped)

    def ros_stamp(self, original_seconds: float):
        value = float(original_seconds) + self.offset
        return _ros_time(value)

    def _publish_static_tf(self, TransformStamped):
        msg = TransformStamped()
        msg.header.stamp = _ros_time(self.clock_start if self.args.time_mode == 'sim' else time.time())
        msg.header.frame_id = self.args.base_frame
        msg.child_frame_id = self.args.lidar_frame
        msg.transform.translation.x = float(self.args.lidar_x)
        msg.transform.translation.y = float(self.args.lidar_y)
        msg.transform.translation.z = float(self.args.lidar_z)
        msg.transform.rotation.w = 1.0
        self.static_tf_pub.sendTransform(msg)

    def publish_frame(self, index: int):
        frame = self.frames[index]
        stamp = self.ros_stamp(frame.stamp)
        if self.args.time_mode == 'sim':
            from rosgraph_msgs.msg import Clock
            clock = Clock()
            clock.clock = stamp
            self.clock_pub.publish(clock)

        from geometry_msgs.msg import TransformStamped
        from nav_msgs.msg import Odometry
        from sensor_msgs.msg import LaserScan
        pose = self.poses[index]
        odom = Odometry()
        odom.header.stamp = stamp
        odom.header.frame_id = self.args.odom_frame
        odom.child_frame_id = self.args.base_frame
        odom.pose.pose.position.x = float(pose[0])
        odom.pose.pose.position.y = float(pose[1])
        odom.pose.pose.orientation.z = math.sin(float(pose[2]) / 2.0)
        odom.pose.pose.orientation.w = math.cos(float(pose[2]) / 2.0)
        enc = self.synced[index]
        if enc is not None:
            steer = float(enc['enc_angle'])
            speed = WHEEL_SPEED_SCALE * float(enc['enc_v'])
            odom.twist.twist.linear.x = speed * math.cos(steer)
            odom.twist.twist.angular.z = speed * math.sin(steer) / max(WHEELBASE_M, 1e-6)
        # The covariance is intentionally finite and conservative.  It is a
        # wheel prior; Cartographer is expected to correct it with scans.
        odom.pose.covariance[0] = odom.pose.covariance[7] = 0.08 ** 2
        odom.pose.covariance[35] = math.radians(2.5) ** 2
        odom.twist.covariance[0] = odom.twist.covariance[35] = 0.2 ** 2
        self.odom_pub.publish(odom)

        angle_min, increment, ranges, angle_error = scan_geometry(frame.xy)
        scan = LaserScan()
        scan.header.stamp = stamp
        scan.header.frame_id = self.args.lidar_frame
        scan.angle_min = angle_min
        scan.angle_increment = increment
        scan.angle_max = angle_min + increment * (len(ranges) - 1)
        scan.range_min = float(MIN_RANGE)
        scan.range_max = float(self.args.max_range)
        scan.ranges = [float(value) for value in ranges]
        dt = 0.0
        finite_t = frame.point_times[np.isfinite(frame.point_times)]
        if len(finite_t) > 1:
            dt = float(np.median(np.diff(finite_t)))
        scan.time_increment = max(0.0, dt)
        scan.scan_time = self._scan_period(index)
        self.scan_pub.publish(scan)

        tf = TransformStamped()
        tf.header.stamp = stamp
        tf.header.frame_id = self.args.odom_frame
        tf.child_frame_id = self.args.base_frame
        tf.transform.translation.x = float(pose[0])
        tf.transform.translation.y = float(pose[1])
        tf.transform.rotation.z = odom.pose.pose.orientation.z
        tf.transform.rotation.w = odom.pose.pose.orientation.w
        self.tf_pub.sendTransform(tf)
        if angle_error > max(abs(increment) * 0.05, 1e-5) and index == 0:
            self.node.get_logger().warning(
                f'non-uniform scan bearings: max step error={angle_error:.3g} rad'
            )

    def _scan_period(self, index):
        if index + 1 < len(self.frames):
            return max(0.0, self.frames[index + 1].stamp - self.frames[index].stamp)
        if index:
            return max(0.0, self.frames[index].stamp - self.frames[index - 1].stamp)
        return 0.1

    def run(self):
        """Publish frames at recorded rate and optionally save pbstream."""
        from rclpy.executors import SingleThreadedExecutor
        executor = SingleThreadedExecutor()
        executor.add_node(self.node)
        rate = float(self.args.rate)
        for index, frame in enumerate(self.frames):
            self.publish_frame(index)
            executor.spin_once(timeout_sec=0.0)
            if rate > 0.0 and index + 1 < len(self.frames):
                delay = max(0.0, self._scan_period(index) / rate)
                deadline = time.monotonic() + delay
                while time.monotonic() < deadline and self.rclpy.ok():
                    executor.spin_once(timeout_sec=min(0.02, deadline - time.monotonic()))
        settle = max(0.0, float(self.args.settle))
        deadline = time.monotonic() + settle
        while time.monotonic() < deadline and self.rclpy.ok():
            executor.spin_once(timeout_sec=min(0.05, deadline - time.monotonic()))
        if self.args.write_state:
            self._write_state(executor, Path(self.args.write_state))
        executor.remove_node(self.node)

    def _write_state(self, executor, path: Path):
        from cartographer_ros_msgs.srv import WriteState
        client = self.node.create_client(WriteState, '/write_state')
        if not client.wait_for_service(timeout_sec=float(self.args.service_timeout)):
            self.node.get_logger().error('Cartographer /write_state service is unavailable')
            return
        request = WriteState.Request()
        request.filename = str(path)
        request.include_unfinished_submaps = True
        future = client.call_async(request)
        executor.spin_until_future_complete(future, timeout_sec=float(self.args.service_timeout))
        if not future.done() or not future.result().status.code == 0:
            self.node.get_logger().error(f'failed to write pbstream: {path}')
        else:
            self.node.get_logger().info(f'wrote Cartographer state: {path}')


def build_argument_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', required=True)
    parser.add_argument('--scan-topic', default='/scan')
    parser.add_argument('--odom-topic', default='/odom')
    parser.add_argument('--base-frame', default='base_link')
    parser.add_argument('--odom-frame', default='wheel_odom')
    parser.add_argument('--lidar-frame', default='lidar_top')
    parser.add_argument('--lidar-x', type=float, default=UP_LIDAR_X)
    parser.add_argument('--lidar-y', type=float, default=0.0)
    parser.add_argument('--lidar-z', type=float, default=UP_LIDAR_Z)
    parser.add_argument('--max-range', type=float, default=30.0)
    parser.add_argument('--rate', type=float, default=1.0,
                        help='playback speed; 0 publishes as fast as possible')
    parser.add_argument('--time-mode', choices=('sim', 'absolute'), default='sim')
    parser.add_argument('--clock-start', type=float, default=0.001)
    parser.add_argument('--write-state', default='', metavar='PBSTREAM')
    parser.add_argument('--settle', type=float, default=2.0)
    parser.add_argument('--service-timeout', type=float, default=30.0)
    return parser


def main(argv=None):
    # launch_ros appends ``--ros-args`` and a temporary parameter file even
    # though this adapter receives all operational settings as CLI arguments.
    # Keep those ROS arguments for rclpy and parse only our own options.
    args, _ros_args = build_argument_parser().parse_known_args(argv)
    import rclpy
    rclpy.init(args=None)
    replay = None
    try:
        replay = NpzCartographerReplay(args)
        replay.node.get_logger().info(
            f'replaying {len(replay.frames)} scans from {args.run_dir} '
            f'at {args.rate:g}x; topics {args.scan_topic}, {args.odom_topic}'
        )
        replay.run()
    except Exception as exc:
        if replay is not None:
            replay.node.get_logger().error(str(exc))
        else:
            print(f'npz_cartographer_replay: {exc}')
        return 2
    finally:
        rclpy.shutdown()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
