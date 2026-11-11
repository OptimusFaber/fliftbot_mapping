#!/usr/bin/env python3
"""Direct UDP LiDAR + ROS MC odometry localization against an occupancy map.

Only files inside the copied fliftbot_lidar_nav directory are needed at runtime. Requires
NumPy, Pillow, SciPy and ROS 2 Python packages (rclpy, fliftbot_interfaces).
Pose output is expressed in the selected map's marker-98 frame.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import socket
import sys
import threading
import time
from pathlib import Path

import numpy as np
import yaml
from PIL import Image
from scipy.ndimage import distance_transform_edt
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import receive_scan as dephan  # noqa: E402
import receive_pandar40p as pandar  # noqa: E402
from marker98_pose import pose_in_marker98_frame  # noqa: E402
from robot_identity import mc_topic  # noqa: E402
from wheel_motion import telemetry_twist


def wrap(a: float) -> float:
    return math.atan2(math.sin(a), math.cos(a))


def integrate_motion(pose: tuple[float, float, float], samples: list[tuple[float, float, float]],
                     start_t: float, end_t: float) -> tuple[float, float, float]:
    """Integrate timestamped (unix_s, forward_mps, yaw_rate_rps) samples.

    Piecewise constant hold. Caller should pass at least one sample at or
    before start_t. Samples are interpolated only by the hold boundary.
    """
    x, y, a = pose
    if end_t <= start_t or not samples:
        return pose
    ordered = sorted(samples, key=lambda sample: sample[0])
    before = [sample for sample in ordered if sample[0] <= start_t]
    if not before:
        return pose
    current = before[-1]
    cursor = start_t
    for sample in [s for s in ordered if start_t < s[0] < end_t] + [(end_t, 0.0, 0.0)]:
        dt = sample[0] - cursor
        v, wz = current[1], current[2]
        if dt > 0:
            mid = a + 0.5 * wz * dt
            x += v * math.cos(mid) * dt
            y += v * math.sin(mid) * dt
            a = wrap(a + wz * dt)
        cursor = sample[0]
        if sample[0] < end_t:
            current = sample
    return x, y, a


def read_yaml_value(text: str, key: str) -> str:
    for raw in text.splitlines():
        line = raw.split("#", 1)[0].strip()
        if line.startswith(key + ":"):
            return line.split(":", 1)[1].strip()
    raise ValueError(f"missing {key!r} in map YAML")


def load_map(path: Path):
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    image_name = config["image"]
    res = float(config["resolution"])
    origin = [float(v) for v in config["origin"]]
    threshold = float(config["occupied_thresh"])
    negate = int(config["negate"])
    if len(origin) != 3 or abs(origin[2]) > 1e-9:
        raise ValueError("rotated occupancy YAML origin is unsupported; export an unrotated map")
    if not math.isfinite(res) or res <= 0 or not all(math.isfinite(v) for v in origin):
        raise ValueError("invalid map resolution/origin")
    pixels = np.asarray(Image.open(path.parent / image_name).convert("L"))
    probability = pixels / 255.0 if negate else (255.0 - pixels) / 255.0
    occupied = probability > threshold
    rows, cols = np.nonzero(occupied)
    points = np.column_stack((origin[0] + (cols + .5) * res,
                              origin[1] + (pixels.shape[0] - rows - .5) * res))
    if len(points) < 20:
        raise ValueError(f"{path}: occupancy map has too few occupied cells")
    return points, distance_transform_edt(~occupied) * res, res, origin[:2], pixels.shape


def transform(points: np.ndarray, pose: tuple[float, float, float]) -> np.ndarray:
    c, s = math.cos(pose[2]), math.sin(pose[2])
    return points @ np.array(((c, s), (-s, c))) + np.asarray(pose[:2])


def distance_score(points, pose, field, resolution, origin, shape):
    world = transform(points, pose)
    col = np.floor((world[:, 0] - origin[0]) / resolution).astype(int)
    row = shape[0] - 1 - np.floor((world[:, 1] - origin[1]) / resolution).astype(int)
    valid = (row >= 0) & (row < shape[0]) & (col >= 0) & (col < shape[1])
    if valid.sum() < max(20, len(points) // 5):
        return float("inf")
    d = field[row[valid], col[valid]]
    return float(np.mean(np.minimum(d, 1.0)) + .5 * (1.0 - valid.mean()))


def global_pose(points, field, resolution, origin, shape):
    sample = points[::max(1, len(points) // 240)][:240]
    width, height = shape[1] * resolution, shape[0] * resolution
    best = (float("inf"), (0.0, 0.0, 0.0))
    for angle in np.arange(-math.pi, math.pi, math.radians(15)):
        for x in np.arange(origin[0], origin[0] + width, .5):
            for y in np.arange(origin[1], origin[1] + height, .5):
                score = distance_score(sample, (x, y, angle), field, resolution, origin, shape)
                if score < best[0]:
                    best = score, (float(x), float(y), float(angle))
    pose = best[1]
    for step_xy, step_a in ((.15, 5.0), (.04, 1.5)):
        candidates = []
        for dx in np.arange(-2 * step_xy, 2.01 * step_xy, step_xy):
            for dy in np.arange(-2 * step_xy, 2.01 * step_xy, step_xy):
                for da in np.arange(-2 * step_a, 2.01 * step_a, step_a):
                    p = (pose[0] + dx, pose[1] + dy, wrap(pose[2] + math.radians(da)))
                    candidates.append((distance_score(sample, p, field, resolution, origin, shape), p))
        pose = min(candidates, key=lambda item: item[0])[1]
    return pose


def icp(points, map_tree, initial, max_distance=.45):
    pose = tuple(initial)
    used = points[::max(1, len(points) // 800)][:800]
    rmse, ratio = float("inf"), 0.0
    for _ in range(20):
        moved = transform(used, pose)
        distances, indices = map_tree.query(moved)
        valid = distances < max_distance
        ratio = float(valid.mean())
        if valid.sum() < 30:
            break
        keep = np.flatnonzero(valid)
        keep = keep[np.argsort(distances[keep])[:max(30, int(len(keep) * .75))]]
        src, dst = moved[keep], map_tree.data[indices[keep]]
        sm, dm = src.mean(0), dst.mean(0)
        u, _, vt = np.linalg.svd((src - sm).T @ (dst - dm))
        rot = vt.T @ u.T
        if np.linalg.det(rot) < 0:
            vt[-1] *= -1
            rot = vt.T @ u.T
        trans = dm - rot @ sm
        da = math.atan2(rot[1, 0], rot[0, 0])
        px, py = rot @ np.asarray(pose[:2]) + trans
        pose = float(px), float(py), wrap(pose[2] + da)
        new_rmse = float(np.sqrt(np.mean(distances[keep] ** 2)))
        if abs(rmse - new_rmse) < 1e-4:
            rmse = new_rmse
            break
        rmse = new_rmse
    return pose, rmse, ratio


class MCTelemetry:
    """Thread-safe BEST_EFFORT subscriber and bounded motion sample history."""
    def __init__(self, args):
        import rclpy
        from fliftbot_interfaces.msg import MC
        from rclpy.executors import SingleThreadedExecutor
        from rclpy.node import Node
        from rclpy.qos import (DurabilityPolicy, HistoryPolicy, QoSProfile,
                               ReliabilityPolicy)

        class Subscriber(Node):
            def __init__(self, owner):
                super().__init__("direct_udp_mc_localizer")
                self.owner = owner
                qos = QoSProfile(history=HistoryPolicy.KEEP_LAST, depth=100,
                                 reliability=ReliabilityPolicy.BEST_EFFORT,
                                 durability=DurabilityPolicy.VOLATILE)
                self.create_subscription(MC, args.mc_topic, self.callback, qos)

            def callback(self, msg):
                stamp = float(msg.feedback_timestamp)
                if not math.isfinite(stamp) or stamp <= 0:
                    stamp = self.get_clock().now().nanoseconds * 1e-9
                v, wz = telemetry_twist(float(msg.enc_lin_vel_mps),
                                        float(msg.enc_whl_yaw_dgr),
                                        speed_sign=args.speed_sign,
                                        steering_zero_deg=args.steering_zero_deg,
                                        wheelbase=args.wheelbase, model=args.odom_model, steering_sign=args.steering_sign)
                self.owner.append((stamp, v, wz))

        self.samples = []
        self.lock = threading.Lock()
        self.node = Subscriber(self)
        self.executor = SingleThreadedExecutor()
        self.executor.add_node(self.node)
        self.thread = threading.Thread(target=self.executor.spin, daemon=True)
        self.thread.start()

    def append(self, sample):
        with self.lock:
            if self.samples and sample[0] <= self.samples[-1][0]:
                return
            self.samples.append(sample)
            del self.samples[:-3000]

    def interval(self, start, end):
        with self.lock:
            return [s for s in self.samples if s[0] <= end and s[0] >= start - 1.0]

    def close(self):
        self.executor.shutdown(timeout_sec=1.0)
        self.node.destroy_node()


def dephan_scans(sock, source):
    current = None
    scan = [None] * dephan.POINTS_PER_SCAN
    while True:
        data, address = sock.recvfrom(2048)
        if (source and address[0] != source) or len(data) != dephan.PACKET_SIZE:
            continue
        host_t = time.time()
        packet = dephan.decode_packet(data, host_t)
        if current is None:
            current = packet.revolution
        elif packet.revolution != current:
            rows = list(dephan.scan_rows(scan))
            if len(rows) >= 100:
                yield max(r[1] for r in rows), np.asarray([(r[6], r[7]) for r in rows])
            scan = [None] * dephan.POINTS_PER_SCAN
            current = packet.revolution
        start = packet.encoder
        end = min(start + dephan.POINTS_PER_PACKET, dephan.POINTS_PER_SCAN)
        if start < dephan.POINTS_PER_SCAN:
            scan[start:end] = [(d, i, packet.timestamp, host_t)
                               for d, i in packet.points[:end - start]]


def pandar_scans(sock, source):
    last_az, points, stamp = None, [], 0.0
    while True:
        data, address = sock.recvfrom(2048)
        if (source and address[0] != source) or len(data) not in pandar.VALID_SIZES:
            continue
        host_t = time.time()
        try:
            packet = pandar.decode_packet(data, host_t, {pandar.HORIZON_CHANNEL})
        except ValueError:
            continue
        if last_az is not None and last_az - packet.azimuth0_deg > 180:
            if len(points) >= 100:
                yield stamp, np.asarray(points)
            points, stamp = [], 0.0
        for _, channel, azimuth, distance, _, t_host, _ in packet.points:
            if .3 <= distance <= 12.0:
                x, y, _ = pandar.xyz(distance, azimuth, channel)
                points.append((x, y))
                stamp = max(stamp, t_host)
        last_az = packet.azimuth0_deg


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sensor", choices=("pandar", "dephan"), required=True)
    p.add_argument("--map", type=Path)
    p.add_argument("--map-dir", type=Path)
    p.add_argument("--steering-sign", type=float, choices=(-1., 1.), default=None)
    p.add_argument("--marker-reference", type=Path)
    p.add_argument("--bind", default="0.0.0.0")
    p.add_argument("--port", type=int)
    p.add_argument("--source")
    p.add_argument("--mc-topic",
                   help="default is derived from hostname suffix")
    p.add_argument("--speed-sign", type=float, choices=(-1.0, 1.0), default=None)
    p.add_argument("--steering-zero-deg", type=float, default=None,
                   help="raw MC wheel angle corresponding to straight travel")
    p.add_argument("--sensor-yaw-deg", type=float, default=None,
                   help="sensor XY axes to configured base/map axes")
    p.add_argument("--sensor-x", type=float,
                   help="override profile sensor X offset in metres")
    p.add_argument("--sensor-y", type=float,
                   help="override profile sensor Y offset in metres")
    p.add_argument("--wheelbase", type=float)
    p.add_argument("--odom-model", choices=("drive_wheel", "legacy_sin", "bicycle"))
    p.add_argument("--initial", nargs=3, type=float, metavar=("X_MAP", "Y_MAP", "ALPHA_MAP"))
    p.add_argument("--out", type=Path)
    p.add_argument("--min-overlap", type=float, default=.25)
    p.add_argument("--max-rmse", type=float, default=.35)
    return p.parse_args()


def main():
    args = parse_args()
    args.mc_topic = args.mc_topic or mc_topic()
    import rclpy
    rclpy.init(args=None)
    from lidar_profiles import load_profile
    from map_bundle import read_reference, read_y_axis_sign
    profile = load_profile(args.sensor, args.map_dir)
    if args.map is not None and args.marker_reference is None:
        raise ValueError("--map override requires --marker-reference; prefer --map-dir")
    map_path = args.map or profile["map"]
    ref_path = args.marker_reference or profile["reference"]
    port = args.port or profile["udp_port"]
    source = profile["source_ip"] if args.source is None else args.source
    mount_x = profile["mount"][0] if args.sensor_x is None else args.sensor_x
    mount_y = profile["mount"][1] if args.sensor_y is None else args.sensor_y
    if args.sensor_yaw_deg is None:
        args.sensor_yaw_deg = profile["sensor_yaw_deg"]
    odom = profile["odometry"]
    for attribute, key in (("speed_sign", "speed_sign"), ("wheelbase", "wheelbase_m"),
                           ("steering_zero_deg", "steering_zero_deg"),
                           ("steering_sign", "steering_sign"), ("odom_model", "model")):
        if getattr(args, attribute) is None:
            setattr(args, attribute, odom.get(key, 1.0 if key == "steering_sign" else None))
    if args.out is None:
        args.out = ROOT / "results" / f"{args.sensor}_pose.csv"
    marker_x, marker_y, marker_a = read_reference(ref_path)
    y_axis_sign = read_y_axis_sign(ref_path)
    map_points, field, resolution, origin, shape = load_map(map_path)
    tree = cKDTree(map_points)
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 16 << 20)
    sock.bind((args.bind, port))
    mc = MCTelemetry(args)
    scans = dephan_scans(sock, source) if args.sensor == "dephan" else pandar_scans(sock, source)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fresh = not args.out.exists() or args.out.stat().st_size == 0
    pose = tuple(args.initial) if args.initial else None
    last_scan_t = None
    print(json.dumps({"status": "listening", "sensor": args.sensor, "udp": f"{args.bind}:{port}",
                      "source": source, "mc_topic": args.mc_topic, "map": str(map_path),
                      "marker_reference": str(ref_path), "csv": str(args.out)}), flush=True)
    try:
        with args.out.open("a", newline="", encoding="utf-8", buffering=1) as f:
            writer = csv.writer(f)
            if fresh:
                writer.writerow(("x", "y", "alpha", "ts"))
            for ts, sensor_points in scans:
                sensor_yaw = math.radians(args.sensor_yaw_deg)
                c, s = math.cos(sensor_yaw), math.sin(sensor_yaw)
                base_points = sensor_points @ np.array(((c, s), (-s, c))) + np.array((mount_x, mount_y))
                if pose is None:
                    print(json.dumps({"status": "global_search", "ts": ts}), flush=True)
                    pose = global_pose(base_points, field, resolution, origin, shape)
                elif last_scan_t is not None:
                    samples = mc.interval(last_scan_t, ts)
                    pred_samples = [(t, v, wz) for t, v, wz in samples]
                    pose = integrate_motion(pose, pred_samples, last_scan_t, ts)
                candidate, rmse, overlap = icp(base_points, tree, pose)
                last_scan_t = ts
                if overlap < args.min_overlap or rmse > args.max_rmse:
                    print(json.dumps({"status": "reject", "ts": ts, "rmse": rmse,
                                      "overlap": overlap}), file=sys.stderr, flush=True)
                    continue
                pose = candidate
                x, y, alpha = pose_in_marker98_frame(*pose, marker_x, marker_y, marker_a,
                                                    y_axis_sign=y_axis_sign)
                record = {"x": x, "y": y, "alpha": alpha, "ts": ts}
                writer.writerow((f"{x:.6f}", f"{y:.6f}", f"{alpha:.6f}", f"{ts:.9f}"))
                print(json.dumps(record, separators=(",", ":")), flush=True)
    except KeyboardInterrupt:
        pass
    finally:
        sock.close()
        mc.close()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
