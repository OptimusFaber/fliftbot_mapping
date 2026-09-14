#!/usr/bin/env python3
"""Direct-UDP Pandar40P/DePHAN localization against a Cartographer PGM map."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import math
import socket
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import distance_transform_edt
from scipy.spatial import cKDTree


ROOT = Path(__file__).resolve().parents[3]


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


# Sensor drivers are loaded lazily in ``main``.  Pandar-only deployments do
# not need the optional DePHAN checkout (and vice versa).
DEPHAN = None
PANDAR = None


def yaml_value(text: str, key: str):
    for raw in text.splitlines():
        line = raw.split('#', 1)[0].strip()
        if line.startswith(key + ':'):
            return line.split(':', 1)[1].strip()
    raise ValueError(f'missing {key!r} in map yaml')


def load_map(yaml_path: Path):
    text = yaml_path.read_text(encoding='utf-8')
    image_name = yaml_value(text, 'image').strip('"\'')
    resolution = float(yaml_value(text, 'resolution'))
    origin = [float(v) for v in yaml_value(text, 'origin').strip('[]').split(',')]
    occupied_thresh = float(yaml_value(text, 'occupied_thresh'))
    negate = int(yaml_value(text, 'negate'))
    pixels = np.asarray(Image.open(yaml_path.parent / image_name).convert('L'))
    probability = pixels / 255.0 if negate else (255.0 - pixels) / 255.0
    occupied = probability > occupied_thresh
    rows, cols = np.nonzero(occupied)
    points = np.column_stack((
        origin[0] + (cols + 0.5) * resolution,
        origin[1] + (pixels.shape[0] - rows - 0.5) * resolution,
    ))
    if len(points) < 20:
        raise ValueError(f'{yaml_path}: map contains too few occupied cells')
    # Metric distance-to-wall field, indexed in image row/column coordinates.
    distance = distance_transform_edt(~occupied) * resolution
    return points, distance, resolution, origin[:2], pixels.shape


def transform(points, pose):
    c, s = math.cos(pose[2]), math.sin(pose[2])
    return points @ np.array(((c, s), (-s, c))) + np.asarray(pose[:2])


def normalize(a):
    return math.atan2(math.sin(a), math.cos(a))


def distance_score(points, pose, field, resolution, origin, shape):
    world = transform(points, pose)
    col = ((world[:, 0] - origin[0]) / resolution).astype(int)
    row = shape[0] - 1 - ((world[:, 1] - origin[1]) / resolution).astype(int)
    valid = (row >= 0) & (row < shape[0]) & (col >= 0) & (col < shape[1])
    if valid.sum() < max(20, len(points) // 5):
        return float('inf')
    distances = field[row[valid], col[valid]]
    return float(np.mean(np.minimum(distances, 1.0)) + 0.5 * (1.0 - valid.mean()))


def global_pose(points, field, resolution, origin, shape):
    sample = points[::max(1, len(points) // 240)][:240]
    x0, y0 = origin
    width, height = shape[1] * resolution, shape[0] * resolution
    best = (float('inf'), (0.0, 0.0, 0.0))
    for angle in np.arange(-math.pi, math.pi, math.radians(15.0)):
        for x in np.arange(x0, x0 + width, 0.5):
            for y in np.arange(y0, y0 + height, 0.5):
                score = distance_score(sample, (x, y, angle), field, resolution, origin, shape)
                if score < best[0]:
                    best = (score, (float(x), float(y), float(angle)))
    pose = best[1]
    for step_xy, step_a in ((0.15, 5.0), (0.04, 1.5)):
        candidates = []
        for dx in np.arange(-2 * step_xy, 2.01 * step_xy, step_xy):
            for dy in np.arange(-2 * step_xy, 2.01 * step_xy, step_xy):
                for da in np.arange(-2 * step_a, 2.01 * step_a, step_a):
                    p = (pose[0] + dx, pose[1] + dy, normalize(pose[2] + math.radians(da)))
                    candidates.append((distance_score(sample, p, field, resolution, origin, shape), p))
        pose = min(candidates, key=lambda item: item[0])[1]
    return pose


def icp(points, map_tree, initial, max_distance=0.45):
    pose = tuple(initial)
    used = points[::max(1, len(points) // 800)][:800]
    rmse = float('inf')
    ratio = 0.0
    for _ in range(20):
        moved = transform(used, pose)
        distances, indices = map_tree.query(moved)
        valid = distances < max_distance
        ratio = float(valid.mean())
        if valid.sum() < 30:
            break
        keep = np.flatnonzero(valid)
        keep = keep[np.argsort(distances[keep])[:max(30, int(len(keep) * 0.75))]]
        source, target = moved[keep], map_tree.data[indices[keep]]
        sm, tm = source.mean(0), target.mean(0)
        u, _, vt = np.linalg.svd((source - sm).T @ (target - tm))
        rotation = vt.T @ u.T
        if np.linalg.det(rotation) < 0:
            vt[-1] *= -1
            rotation = vt.T @ u.T
        translation = tm - rotation @ sm
        da = math.atan2(rotation[1, 0], rotation[0, 0])
        px, py = rotation @ np.asarray(pose[:2]) + translation
        pose = (float(px), float(py), normalize(pose[2] + da))
        new_rmse = float(np.sqrt(np.mean(distances[keep] ** 2)))
        if abs(rmse - new_rmse) < 1e-4:
            rmse = new_rmse
            break
        rmse = new_rmse
    return pose, rmse, ratio


def sensor_to_base(points, x, y, yaw):
    # Sensor pose is base<-sensor; map matching estimates map<-base.
    c, s = math.cos(yaw), math.sin(yaw)
    return points @ np.array(((c, s), (-s, c))) + np.array((x, y))


def dephan_scans(sock, source):
    current = None
    scan = [None] * DEPHAN.POINTS_PER_SCAN
    while True:
        data, address = sock.recvfrom(2048)
        if source and address[0] != source or len(data) != DEPHAN.PACKET_SIZE:
            continue
        host_ts = time.time()
        packet = DEPHAN.decode_packet(data, host_ts)
        if current is None:
            current = packet.revolution
        elif packet.revolution != current:
            rows = list(DEPHAN.scan_rows(scan))
            if len(rows) >= 100:
                scan_ts = max(r[1] for r in rows)
                yield scan_ts, np.asarray([(r[6], r[7]) for r in rows], dtype=float)
            scan = [None] * DEPHAN.POINTS_PER_SCAN
            current = packet.revolution
        start = packet.encoder
        end = min(start + DEPHAN.POINTS_PER_PACKET, DEPHAN.POINTS_PER_SCAN)
        if start < DEPHAN.POINTS_PER_SCAN:
            scan[start:end] = [(d, i, packet.timestamp, host_ts)
                               for d, i in packet.points[:end - start]]


def pandar_scans(sock, source):
    last_az = None
    points = []
    ts = 0.0
    while True:
        data, address = sock.recvfrom(2048)
        if source and address[0] != source or len(data) not in PANDAR.VALID_SIZES:
            continue
        host_ts = time.time()
        try:
            packet = PANDAR.decode_packet(data, host_ts, {PANDAR.HORIZON_CHANNEL})
        except ValueError:
            continue
        if last_az is not None and last_az - packet.azimuth0_deg > 180.0:
            if len(points) >= 100:
                yield ts, np.asarray(points, dtype=float)
            points = []
        for _, channel, azimuth, distance, _, t_host, _ in packet.points:
            if 0.3 <= distance <= 12.0:
                x, y, _ = PANDAR.xyz(distance, azimuth, channel)
                points.append((x, y))
                ts = max(ts, t_host)
        last_az = packet.azimuth0_deg


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--sensor', required=True, choices=('pandar', 'dephan'))
    p.add_argument('--map', required=True, type=Path, metavar='MAP.YAML')
    p.add_argument('--out', required=True, type=Path)
    p.add_argument('--bind', default='0.0.0.0')
    p.add_argument('--port', type=int)
    p.add_argument('--source')
    p.add_argument('--initial', nargs=3, type=float, metavar=('X', 'Y', 'ALPHA'))
    p.add_argument('--sensor-x', type=float)
    p.add_argument('--sensor-y', type=float)
    p.add_argument('--sensor-yaw', type=float, default=0.0, help='radians')
    p.add_argument('--min-overlap', type=float, default=0.25)
    p.add_argument('--max-rmse', type=float, default=0.35)
    return p.parse_args()


def main():
    global DEPHAN, PANDAR
    args = parse_args()
    if args.sensor == 'dephan':
        DEPHAN = load_module(
            'fliftbot_dephan_udp', ROOT / 'DePHAN_LiDAR/receive_scan.py')
    else:
        PANDAR = load_module(
            'fliftbot_pandar_udp', ROOT / 'Pandar_LiDAR/receive_pandar40p.py')
    defaults = {'pandar': (2368, '192.168.0.223', 0.590, 0.035),
                'dephan': (50007, '192.168.0.120', 1.015, 0.0)}
    port, source, mount_x, mount_y = defaults[args.sensor]
    port = args.port if args.port is not None else port
    source = args.source if args.source is not None else source
    mount_x = args.sensor_x if args.sensor_x is not None else mount_x
    mount_y = args.sensor_y if args.sensor_y is not None else mount_y
    map_points, field, resolution, origin, shape = load_map(args.map)
    tree = cKDTree(map_points)

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 16 << 20)
    sock.bind((args.bind, port))
    scans = dephan_scans(sock, source) if args.sensor == 'dephan' else pandar_scans(sock, source)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    new_file = not args.out.exists() or args.out.stat().st_size == 0
    pose = tuple(args.initial) if args.initial else None
    print(f'{args.sensor}: UDP {args.bind}:{port}, source={source}, map={args.map}')
    with args.out.open('a', newline='', encoding='utf-8', buffering=1) as output:
        writer = csv.writer(output)
        if new_file:
            writer.writerow(('ts', 'x', 'y', 'alpha'))
        try:
            for ts, sensor_points in scans:
                base_points = sensor_to_base(sensor_points, mount_x, mount_y, args.sensor_yaw)
                if pose is None:
                    print('First scan: global localization...')
                    pose = global_pose(base_points, field, resolution, origin, shape)
                candidate, rmse, overlap = icp(base_points, tree, pose)
                if overlap >= args.min_overlap and rmse <= args.max_rmse:
                    pose = candidate
                    writer.writerow((f'{ts:.9f}', f'{pose[0]:.6f}',
                                     f'{pose[1]:.6f}', f'{pose[2]:.6f}'))
                    print(f'ts={ts:.3f} x={pose[0]:.3f} y={pose[1]:.3f} '
                          f'alpha={pose[2]:.3f} rmse={rmse:.3f} overlap={overlap:.2f}')
                else:
                    print(f'REJECT rmse={rmse:.3f} overlap={overlap:.2f}', file=sys.stderr)
        except KeyboardInterrupt:
            print('\nStopped')
        finally:
            sock.close()


if __name__ == '__main__':
    main()
