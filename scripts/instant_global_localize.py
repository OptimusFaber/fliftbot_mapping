#!/usr/bin/env python3
"""Localize one complete direct-UDP LiDAR revolution against a saved map.

No ROS lidar topic or MC telemetry is used. The script waits for one full
Pandar40P or DePHAN revolution, searches the occupancy map globally in x/y/yaw,
refines the best pose with ICP and prints one JSON result in marker-98 frame.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import socket
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy.spatial import cKDTree

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))

import direct_udp_mc_localizer as matcher  # reuse tested map matching and UDP decoders
import receive_scan as dephan
import receive_pandar40p as pandar
from marker98_pose import pose_in_marker98_frame
from lidar_profiles import load_profile


PROFILES = {sensor: load_profile(sensor) for sensor in ("dephan", "pandar")}


def create_run_dir(parent: Path) -> Path:
    """Create a unique UTC timestamp directory for this invocation."""
    parent.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    run_dir = parent / stamp
    suffix = 1
    while True:
        try:
            run_dir.mkdir()
            return run_dir
        except FileExistsError:
            run_dir = parent / f"{stamp}_{suffix}"
            suffix += 1


def save_map_snapshot(map_path: Path, run_dir: Path) -> dict:
    """Copy map YAML and its referenced image next to the diagnostic results."""
    copied = {}
    if not map_path.is_file():
        return {"error": f"map YAML not found: {map_path}"}
    yaml_copy = run_dir / "map.yaml"
    shutil.copy2(map_path, yaml_copy)
    copied["yaml"] = yaml_copy.name
    try:
        image_name = matcher.read_yaml_value(
            map_path.read_text(encoding="utf-8"), "image").strip("\"'")
        image_path = (map_path.parent / image_name).resolve()
        if not image_path.is_file():
            copied["image_error"] = f"map image not found: {image_path}"
        else:
            image_copy = run_dir / Path(image_name).name
            shutil.copy2(image_path, image_copy)
            copied["image"] = image_copy.name
    except Exception as exc:
        copied["image_error"] = f"{type(exc).__name__}: {exc}"
    return copied


def save_points(path: Path, points: np.ndarray | None) -> None:
    if points is None:
        return
    points = np.asarray(points, dtype=float).reshape((-1, 2))
    np.savetxt(path, points, delimiter=",", header="x_m,y_m", comments="",
               fmt="%.9f")


def finite_or_none(value):
    if isinstance(value, dict):
        return {key: finite_or_none(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [finite_or_none(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if math.isfinite(float(value)) else None
    return value


def write_result(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(finite_or_none(payload), indent=2,
                               ensure_ascii=False, allow_nan=False) + "\n",
                    encoding="utf-8")


def pixel_for_xy(x: float, y: float, resolution: float, origin: list[float],
                 shape: tuple[int, int]) -> tuple[int, int]:
    col = math.floor((x - origin[0]) / resolution)
    row = shape[0] - 1 - math.floor((y - origin[1]) / resolution)
    return col, row


def save_overlay(map_path: Path, run_dir: Path, marker_xy: tuple[float, float],
                 resolution: float, origin: list[float], shape: tuple[int, int],
                 base_points: np.ndarray | None,
                 map_pose: tuple[float, float, float] | None) -> None:
    """Render the occupancy image, marker, robot and scan in map coordinates."""
    image_name = matcher.read_yaml_value(
        map_path.read_text(encoding="utf-8"), "image").strip("\"'")
    with Image.open(map_path.parent / image_name) as source:
        canvas = source.convert("RGB")
    draw = ImageDraw.Draw(canvas)
    marker_px = pixel_for_xy(*marker_xy, resolution, origin, shape)
    mx, my = marker_px
    draw.ellipse((mx - 7, my - 7, mx + 7, my + 7), fill=(0, 210, 70),
                 outline=(0, 0, 0), width=2)
    draw.text((mx + 9, my - 9), "marker98", fill=(0, 130, 35),
              stroke_width=2, stroke_fill=(255, 255, 255))

    if base_points is not None and map_pose is not None:
        c, s = math.cos(map_pose[2]), math.sin(map_pose[2])
        moved = base_points @ np.array(((c, s), (-s, c)))
        moved += np.asarray(map_pose[:2])
        stride = max(1, len(moved) // 12000)
        for x, y in moved[::stride]:
            px, py = pixel_for_xy(float(x), float(y), resolution, origin, shape)
            if 0 <= px < canvas.width and 0 <= py < canvas.height:
                draw.ellipse((px - 1, py - 1, px + 1, py + 1), fill=(235, 30, 30))

    if map_pose is not None:
        rx, ry = pixel_for_xy(map_pose[0], map_pose[1], resolution, origin, shape)
        draw.ellipse((rx - 6, ry - 6, rx + 6, ry + 6), fill=(30, 100, 255),
                     outline=(0, 0, 0), width=2)
        heading = (map_pose[0] + 0.6 * math.cos(map_pose[2]),
                   map_pose[1] + 0.6 * math.sin(map_pose[2]))
        hx, hy = pixel_for_xy(*heading, resolution, origin, shape)
        draw.line((rx, ry, hx, hy), fill=(30, 100, 255), width=3)
        draw.text((rx + 8, ry + 7), "robot", fill=(0, 55, 180),
                  stroke_width=2, stroke_fill=(255, 255, 255))
    else:
        draw.text((8, 8), "Pose unavailable; map and origin shown",
                  fill=(180, 0, 0), stroke_width=2, stroke_fill=(255, 255, 255))
    canvas.save(run_dir / "localization_overlay.png")


def distinct_hypotheses(observations: list[tuple[float, tuple[float, float, float]]],
                        limit: int = 10) -> list[dict]:
    """Keep low-score pose hypotheses from different map regions/orientations."""
    candidates = sorted((item for item in observations if math.isfinite(item[0])),
                        key=lambda item: item[0])
    chosen = []
    for score, pose in candidates:
        separate = True
        for prior in chosen:
            distance = math.hypot(pose[0] - prior["map_x"],
                                  pose[1] - prior["map_y"])
            delta_yaw = abs(math.atan2(math.sin(pose[2] -
                                                   math.radians(prior["map_alpha_deg"])),
                                       math.cos(pose[2] -
                                                math.radians(prior["map_alpha_deg"]))))
            # Preserve a candidate if it is a different place, or a markedly
            # different orientation at nearly the same place.
            if distance < 1.0 and not (distance > 0.25 and delta_yaw > math.radians(30)):
                separate = False
                break
        if separate:
            chosen.append({"map_x": pose[0], "map_y": pose[1],
                           "map_alpha_deg": math.degrees(pose[2]),
                           "global_score_m": score})
            if len(chosen) >= limit:
                break
    return chosen


def one_dephan_revolution(sock: socket.socket, source: str):
    scan = [None] * dephan.POINTS_PER_SCAN
    revolution = None
    discard_first_partial = True
    while True:
        packet_data, address = sock.recvfrom(2048)
        if (source and address[0] != source) or len(packet_data) != dephan.PACKET_SIZE:
            continue
        host_ts = time.time()
        packet = dephan.decode_packet(packet_data, host_ts)
        if revolution is None:
            revolution = packet.revolution
        elif packet.revolution != revolution:
            # The receiver joined at an arbitrary point in the first turn;
            # discard that partial revolution and collect the next one whole.
            if discard_first_partial:
                discard_first_partial = False
                revolution = packet.revolution
                scan = [None] * dephan.POINTS_PER_SCAN
                start = packet.encoder
                end = min(start + dephan.POINTS_PER_PACKET, dephan.POINTS_PER_SCAN)
                if start < dephan.POINTS_PER_SCAN:
                    scan[start:end] = [
                        (distance, intensity, packet.timestamp, host_ts)
                        for distance, intensity in packet.points[:end - start]
                    ]
                continue
            rows = list(dephan.scan_rows(scan))
            if len(rows) >= 1000:
                points = np.asarray([(row[6], row[7]) for row in rows], dtype=float)
                return max(row[1] for row in rows), points, len(rows), revolution
            scan = [None] * dephan.POINTS_PER_SCAN
            revolution = packet.revolution
        start = packet.encoder
        end = min(start + dephan.POINTS_PER_PACKET, dephan.POINTS_PER_SCAN)
        if start < dephan.POINTS_PER_SCAN:
            scan[start:end] = [
                (distance, intensity, packet.timestamp, host_ts)
                for distance, intensity in packet.points[:end - start]
            ]


def one_pandar_revolution(sock: socket.socket, source: str):
    last_azimuth = None
    points = []
    stamp = 0.0
    discard_first_partial = True
    while True:
        packet_data, address = sock.recvfrom(4096)
        if (source and address[0] != source) or len(packet_data) not in pandar.VALID_SIZES:
            continue
        host_ts = time.time()
        try:
            packet = pandar.decode_packet(packet_data, host_ts, {pandar.HORIZON_CHANNEL})
        except ValueError:
            continue
        if last_azimuth is not None and last_azimuth - packet.azimuth0_deg > 180.0:
            if discard_first_partial:
                # First observed revolution started before we bound the UDP socket.
                discard_first_partial = False
                points = []
                stamp = 0.0
                last_azimuth = packet.azimuth0_deg
                continue
            if len(points) >= 1000:
                return stamp, np.asarray(points, dtype=float), len(points), stamp
            points = []
            stamp = 0.0
        for _, channel, azimuth, distance, _, point_ts, _ in packet.points:
            if 0.15 <= distance <= 30.0:
                x, y, _ = pandar.xyz(distance, azimuth, channel)
                points.append((x, y))
                stamp = max(stamp, point_ts)
        last_azimuth = packet.azimuth0_deg


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sensor", choices=tuple(PROFILES), required=True)
    parser.add_argument("--map-dir", type=Path, help="calibrated fliftbot_map_build deploy bundle")
    parser.add_argument("--bind", default="0.0.0.0")
    parser.add_argument("--port", type=int)
    parser.add_argument("--source", default=None)
    parser.add_argument("--scan-timeout", type=float, default=15.0,
                        help="seconds to wait for a complete revolution")
    parser.add_argument("--sensor-x", type=float)
    parser.add_argument("--sensor-y", type=float)
    parser.add_argument("--sensor-yaw-deg", type=float,
                        help="override the sensor-profile yaw calibration")
    parser.add_argument("--results-dir", type=Path,
                        default=ROOT / "results/instant_localization",
                        help="parent directory for per-run timestamped artifacts")
    parser.add_argument("--out", type=Path,
                        help="optional legacy CSV output (also saved in the run folder)")
    parser.add_argument("--min-overlap", type=float, default=0.25)
    parser.add_argument("--max-rmse", type=float, default=0.35)
    args = parser.parse_args()

    run_dir = create_run_dir(args.results_dir)
    profile = load_profile(args.sensor, args.map_dir)
    port = args.port or profile["udp_port"]
    source = profile["source_ip"] if args.source is None else args.source
    mount_x = profile["mount"][0] if args.sensor_x is None else args.sensor_x
    mount_y = profile["mount"][1] if args.sensor_y is None else args.sensor_y
    sensor_yaw_deg = (profile["sensor_yaw_deg"] if args.sensor_yaw_deg is None
                      else args.sensor_yaw_deg)
    config = {
        "sensor": args.sensor,
        "udp_bind": args.bind,
        "udp_port": port,
        "source_ip": source,
        "scan_timeout_s": args.scan_timeout,
        "map_path": str(profile["map"]),
        "marker_reference_path": str(profile["reference"]),
        "sensor_mount_x_m": mount_x,
        "sensor_mount_y_m": mount_y,
        "sensor_yaw_deg": sensor_yaw_deg,
        "min_overlap": args.min_overlap,
        "max_rmse_m": args.max_rmse,
        "results_dir": str(run_dir),
        "legacy_out_csv": str(args.out) if args.out else None,
    }
    result = {"status": "starting", "sensor": args.sensor,
              "parameters": config, "artifacts": {}}
    map_pose = None
    sensor_points = None
    base_points = None
    marker_xy = None
    resolution = origin = shape = None
    sock = None

    try:
        result["artifacts"].update(save_map_snapshot(profile["map"], run_dir))
        if profile["reference"].is_file():
            shutil.copy2(profile["reference"],
                         run_dir / "origin_reference.json")
            result["artifacts"]["origin_reference"] = "origin_reference.json"
        map_points, field, resolution, origin, shape = matcher.load_map(profile["map"])
        tree = cKDTree(map_points)
        reference = json.loads(profile["reference"].read_text(encoding="utf-8"))
        from map_bundle import read_reference
        marker_x, marker_y, marker_yaw = read_reference(profile["reference"])
        from map_bundle import read_y_axis_sign
        y_axis_sign = read_y_axis_sign(profile["reference"])
        marker_xy = (marker_x, marker_y)
        config["map_reference"] = reference
        config["map_resolution_m"] = resolution
        config["map_origin_m"] = origin
        config["map_shape_pixels"] = list(shape)

        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 16 << 20)
        sock.settimeout(args.scan_timeout)
        sock.bind((args.bind, port))
        print(json.dumps({"status": "waiting_for_one_full_revolution", "sensor": args.sensor,
                          "udp": f"{args.bind}:{port}", "source": source,
                          "map": str(profile["map"]), "results": str(run_dir)}), flush=True)

        capture = one_dephan_revolution if args.sensor == "dephan" else one_pandar_revolution
        scan_ts, sensor_points, point_count, revolution = capture(sock, source)
        result.update({"scan_timestamp": scan_ts, "revolution": revolution,
                       "point_count": point_count})
        save_points(run_dir / "raw_scan_points.csv", sensor_points)
        result["artifacts"]["raw_scan_points"] = "raw_scan_points.csv"

        yaw = math.radians(sensor_yaw_deg)
        c, s = math.cos(yaw), math.sin(yaw)
        base_points = sensor_points @ np.array(((c, s), (-s, c)))
        base_points += np.array((mount_x, mount_y))
        save_points(run_dir / "base_scan_points.csv", base_points)
        result["artifacts"]["base_scan_points"] = "base_scan_points.csv"

        print(json.dumps({"status": "global_search", "points": point_count,
                          "revolution": revolution, "ts": scan_ts,
                          "results": str(run_dir)}), flush=True)

        # Record the exact search evaluations made by the existing matcher so
        # low-scoring, spatially distinct alternatives remain inspectable. The
        # wrapper returns each score unchanged, so pose selection is untouched.
        score_fn = matcher.distance_score
        observations: list[tuple[float, tuple[float, float, float]]] = []

        def recording_score(points, pose, candidate_field, candidate_resolution,
                            candidate_origin, candidate_shape):
            score_value = score_fn(points, pose, candidate_field,
                                   candidate_resolution, candidate_origin,
                                   candidate_shape)
            observations.append((score_value, tuple(pose)))
            return score_value

        matcher.distance_score = recording_score
        try:
            map_pose = matcher.global_pose(base_points, field, resolution, origin, shape)
        finally:
            matcher.distance_score = score_fn
        hypotheses = distinct_hypotheses(observations)
        result["global_hypotheses"] = hypotheses
        result["global_search_evaluations"] = len(observations)

        refined_pose, rmse, overlap = matcher.icp(base_points, tree, map_pose)
        map_pose = refined_pose
        score = score_fn(base_points[::max(1, len(base_points) // 500)],
                         refined_pose, field, resolution, origin, shape)
        x, y, alpha_rad = pose_in_marker98_frame(
            *refined_pose, marker_x, marker_y, marker_yaw, y_axis_sign=y_axis_sign)
        alpha_deg = math.degrees(alpha_rad)
        accepted = overlap >= args.min_overlap and rmse <= args.max_rmse
        pose_result = {
            "x": x, "y": y, "alpha": alpha_deg, "alpha_unit": "deg",
            "ts": scan_ts,
            "map_x": refined_pose[0], "map_y": refined_pose[1],
            "map_alpha": math.degrees(refined_pose[2]),
            "map_alpha_unit": "deg",
            "marker_id": reference.get("marker_id"), "frame": "navigation_local", "points": point_count,
            "map_score_m": score, "icp_rmse_m": rmse, "overlap": overlap,
        }
        result.update(pose_result)
        result["status"] = "localized" if accepted else "low_confidence"
        print(json.dumps({"status": result["status"], "sensor": args.sensor,
                          "x": x, "y": y, "alpha": alpha_deg, "ts": scan_ts,
                          "map_x": refined_pose[0], "map_y": refined_pose[1],
                          "map_alpha": math.degrees(refined_pose[2]),
                          "alpha_unit": "deg", "marker_id": reference.get("marker_id"), "frame": "navigation_local",
                          "points": point_count, "map_score_m": score,
                          "icp_rmse_m": rmse, "overlap": overlap,
                          "results": str(run_dir)},
                         separators=(",", ":")), flush=True)

        with (run_dir / "pose.csv").open("w", newline="", encoding="utf-8") as output:
            writer = csv.writer(output)
            writer.writerow(("x_m", "y_m", "alpha_deg", "ts_unix_s",
                             "map_x_m", "map_y_m", "map_alpha_deg",
                             "map_score_m", "icp_rmse_m", "overlap"))
            writer.writerow((x, y, alpha_deg, scan_ts, refined_pose[0],
                             refined_pose[1], math.degrees(refined_pose[2]),
                             score, rmse, overlap))
        result["artifacts"]["pose_csv"] = "pose.csv"
        if args.out:
            args.out.parent.mkdir(parents=True, exist_ok=True)
            is_new = not args.out.exists() or args.out.stat().st_size == 0
            with args.out.open("a", encoding="utf-8") as output:
                if is_new:
                    output.write("x,y,alpha_deg,ts\n")
                output.write(f"{x:.6f},{y:.6f},{alpha_deg:.6f},{scan_ts:.9f}\n")
            result["artifacts"]["legacy_out_csv"] = str(args.out)
        return 0 if accepted else 3
    except socket.timeout:
        result.update({"status": "error", "reason": "scan_timeout",
                       "error": f"no full revolution received within {args.scan_timeout:g} s"})
        print(json.dumps({"status": "error", "reason": "scan_timeout",
                          "timeout_s": args.scan_timeout,
                          "results": str(run_dir)}, separators=(",", ":")),
              file=sys.stderr, flush=True)
        return 2
    except Exception as exc:
        result.update({"status": "error", "reason": "exception",
                       "error_type": type(exc).__name__, "error": str(exc)})
        print(json.dumps({"status": "error", "reason": "exception",
                          "error_type": type(exc).__name__, "error": str(exc),
                          "results": str(run_dir)}, separators=(",", ":")),
              file=sys.stderr, flush=True)
        return 2
    finally:
        if sock is not None:
            sock.close()
        if marker_xy is not None and resolution is not None:
            try:
                save_overlay(profile["map"], run_dir, marker_xy,
                             resolution, origin, shape, base_points, map_pose)
                result["artifacts"]["visualization"] = "localization_overlay.png"
            except Exception as exc:
                result["visualization_error"] = f"{type(exc).__name__}: {exc}"
        write_result(run_dir / "result.json", result)


if __name__ == "__main__":
    raise SystemExit(main())
