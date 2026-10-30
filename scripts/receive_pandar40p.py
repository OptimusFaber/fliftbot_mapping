#!/usr/bin/env python3
"""Receive Pandar40P UDP packets without ROS (Dephan-style).

The lidar sends point-cloud UDP from 192.168.0.223 to the host:2368.
Stdlib only. Raw capture keeps the datagram plus host receive time; the
packet tail already contains the device UTC + microsecond timestamp.
Per-point times are packet time plus Hesai firetime/block offsets.

Print stats:

    python3 receive_pandar40p.py

Record ~240 s:

    python3 receive_pandar40p.py --out scan.p40p --duration 240
"""

from __future__ import annotations

import argparse
import csv
import math
import socket
import struct
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


PACKET_SIZE = 1262
PACKET_SIZE_AC = 1256
PACKET_SIZE_SEQ = 1266
MAX_PACKET = PACKET_SIZE_SEQ
BLOCK_NUM = 10
LASER_NUM = 40
DISTANCE_UNIT = 0.004
BLOCK_BYTES = 4 + LASER_NUM * 3  # azimuth header + 40 channels
BODY_BYTES = BLOCK_BYTES * BLOCK_NUM  # 1240
TAIL_TS_OFF = BODY_BYTES + 10  # uint32 microseconds
TAIL_RETURN_OFF = BODY_BYTES + 14
TAIL_UTC_OFF = BODY_BYTES + 16  # 6 bytes
BLOCK_NS_OFFSET1 = -28580
BLOCK_NS_OFFSET2 = -55560
HORIZON_CHANNEL = 13  # 1-based, elevation ≈ 0°
RAW_MAGIC = b"P40PRAW1"
RAW_RECORD = struct.Struct(f"<dH{MAX_PACKET}s")
VALID_SIZES = {PACKET_SIZE_AC, PACKET_SIZE, PACKET_SIZE_SEQ}

# Pandar40P.csv: channel -> (elevation_deg, azimuth_offset_deg)
CHANNEL_TABLE = {
    1: (15.21, -1.042),
    2: (11.36, -1.042),
    3: (8.387, -1.042),
    4: (5.385, -1.042),
    5: (3.368, -1.042),
    6: (2.356, -1.042),
    7: (2.016, 3.125),
    8: (1.679, -5.208),
    9: (1.341, -1.042),
    10: (1.003, 3.125),
    11: (0.665, -5.208),
    12: (0.328, -1.042),
    13: (-0.009, 3.125),
    14: (-0.347, -5.208),
    15: (-0.685, -1.042),
    16: (-1.023, 3.125),
    17: (-1.36, -5.208),
    18: (-1.7, -1.042),
    19: (-2.037, 3.125),
    20: (-2.372, -5.208),
    21: (-2.712, -1.042),
    22: (-3.047, 3.125),
    23: (-3.384, -5.208),
    24: (-3.722, -1.042),
    25: (-4.057, 3.125),
    26: (-4.392, -5.208),
    27: (-4.729, -1.042),
    28: (-5.063, 3.125),
    29: (-5.398, -5.208),
    30: (-5.733, -1.042),
    31: (-6.735, -1.042),
    32: (-7.731, -1.042),
    33: (-8.732, -1.042),
    34: (-9.557, -1.042),
    35: (-10.704, -1.042),
    36: (-11.678, -1.042),
    37: (-12.646, -1.042),
    38: (-13.602, -1.042),
    39: (-18.561, -1.042),
    40: (-24.569, -1.042),
}

# Firetime correction file, microseconds, channel 1..40
FIRETIME_US = {
    1: -42.22,
    2: -28.47,
    3: -16.04,
    4: -3.62,
    5: -45.49,
    6: -31.74,
    7: -47.46,
    8: -54.67,
    9: -20.62,
    10: -33.71,
    11: -40.91,
    12: -8.19,
    13: -20.62,
    14: -27.16,
    15: -50.73,
    16: -8.19,
    17: -14.74,
    18: -36.98,
    19: -45.49,
    20: -52.7,
    21: -23.89,
    22: -31.74,
    23: -38.95,
    24: -11.47,
    25: -18.65,
    26: -25.19,
    27: -48.76,
    28: -6.23,
    29: -12.77,
    30: -35.01,
    31: -21.92,
    32: -9.5,
    33: -43.52,
    34: -29.77,
    35: -17.35,
    36: -4.92,
    37: -42.22,
    38: -28.47,
    39: -16.04,
    40: -3.62,
}

CSV_FIELDS = (
    "timestamp_device_s",
    "timestamp_host_s",
    "timestamp_point_s",
    "channel",
    "azimuth_deg",
    "elevation_deg",
    "distance_m",
    "intensity",
    "x_m",
    "y_m",
    "z_m",
)


def device_unix(utc: bytes, timestamp_us: int) -> float:
    """UTC fields from the Pandar40P tail plus the 32-bit microsecond stamp."""
    if len(utc) < 6:
        return timestamp_us / 1e6
    year, month, day, hour, minute, second = utc[:6]
    year = int(year)
    if year < 70:
        year += 2000
    elif year < 1900:
        year += 1900
    try:
        dt = datetime(
            year, int(month), int(day), int(hour), int(minute), int(second),
            tzinfo=timezone.utc,
        )
    except ValueError:
        return timestamp_us / 1e6
    return dt.timestamp() + timestamp_us / 1e6


def block_ns_offset(block_id: int, dual_return: bool) -> int:
    stride = 2 if dual_return else 1
    return BLOCK_NS_OFFSET1 + BLOCK_NS_OFFSET2 * ((BLOCK_NUM - block_id - 1) // stride)


def point_time_s(host_t: float, device_t: float, block_id: int, channel: int, dual_return: bool) -> tuple[float, float]:
    """Return (timestamp_point_on_host_clock, timestamp_point_on_device_clock)."""
    offset_s = (block_ns_offset(block_id, dual_return) + FIRETIME_US[channel] * 1000.0) * 1e-9
    return host_t + offset_s, device_t + offset_s


@dataclass
class Packet:
    host_time: float
    timestamp_device: float
    timestamp_us: int
    azimuth0_deg: float
    dual_return: bool
    raw: bytes
    # (block_id, channel 1..40, azimuth_deg, distance_m, intensity, t_host, t_device)
    points: list[tuple[int, int, float, float, int, float, float]]


def decode_packet(
    data: bytes,
    host_time: float = 0.0,
    channels: set[int] | None = None,
) -> Packet:
    if data[:2] != b"\xff\xee":
        raise ValueError("not a Pandar40P point-cloud packet")
    # The 1256-byte AC packet ends immediately after return mode/factory
    # information and has no six-byte UTC field.  Newer 1262/1266-byte
    # packets append UTC (and optionally a sequence number).
    if len(data) < BODY_BYTES + 16:
        raise ValueError(f"packet too short: {len(data)}")
    timestamp_us = struct.unpack_from("<I", data, TAIL_TS_OFF)[0]
    return_mode = data[TAIL_RETURN_OFF]
    utc = data[TAIL_UTC_OFF:TAIL_UTC_OFF + 6]
    device_t = device_unix(utc, timestamp_us)
    dual = return_mode >= 0x39
    azimuth0 = struct.unpack_from("<H", data, 2)[0] / 100.0
    points: list[tuple[int, int, float, float, int, float, float]] = []
    selected_channels = (
        range(1, LASER_NUM + 1)
        if channels is None
        else sorted(channel for channel in channels if 1 <= channel <= LASER_NUM)
    )
    for block_id in range(BLOCK_NUM):
        base = block_id * BLOCK_BYTES
        azimuth_deg = struct.unpack_from("<H", data, base + 2)[0] / 100.0
        ch_off = base + 4
        for channel in selected_channels:
            dist_raw, intensity = struct.unpack_from(
                "<HB", data, ch_off + (channel - 1) * 3
            )
            distance_m = dist_raw * DISTANCE_UNIT
            t_host, t_dev = point_time_s(host_time, device_t, block_id, channel, dual)
            points.append(
                (block_id, channel, azimuth_deg, distance_m, intensity, t_host, t_dev)
            )
    return Packet(
        host_time=host_time,
        timestamp_device=device_t,
        timestamp_us=timestamp_us,
        azimuth0_deg=azimuth0,
        dual_return=dual,
        raw=data,
        points=points,
    )


def iter_raw_records(path: str):
    with open(path, "rb") as handle:
        magic = handle.read(len(RAW_MAGIC))
        if magic != RAW_MAGIC:
            raise ValueError(f"{path} is not a P40PRAW1 capture")
        while True:
            blob = handle.read(RAW_RECORD.size)
            if not blob:
                return
            if len(blob) != RAW_RECORD.size:
                raise ValueError(
                    f"{path}: truncated P40PRAW1 record: {len(blob)} of {RAW_RECORD.size}"
                )
            host_time, nbytes, raw = RAW_RECORD.unpack(blob)
            yield host_time, raw[:nbytes]


def xyz(distance_m: float, azimuth_deg: float, channel: int) -> tuple[float, float, float]:
    elev_deg, az_off = CHANNEL_TABLE[channel]
    az = math.radians(azimuth_deg + az_off)
    el = math.radians(elev_deg)
    xy = distance_m * math.cos(el)
    return xy * math.sin(az), xy * math.cos(az), distance_m * math.sin(el)


def ipv4_addresses() -> list[tuple[str, str]]:
    """Return (iface, ipv4) from `ip -4 -br addr`. Empty if ip is missing."""
    import subprocess
    try:
        out = subprocess.check_output(
            ["ip", "-4", "-br", "addr"], text=True, stderr=subprocess.DEVNULL
        )
    except (OSError, subprocess.CalledProcessError):
        return []
    rows = []
    for line in out.splitlines():
        parts = line.split()
        if len(parts) < 3:
            continue
        iface, _state, *addrs = parts
        for addr in addrs:
            rows.append((iface, addr.split("/")[0]))
    return rows


def warn_if_host_off_subnet(source: str) -> None:
    if not source:
        return
    addrs = ipv4_addresses()
    src_net = ".".join(source.split(".")[:3])
    matching = [(iface, ip) for iface, ip in addrs if ip.startswith(src_net + ".")]
    if matching:
        shown = ", ".join(f"{iface}={ip}" for iface, ip in matching)
        print(f"Host IPv4 on lidar subnet {src_net}.0/24: {shown}")
        return
    listed = ", ".join(f"{iface}={ip}" for iface, ip in addrs) or "(none)"
    print(
        f"ERROR: no IPv4 on {src_net}.0/24. Lidar {source} cannot deliver UDP here.\n"
        f"  Current IPv4: {listed}\n"
        f"  Ethernet is often UP with no address. On Jetson:\n"
        f"    sudo ip addr add 192.168.0.100/24 dev enP8p1s0\n"
        f"    ping -c 3 {source}\n"
        f"  Web UI Destination IP must be 192.168.0.100 (not 192.168.100.100).",
        flush=True,
    )


def default_run_dir(place: str) -> Path:
    root = Path(__file__).resolve().parents[1] / "ros-data" / "recordings" / "real"
    now = datetime.now()
    day = now.strftime("%Y%m%d")
    stamp = now.strftime("%H%M%S")
    out = root / day / f"{stamp}_pandar40p_{place}"
    out.mkdir(parents=True, exist_ok=True)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Receive Pandar40P UDP scans")
    parser.add_argument("--bind", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=2368)
    parser.add_argument("--source", default="192.168.0.223",
                        help="accepted LiDAR IP; empty accepts any")
    parser.add_argument("--out", metavar="PATH", help="raw capture (P40PRAW1)")
    parser.add_argument("--place", default="",
                        help="if set and --out is omitted, write ros-data/.../scan.p40p")
    parser.add_argument(
        "--csv",
        metavar="PATH",
        help="debug dump of every point (huge). Prefer --out",
    )
    parser.add_argument("--horizon-csv", metavar="PATH",
                        help="debug dump of channel 13 only")
    parser.add_argument("--duration", type=float, metavar="SECONDS")
    args = parser.parse_args()
    if args.duration is not None and args.duration <= 0:
        parser.error("--duration must be positive")
    if args.duration is not None and not args.out and not args.place and not args.csv and not args.horizon_csv:
        parser.error("--duration requires --out, --place, and/or a CSV path")

    out_path = Path(args.out) if args.out else None
    if out_path is None and args.place:
        run_dir = default_run_dir(args.place)
        out_path = run_dir / "scan.p40p"
        (run_dir / "run_meta.json").write_text(
            '{"lidar":"pandar40p","sensor":"Pandar40P"}\n',
            encoding="utf-8",
        )
        print(f"Run dir -> {run_dir}")

    warn_if_host_off_subnet(args.source)
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 16 << 20)
    sock.bind((args.bind, args.port))
    actual_buf = sock.getsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF)
    print(
        f"Listening on {args.bind}:{args.port}; UDP rcvbuf={actual_buf} bytes; "
        f"source={args.source or 'any'}; Ctrl+C to stop"
    )

    raw_file = None
    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        raw_file = out_path.open("wb", buffering=1 << 20)
        raw_file.write(RAW_MAGIC)
        print(f"Raw packets -> {out_path}")

    csv_file = writer = None
    if args.csv:
        csv_file = open(args.csv, "w", newline="", encoding="utf-8", buffering=1 << 20)
        writer = csv.writer(csv_file)
        writer.writerow(CSV_FIELDS)
        print(f"CSV (debug, large) -> {args.csv}")
    hz_file = hz_writer = None
    if args.horizon_csv:
        hz_file = open(args.horizon_csv, "w", newline="", encoding="utf-8", buffering=1 << 20)
        hz_writer = csv.writer(hz_file)
        hz_writer.writerow(CSV_FIELDS)
        print(f"Horizon CSV -> {args.horizon_csv}")

    packet_count = 0
    csv_points = 0
    started = None
    last_status = time.monotonic()
    last_flush = time.monotonic()
    last_az = None
    frames = 0

    def write_rows(packet: Packet) -> None:
        nonlocal csv_points
        for block_id, channel, azimuth_deg, distance_m, intensity, t_host, t_dev in packet.points:
            if distance_m <= 0.0:
                continue
            x, y, z = xyz(distance_m, azimuth_deg, channel)
            row = (
                packet.timestamp_device,
                packet.host_time,
                t_host,
                channel,
                azimuth_deg,
                CHANNEL_TABLE[channel][0],
                distance_m,
                intensity,
                x, y, z,
            )
            if writer is not None:
                writer.writerow(row)
                csv_points += 1
            if hz_writer is not None and channel == HORIZON_CHANNEL:
                hz_writer.writerow(row)

    def print_status(azimuth: float) -> None:
        elapsed = time.monotonic() - started if started else 0.0
        rate = packet_count / elapsed if elapsed else 0.0
        raw_mb = packet_count * RAW_RECORD.size / 1e6
        print(
            f"az={azimuth:6.2f} frames={frames} packets={packet_count} "
            f"elapsed={elapsed:.1f}s rate={rate:.1f} pkt/s raw={raw_mb:.1f} MB"
        )

    try:
        while True:
            data, address = sock.recvfrom(2048)
            if args.source and address[0] != args.source:
                continue
            if len(data) not in VALID_SIZES or data[:2] != b"\xff\xee":
                continue
            host_t = time.time()
            if started is None:
                started = time.monotonic()
            if raw_file is not None:
                raw_file.write(RAW_RECORD.pack(host_t, len(data), data.ljust(MAX_PACKET, b"\x00")))
            azimuth = struct.unpack_from("<H", data, 2)[0] / 100.0
            if last_az is not None and last_az - azimuth > 180.0:
                frames += 1
            last_az = azimuth
            packet_count += 1
            if writer is not None or hz_writer is not None:
                write_rows(decode_packet(data, host_t))
            now = time.monotonic()
            if now - last_status >= 1.0:
                print_status(azimuth)
                last_status = now
            if raw_file is not None and now - last_flush >= 1.0:
                raw_file.flush()
                last_flush = now
            if (
                args.duration is not None
                and started is not None
                and now - started >= args.duration
            ):
                return
    except KeyboardInterrupt:
        print("\nStopped")
    finally:
        if raw_file is not None:
            raw_file.flush()
            raw_file.close()
            size_mb = out_path.stat().st_size / 1e6
            print(f"Saved {packet_count} packets ({size_mb:.1f} MB) to {out_path}")
        if csv_file is not None:
            csv_file.close()
            print(f"Saved {csv_points} CSV points to {args.csv}")
        if hz_file is not None:
            hz_file.close()


if __name__ == "__main__":
    main()
