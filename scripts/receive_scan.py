#!/usr/bin/env python3
"""Receive and decode DEPHAN M360 UDP scans without external dependencies."""

from __future__ import annotations

import argparse
import csv
import math
import socket
import struct
import time
from dataclasses import dataclass
from pathlib import Path


PACKET_SIZE = 936
POINTS_PER_PACKET = 115
POINTS_PER_SCAN = 2300
HEADER = struct.Struct("<HHHHII")
POINT = struct.Struct("<II")
INVALID_DISTANCE = 0xFFFFFFFF
# Host unix time (float64) + raw UDP datagram. ~944 B/packet, ~113 MB / 240 s at 25 Hz.
RAW_MAGIC = b"M360RAW1"
RAW_RECORD = struct.Struct(f"<d{PACKET_SIZE}s")
CSV_FIELDS = (
    "timestamp_device_s",
    "timestamp_host_s",
    "index",
    "angle_rad",
    "distance_m",
    "intensity",
    "x_m",
    "y_m",
)


@dataclass(frozen=True)
class Packet:
    version: int
    revolution: int
    frequency_hz: int
    encoder: int
    timestamp_seconds: int
    timestamp_fraction: int
    points: list[tuple[int, int]]
    host_time: float = 0.0
    raw: bytes = b""

    @property
    def timestamp(self) -> float:
        # TimeStamp is an unsigned 64-bit fixed-point value transmitted
        # little-endian.  The first uint32 in the byte stream is therefore
        # the low (fractional) word and the second is the high (seconds) word.
        return self.timestamp_fraction + self.timestamp_seconds / 2**32


def decode_packet(data: bytes, host_time: float = 0.0) -> Packet:
    if len(data) != PACKET_SIZE:
        raise ValueError(f"expected {PACKET_SIZE} bytes, received {len(data)}")

    version, revolution, frequency_hz, encoder, ts_a, ts_b = HEADER.unpack_from(data)
    points = [
        POINT.unpack_from(data, HEADER.size + i * POINT.size)
        for i in range(POINTS_PER_PACKET)
    ]
    return Packet(
        version, revolution, frequency_hz, encoder, ts_a, ts_b, points,
        host_time=host_time, raw=data,
    )


def iter_raw_records(path: str):
    """Yield (host_time, raw_datagram) from an M360RAW1 capture."""
    with open(path, "rb") as handle:
        magic = handle.read(len(RAW_MAGIC))
        if magic != RAW_MAGIC:
            raise ValueError(f"{path} is not an M360RAW1 capture")
        while True:
            blob = handle.read(RAW_RECORD.size)
            if not blob:
                return
            if len(blob) != RAW_RECORD.size:
                raise ValueError(
                    f"{path}: truncated M360RAW1 record: "
                    f"{len(blob)} of {RAW_RECORD.size} bytes"
                )
            host_time, raw = RAW_RECORD.unpack(blob)
            yield host_time, raw


def scan_rows(scan: list[tuple[int, int, float, float] | None]):
    for index, measurement in enumerate(scan):
        if measurement is None:
            continue
        distance_mm, intensity, timestamp_device, timestamp_host = measurement
        if distance_mm == INVALID_DISTANCE:
            continue
        angle = index * 2.0 * math.pi / POINTS_PER_SCAN
        distance_m = distance_mm / 1000.0
        yield (
            timestamp_device,
            timestamp_host,
            index,
            angle,
            distance_m,
            intensity,
            distance_m * math.cos(angle),
            distance_m * math.sin(angle),
        )


def open_csv(path: str):
    output = open(path, "w", newline="", encoding="utf-8", buffering=1 << 20)
    writer = csv.writer(output)
    writer.writerow(CSV_FIELDS)
    return output, writer


def write_scan(writer, output, scan) -> int:
    count = 0
    for row in scan_rows(scan):
        writer.writerow(row)
        count += 1
    output.flush()
    return count


def finish_revolution(scan):
    received = 0
    valid = 0
    for point in scan:
        if point is None:
            continue
        received += 1
        if point[0] != INVALID_DISTANCE:
            valid += 1
    return received, valid


def assemble_from_packets(packets: list[Packet]):
    """Rebuild revolutions from decoded packets; used by exporters."""
    scans: list[list] = []
    current_rev = None
    scan: list = [None] * POINTS_PER_SCAN
    for packet in packets:
        if current_rev is None:
            current_rev = packet.revolution
        elif packet.revolution != current_rev:
            scans.append(scan)
            scan = [None] * POINTS_PER_SCAN
            current_rev = packet.revolution
        start = packet.encoder
        end = min(start + POINTS_PER_PACKET, POINTS_PER_SCAN)
        if start < POINTS_PER_SCAN:
            scan[start:end] = [
                (distance, intensity, packet.timestamp, packet.host_time)
                for distance, intensity in packet.points[: end - start]
            ]
    if current_rev is not None:
        scans.append(scan)
    return scans


def main() -> None:
    parser = argparse.ArgumentParser(description="Receive DEPHAN M360 UDP scans")
    parser.add_argument("--bind", default="0.0.0.0", help="local address (default: all interfaces)")
    parser.add_argument("--port", type=int, default=50007, help="local UDP port")
    parser.add_argument("--source", default="192.168.0.120", help="accepted LiDAR IP; empty accepts any")
    parser.add_argument(
        "--out",
        metavar="PATH",
        help="raw capture (M360RAW1). ~9 MB / 20 s, ~113 MB / 240 s at 25 Hz",
    )
    parser.add_argument("--csv", metavar="PATH", help="optional bulky text dump; prefer --out")
    parser.add_argument(
        "--duration",
        type=float,
        metavar="SECONDS",
        help="stop this many seconds after the first LiDAR packet",
    )
    parser.add_argument(
        "--min-points",
        type=int,
        default=1,
        help="CSV only: write a revolution if at least this many points arrived",
    )
    args = parser.parse_args()
    if args.duration is not None and args.duration <= 0:
        parser.error("--duration must be positive")
    if args.duration is not None and not args.out and not args.csv:
        parser.error("--duration requires --out and/or --csv")
    if args.min_points < 1:
        parser.error("--min-points must be >= 1")

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 16 << 20)
    sock.bind((args.bind, args.port))
    actual_buf = sock.getsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF)
    print(
        f"Listening on {args.bind}:{args.port}; UDP rcvbuf={actual_buf} bytes; "
        "Ctrl+C to stop"
    )

    raw_file = None
    if args.out:
        raw_path = Path(args.out)
        raw_file = raw_path.open("wb", buffering=1 << 20)
        raw_file.write(RAW_MAGIC)
        print(f"Raw packets -> {raw_path}")
    csv_file = None
    writer = None
    if args.csv:
        csv_file, writer = open_csv(args.csv)
        print(f"CSV (debug, large) -> {args.csv}")

    current_revolution = None
    scan: list[tuple[int, int, float, float] | None] = [None] * POINTS_PER_SCAN
    packet_count = 0
    saved_scans = 0
    saved_points = 0
    skipped_scans = 0
    started = None
    last_status = time.monotonic()
    last_flush = time.monotonic()

    def emit_scan(frequency_hz: int) -> None:
        nonlocal scan, saved_scans, saved_points, skipped_scans
        if writer is None:
            scan = [None] * POINTS_PER_SCAN
            return
        received, valid = finish_revolution(scan)
        if received >= args.min_points:
            saved_points += write_scan(writer, csv_file, scan)
            saved_scans += 1
        else:
            skipped_scans += 1
        scan = [None] * POINTS_PER_SCAN

    def print_status(frequency_hz: int, received: int | None = None) -> None:
        elapsed = time.monotonic() - started if started else 0.0
        rate = packet_count / elapsed if elapsed else 0.0
        raw_mb = packet_count * RAW_RECORD.size / 1e6
        extra = ""
        if received is not None:
            extra = f" received={received:4d}/{POINTS_PER_SCAN}"
        print(
            f"HZ={frequency_hz:2d}{extra} packets={packet_count} "
            f"elapsed={elapsed:.1f}s rate={rate:.1f} pkt/s raw={raw_mb:.1f} MB"
        )

    try:
        while True:
            data, address = sock.recvfrom(2048)
            if args.source and address[0] != args.source:
                continue
            if len(data) != PACKET_SIZE:
                continue
            host_t = time.time()
            if started is None:
                started = time.monotonic()
            if raw_file is not None:
                raw_file.write(RAW_RECORD.pack(host_t, data))

            _version, revolution, frequency_hz, _encoder, _ts_a, _ts_b = HEADER.unpack_from(data)
            packet_count += 1

            if writer is not None:
                packet = decode_packet(data, host_t)
                if current_revolution is None:
                    current_revolution = packet.revolution
                elif packet.revolution != current_revolution:
                    emit_scan(packet.frequency_hz)
                    current_revolution = packet.revolution
                start = packet.encoder
                end = min(start + POINTS_PER_PACKET, POINTS_PER_SCAN)
                if start < POINTS_PER_SCAN:
                    scan[start:end] = [
                        (distance, intensity, packet.timestamp, host_t)
                        for distance, intensity in packet.points[: end - start]
                    ]

            now = time.monotonic()
            if now - last_status >= 1.0:
                print_status(frequency_hz)
                last_status = now
            if raw_file is not None and now - last_flush >= 1.0:
                raw_file.flush()
                last_flush = now
            if (
                args.duration is not None
                and started is not None
                and now - started >= args.duration
            ):
                if writer is not None and current_revolution is not None:
                    emit_scan(frequency_hz)
                return
    except KeyboardInterrupt:
        if writer is not None and current_revolution is not None:
            emit_scan(0)
        print("\nStopped")
    finally:
        if raw_file is not None:
            raw_file.flush()
            raw_file.close()
            size_mb = Path(args.out).stat().st_size / 1e6
            print(f"Saved {packet_count} packets ({size_mb:.1f} MB) to {args.out}")
        if csv_file is not None:
            csv_file.close()
            print(
                f"Saved {saved_points} CSV points from {saved_scans} revolutions "
                f"({skipped_scans} skipped) to {args.csv}"
            )


if __name__ == "__main__":
    main()
