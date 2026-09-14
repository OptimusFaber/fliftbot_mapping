#!/usr/bin/python3
"""Validate the sqlite3 rosbag2 contract needed by offline Cartographer.

This deliberately uses only Python's sqlite3 module.  It therefore works even
when the ROS Python extension is not installed for the system interpreter and
can inspect an interrupted recording after its recorder process is stopped.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path


def _db_files(path: Path) -> list[Path]:
    if path.is_file() and path.suffix == ".db3":
        return [path]
    return sorted(path.glob("*.db3"))


def inspect(path: Path, required: list[str], optional: list[str]) -> dict:
    files = _db_files(path)
    if not files:
        raise ValueError(f"no sqlite3 bag files (*.db3) found below {path}")
    counts: dict[str, int] = {}
    first: dict[str, int] = {}
    last: dict[str, int] = {}
    nonmonotonic: list[dict] = []
    types: dict[str, str] = {}
    for db in files:
        try:
            connection = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
        except sqlite3.Error as exc:
            raise ValueError(f"cannot open {db}: {exc}") from exc
        with connection:
            topic_rows = connection.execute("SELECT id, name, type FROM topics").fetchall()
            topic_by_id = {int(row[0]): str(row[1]) for row in topic_rows}
            for row in topic_rows:
                types[str(row[1])] = str(row[2])
            previous: dict[str, int] = {}
            try:
                rows = connection.execute(
                    "SELECT timestamp, topic_id FROM messages ORDER BY id"
                )
            except sqlite3.Error as exc:
                raise ValueError(f"invalid rosbag2 schema in {db}: {exc}") from exc
            for stamp, topic_id in rows:
                topic = topic_by_id.get(int(topic_id), f"<topic:{topic_id}>")
                stamp = int(stamp)
                counts[topic] = counts.get(topic, 0) + 1
                first[topic] = min(first.get(topic, stamp), stamp)
                last[topic] = max(last.get(topic, stamp), stamp)
                if topic in previous and stamp < previous[topic]:
                    nonmonotonic.append({"topic": topic, "db": str(db), "previous": previous[topic], "current": stamp})
                previous[topic] = stamp
        connection.close()

    missing = [name for name in required if counts.get(name, 0) == 0]
    missing_optional = [name for name in optional if counts.get(name, 0) == 0]
    zero_stamp = [name for name in required if counts.get(name, 0) and first.get(name, 0) <= 0]
    return {
        "path": str(path),
        "files": [str(item) for item in files],
        "topics": {
            name: {"type": types.get(name), "count": counts.get(name, 0),
                   "first_timestamp_ns": first.get(name), "last_timestamp_ns": last.get(name)}
            for name in sorted(counts)
        },
        "missing_required": missing,
        "missing_optional": missing_optional,
        "zero_or_negative_first_timestamp": zero_stamp,
        "nonmonotonic": nonmonotonic,
        "ok": not missing and not zero_stamp and not nonmonotonic,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bag", type=Path, help="rosbag2 directory or one .db3 file")
    parser.add_argument("--scan-topic", default="/lidar_top/scan")
    parser.add_argument("--odom-topic", default="/wheel_encoder_odom")
    parser.add_argument("--clock", action="store_true", help="require /clock (simulation)")
    parser.add_argument("--json", dest="json_path", type=Path, help="write the report atomically")
    args = parser.parse_args()
    required = [args.scan_topic, args.odom_topic, "/tf"]
    # /tf_static is not required when the scan frame is base_link (as in the
    # real HTTP bridge), but it is still recorded and reported when available.
    optional = ["/tf_static"]
    if args.clock:
        required.append("/clock")
    try:
        report = inspect(args.bag, required, optional)
    except (OSError, ValueError, sqlite3.Error) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    if args.json_path:
        temporary = args.json_path.with_name(args.json_path.name + ".tmp")
        args.json_path.parent.mkdir(parents=True, exist_ok=True)
        temporary.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        temporary.replace(args.json_path)
    for topic in required + optional:
        print(f"{topic}: {report['topics'].get(topic, {}).get('count', 0)} messages")
    if report["nonmonotonic"]:
        print(f"ERROR: {len(report['nonmonotonic'])} non-monotonic timestamp sequences", file=sys.stderr)
    if report["missing_required"]:
        print("ERROR: missing required topics: " + ", ".join(report["missing_required"]), file=sys.stderr)
    if report["missing_optional"]:
        print("ERROR: missing requested optional topics: " + ", ".join(report["missing_optional"]), file=sys.stderr)
    if report["zero_or_negative_first_timestamp"]:
        print("ERROR: required topic starts at a non-positive timestamp: " + ", ".join(report["zero_or_negative_first_timestamp"]), file=sys.stderr)
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
