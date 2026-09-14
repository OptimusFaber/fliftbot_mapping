#!/usr/bin/env /usr/bin/python3
"""Export Cartographer's optimized trajectory through TrajectoryQuery.

Start ``cartographer_node`` with ``-load_state_filename`` first.  The script
uses the service rather than /tracked_pose so loop-closure optimized poses are
exported.  It intentionally imports ROS only in ``main`` so the normalisation
code remains unit-testable on machines without ROS Python bindings.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pbstream", required=True, help="source pbstream, recorded in JSON metadata")
    parser.add_argument("--output", required=True, help="output JSON path")
    parser.add_argument("--trajectory-id", type=int, default=0)
    parser.add_argument("--service", default="/trajectory_query")
    parser.add_argument("--timeout", type=float, default=30.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        import rclpy
        from cartographer_ros_msgs.srv import TrajectoryQuery
        from flifltbot.common.cartographer_trajectory import build_trajectory_json
    except ImportError as exc:
        print(f"ROS 2 Python bindings are unavailable: {exc}", file=sys.stderr)
        print("Run this command with /usr/bin/python3 after sourcing /opt/ros/humble.", file=sys.stderr)
        return 2

    rclpy.init()
    node = rclpy.create_node("fliftbot_cartographer_trajectory_export")
    client = node.create_client(TrajectoryQuery, args.service)
    deadline = time.monotonic() + args.timeout
    try:
        while not client.wait_for_service(timeout_sec=min(1.0, max(0.0, deadline - time.monotonic()))):
            if time.monotonic() >= deadline:
                print(f"Timed out waiting for {args.service}", file=sys.stderr)
                return 2
        request = TrajectoryQuery.Request()
        request.trajectory_id = args.trajectory_id
        future = client.call_async(request)
        while rclpy.ok() and not future.done():
            rclpy.spin_once(node, timeout_sec=0.2)
        response = future.result()
        if response is None:
            print("TrajectoryQuery returned no response", file=sys.stderr)
            return 2
        # StatusResponse.code == 0 is OK in cartographer_ros_msgs.
        status = getattr(response, "status", None)
        if status is not None and getattr(status, "code", 0) != 0:
            print(f"TrajectoryQuery failed: {status.message}", file=sys.stderr)
            return 2
        response_header = getattr(response, "header", None)
        document = build_trajectory_json(
            response.trajectory,
            source_pbstream=os.path.abspath(args.pbstream),
            trajectory_id=args.trajectory_id,
            # Humble's TrajectoryQuery response has no header field; newer
            # builds may add one.  Cartographer's map frame is the stable
            # fallback for both service versions.
            frame_id=getattr(response_header, "frame_id", "map") or "map",
        )
        output = os.path.abspath(args.output)
        os.makedirs(os.path.dirname(output) or ".", exist_ok=True)
        temporary = output + ".tmp"
        with open(temporary, "w", encoding="utf-8") as handle:
            json.dump(document, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, output)
        print(f"Exported {len(document['poses'])} optimized poses to {output}")
        return 0
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
