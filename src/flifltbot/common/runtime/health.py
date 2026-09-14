"""Lightweight runtime health checks for mapping inputs."""

from __future__ import annotations

import subprocess

from .source_detect import detect_source

TOPICS = {
    'lidar': [
        '/lidar_top/points',
        '/lidar_top/scan',
        '/lidar_left/scan',
        '/lidar_right/scan',
    ],
    'odom': ['/odom'],
}


def _topic_list():
    try:
        proc = subprocess.run(
            ['ros2', 'topic', 'list'],
            check=True,
            text=True,
            capture_output=True,
        )
    except Exception:
        return set()
    return {line.strip() for line in proc.stdout.splitlines() if line.strip()}


def check_inputs(source='auto'):
    topics = _topic_list()
    resolved = detect_source(source)
    return {
        'source': resolved,
        'topics_seen': sorted(topics),
        'lidar_alive': any(topic in topics for topic in TOPICS['lidar']),
        'odom_alive': any(topic in topics for topic in TOPICS['odom']),
        'encoder_alive': '/odom' in topics,
    }


def summary_lines(report):
    return [
        f"source={report['source']}",
        f"lidar_alive={'yes' if report['lidar_alive'] else 'no'}",
        f"odom_alive={'yes' if report['odom_alive'] else 'no'}",
        f"encoder_alive={'yes' if report['encoder_alive'] else 'no'}",
    ]
