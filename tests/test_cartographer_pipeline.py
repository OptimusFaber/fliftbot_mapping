"""Pure-Python contract tests for the Cartographer offline artifacts."""

import json
import sqlite3
import subprocess
from pathlib import Path

from flifltbot.common.cartographer_trajectory import build_trajectory_json
from flifltbot.common.cartographer_trajectory import quaternion_to_euler


def _pose(stamp, x, y, yaw):
    import math
    return {
        "header": {"stamp": {"sec": stamp, "nanosec": 0}},
        "pose": {
            "position": {"x": x, "y": y, "z": 0.0},
            "orientation": {"x": 0.0, "y": 0.0,
                            "z": math.sin(yaw / 2), "w": math.cos(yaw / 2)},
        },
    }


def test_trajectory_is_normalized_with_rotation():
    result = build_trajectory_json(
        [_pose(10, 5.0, 7.0, 0.5), _pose(11, 6.0, 7.0, 0.5)],
        source_pbstream="map.pbstream", trajectory_id=0,
    )
    assert result["poses"][0]["pose"]["x"] == 0.0
    assert result["poses"][0]["pose"]["y"] == 0.0
    assert abs(result["poses"][1]["pose"]["yaw"]) < 1e-12
    assert result["poses"][1]["global"]["x"] == 6.0
    assert abs(result["origin"]["global"]["yaw"] - 0.5) < 1e-12


def test_zero_quaternion_is_safe():
    assert quaternion_to_euler(0, 0, 0, 0) == (0.0, 0.0, 0.0)


def test_sqlite_validator_accepts_minimal_bag(tmp_path):
    bag = tmp_path / "run"
    bag.mkdir()
    db = bag / "run_0.db3"
    con = sqlite3.connect(db)
    con.executescript("""
      CREATE TABLE topics (id INTEGER PRIMARY KEY, name TEXT NOT NULL, type TEXT NOT NULL,
                           serialization_format TEXT NOT NULL, offered_qos_profiles TEXT NOT NULL);
      CREATE TABLE messages (id INTEGER PRIMARY KEY AUTOINCREMENT, topic_id INTEGER NOT NULL,
                             timestamp INTEGER NOT NULL, data BLOB NOT NULL);
    """)
    topics = ["/lidar_top/scan", "/wheel_encoder_odom", "/tf", "/tf_static", "/clock"]
    for index, topic in enumerate(topics, 1):
        con.execute("INSERT INTO topics VALUES (?, ?, ?, 'cdr', '')", (index, topic, "test_msgs/msg/Test"))
        con.execute("INSERT INTO messages(topic_id,timestamp,data) VALUES (?, ?, X'00')", (index, 1_000 + index))
    con.commit(); con.close()
    script = Path(__file__).parents[1] / "scripts/mapping/validate_cartographer_recording.py"
    report = tmp_path / "report.json"
    proc = subprocess.run(["/usr/bin/python3", str(script), str(bag), "--clock", "--json", str(report)])
    assert proc.returncode == 0
    assert json.loads(report.read_text())["ok"] is True
