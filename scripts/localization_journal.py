"""ROS-independent, append-only localization journal and scenario contract."""
from __future__ import annotations

import json
import math
from pathlib import Path
import time
import uuid

DEFAULTS = {
    "scenario_id": "unspecified",
    "position_error_m": 0.05,
    "orientation_error_deg": None,
    "max_pose_age_s": 0.5,
    "max_reference_gap_s": 0.2,
    "min_event_duration_s": 0.5,
    "min_availability": 0.99,
    "min_reference_coverage": 0.95,
    "required_within_threshold_fraction": 0.95,
    "max_failure_episodes": 0,
    "jump_position_m": 0.5,
    "jump_orientation_deg": 20.0,
    "max_sensor_gap_s": 0.5,
}


def load_scenario(path=None):
    supplied = json.loads(Path(path).read_text()) if path else {}
    unknown = set(supplied) - set(DEFAULTS) - {"description", "route_id", "route_start_s", "route_end_s"}
    if unknown:
        raise ValueError(f"unknown scenario fields: {sorted(unknown)}")
    cfg = {**DEFAULTS, **supplied}
    if not isinstance(cfg["scenario_id"], str) or not cfg["scenario_id"].strip():
        raise ValueError("scenario_id must be a nonempty string")
    for key in DEFAULTS:
        if key == "scenario_id" or (key == "orientation_error_deg" and cfg[key] is None):
            continue
        value = cfg[key]
        if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value) or value < 0:
            raise ValueError(f"{key} must be finite and nonnegative")
        if key != "max_failure_episodes" and value == 0:
            raise ValueError(f"{key} must be positive")
    for key in ("min_availability", "min_reference_coverage", "required_within_threshold_fraction"):
        if cfg[key] > 1:
            raise ValueError(f"{key} must be <= 1")
    if int(cfg["max_failure_episodes"]) != cfg["max_failure_episodes"]:
        raise ValueError("max_failure_episodes must be an integer")
    for key in ("route_start_s", "route_end_s"):
        if key in cfg and (not isinstance(cfg[key], (int, float)) or not math.isfinite(cfg[key])):
            raise ValueError(f"{key} must be finite")
    if cfg.get("route_end_s", math.inf) <= cfg.get("route_start_s", -math.inf):
        raise ValueError("route_end_s must be after route_start_s")
    return cfg


def wrap(angle):
    return (angle + math.pi) % (2 * math.pi) - math.pi


def json_safe(value):
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return value


class Journal:
    def __init__(self, path, scenario, metadata=None):
        self.scenario = scenario
        self.run_id = uuid.uuid4().hex
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.stream = path.open("x", encoding="utf-8")
        self.last_stamp = None
        self.last_pose = None
        self.state = None
        self.sensor_states = {}
        self.last_flush = time.monotonic()
        self.write("run_start", scenario=scenario, metadata=metadata or {}, wall_time_s=time.time())

    def write(self, kind, **fields):
        record = {"schema": "fliftbot.localization_journal.v1", "run_id": self.run_id,
                  "scenario_id": self.scenario["scenario_id"], "kind": kind, **fields}
        self.stream.write(json.dumps(json_safe(record), ensure_ascii=False, allow_nan=False) + "\n")
        if kind != "sample" or time.monotonic() - self.last_flush >= 1:
            self.stream.flush()
            self.last_flush = time.monotonic()

    def observe(self, now, stamp=None, pose=None, reason=None, **fields):
        finite = stamp is not None and math.isfinite(stamp) and pose is not None and all(math.isfinite(v) for v in pose)
        age = now - stamp if finite else None
        fresh = finite and -0.25 <= age <= self.scenario["max_pose_age_s"]
        new = finite and (self.last_stamp is None or stamp > self.last_stamp + 1e-6)
        state = "available" if fresh else (reason or ("stale_pose" if finite else "no_pose"))
        if state != self.state:
            self.write("event", time_s=now, event="state_changed", previous=self.state, state=state)
            self.state = state
        for sensor in ("scan", "odom"):
            key = sensor + "_available"
            if key in fields and fields[key] != self.sensor_states.get(sensor):
                self.write("event", time_s=now, event="sensor_state_changed", sensor=sensor,
                           available=fields[key])
                self.sensor_states[sensor] = fields[key]
        jump = False
        if new and fresh and self.last_pose is not None:
            distance = math.hypot(pose[0] - self.last_pose[0], pose[1] - self.last_pose[1])
            angle = abs(math.degrees(wrap(pose[2] - self.last_pose[2])))
            jump = distance > self.scenario["jump_position_m"] or angle > self.scenario["jump_orientation_deg"]
            if jump:
                self.write("event", time_s=now, event="pose_jump", distance_m=distance,
                           orientation_deg=angle, sample_interval_s=stamp-self.last_stamp)
        self.write("sample", time_s=now, stamp_s=stamp if finite else None,
                   pose=list(pose) if finite else None, age_s=age, fresh=bool(fresh),
                   new_sample=bool(new), state=state, jump=jump, **fields)
        if new:
            self.last_stamp = stamp
            self.last_pose = tuple(pose) if fresh else None

    def close(self, now=None):
        if not self.stream.closed:
            self.write("run_end", time_s=now, wall_time_s=time.time())
            self.stream.close()
