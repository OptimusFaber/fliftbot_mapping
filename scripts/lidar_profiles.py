"""Load the shared direct-UDP LiDAR profiles."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PROFILE_PATH = ROOT / "config" / "lidar_profiles.json"


def load_profile(sensor: str, map_dir: Path | None = None) -> dict:
    profiles = json.loads(PROFILE_PATH.read_text(encoding="utf-8"))
    try:
        profile = profiles[sensor]
    except KeyError as exc:
        raise ValueError(f"unknown LiDAR profile {sensor!r}") from exc
    if map_dir is not None:
        from map_bundle import validate_bundle
        report = validate_bundle(map_dir)
        calibration = report["navigation_config"]
        supported = {"dephan": ("dephan_m360",), "pandar": ("pandar40p",)}
        if calibration["lidar_type"] not in supported[sensor]:
            raise ValueError(f"bundle LiDAR {calibration['lidar_type']} does not match {sensor}")
        profile = {**profile, **calibration}
        map_dir = Path(report["map_dir"])
        reference = map_dir / "origin_reference.json"
    else:
        map_dir = ROOT / profile["map_directory"]
        reference = map_dir / "marker98_map_reference.json"
    mount = profile["scan_to_base"]
    return {
        **profile,
        "map_dir": map_dir,
        "map": map_dir / "map.yaml",
        "pbstream": map_dir / "map.pbstream",
        "reference": reference,
        "mount": (float(mount["x_m"]), float(mount["y_m"])),
        "sensor_yaw_deg": float(mount["yaw_deg"]),
    }
