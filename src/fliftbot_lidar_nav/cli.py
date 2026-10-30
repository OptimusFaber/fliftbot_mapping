"""Common source/installed entry point for Fliftbot LiDAR navigation."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys


def resource_root() -> Path:
    source = Path(__file__).resolve().parents[2]
    if (source / "scripts/map_bundle.py").is_file():
        return source
    from ament_index_python.packages import get_package_share_directory
    return Path(get_package_share_directory("fliftbot_lidar_nav"))


def _main() -> int:
    os.environ.setdefault("FLIFTBOT_LIDAR_NAV_PYTHON", sys.executable)
    root = resource_root()
    parser = argparse.ArgumentParser(description="Fliftbot LiDAR localization; maps are built by fliftbot_map_build")
    commands = parser.add_subparsers(dest="command", required=True)
    check = commands.add_parser("check-map", help="verify deploy files, hashes, origin and calibration")
    check.add_argument("--map-dir", type=Path, required=True)
    commands.add_parser("verify", help="offline syntax, bundle and localization regression checks")
    for name, help_text in (("instant", "one full UDP revolution without ROS"),
                            ("localize-recording", "canonical NPZ replay without ROS"),
                            ("sim", "live localization on fliftbot_sim sensor topics"),
                            ("live", "UDP + MC + Cartographer + navigation pose output")):
        commands.add_parser(name, help=help_text, add_help=False)
    args, forwarded = parser.parse_known_args()
    sys.path.insert(0, str(root / "scripts"))
    if args.command == "check-map":
        if forwarded:
            parser.error(f"unknown arguments: {' '.join(forwarded)}")
        from map_bundle import validate_bundle
        print(json.dumps(validate_bundle(args.map_dir), indent=2))
        return 0
    if args.command == "verify":
        return subprocess.call([sys.executable, str(root / "scripts/verify.py"), *forwarded])
    if args.command == "sim":
        sim = argparse.ArgumentParser(description="Localize live fliftbot_sim on a deploy map; isolated wheel-only TF")
        sim.add_argument("--map-dir", type=Path, required=True)
        config, launch_arguments = sim.parse_known_args(forwarded)
        from map_bundle import validate_bundle
        validate_bundle(config.map_dir)
        command = ["ros2", "launch", str(root / "launch/sim_localization.launch.py"),
                   f"map_dir:={config.map_dir.resolve()}", *launch_arguments]
    elif args.command == "live":
        live = argparse.ArgumentParser(description="Launch UDP/MC pure localization using calibrated map bundle")
        live.add_argument("--sensor", choices=("dephan", "pandar"), required=True)
        live.add_argument("--map-dir", type=Path, required=True)
        config, launch_arguments = live.parse_known_args(forwarded)
        from lidar_profiles import load_profile
        load_profile(config.sensor, config.map_dir)
        command = ["ros2", "launch", str(root / "launch/mc_lidar_localization.launch.py"),
                   f"sensor:={config.sensor}", f"map_dir:={config.map_dir.resolve()}", *launch_arguments]
    else:
        script = {"instant": "instant_global_localize.py",
                  "localize-recording": "localize_recording.py"}[args.command]
        command = [sys.executable, str(root / "scripts" / script), *forwarded]
    os.execvp(command[0], command)
    return 0


def main() -> int:
    try:
        return _main()
    except (ValueError, OSError, KeyError) as exc:
        print(f"LiDAR navigation: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
