#!/usr/bin/env python3
"""Record a map-extension run while localizing against an existing deploy map."""
import argparse
import json
from pathlib import Path
import signal
import subprocess
import time
from robot_identity import labels_topic


ROOT = Path(__file__).resolve().parents[2]
COLLECT = ROOT / 'fliftbot_data_collect' / 'collect.sh'
NAV = ROOT / 'fliftbot_lidar_nav' / 'nav.sh'


def stop(process):
    if process is None or process.poll() is not None:
        return
    process.send_signal(signal.SIGINT)
    try:
        process.wait(timeout=12)
    except subprocess.TimeoutExpired:
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--map-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--sensor', choices=('rplidar', 'dephan'), default='dephan')
    parser.add_argument('--initial-localization', choices=('lidar', 'markers'), required=True)
    parser.add_argument('--marker-pose-topic', default='')
    parser.add_argument('--video', choices=('on', 'off'), required=True)
    parser.add_argument('--mc-topic', default='')
    parser.add_argument('--duration', type=float)
    parser.add_argument('--stable-scans', type=int, default=5)
    parser.add_argument('--send-to-navigation', action='store_true')
    parser.add_argument('--check-only', action='store_true', help='validate the map and arguments without starting hardware')
    args = parser.parse_args()
    if args.initial_localization == 'markers' and not args.marker_pose_topic:
        try:
            args.marker_pose_topic = labels_topic()
        except ValueError as exc:
            parser.error(str(exc) + '; set --marker-pose-topic explicitly')
    if args.video == 'off' and not args.mc_topic:
        parser.error('--mc-topic required when video is off')
    if args.duration is not None and args.duration <= 0:
        parser.error('--duration must be positive')
    if args.stable_scans < 2:
        parser.error('--stable-scans must be >= 2')
    if args.initial_localization == 'lidar' and args.send_to_navigation:
        parser.error('LiDAR-only acquisition is not validated for navigation output on this warehouse map')
    map_dir = args.map_dir.resolve()
    for name in ('map.pbstream', 'map.yaml', 'origin_reference.json', 'navigation_config.json'):
        if not (map_dir / name).is_file():
            parser.error(f'missing map file: {map_dir / name}')
    config = json.loads((map_dir / 'navigation_config.json').read_text())
    if config.get('lidar_type') != {'rplidar': 'rplidar', 'dephan': 'dephan_m360'}[args.sensor]:
        parser.error(f"map LiDAR type {config.get('lidar_type')!r} does not match --sensor {args.sensor}")
    if args.initial_localization == 'markers' and not (map_dir / 'pbstream_to_local_matrix.npy').is_file():
        parser.error('marker mode requires pbstream_to_local_matrix.npy')
    if args.initial_localization == 'markers':
        reference = json.loads((map_dir / 'origin_reference.json').read_text())
        if reference.get('origin_name') not in ('gt_marker_98', 'web_marker_057'):
            parser.error('marker mode requires a verified labels-coordinate origin (gt_marker_98 or web_marker_057)')
    output = args.output.resolve()
    if output.exists():
        parser.error(f'output must be a new directory: {output}')
    subprocess.run([str(NAV), 'check-map', '--map-dir', str(map_dir)], check=True)
    if args.check_only:
        print(f'Preflight accepted: {map_dir}; sensor={args.sensor}; initial={args.initial_localization}')
        return
    output.mkdir(parents=True)
    recording = output / 'recording'
    logs = output / 'localization'
    logs.mkdir()
    (output / 'run.json').write_text(json.dumps({
        'map_dir': str(map_dir), 'sensor': args.sensor, 'initial_localization': args.initial_localization,
        'marker_pose_topic': args.marker_pose_topic, 'video': args.video,
        'send_to_navigation': args.send_to_navigation, 'stable_scans': args.stable_scans,
    }, indent=2) + '\n')
    record_command = [str(COLLECT), 'record', '--sensor', args.sensor, '--recording', str(recording),
                      '--odometry-source', 'navigation' if args.video == 'on' else 'ros']
    if args.mc_topic:
        record_command += ['--mc-topic', args.mc_topic]
    if args.duration is not None:
        record_command += ['--duration', str(args.duration)]
    nav_command = [str(NAV), 'live', '--sensor', args.sensor, '--map-dir', str(map_dir),
                   f'initial_localization:={args.initial_localization}',
                   f'marker_input_topic:={args.marker_pose_topic}',
                   f'stable_scans:={args.stable_scans}',
                   f'marker_pose_csv:={logs / "poses.csv"}',
                   f'journal:={logs / "journal.jsonl"}',
                   f'send_to_navigation:={str(args.send_to_navigation).lower()}']
    if args.mc_topic:
        nav_command += [f'mc_topic:={args.mc_topic}']
    if args.sensor == 'rplidar':
        nav_command += [f'rplidar_recording:={recording / "scan.rplidar"}']
    else:
        nav_command += [f'dephan_recording:={recording / "scan.m360"}']
    recorder = nav = None
    try:
        recorder = subprocess.Popen(record_command, start_new_session=True)
        scan = recording / ('scan.rplidar' if args.sensor == 'rplidar' else 'scan.m360')
        deadline = time.monotonic() + 20
        while (not scan.exists() or scan.stat().st_size <= 8) and time.monotonic() < deadline:
            if recorder.poll() is not None:
                raise RuntimeError(f'recorder exited early: {recorder.returncode}')
            time.sleep(.2)
        if not scan.exists() or scan.stat().st_size <= 8:
            raise RuntimeError('no LiDAR scans within 20 seconds')
        with (logs / 'launch.log').open('w') as stream:
            nav = subprocess.Popen(nav_command, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
            while recorder.poll() is None and nav.poll() is None:
                time.sleep(.5)
            if nav.poll() is not None and recorder.poll() is None:
                raise RuntimeError(f'localization exited early: {nav.returncode}; see {logs / "launch.log"}')
            if recorder.returncode:
                raise RuntimeError(f'recorder failed: {recorder.returncode}')
    except KeyboardInterrupt:
        pass
    finally:
        stop(nav)
        stop(recorder)
    print(f'Run saved: {output}')


if __name__ == '__main__':
    main()
