"""Shared on-disk layout for recorded runs and rendered results."""

from __future__ import annotations

from datetime import datetime
import os
import re
from pathlib import Path


def _discover_project_root() -> Path:
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / 'setup.py').exists() and (parent / 'src').exists():
            return parent
    return Path.cwd().resolve()


PROJECT_ROOT = _discover_project_root()
DATA_ROOT = Path(
    os.environ.get('ROS_DATA_ROOT', PROJECT_ROOT.parent / 'ros-data')
).expanduser().resolve()
RECORDINGS_ROOT = DATA_ROOT / 'recordings'
RESULTS_ROOT = DATA_ROOT / 'results'
# Compatibility name for older imports. It no longer points inside the repo.
MAPS_ROOT = RECORDINGS_ROOT


def _normalize_source(source: str | None) -> str:
    if source in ('sim', 'real'):
        return source
    return 'real'


def _date_from_run_name(run_name: str, when: datetime | None = None) -> str:
    marker = run_name.find('run_')
    if marker >= 0 and len(run_name) >= marker + 12:
        candidate = run_name[marker + 4:marker + 12]
        if candidate.isdigit():
            return candidate
    return (when or datetime.now()).strftime('%Y%m%d')


def _canonical_run_name(source: str, run_name: str, scene: str | None, lidar: str | None) -> str:
    if re.fullmatch(r'\d{6}_[A-Za-z0-9]+_[A-Za-z0-9_]+', run_name):
        return run_name
    scene = scene or os.environ.get('SIM_SCENE') or ('warehouse' if source == 'sim' else 'room')
    lidar = lidar or os.environ.get('SIM_LIDAR') or ('pandar40p' if source == 'sim' else 'rplidar')
    match = re.search(r'(\d{6})(?!\d)', run_name)
    stamp = match.group(1) if match else datetime.now().strftime('%H%M%S')
    return f'{stamp}_{lidar}_{scene}'


def make_run_directories(
    source: str,
    when: datetime | None = None,
    run_name: str | None = None,
    *,
    scene: str | None = None,
    lidar: str | None = None,
):
    when = when or datetime.now()
    source_name = _normalize_source(source)
    run_name = run_name or f"run_{when.strftime('%H%M%S')}"
    run_name = _canonical_run_name(source_name, run_name, scene, lidar)
    date_name = _date_from_run_name(run_name, when=when)

    maps_dir = RECORDINGS_ROOT / source_name / date_name / run_name
    results_dir = RESULTS_ROOT / source_name / date_name / run_name
    maps_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    return maps_dir, results_dir


def default_maps_run_dir(source: str = 'sim', when: datetime | None = None) -> Path:
    """ros-data/recordings/{sim|real}/YYYYMMDD/SCENE_LIDAR_run_HHMMSS/."""
    maps_dir, _results = make_run_directories(source, when=when)
    return maps_dir


def is_legacy_recording_dir(path: str | None) -> bool:
    raw = (path or '').strip().rstrip('/')
    if not raw or raw in ('recorded-data/runs', 'recorded-data', 'maps'):
        return True
    return raw.endswith('recorded-data/runs') or raw.endswith('recorded-data')


def infer_results_dir(local_dir: str | Path | None, source: str | None = None, run_name: str | None = None):
    source_name = _normalize_source(source)
    if local_dir is not None:
        run_path = Path(local_dir).resolve()
        if run_name is None:
            run_name = run_path.name
        parts = run_path.parts
        if 'recordings' in parts:
            idx = parts.index('recordings')
            if idx + 3 < len(parts):
                source_name = parts[idx + 1]
                date_name = parts[idx + 2]
                results_dir = RESULTS_ROOT / source_name / date_name / run_path.name
                results_dir.mkdir(parents=True, exist_ok=True)
                return results_dir
        if 'maps' in parts:  # read-only compatibility with historical paths
            idx = parts.index('maps')
            if idx + 3 < len(parts):
                date_name = parts[idx + 1]
                source_name = parts[idx + 2]
                results_dir = RESULTS_ROOT / source_name / date_name / run_path.name
                results_dir.mkdir(parents=True, exist_ok=True)
                return results_dir
    run_name = run_name or 'run_unknown'
    date_name = _date_from_run_name(run_name)
    run_name = _canonical_run_name(source_name, run_name, None, None)
    results_dir = RESULTS_ROOT / source_name / date_name / run_name
    results_dir.mkdir(parents=True, exist_ok=True)
    return results_dir
