"""Shared on-disk layout for recorded runs and rendered results."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path


def _discover_project_root() -> Path:
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / 'setup.py').exists() and (parent / 'src').exists():
            return parent
    return Path.cwd().resolve()


PROJECT_ROOT = _discover_project_root()
MAPS_ROOT = PROJECT_ROOT / 'maps'
RESULTS_ROOT = PROJECT_ROOT / 'results'


def _normalize_source(source: str | None) -> str:
    if source in ('sim', 'real'):
        return source
    return 'real'


def _date_from_run_name(run_name: str, when: datetime | None = None) -> str:
    if run_name.startswith('run_') and len(run_name) >= 12:
        candidate = run_name[4:12]
        if candidate.isdigit():
            return candidate
    return (when or datetime.now()).strftime('%Y%m%d')


def make_run_directories(source: str, when: datetime | None = None, run_name: str | None = None):
    when = when or datetime.now()
    source_name = _normalize_source(source)
    run_name = run_name or f"run_{when.strftime('%H%M%S')}"
    date_name = _date_from_run_name(run_name, when=when)

    maps_dir = MAPS_ROOT / date_name / source_name / run_name
    results_dir = RESULTS_ROOT / date_name / source_name / run_name
    maps_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    return maps_dir, results_dir


def default_maps_run_dir(source: str = 'sim', when: datetime | None = None) -> Path:
    """maps/YYYYMMDD/{sim|real}/run_HHMMSS/"""
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
        if 'maps' in parts:
            idx = parts.index('maps')
            if idx + 3 < len(parts):
                date_name = parts[idx + 1]
                source_name = parts[idx + 2]
                run_name = run_path.name
                extra = parts[idx + 3 : -1]
                results_dir = RESULTS_ROOT.joinpath(
                    date_name, source_name, *extra, run_name
                )
                results_dir.mkdir(parents=True, exist_ok=True)
                return results_dir
    run_name = run_name or 'run_unknown'
    date_name = _date_from_run_name(run_name)
    results_dir = RESULTS_ROOT / date_name / source_name / run_name
    results_dir.mkdir(parents=True, exist_ok=True)
    return results_dir
