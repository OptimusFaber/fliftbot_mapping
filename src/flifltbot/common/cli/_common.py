from __future__ import annotations

from datetime import datetime
from pathlib import Path
from flifltbot.common.output.layout import PROJECT_ROOT, make_run_directories
from flifltbot.common.runtime.source_detect import detect_source


def parse_cameras(value):
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        return list(value)
    return [item.strip() for item in str(value).split(',') if item.strip()]


def resolve_path(path, *, must_exist=False):
    """Resolve a path relative to cwd, falling back to the project root.

    Shell wrappers ``cd`` into ``src/``. Canonical recordings should normally
    be passed as absolute paths under ``ros-data/recordings``.
    """
    raw = Path(path).expanduser()
    if raw.is_absolute():
        resolved = raw.resolve()
    else:
        cwd_candidate = (Path.cwd() / raw).resolve()
        root_candidate = (PROJECT_ROOT / raw).resolve()
        if cwd_candidate.exists():
            resolved = cwd_candidate
        elif root_candidate.exists():
            resolved = root_candidate
        else:
            resolved = cwd_candidate
    if must_exist and not resolved.exists():
        raise FileNotFoundError(
            f'path not found: {path} (tried {Path.cwd() / raw} and {PROJECT_ROOT / raw})'
        )
    return resolved


def ensure_dir(path):
    p = resolve_path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def resolve_run_dir(path):
    run_dir = resolve_path(path, must_exist=True)
    if not run_dir.is_dir():
        raise NotADirectoryError(f'run dir is not a directory: {run_dir}')
    return run_dir


def resolve_output_dir(path, source='auto'):
    if path:
        return ensure_dir(path)
    when = datetime.now()
    resolved = detect_source(source)
    if resolved not in ('sim', 'real'):
        resolved = 'real'
    out_dir, _results_dir = make_run_directories(
        resolved, when=when, run_name=f"run_{when.strftime('%H%M%S')}"
    )
    return out_dir
