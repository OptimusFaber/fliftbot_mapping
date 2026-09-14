"""Helpers for choosing sim/real/offline mapping source."""

from __future__ import annotations

from pathlib import Path
import os


def detect_source(source='auto', run_dir=None):
    if source and source != 'auto':
        return source
    if run_dir and Path(run_dir).exists():
        return 'offline_run'
    if os.environ.get('ROS_DOMAIN_ID') is not None and os.environ.get('IGN_GAZEBO_RESOURCE_PATH'):
        return 'sim'
    if os.environ.get('USE_SIM_TIME') == '1':
        return 'sim'
    return 'real'
