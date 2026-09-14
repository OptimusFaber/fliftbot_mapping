#!/usr/bin/env python3
"""Run V5 localization and render sequentially numbered physical points."""
from pathlib import Path
import runpy
import sys

sys.argv.extend(["--output-profile", "points"])
runpy.run_path(str(Path(__file__).with_name("ceiling_marker_localization_v5.py")), run_name="__main__")
