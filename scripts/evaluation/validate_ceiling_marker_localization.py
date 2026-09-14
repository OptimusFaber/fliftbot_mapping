#!/usr/bin/env python3
"""Run V5 localization and offline validation against explicitly supplied GT."""
from pathlib import Path
import runpy
import sys

if "--gt" not in sys.argv and not any(value in sys.argv for value in ("-h", "--help")):
    raise SystemExit("validation requires: --gt /path/to/ceiling_labels.json")
sys.argv.extend(["--output-profile", "all"])
runpy.run_path(str(Path(__file__).with_name("ceiling_marker_localization_v5.py")), run_name="__main__")
