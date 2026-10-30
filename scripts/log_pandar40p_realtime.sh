#!/usr/bin/env bash
# Raw UDP logger. The resulting P40PRAW1 file is available for offline conversion.
set -euo pipefail
export PYTHONNOUSERSITE=1

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PANDAR_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
WS_DIR="$(cd "$PANDAR_ROOT/.." && pwd)"

OUT="${OUT:-${ROS_DATA_ROOT:-$PANDAR_ROOT/results}/pandar_capture.p40p}"
exec python3 "$SCRIPT_DIR/receive_pandar40p.py" --out "$OUT" "$@"
