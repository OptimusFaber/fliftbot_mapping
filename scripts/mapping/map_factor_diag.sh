#!/usr/bin/env bash
# Factor diagnostics vs warehouse GT. /odom = GT only (not in SLAM).
set -euo pipefail
export PYTHONNOUSERSITE=1
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
set +u
source /opt/ros/humble/setup.bash 2>/dev/null || true
source "${ROOT}/../install/setup.bash" 2>/dev/null || true
export PYTHONPATH="${ROOT}/src:${PYTHONPATH:-}"
cd "${ROOT}/src"
exec python3 -m flifltbot.common.cli.map_factor_diag "$@"
