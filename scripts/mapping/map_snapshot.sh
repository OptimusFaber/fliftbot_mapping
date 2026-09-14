#!/usr/bin/env bash
set -euo pipefail
export PYTHONNOUSERSITE=1
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
set +u
source /opt/ros/humble/setup.bash
source "${ROOT}/../install/setup.bash" 2>/dev/null || true
set -u
export PYTHONPATH="${ROOT}/src:${PYTHONPATH:-}"
cd "${ROOT}/src"
exec python3 -m flifltbot.common.cli.map_snapshot "$@"
