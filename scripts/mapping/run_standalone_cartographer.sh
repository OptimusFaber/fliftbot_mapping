#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
BUILD="${ROOT}/standalone_cartographer/build"
SOURCE="${ROOT}/standalone_cartographer"
if [[ ! -x "${BUILD}/fliftbot_cartographer_offline" ]]; then
  cmake -S "${SOURCE}" -B "${BUILD}" -DCMAKE_PREFIX_PATH=/opt/ros/humble
  cmake --build "${BUILD}" -j"$(nproc)"
fi
# Keep GT/navigation artefacts next to a recording harmless: the standalone
# runner is allowed to consume only horizon/*.npz, encoder.jsonl and the
# optional lidar.config.  Forward all flags; the checker consumes --run_dir
# and ignores Cartographer-specific options.
for arg in "$@"; do
  if [[ "$arg" == "--help" || "$arg" == "-h" ]]; then
    exec "${BUILD}/fliftbot_cartographer_offline" "$@"
  fi
done
python3 "${ROOT}/scripts/evaluation/check_standalone_inputs.py" "$@"
exec "${BUILD}/fliftbot_cartographer_offline" "$@"
