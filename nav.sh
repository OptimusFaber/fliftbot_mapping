#!/usr/bin/env bash
set -eo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
export PYTHONNOUSERSITE=1
export PYTHONPATH="${ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
PYTHON_BIN="${FLIFTBOT_LIDAR_NAV_PYTHON:-${ROOT}/.venv/bin/python}"
[[ -x "${PYTHON_BIN}" ]] || PYTHON_BIN=/usr/bin/python3
export FLIFTBOT_LIDAR_NAV_PYTHON="${PYTHON_BIN}"
exec "${PYTHON_BIN}" -m fliftbot_lidar_nav.cli "$@"
