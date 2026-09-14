#!/usr/bin/env bash
# Record the canonical rosbag2 input for an offline Cartographer run.
# Run this on the robot/edge computer: a remote recorder cannot recover data
# that was lost during a network outage.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DATA_ROOT="${ROS_DATA_ROOT:-$(cd "${ROOT}/.." && pwd)/ros-data}"
KIND=sim
SCAN_TOPIC=/lidar_top/scan
ODOM_TOPIC=/wheel_encoder_odom
OUTPUT=""
MAX_DURATION=300
SIM_TIME=0
SCENE=room

usage() {
  cat <<'EOF'
Usage: record_cartographer_run.sh [options]
  --output DIR          run directory (default: ros-data/recordings/<kind>/...)
  --kind sim|real       recording class (default: sim)
  --scene NAME          scene label for the output name (default: room)
  --scan-topic TOPIC    LaserScan topic (default: /lidar_top/scan)
  --odom-topic TOPIC    Odometry topic (default: /wheel_encoder_odom)
  --sim-time            record /clock and use simulated time
  --chunk-seconds N     rotate sqlite bag files every N seconds (default: 300)
EOF
}

while (($#)); do
  case "$1" in
    --output) OUTPUT="${2:?--output needs a directory}"; shift 2 ;;
    --kind) KIND="${2:?--kind needs sim or real}"; shift 2 ;;
    --scene) SCENE="${2:?--scene needs a name}"; shift 2 ;;
    --scan-topic) SCAN_TOPIC="${2:?--scan-topic needs a topic}"; shift 2 ;;
    --odom-topic) ODOM_TOPIC="${2:?--odom-topic needs a topic}"; shift 2 ;;
    --sim-time) SIM_TIME=1; shift ;;
    --chunk-seconds) MAX_DURATION="${2:?--chunk-seconds needs seconds}"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done
case "${KIND}" in sim|real) ;; *) echo "--kind must be sim or real" >&2; exit 2 ;; esac
if ! [[ "${MAX_DURATION}" =~ ^[1-9][0-9]*$ ]]; then
  echo "--chunk-seconds must be a positive integer" >&2; exit 2
fi
SCENE="${SCENE,,}"
SCENE="${SCENE//[^a-z0-9_]/_}"
[[ -n "${SCENE}" ]] || { echo "--scene is empty after normalization" >&2; exit 2; }
if [[ -z "${OUTPUT}" ]]; then
  # Do not introduce a second payload format into the canonical NPZ tree.
  OUTPUT="${DATA_ROOT}/cartographer_bags/${KIND}/$(date -u +%Y%m%d)/$(date -u +%H%M%S)_cartographer_${SCENE}"
fi
if [[ -e "${OUTPUT}/metadata.yaml" || -e "${OUTPUT}/recording_manifest.json" ]]; then
  echo "Refusing to reuse an existing recording: ${OUTPUT}" >&2
  exit 2
fi
mkdir -p "${OUTPUT}"

set +u
source /opt/ros/humble/setup.bash
set -u
command -v ros2 >/dev/null || { echo "ros2 is unavailable" >&2; exit 2; }
topics=("${SCAN_TOPIC}" "${ODOM_TOPIC}" /tf /tf_static)
if (( SIM_TIME )); then topics+=(/clock); fi

started_at="$(date --iso-8601=seconds --utc)"
manifest="${OUTPUT}/recording_manifest.json"
tmp_manifest="${manifest}.tmp"
clock_entry=""
if (( SIM_TIME )); then clock_entry=', "/clock"'; fi
printf '{\n  "schema": "fliftbot.cartographer_recording.v1",\n  "kind": "%s",\n  "scene": "%s",\n  "scan_topic": "%s",\n  "odom_topic": "%s",\n  "sim_time": %s,\n  "chunk_seconds": %s,\n  "started_at": "%s",\n  "status": "recording",\n  "topics": ["%s", "%s", "/tf", "/tf_static"%s]\n}\n' \
  "${KIND}" "${SCENE}" "${SCAN_TOPIC}" "${ODOM_TOPIC}" "$([[ ${SIM_TIME} == 1 ]] && echo true || echo false)" \
  "${MAX_DURATION}" "${started_at}" "${SCAN_TOPIC}" "${ODOM_TOPIC}" "${clock_entry}" > "${tmp_manifest}"
mv -f "${tmp_manifest}" "${manifest}"

record_pid=""
stopped=0
finish() {
  local rc=$?
  (( stopped )) && return
  stopped=1
  if [[ -n "${record_pid}" ]] && kill -0 "${record_pid}" 2>/dev/null; then
    kill -INT "${record_pid}" 2>/dev/null || true
    wait "${record_pid}" 2>/dev/null || true
  fi
  finished_at="$(date --iso-8601=seconds --utc)"
  sed -e 's/"status": "recording"/"status": "complete"/' \
      -e "s/  \"started_at\": \"[^\"]*\",/  \"started_at\": \"${started_at}\",\n  \"finished_at\": \"${finished_at}\",/" \
      "${manifest}" > "${tmp_manifest}" && mv -f "${tmp_manifest}" "${manifest}"
  return "${rc}"
}
trap finish EXIT INT TERM

echo "Recording on local filesystem: ${OUTPUT}"
echo "Topics: ${topics[*]}"
echo "Use Ctrl-C to finish."
ros2 bag record --storage sqlite3 --output "${OUTPUT}" \
  --max-bag-duration "${MAX_DURATION}" --max-cache-size 0 \
  --polling-interval 250 $([[ ${SIM_TIME} == 1 ]] && echo --use-sim-time) "${topics[@]}" &
record_pid=$!
wait "${record_pid}"
