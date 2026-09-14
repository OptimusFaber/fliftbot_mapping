#!/usr/bin/env bash
# Validate a rosbag2 run, build a Cartographer pbstream/map, and export the
# loop-closure-optimized trajectory to JSON.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
BAG=""; OUT=""; SCAN_TOPIC=/lidar_top/scan; ODOM_TOPIC=/wheel_encoder_odom
CONFIG_DIR="${ROOT}/config/cartographer"; CONFIG_BASE=fliftbot_2d.lua
URDF="${ROOT}/config/cartographer/fliftbot_base.urdf"
SIM_TIME=0; RESOLUTION=0.05; TRAJECTORY_ID=0
usage() {
  cat <<'EOF'
Usage: offline_cartographer.sh --bag RUN_DIR [options]
  --bag DIR[,DIR...]       rosbag2 run(s); split chunks in one DIR are read together
  --output DIR             result directory (default: RUN_DIR/cartographer)
  --scan-topic TOPIC       (default: /lidar_top/scan)
  --odom-topic TOPIC       (default: /wheel_encoder_odom)
  --sim-time               require /clock in validation
  --configuration-dir DIR  Cartographer Lua directory
  --configuration-file F   Lua basename (default: fliftbot_2d.lua)
  --urdf FILE               robot URDF (default: fliftbot_base.urdf)
  --resolution M           occupancy map resolution (default: 0.05)
  --trajectory-id N        trajectory service id (default: 0)
EOF
}
while (($#)); do
  case "$1" in
    --bag) BAG="${2:?--bag needs a rosbag2 directory}"; shift 2 ;;
    --output) OUT="${2:?--output needs a directory}"; shift 2 ;;
    --scan-topic) SCAN_TOPIC="${2:?--scan-topic needs a topic}"; shift 2 ;;
    --odom-topic) ODOM_TOPIC="${2:?--odom-topic needs a topic}"; shift 2 ;;
    --sim-time) SIM_TIME=1; shift ;;
    --configuration-dir) CONFIG_DIR="${2:?--configuration-dir needs a directory}"; shift 2 ;;
    --configuration-file) CONFIG_BASE="${2:?--configuration-file needs a file}"; shift 2 ;;
    --urdf) URDF="${2:?--urdf needs a file}"; shift 2 ;;
    --resolution) RESOLUTION="${2:?--resolution needs metres}"; shift 2 ;;
    --trajectory-id) TRAJECTORY_ID="${2:?--trajectory-id needs an id}"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done
[[ -n "${BAG}" ]] || { echo "--bag is required" >&2; usage >&2; exit 2; }
[[ -f "${CONFIG_DIR}/${CONFIG_BASE}" ]] || { echo "Cartographer config not found: ${CONFIG_DIR}/${CONFIG_BASE}" >&2; exit 2; }
[[ -f "${URDF}" ]] || { echo "Cartographer URDF not found: ${URDF}" >&2; exit 2; }
if [[ -z "${OUT}" ]]; then OUT="${BAG%%,*}/cartographer"; fi
mkdir -p "${OUT}"
set +u
source /opt/ros/humble/setup.bash
source "${ROOT}/../install/setup.bash" 2>/dev/null || true
set -u
export PYTHONPATH="${ROOT}/src:${PYTHONPATH:-}"
validator="${ROOT}/scripts/mapping/validate_cartographer_recording.py"
IFS=',' read -r -a bags <<< "${BAG}"
for item in "${bags[@]}"; do
  [[ -d "${item}" || -f "${item}" ]] || { echo "Bag not found: ${item}" >&2; exit 2; }
  validation_args=("${item}" --scan-topic "${SCAN_TOPIC}" --odom-topic "${ODOM_TOPIC}" --json "${OUT}/validation_$(basename "${item}").json")
  if (( SIM_TIME )); then validation_args+=(--clock); fi
  /usr/bin/python3 "${validator}" "${validation_args[@]}"
  ros2 run cartographer_ros cartographer_rosbag_validate \
    -bag_filename="${item}"
done
PBSTREAM="${OUT}/map.pbstream"; MAP_STEM="${OUT}/map"
bag_list="$(IFS=,; echo "${bags[*]}")"
echo "Running cartographer_offline_node; output: ${PBSTREAM}"
ros2 run cartographer_ros cartographer_offline_node \
  -configuration_directory="${CONFIG_DIR}" \
  -configuration_basenames="${CONFIG_BASE}" -bag_filenames="${bag_list}" \
  -urdf_filenames="${URDF}" \
  -save_state_filename="${PBSTREAM}" -load_frozen_state=false \
  -use_bag_transforms=true -keep_running=false --ros-args \
  -r scan:="${SCAN_TOPIC}" -r odom:="${ODOM_TOPIC}"
[[ -s "${PBSTREAM}" ]] || { echo "Cartographer did not create ${PBSTREAM}" >&2; exit 1; }
echo "Rendering occupancy map"
ros2 run cartographer_ros cartographer_pbstream_to_ros_map \
  -pbstream_filename="${PBSTREAM}" \
  -map_filestem="${MAP_STEM}" -resolution="${RESOLUTION}"
NODE_LOG="${OUT}/trajectory_query_node.log"
echo "Starting Cartographer node to query optimized trajectory"
ros2 run cartographer_ros cartographer_node \
  -configuration_directory="${CONFIG_DIR}" \
  -configuration_basename="${CONFIG_BASE}" -load_state_filename="${PBSTREAM}" \
  -load_frozen_state=true -start_trajectory_with_default_topics=false \
  --ros-args -p use_sim_time:=false >"${NODE_LOG}" 2>&1 &
NODE_PID=$!
cleanup_node() {
  if kill -0 "${NODE_PID}" 2>/dev/null; then
    kill -INT "${NODE_PID}" 2>/dev/null || true
    wait "${NODE_PID}" 2>/dev/null || true
  fi
}
trap cleanup_node EXIT INT TERM
/usr/bin/python3 "${ROOT}/scripts/mapping/export_cartographer_trajectory.py" \
  --pbstream "${PBSTREAM}" --trajectory-id "${TRAJECTORY_ID}" \
  --output "${OUT}/trajectory.json" --timeout 60
echo "Offline Cartographer result: ${OUT}"
