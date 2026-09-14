#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
RUN=""; OUT=""; RATE=1.0; RESOLUTION=0.05
usage() { echo "Usage: $0 --run-dir RUN [--output DIR] [--rate N] [--resolution M]"; }
while (($#)); do
  case "$1" in
    --run-dir) RUN="${2:?--run-dir needs a directory}"; shift 2 ;;
    --output) OUT="${2:?--output needs a directory}"; shift 2 ;;
    --rate) RATE="${2:?--rate needs a number}"; shift 2 ;;
    --resolution) RESOLUTION="${2:?--resolution needs metres}"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done
[[ -d "$RUN" ]] || { echo "run directory not found: $RUN" >&2; exit 2; }
[[ -n "$OUT" ]] || OUT="$RUN/cartographer_npz"
# The ROS compatibility replay has the same NPZ+encoder input boundary as the
# standalone runner.  Keep evaluation/GT artefacts from being selected here
# too; nav_msgs below is only the ROS message type for the encoder prior.
/usr/bin/python3 "$ROOT/scripts/evaluation/check_standalone_inputs.py" \
  --run_dir "$RUN"
mkdir -p "$OUT"
PBSTREAM="$OUT/map.pbstream"
set +u
source /opt/ros/humble/setup.bash
source "$ROOT/../install/setup.bash" 2>/dev/null || true
set -u
export PYTHONPATH="$ROOT/src:${PYTHONPATH:-}"
# Isolate this offline run from live ROS nodes and stale replay processes.
# Respect an explicitly supplied domain for advanced/debug use.
export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-$((20 + $$ % 180))}"
# Put the launch tree in its own process group. Killing only the `ros2 launch`
# parent leaves Cartographer and occupancy-grid children alive; subsequent
# runs then share `/write_state` and can serialize the wrong trajectory.
setsid ros2 launch flifltbot cartographer_npz_replay.launch.py \
  run_dir:="$RUN" write_state:="$PBSTREAM" rate:="$RATE" settle:=5.0 \
  >"$OUT/replay.log" 2>&1 &
LAUNCH_PID=$!
cleanup() {
  if kill -0 "$LAUNCH_PID" 2>/dev/null; then
    kill -INT -- "-$LAUNCH_PID" 2>/dev/null || true
    for _ in $(seq 1 20); do
      kill -0 "$LAUNCH_PID" 2>/dev/null || return
      sleep 0.5
    done
    kill -TERM -- "-$LAUNCH_PID" 2>/dev/null || true
    wait "$LAUNCH_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM
deadline=$((SECONDS + 900))
while [[ ! -s "$PBSTREAM" ]]; do
  if ! kill -0 "$LAUNCH_PID" 2>/dev/null; then
    echo "Replay exited before writing $PBSTREAM" >&2
    tail -80 "$OUT/replay.log" >&2 || true
    exit 1
  fi
  if (( SECONDS >= deadline )); then echo "Timeout waiting for pbstream" >&2; exit 1; fi
  sleep 2
done
# /write_state returns after serializing the state callback has been queued;
# give Cartographer time to finish flushing gzip before stopping its process.
sleep 10
cleanup
trap - EXIT INT TERM
ros2 run cartographer_ros cartographer_pbstream_to_ros_map \
  -pbstream_filename="$PBSTREAM" -map_filestem="$OUT/map" \
  -resolution="$RESOLUTION"
setsid ros2 run cartographer_ros cartographer_node \
  -configuration_directory="$ROOT/config/cartographer" \
  -configuration_basename=fliftbot_2d.lua -load_state_filename="$PBSTREAM" \
  -load_frozen_state=true -start_trajectory_with_default_topics=false \
  --ros-args -p use_sim_time:=false >"$OUT/trajectory_query_node.log" 2>&1 &
QUERY_PID=$!
cleanup_query() {
  if kill -0 "$QUERY_PID" 2>/dev/null; then
    kill -INT -- "-$QUERY_PID" 2>/dev/null || true
    for _ in $(seq 1 20); do
      kill -0 "$QUERY_PID" 2>/dev/null || return
      sleep 0.5
    done
    kill -TERM -- "-$QUERY_PID" 2>/dev/null || true
    wait "$QUERY_PID" 2>/dev/null || true
  fi
}
trap cleanup_query EXIT INT TERM
/usr/bin/python3 "$ROOT/scripts/mapping/export_cartographer_trajectory.py" \
  --pbstream "$PBSTREAM" --output "$OUT/trajectory.json" --timeout 60
cleanup_query
trap - EXIT INT TERM
echo "Direct NPZ Cartographer result: $OUT"
