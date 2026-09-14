# Standalone Cartographer: LiDAR timing and frames

## What the runner receives

The standalone runner consumes `horizon/*.npz`. `xy` is either sensor-local
or already in the rotation-axis/body frame, as declared by
`lidar.config.points_frame` (legacy RPLIDAR root files default to `tracking`).
Cartographer's `TimedPointCloudData` uses the same contract as
`cartographer_ros`: `ranges` are endpoint coordinates in the tracking frame,
and `origin` is the sensor origin in that frame. Both are required for correct
ray tracing; a body-frame endpoint must not receive the mount translation a
second time.

## Real per-beam timestamps

Pandar/Hesai recordings contain distinct, monotonic `t[i]` values. They are
used unchanged and the scan timestamp is the latest point time, so offsets are
`dt[i] = t[i] - t_scan <= 0`. The runner never synthesizes timing for such
scans.

The HTTP RPLIDAR recordings contain one repeated value in `t[:]`. This is a
timestamp for the returned revolution, not evidence that all beams were
measured simultaneously. When enabled for a verified timestamp contract, the
runner detects this collapsed array and reconstructs deterministic beam times
from firing order and azimuth:

```text
T       = robust median of adjacent frame timestamp intervals
theta_i = unwrapped sensor bearing of beam i
phase_i = clamp(direction * (theta_i - theta_0) / (2*pi), 0, 1)
t_i     = t_end - T + phase_i*T
dt_i    = t_i - t_end
```

`direction` is inferred from ordered bearings (the real samples are
clockwise). The per-frame period is bounded by both neighbouring frame
intervals to prevent overlapping end-stamped sweeps when HTTP polling jitters.
Small azimuth jitter is monotonically clamped because Cartographer's
range-data collator requires non-decreasing point times.

This reconstruction is physically justified only for a complete ordered
revolution with an estimable period. It is not applied to Pandar or to a
non-collapsed RPLIDAR `t` array. The HTTP recorder's timestamp is treated as
`end` by `auto`, because the endpoint returns after collecting the revolution.
A driver that timestamps the first firing can be tested with
`--rplidar_timestamp_convention=start`; `--synthesize_rplidar_times=false`
disables reconstruction (and is the safe default for legacy files without
timestamp metadata) for an A/B run. The selected convention, period,
direction, and number of synthesized scans are written to `run_stats.json`.

## Reproduction / A-B commands

```bash
BIN=fliftbot_mapping/standalone_cartographer/build/fliftbot_cartographer_offline
CFG=fliftbot_mapping/config/cartographer
RUN=ros-data/recordings/real/20260901/182826_rplidar_room

$BIN --run_dir "$RUN" --output /tmp/rplidar_end \
  --configuration_dir "$CFG" --configuration_file fliftbot_2d_offline.lua \
  --max_range=30 --rplidar_timestamp_convention=end
$BIN --run_dir "$RUN" --output /tmp/rplidar_no_timing \
  --configuration_dir "$CFG" --configuration_file fliftbot_2d_offline.lua \
  --max_range=30 --synthesize_rplidar_times=false
$BIN --run_dir "$RUN" --output /tmp/rplidar_start \
  --configuration_dir "$CFG" --configuration_file fliftbot_2d_offline.lua \
  --max_range=30 --rplidar_timestamp_convention=start
```

Use `validate_standalone_map.py` after a run for map extents and structure. Do
not use GT/navigation files as runner input.
