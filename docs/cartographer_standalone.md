# Standalone Cartographer

`fliftbot_cartographer_offline` feeds the core Cartographer `MapBuilder` directly
from a canonical recording. It does not start ROS, create topics, use TF, or
write a rosbag. The only required input is:

```text
RUN/
  horizon/*.npz
  encoder.jsonl
  lidar.config             # optional; position/orientation defaults to zero
```

Build and run:

```bash
./fliftbot_mapping/scripts/mapping/run_standalone_cartographer.sh \
  --run_dir /home/optimus/ros-projects/ros-data/recordings/real/20260901/182826_rplidar_room \
  --output /home/optimus/ros-projects/ros-data/results/real/20260901/cartographer_standalone \
  --configuration_dir /home/optimus/ros-projects/fliftbot_mapping/config/cartographer
```

For the RPLIDAR recordings, the tested offline profile enables the correlative
front-end and gives the encoder prior a moderate weight:

```bash
... --configuration_file fliftbot_2d_offline.lua
```

The output contains `map.pbstream`, `map.png`, `map.pgm`, `map.yaml`,
`trajectory.json`, and `run_stats.json`. NPZ `xy` frame is explicit when
`lidar.config.points_frame` is present. The `--points_frame sensor|tracking`
override is available; `auto` preserves the two existing formats (RPLIDAR
runtime root horizon files are already body/tracking-frame, while DePHAN M360
conversion files are sensor-local). By default the standalone runner uses the
physical LiDAR as Cartographer's tracking frame: sensor-local points remain
local, while encoder poses of the rotation axis are composed as
`T_world_lidar = T_world_rotation_axis * T_rotation_axis_lidar`. This accounts
for the LiDAR lever arm during turns and keeps range data in its natural sensor
frame. `--tracking_frame=rotation_axis` retains the older diagnostic
formulation, with points/origin transformed into the rotation-axis frame.
Already tracking-frame point clouds require that diagnostic mode. Point
timestamps are retained as non-positive offsets from the last point in each
scan; this is important for DePHAN scans with a measurable sweep time. For
legacy HTTP RPLIDAR files whose `t[:]` is one repeated frame timestamp, beam
times can be reconstructed with the verified `start/end` convention flag; the
safe default leaves them disabled because old files do not record that
timestamp semantic. See `cartographer_standalone_timing.md` for the formula,
switch, and diagnostics in `run_stats.json`.

Before launching, `run_standalone_cartographer.sh` runs
`scripts/evaluation/check_standalone_inputs.py`. The runner consumes only the
lidar arrays it needs (`xy`, `t`, `stamp`, `frame`) and encoder fields
(`enc_*`) from the recording. The guard rejects `nav_*`, `odom_*`, GT,
floorplan, and alignment fields/files. Evaluation tools may consume GT only
when explicitly invoked after the map has been built (for example, the
validator's optional `--gt` argument); those files are never read by the
standalone runner.

The encoder model uses the recording contract: validated real RPLIDAR HTTP
recordings use `enc_v` directly (`+1`), while legacy DePHAN/Pandar recordings
retain `WHEEL_SPEED_SCALE=-1`; wheelbase is 1.410 m. The sign can be overridden
with `encoder_speed_sign` in `lidar.config` or the CLI
`--encoder_speed_sign=1/-1`, and is written to `run_stats.json`. The result is
passed as soft Cartographer odometry. Low-speed
motion is suppressed only below 0.008 m/s, matching the main mapping pipeline.
The calibration flags `--wheelbase`, `--steering_offset_deg`, and
`--odometry_time_offset` are available for controlled A/B diagnostics. Use
`--use_odometry=false` to assess LiDAR-only behavior. `--max_scans N` is useful
for a quick smoke test, not for a final map.
