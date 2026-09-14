# External SLAM baselines

The repository now exposes `odom_fixed_v2` in the offline mapper. It uses the
wheel bicycle integration only as a soft prior, scan-to-submap ICP with
anisotropic information, and the existing pose graph/loop closure frontend.

Cartographer and slam_toolbox are provided as ROS 2 configuration baselines in
`config/cartographer` and `config/slam_toolbox`. They require a ROS bag or a
replay/conversion node publishing `sensor_msgs/LaserScan` on `/scan`,
`nav_msgs/Odometry` on `/wheel_odom`, and TF from `base_link` to the LiDAR
frame; the offline NPZ recording is not silently converted into a fake SLAM
result.

## Cartographer 2D

The package installs `launch/cartographer_mapping.launch.py`. Build and source
the workspace, then start it while the replay/driver is publishing the topics:

```bash
source /opt/ros/humble/setup.bash
source install/setup.bash
ros2 launch flifltbot cartographer_mapping.launch.py \
  use_sim_time:=false scan_topic:=/scan odom_topic:=/wheel_odom
```

For a simulation bag use `use_sim_time:=true` and run `ros2 bag play BAG --clock`
in another terminal. The launch starts both `cartographer_node` and
`cartographer_occupancy_grid_node`; the default configuration is
`config/cartographer/fliftbot_2d.lua`. Save the resulting `/map` with
`ros2 run nav2_map_server map_saver_cli -f OUTPUT_PREFIX`.

The launch does not create TF or convert NPZ files. Before starting it, verify
`ros2 topic info /scan`, `ros2 topic info /wheel_odom`, and
`ros2 run tf2_ros tf2_echo base_link lidar_link`. Change the topic names with
`scan_topic:=...` and `odom_topic:=...` when replaying a bag with different
names.

Build the local baseline:

```bash
./fliftbot_mapping/scripts/mapping/map_build.sh \
  --run-dir ./ros-data/recordings/real/20260831/nav_video_lidar_run_135703 \
  --method odom_fixed_v2 --source real --cameras up
```

Cartographer needs `cartographer_ros` and a replay/conversion node. slam_toolbox
needs `slam_toolbox` and a replay/conversion node. Both use wheel odometry as a
prior and perform global loop-closure optimization; neither can resolve an
infinitely repetitive corridor without a return landmark or another sensor.
