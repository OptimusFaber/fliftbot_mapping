"""Direct UDP LiDAR + hostname-selected telemetry + Cartographer localization."""

from pathlib import Path
import sys
import os
from datetime import datetime

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


ROOT = Path(__file__).resolve().parents[1]
PYTHON = os.environ.get("FLIFTBOT_LIDAR_NAV_PYTHON", sys.executable)
SCRIPTS = ROOT / "scripts"
CONFIG = ROOT / "config" / "cartographer"
sys.path.insert(0, str(SCRIPTS))
from robot_identity import mc_topic, robot_namespace
from lidar_profiles import load_profile


def _nodes(context):
    sensor = LaunchConfiguration("sensor").perform(context)
    if sensor not in ("pandar", "dephan"):
        raise RuntimeError("sensor must be exactly 'pandar' or 'dephan'")
    map_dir = LaunchConfiguration("map_dir").perform(context)
    profile = load_profile(sensor, Path(map_dir) if map_dir else None)
    profile_dir = profile["map_dir"]
    map_arg = LaunchConfiguration("map_pbstream").perform(context)
    if map_arg and not map_dir:
        profile = load_profile(sensor, Path(map_arg).resolve().parent)
        profile_dir = profile["map_dir"]
    ref_arg = LaunchConfiguration("marker_reference").perform(context)
    origin_ref_arg = LaunchConfiguration("origin_reference").perform(context)
    if ref_arg and origin_ref_arg:
        raise RuntimeError("set only one of marker_reference or origin_reference")
    if (ref_arg or origin_ref_arg) and not map_arg:
        raise RuntimeError("a custom reference requires map_pbstream")
    map_pbstream = map_arg or str(profile_dir / "map.pbstream")
    if not Path(map_pbstream).is_file():
        raise RuntimeError(f"map.pbstream not found: {map_pbstream}")
    if map_arg and not (ref_arg or origin_ref_arg):
        sibling_origin = Path(map_arg).resolve().parent / "origin_reference.json"
        if not sibling_origin.is_file():
            raise RuntimeError(
                "custom map requires origin_reference/marker_reference, or "
                f"a sibling {sibling_origin.name}"
            )
        origin_ref_arg = str(sibling_origin)
    pose_reference = origin_ref_arg or ref_arg or str(
        profile["reference"])
    topic_arg = LaunchConfiguration("mc_topic").perform(context)
    resolved_mc_topic = topic_arg or mc_topic()
    csv_arg = LaunchConfiguration("marker_pose_csv").perform(context)
    if csv_arg:
        pose_csv = csv_arg
    else:
        log_dir = ROOT / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        pose_csv = str(log_dir / f"{resolved_mc_topic.strip('/').split('/')[0]}_{sensor}_{stamp}.csv")
    print(f"robot telemetry: {resolved_mc_topic}")
    print(f"position log: {pose_csv}")
    sensor_x_arg = LaunchConfiguration("sensor_x").perform(context)
    sensor_y_arg = LaunchConfiguration("sensor_y").perform(context)
    sensor_yaw_arg = LaunchConfiguration("sensor_yaw_deg").perform(context)
    sensor_x = sensor_x_arg or str(profile["mount"][0])
    sensor_y = sensor_y_arg or str(profile["mount"][1])
    sensor_yaw = sensor_yaw_arg or str(profile["sensor_yaw_deg"])
    odometry = profile.get("odometry", {})
    speed_sign_arg = LaunchConfiguration("speed_sign").perform(context)
    steering_zero_arg = LaunchConfiguration("steering_zero_deg").perform(context)
    wheelbase_arg = LaunchConfiguration("wheelbase").perform(context)
    odometry_model = LaunchConfiguration("odom_model").perform(context) or odometry.get("model", "drive_wheel")
    steering_sign = str(odometry.get("steering_sign", 1.0))
    speed_sign = speed_sign_arg or str(odometry.get("speed_sign", -1.0))
    steering_zero = steering_zero_arg or str(odometry.get("steering_zero_deg", 0.0))
    wheelbase = wheelbase_arg or str(odometry.get("wheelbase_m", 1.410))
    send_to_navigation = LaunchConfiguration("send_to_navigation").perform(context).lower()
    if send_to_navigation not in ("1", "true", "yes", "on", "0", "false", "no", "off"):
        raise RuntimeError("send_to_navigation must be true or false")
    print(f"LiDAR calibration: x={sensor_x}, y={sensor_y}, yaw={sensor_yaw} deg")
    print(f"odometry calibration: speed_sign={speed_sign}, "
          f"steering_zero={steering_zero} deg, wheelbase={wheelbase} m, "
          f"model={odometry_model}")
    pose_command = [
        PYTHON, str(SCRIPTS / "live_marker98_pose.py"),
        "--marker-reference", pose_reference,
        "--csv", pose_csv,
        "--rate", LaunchConfiguration("pose_rate_hz"),
        "--sensor-send-rate", LaunchConfiguration("navigation_send_rate_hz"),
        "--sensor-socket", LaunchConfiguration("continuous_nav_sensor_socket"),
        "--max-sensor-age", LaunchConfiguration("max_sensor_age"),
        "--pose-offset-x-m", LaunchConfiguration("pose_offset_x_m"),
        "--pose-offset-y-m", LaunchConfiguration("pose_offset_y_m"),
    ]
    if send_to_navigation in ("0", "false", "no", "off"):
        pose_command.append("--no-send-to-navigation")
    print(f"ContinuousNav output: {send_to_navigation}, "
          f"rate={LaunchConfiguration('navigation_send_rate_hz').perform(context)} Hz")

    nodes = [
        ExecuteProcess(
            cmd=[
                PYTHON, str(SCRIPTS / "direct_udp_scan_publisher.py"),
                "--sensor", sensor,
                "--scan-topic", LaunchConfiguration("scan_topic"),
                "--scan-rate-hz", str(profile.get("scan_rate_hz", 20.0)),
                "--range-max", str(profile.get("max_range_m", 30.0)),
                "--bind", LaunchConfiguration("udp_bind"),
                "--sensor-yaw-deg", sensor_yaw,
                "--sensor-x", sensor_x,
                "--sensor-y", sensor_y,
            ],
            output="screen",
        ),
        ExecuteProcess(
            cmd=[
                PYTHON, str(SCRIPTS / "mc_telemetry_odom_adapter.py"),
                "--topic", resolved_mc_topic,
                "--odom-topic", LaunchConfiguration("odom_topic"),
                "--speed-sign", speed_sign,
                "--steering-zero-deg", steering_zero,
                "--wheelbase", wheelbase,
                "--model", odometry_model,
                "--steering-sign", steering_sign,
            ],
            output="screen",
        ),
        ExecuteProcess(
            cmd=pose_command,
            output="screen",
        ),
        Node(
            package="cartographer_ros", executable="cartographer_node",
            name="cartographer_node", output="screen",
            parameters=[{"use_sim_time": False}],
            additional_env={
                "FLIFT_PUBLISH_TO_TF": LaunchConfiguration("publish_tf"),
                "FLIFT_PUBLISH_TRACKED_POSE": LaunchConfiguration("publish_tracked_pose"),
                "FLIFT_POSE_PUBLISH_PERIOD_SEC": LaunchConfiguration("pose_publish_period_sec"),
                "FLIFT_TRAJECTORY_PUBLISH_PERIOD_SEC": LaunchConfiguration("trajectory_publish_period_sec"),
                "FLIFT_SUBMAP_PUBLISH_PERIOD_SEC": LaunchConfiguration("submap_publish_period_sec"),
                "FLIFT_RANGEFINDER_SAMPLING_RATIO": LaunchConfiguration("rangefinder_sampling_ratio"),
                "FLIFT_GLOBAL_SAMPLING_RATIO": LaunchConfiguration("global_sampling_ratio"),
                "FLIFT_OPTIMIZE_EVERY_N_NODES": LaunchConfiguration("optimize_every_n_nodes"),
                "FLIFT_MOTION_FILTER_MAX_TIME_SEC": LaunchConfiguration("motion_filter_max_time_sec"),
                "FLIFT_MOTION_FILTER_MAX_DISTANCE_M": LaunchConfiguration("motion_filter_max_distance_m"),
                "FLIFT_MOTION_FILTER_MAX_ANGLE_DEG": LaunchConfiguration("motion_filter_max_angle_deg"),
                "FLIFT_USE_ONLINE_CORRELATIVE_SCAN_MATCHING": LaunchConfiguration("use_online_correlative_scan_matching"),
                "FLIFT_CORRELATIVE_LINEAR_WINDOW_M": LaunchConfiguration("correlative_linear_window_m"),
                "FLIFT_CORRELATIVE_ANGULAR_WINDOW_DEG": LaunchConfiguration("correlative_angular_window_deg"),
                "FLIFT_LOG_MATCHES": LaunchConfiguration("log_matches"),
                "FLIFT_LOG_RESIDUAL_HISTOGRAMS": LaunchConfiguration("log_residual_histograms"),
            },
            arguments=[
                "-configuration_directory", LaunchConfiguration("configuration_directory"),
                "-configuration_basename", LaunchConfiguration("configuration_basename"),
                "-load_state_filename", map_pbstream,
                "-start_trajectory_with_default_topics=false",
            ],
            remappings=[
                ("scan", LaunchConfiguration("scan_topic")),
                ("odom", LaunchConfiguration("odom_topic")),
            ],
        ),
    ]
    nodes.append(ExecuteProcess(
        cmd=[PYTHON, str(SCRIPTS / "initialize_localization.py"),
             "--map-dir", str(Path(map_pbstream).parent),
             "--configuration-directory", LaunchConfiguration("configuration_directory"),
             "--scan-topic", LaunchConfiguration("scan_topic"),
             f"--initial-pose={LaunchConfiguration('initial_pose').perform(context)}"], output="screen"))
    if LaunchConfiguration("publish_occupancy_grid").perform(context).lower() in (
            "1", "true", "yes", "on"):
        nodes.append(Node(
            package="cartographer_ros", executable="cartographer_occupancy_grid_node",
            name="cartographer_occupancy_grid_node", output="screen",
            parameters=[{"use_sim_time": False}, {"resolution": 0.05}],
        ))
    return nodes


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument("map_dir", default_value="",
                              description="fliftbot_map_build calibrated deploy bundle"),
        DeclareLaunchArgument("initial_pose", default_value="",
                              description="optional X_MAP,Y_MAP,YAW_RAD acquisition seed"),
        DeclareLaunchArgument("sensor", default_value="dephan",
                              description="LiDAR profile: pandar or dephan"),
        DeclareLaunchArgument("map_pbstream", default_value="",
                              description="Optional override; pair with an origin/marker reference"),
        DeclareLaunchArgument("marker_reference", default_value="",
                              description="Optional per-map marker98 reference override"),
        DeclareLaunchArgument(
            "origin_reference", default_value="",
            description="Optional arbitrary map-frame origin; alternative to marker_reference",
        ),
        DeclareLaunchArgument("mc_topic", default_value="",
                              description="empty: derive /fliftbotNNN/mc_telemetry from hostname"),
        DeclareLaunchArgument("scan_topic", default_value="/scan"),
        DeclareLaunchArgument("odom_topic", default_value="/wheel_odom"),
        DeclareLaunchArgument("udp_bind", default_value="0.0.0.0"),
        DeclareLaunchArgument("sensor_x", default_value="",
                              description="empty selects the sensor profile default"),
        DeclareLaunchArgument("sensor_y", default_value="",
                              description="empty selects the sensor profile default"),
        DeclareLaunchArgument("sensor_yaw_deg", default_value="",
                              description="empty selects the sensor profile default"),
        DeclareLaunchArgument("speed_sign", default_value="",
                              description="empty selects the map profile default"),
        DeclareLaunchArgument("steering_zero_deg", default_value="",
                              description="empty selects the map profile default"),
        DeclareLaunchArgument("wheelbase", default_value="",
                              description="empty selects the map profile default"),
        DeclareLaunchArgument(
            "odom_model", default_value="",
            description="empty uses bundle/profile; drive_wheel, legacy_sin or bicycle",
        ),
        DeclareLaunchArgument("marker_pose_csv", default_value=""),
        DeclareLaunchArgument(
            "pose_offset_x_m", default_value="0.0",
            description="experimental output-point offset in base_link X (forward), metres",
        ),
        DeclareLaunchArgument(
            "pose_offset_y_m", default_value="0.0",
            description="experimental output-point offset in base_link Y (left), metres",
        ),
        DeclareLaunchArgument("pose_rate_hz", default_value="10.0"),
        DeclareLaunchArgument(
            "publish_tf", default_value="true",
            description="publish map->odom TF; required by pose export and ContinuousNav",
        ),
        DeclareLaunchArgument(
            "publish_tracked_pose", default_value="false",
            description="publish Cartographer tracked_pose topic",
        ),
        DeclareLaunchArgument("pose_publish_period_sec", default_value="0.05"),
        DeclareLaunchArgument(
            "trajectory_publish_period_sec", default_value="3600.0",
            description="use e.g. 0.5 for RViz; one-hour default is effectively off",
        ),
        DeclareLaunchArgument(
            "submap_publish_period_sec", default_value="3600.0",
            description="use e.g. 2.0 for RViz; one-hour default is effectively off",
        ),
        DeclareLaunchArgument("rangefinder_sampling_ratio", default_value="0.5"),
        DeclareLaunchArgument(
            "global_sampling_ratio", default_value="0.2",
            description="fixed for this Cartographer trajectory; not dynamic at runtime",
        ),
        DeclareLaunchArgument("optimize_every_n_nodes", default_value="10"),
        DeclareLaunchArgument("motion_filter_max_time_sec", default_value="0.5"),
        DeclareLaunchArgument("motion_filter_max_distance_m", default_value="0.2"),
        DeclareLaunchArgument("motion_filter_max_angle_deg", default_value="1.0"),
        DeclareLaunchArgument("use_online_correlative_scan_matching", default_value="true"),
        DeclareLaunchArgument("correlative_linear_window_m", default_value="0.15"),
        DeclareLaunchArgument("correlative_angular_window_deg", default_value="10.0"),
        DeclareLaunchArgument("log_matches", default_value="false"),
        DeclareLaunchArgument("log_residual_histograms", default_value="false"),
        DeclareLaunchArgument(
            "send_to_navigation", default_value="true",
            description="send top_navigation measurements to ContinuousNav",
        ),
        DeclareLaunchArgument(
            "navigation_send_rate_hz", default_value="10.0",
            description="fixed maximum top_navigation transmission rate",
        ),
        DeclareLaunchArgument(
            "max_sensor_age", default_value="0.5",
            description="drop delayed poses before sending them to ContinuousNav",
        ),
        DeclareLaunchArgument(
            "continuous_nav_sensor_socket",
            default_value="ipc:///tmp/fliftbot_continuous_nav_sensor.ipc",
            description="SensorSender-compatible ContinuousNav ZMQ endpoint",
        ),
        DeclareLaunchArgument("publish_occupancy_grid", default_value="false",
                              description="enable only when a live /map raster is needed"),
        DeclareLaunchArgument("configuration_directory", default_value=str(CONFIG)),
        DeclareLaunchArgument("configuration_basename",
                              default_value="fliftbot_2d_pure_localization.lua"),
        OpaqueFunction(function=_nodes),
    ])
