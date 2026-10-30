"""fliftbot_sim live localization without its /odom ground truth or TF tree."""
from pathlib import Path
import sys
import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

ROOT = Path(__file__).resolve().parents[1]
PYTHON = os.environ.get("FLIFTBOT_LIDAR_NAV_PYTHON", sys.executable)
sys.path.insert(0, str(ROOT / "scripts"))


def nodes(context):
    from map_bundle import validate_bundle
    map_dir = Path(LaunchConfiguration("map_dir").perform(context))
    validate_bundle(map_dir)
    frames = {"FLIFT_MAP_FRAME": "lidar_nav_map", "FLIFT_TRACKING_FRAME": "lidar_nav_base",
              "FLIFT_ODOM_FRAME": "lidar_nav_odom", "FLIFT_PUBLISHED_FRAME": "lidar_nav_odom"}
    return [
        ExecuteProcess(cmd=[PYTHON, str(ROOT / "scripts/sim_inputs.py"),
                            "--map-dir", str(map_dir), "--ros-args", "-p", "use_sim_time:=true"], output="screen"),
        Node(package="cartographer_ros", executable="cartographer_node", output="screen",
             parameters=[{"use_sim_time": True}],
             arguments=["-configuration_directory", str(ROOT / "config/cartographer"),
                        "-configuration_basename", "fliftbot_2d_pure_localization.lua",
                        "-start_trajectory_with_default_topics=false",
                        "-load_state_filename", str(map_dir / "map.pbstream")],
             remappings=[("scan", "/lidar_nav/scan"), ("odom", "/lidar_nav/wheel_odom")],
             additional_env=frames),
        ExecuteProcess(cmd=[PYTHON, str(ROOT / "scripts/initialize_localization.py"),
                            "--map-dir", str(map_dir), "--configuration-directory", str(ROOT / "config/cartographer"),
                            "--scan-topic", "/lidar_nav/scan", "--base-frame", "lidar_nav_base",
                            f"--initial-pose={LaunchConfiguration('initial_pose').perform(context)}",
                            "--ros-args", "-p", "use_sim_time:=true"], additional_env=frames, output="screen"),
        ExecuteProcess(cmd=[PYTHON, str(ROOT / "scripts/live_marker98_pose.py"),
                            "--marker-reference", str(map_dir / "origin_reference.json"),
                            "--map-frame", "lidar_nav_map", "--base-frame", "lidar_nav_base",
                            "--topic", "/localization/lidar_pose", "--no-send-to-navigation",
                            "--csv", LaunchConfiguration("pose_csv"),
                            "--ros-args", "-p", "use_sim_time:=true"], output="screen"),
    ]


def generate_launch_description():
    return LaunchDescription([DeclareLaunchArgument("map_dir"),
                              DeclareLaunchArgument("initial_pose", default_value=""),
                              DeclareLaunchArgument("pose_csv", default_value=""),
                              OpaqueFunction(function=nodes)])
