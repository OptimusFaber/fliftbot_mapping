"""Start Cartographer and replay a horizon NPZ/encoder recording."""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    share = get_package_share_directory('flifltbot')
    cartographer_launch = os.path.join(
        share, 'launch', 'cartographer_mapping.launch.py'
    )
    run_dir = LaunchConfiguration('run_dir')
    write_state = LaunchConfiguration('write_state')
    return LaunchDescription([
        DeclareLaunchArgument(
            'run_dir',
            description='Recording with horizon/*.npz and encoder.jsonl',
        ),
        DeclareLaunchArgument(
            'write_state', default_value='',
            description='Optional output .pbstream path',
        ),
    DeclareLaunchArgument('rate', default_value='1.0'),
    DeclareLaunchArgument('settle', default_value='2.0'),
        DeclareLaunchArgument('lidar_x', default_value='1.29'),
        DeclareLaunchArgument('lidar_y', default_value='-0.32'),
        DeclareLaunchArgument('lidar_z', default_value='2.03'),
        DeclareLaunchArgument('use_sim_time', default_value='true'),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(cartographer_launch),
            launch_arguments={
                'use_sim_time': LaunchConfiguration('use_sim_time'),
                'scan_topic': '/scan',
                'odom_topic': '/odom',
            }.items(),
        ),
        Node(
            package='flifltbot', executable='npz_cartographer_replay',
            name='npz_cartographer_replay', output='screen',
            arguments=[
                '--run-dir', run_dir,
                '--scan-topic', '/scan', '--odom-topic', '/odom',
                '--time-mode', 'sim', '--rate', LaunchConfiguration('rate'),
                '--lidar-x', LaunchConfiguration('lidar_x'),
                '--lidar-y', LaunchConfiguration('lidar_y'),
                '--lidar-z', LaunchConfiguration('lidar_z'),
                '--settle', LaunchConfiguration('settle'),
                '--write-state', write_state,
            ],
            parameters=[
                {'use_sim_time': LaunchConfiguration('use_sim_time')}
            ],
        ),
    ])
