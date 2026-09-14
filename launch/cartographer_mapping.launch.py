"""Run the Cartographer 2-D baseline for a live or replayed Fliftbot stream.

The sensor/replay side must publish the topics named by ``scan_topic`` and
``odom_topic`` and provide TF from ``base_link`` to the LiDAR frame.  The
recording-to-ROS conversion is intentionally outside this package.
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution

from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    """Create Cartographer and occupancy-grid nodes with Fliftbot defaults."""
    package_share = FindPackageShare('flifltbot')
    config_dir = PathJoinSubstitution([
        package_share, 'config', 'cartographer',
    ])

    use_sim_time = DeclareLaunchArgument('use_sim_time', default_value='false')
    configuration_directory = DeclareLaunchArgument(
        'configuration_directory', default_value=config_dir
    )
    configuration_basename = DeclareLaunchArgument(
        'configuration_basename', default_value='fliftbot_2d.lua'
    )
    scan_topic = DeclareLaunchArgument(
        'scan_topic', default_value='/scan'
    )
    odom_topic = DeclareLaunchArgument(
        'odom_topic', default_value='/wheel_odom'
    )

    cartographer = Node(
        package='cartographer_ros',
        executable='cartographer_node',
        name='cartographer_node',
        output='screen',
        parameters=[{'use_sim_time': LaunchConfiguration('use_sim_time')}],
        arguments=[
            '-configuration_directory',
            LaunchConfiguration('configuration_directory'),
            '-configuration_basename',
            LaunchConfiguration('configuration_basename'),
        ],
        remappings=[
            ('scan', LaunchConfiguration('scan_topic')),
            ('odom', LaunchConfiguration('odom_topic')),
        ],
    )

    occupancy_grid = Node(
        package='cartographer_ros',
        executable='cartographer_occupancy_grid_node',
        name='cartographer_occupancy_grid_node',
        output='screen',
        parameters=[
            {'use_sim_time': LaunchConfiguration('use_sim_time')},
            {'resolution': 0.05},
        ],
    )

    return LaunchDescription(
        [
            use_sim_time,
            configuration_directory,
            configuration_basename,
            scan_topic,
            odom_topic,
            cartographer,
            occupancy_grid,
        ]
    )
