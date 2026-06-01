"""
mmwave_drone_detector — ROS2 Humble launch file

Usage:
  ros2 launch mmwave_drone_detector detector.launch.py
  ros2 launch mmwave_drone_detector detector.launch.py cli_port:=/dev/ttyUSB0 data_port:=/dev/ttyUSB1
  ros2 launch mmwave_drone_detector detector.launch.py min_confidence:=0.4 danger_radius:=1.5
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():

    # ── Launch arguments (overridable on command line) ────────────────────────
    args = [
        DeclareLaunchArgument('cli_port',            default_value='/dev/ttyUSB0'),
        DeclareLaunchArgument('data_port',           default_value='/dev/ttyUSB1'),
        DeclareLaunchArgument('sensor_frame_id',     default_value='radar'),
        DeclareLaunchArgument('min_snr_db',          default_value='6.0'),
        DeclareLaunchArgument('min_range_m',         default_value='0.3'),
        DeclareLaunchArgument('max_range_m',         default_value='8.0'),
        DeclareLaunchArgument('max_azimuth_deg',     default_value='60.0'),
        DeclareLaunchArgument('min_height_m',        default_value='-0.3'),
        DeclareLaunchArgument('max_height_m',        default_value='5.0'),
        DeclareLaunchArgument('min_doppler_ms',      default_value='0.15'),
        DeclareLaunchArgument('min_cluster_points',  default_value='2'),
        DeclareLaunchArgument('max_cluster_dist_m',  default_value='0.8'),
        DeclareLaunchArgument('min_confidence',      default_value='0.3'),
        DeclareLaunchArgument('velocity_alpha',      default_value='0.4'),
        DeclareLaunchArgument('position_alpha',      default_value='0.6'),
    ]

    # ── Node ──────────────────────────────────────────────────────────────────
    node = Node(
        package    = 'mmwave_drone_detector',
        executable = 'drone_detector',
        name       = 'mmwave_drone_detector',
        output     = 'screen',
        parameters = [
            PathJoinSubstitution([
                FindPackageShare('mmwave_drone_detector'),
                'config', 'params.yaml'
            ]),
            # Command-line overrides
            {
                'cli_port':           LaunchConfiguration('cli_port'),
                'data_port':          LaunchConfiguration('data_port'),
                'sensor_frame_id':    LaunchConfiguration('sensor_frame_id'),
                'min_snr_db':         LaunchConfiguration('min_snr_db'),
                'min_range_m':        LaunchConfiguration('min_range_m'),
                'max_range_m':        LaunchConfiguration('max_range_m'),
                'max_azimuth_deg':    LaunchConfiguration('max_azimuth_deg'),
                'min_height_m':       LaunchConfiguration('min_height_m'),
                'max_height_m':       LaunchConfiguration('max_height_m'),
                'min_doppler_ms':     LaunchConfiguration('min_doppler_ms'),
                'min_cluster_points': LaunchConfiguration('min_cluster_points'),
                'max_cluster_dist_m': LaunchConfiguration('max_cluster_dist_m'),
                'min_confidence':     LaunchConfiguration('min_confidence'),
                'velocity_alpha':     LaunchConfiguration('velocity_alpha'),
                'position_alpha':     LaunchConfiguration('position_alpha'),
            }
        ],
    )

    return LaunchDescription(args + [node])
