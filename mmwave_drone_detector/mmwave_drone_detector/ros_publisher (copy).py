#!/usr/bin/env python3
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
"""
mmwave_drone_detector.ros_publisher
=====================================
ROS2 Humble node that reads radar frames, runs drone detection pipeline,
and publishes results on standard ROS2 topics.

Published topics:
  /radar/point_cloud       sensor_msgs/PointCloud2   — filtered radar points
  /radar/drone_detections  DroneDetection[]           — all detected drones
  /radar/drone_pose        geometry_msgs/PoseStamped  — closest drone pose
  /radar/drone_velocity    geometry_msgs/TwistStamped — closest drone velocity

Parameters: see config/params.yaml
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy

import struct
import math
import time
import os
import numpy as np

from std_msgs.msg import Header
from geometry_msgs.msg import (PoseStamped, Point, Quaternion,
                                TwistStamped, Vector3)
from sensor_msgs.msg import PointCloud2, PointField

from mmwave_drone_detector.radar_driver import RadarDriver
from mmwave_drone_detector.drone_detector import DroneDetector


# ── PointCloud2 helpers ───────────────────────────────────────────────────────

def make_pointcloud2(points, frame_id: str, stamp) -> PointCloud2:
    """Convert list of (x,y,z,doppler,snr) tuples to PointCloud2 msg."""
    msg = PointCloud2()
    msg.header.frame_id = frame_id
    msg.header.stamp    = stamp

    fields = [
        PointField(name='x',       offset=0,  datatype=PointField.FLOAT32, count=1),
        PointField(name='y',       offset=4,  datatype=PointField.FLOAT32, count=1),
        PointField(name='z',       offset=8,  datatype=PointField.FLOAT32, count=1),
        PointField(name='doppler', offset=12, datatype=PointField.FLOAT32, count=1),
        PointField(name='snr',     offset=16, datatype=PointField.FLOAT32, count=1),
    ]
    msg.fields    = fields
    msg.is_bigendian = False
    msg.point_step   = 20   # 5 × float32
    msg.row_step     = msg.point_step * len(points)
    msg.height       = 1
    msg.width        = len(points)
    msg.is_dense     = True

    data = bytearray()
    for p in points:
        data += struct.pack('<fffff', p.x, p.y, p.z, p.doppler, p.snr_db)
    msg.data = bytes(data)
    return msg


def yaw_to_quaternion(yaw_rad: float) -> Quaternion:
    """Convert yaw angle to quaternion (pointing toward detected drone)."""
    q = Quaternion()
    q.w = math.cos(yaw_rad / 2)
    q.z = math.sin(yaw_rad / 2)
    q.x = 0.0
    q.y = 0.0
    return q


# ── ROS2 Node ─────────────────────────────────────────────────────────────────

class DroneDetectorNode(Node):

    def __init__(self):
        super().__init__('mmwave_drone_detector')

        # ── Declare parameters ────────────────────────────────────────────────
        self.declare_parameter('cli_port',   '/dev/ttyUSB0')
        self.declare_parameter('data_port',  '/dev/ttyUSB1')
        self.declare_parameter('cfg_path',   '')
        self.declare_parameter('sensor_frame_id', 'radar')

        self.declare_parameter('min_snr_db',         6.0)
        self.declare_parameter('min_range_m',         0.3)
        self.declare_parameter('max_range_m',         8.0)
        self.declare_parameter('max_azimuth_deg',    60.0)
        self.declare_parameter('min_height_m',       -0.3)
        self.declare_parameter('max_height_m',        5.0)
        self.declare_parameter('min_doppler_ms',      0.15)
        self.declare_parameter('min_cluster_points',  2)
        self.declare_parameter('max_cluster_dist_m',  0.8)
        self.declare_parameter('min_confidence',      0.3)
        self.declare_parameter('velocity_alpha',      0.4)
        self.declare_parameter('position_alpha',      0.6)

        self.declare_parameter('topic_pointcloud',  '/radar/point_cloud')
        self.declare_parameter('topic_detections',  '/radar/drone_detections')
        self.declare_parameter('topic_pose',        '/radar/drone_pose')
        self.declare_parameter('topic_velocity',    '/radar/drone_velocity')

        # ── Read parameters ───────────────────────────────────────────────────
        cli_port   = self.get_parameter('cli_port').value
        data_port  = self.get_parameter('data_port').value
        cfg_path   = self.get_parameter('cfg_path').value
        self.frame_id = self.get_parameter('sensor_frame_id').value

        # Find default config if not specified
        if not cfg_path:
            pkg_share = os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                '..', 'config', 'drone_detect.cfg')
            cfg_path = os.path.normpath(pkg_share)
            if not os.path.exists(cfg_path):
                # Try ROS2 share directory
                try:
                    from ament_index_python.packages import get_package_share_directory
                    pkg = get_package_share_directory('mmwave_drone_detector')
                    cfg_path = os.path.join(pkg, 'config', 'drone_detect.cfg')
                except Exception:
                    cfg_path = ''

        # ── Init radar driver ─────────────────────────────────────────────────
        self.driver = RadarDriver(
            cli_port=cli_port,
            data_port=data_port,
            cfg_path=cfg_path,
        )

        # ── Init detector ─────────────────────────────────────────────────────
        self.detector = DroneDetector(
            min_snr_db        = self.get_parameter('min_snr_db').value,
            min_range_m       = self.get_parameter('min_range_m').value,
            max_range_m       = self.get_parameter('max_range_m').value,
            max_azimuth_deg   = self.get_parameter('max_azimuth_deg').value,
            min_height_m      = self.get_parameter('min_height_m').value,
            max_height_m      = self.get_parameter('max_height_m').value,
            min_doppler_ms    = self.get_parameter('min_doppler_ms').value,
            min_cluster_points= self.get_parameter('min_cluster_points').value,
            max_cluster_dist_m= self.get_parameter('max_cluster_dist_m').value,
            min_confidence    = self.get_parameter('min_confidence').value,
            velocity_alpha    = self.get_parameter('velocity_alpha').value,
            position_alpha    = self.get_parameter('position_alpha').value,
        )

        # ── QoS ───────────────────────────────────────────────────────────────
        sensor_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1,
        )
        reliable_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=10,
        )

        # ── Publishers ────────────────────────────────────────────────────────
        self.pub_cloud = self.create_publisher(
            PointCloud2,
            self.get_parameter('topic_pointcloud').value,
            sensor_qos)

        self.pub_pose = self.create_publisher(
            PoseStamped,
            self.get_parameter('topic_pose').value,
            reliable_qos)

        self.pub_vel = self.create_publisher(
            TwistStamped,
            self.get_parameter('topic_velocity').value,
            reliable_qos)

        # DroneDetection array published as a string-encoded topic
        # (avoids needing custom msg build for quick testing)
        # A proper custom msg publisher is set up below if msgs compiled
        try:
            from mmwave_drone_detector.msg import DroneDetection
            self._has_custom_msg = True
            from rclpy.impl.rcutils_logger import RcutilsLogger
            # Use Array wrapper via std_msgs Float32MultiArray fallback?
            # Actually just publish individual DroneDetection msgs
            self.pub_detection = self.create_publisher(
                DroneDetection,
                self.get_parameter('topic_detections').value,
                reliable_qos)
            self.DroneDetectionMsg = DroneDetection
        except ImportError:
            self._has_custom_msg = False
            self.get_logger().warn(
                'DroneDetection custom msg not found. '
                'Only PointCloud2, PoseStamped, TwistStamped will be published. '
                'Run colcon build to generate custom messages.')

        # ── Start radar ───────────────────────────────────────────────────────
        self.driver.start()
        self.get_logger().info(
            f'Radar driver started: {cli_port} / {data_port}')

        # ── Timer to poll radar at ~20Hz ──────────────────────────────────────
        self.create_timer(0.05, self._timer_callback)

        # Stats
        self._frame_count     = 0
        self._detection_count = 0
        self._last_log_time   = time.time()

    def _timer_callback(self):
        frame = self.driver.get_frame(timeout=0.04)
        if frame is None:
            return

        self._frame_count += 1
        stamp = self.get_clock().now().to_msg()

        # Run detection pipeline
        detections = self.detector.process(frame)
        filtered   = self.detector.filtered_points

        # ── Publish point cloud ───────────────────────────────────────────────
        if filtered:
            pc_msg = make_pointcloud2(filtered, self.frame_id, stamp)
            self.pub_cloud.publish(pc_msg)

        # ── Publish detections ────────────────────────────────────────────────
        if detections:
            self._detection_count += len(detections)

            # Find closest drone
            closest = min(detections, key=lambda d: d.range)

            # PoseStamped — closest drone position + orientation toward it
            pose_msg = PoseStamped()
            pose_msg.header.stamp    = stamp
            pose_msg.header.frame_id = self.frame_id
            pose_msg.pose.position.x = closest.position[0]
            pose_msg.pose.position.y = closest.position[1]
            pose_msg.pose.position.z = closest.position[2]
            yaw = math.atan2(closest.position[0], closest.position[1])
            pose_msg.pose.orientation = yaw_to_quaternion(yaw)
            self.pub_pose.publish(pose_msg)

            # TwistStamped — closest drone velocity
            twist_msg = TwistStamped()
            twist_msg.header.stamp    = stamp
            twist_msg.header.frame_id = self.frame_id
            twist_msg.twist.linear.x  = closest.velocity[0]
            twist_msg.twist.linear.y  = closest.velocity[1]
            twist_msg.twist.linear.z  = closest.velocity[2]
            self.pub_vel.publish(twist_msg)

            # Custom DroneDetection msg for each drone
            if self._has_custom_msg:
                for det in detections:
                    dm = self.DroneDetectionMsg()
                    dm.header.stamp    = stamp
                    dm.header.frame_id = self.frame_id
                    dm.drone_id        = det.drone_id
                    dm.position.x      = det.position[0]
                    dm.position.y      = det.position[1]
                    dm.position.z      = det.position[2]
                    dm.velocity.x      = det.velocity[0]
                    dm.velocity.y      = det.velocity[1]
                    dm.velocity.z      = det.velocity[2]
                    dm.speed           = det.speed
                    dm.range           = det.range
                    dm.azimuth_deg     = det.azimuth_deg
                    dm.elevation_deg   = det.elevation_deg
                    dm.confidence      = det.confidence
                    dm.point_count     = det.point_count
                    dm.snr_mean        = det.snr_mean
                    dm.doppler_mean    = det.doppler_mean
                    dm.is_approaching  = det.is_approaching
                    self.pub_detection.publish(dm)

        # ── Log summary every 5 seconds ───────────────────────────────────────
        now = time.time()
        if now - self._last_log_time > 5.0:
            fps = self._frame_count / max(1, now - self._last_log_time)
            self.get_logger().info(
                f'Radar: {fps:.1f} fps | '
                f'Filtered pts: {len(filtered)} | '
                f'Detections: {len(detections)} | '
                f'Total detections: {self._detection_count}')
            self._frame_count   = 0
            self._last_log_time = now

    def destroy_node(self):
        self.driver.stop()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = DroneDetectorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
