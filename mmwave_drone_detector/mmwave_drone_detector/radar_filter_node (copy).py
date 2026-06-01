#!/usr/bin/env python3
"""
mmwave_drone_detector.radar_filter_node
=========================================
Drone detection node inspired by nhma20/spherical-radar-drone,
adapted for the IWR6843ISK compressed TLV 1020 point cloud format.

Key differences from the original:
  - Subscribes to /radar/point_cloud (our PointCloud2 with x,y,z,doppler,snr)
  - Uses frame buffering + voxel downsampling (nhma20 approach)
  - DBSCAN clustering (sklearn if available, numpy fallback)
  - Publishes filtered cloud, best target, velocity

Published topics:
  /radar/filtered_pcl      sensor_msgs/PointCloud2   cleaned point cloud
  /radar/best_target       geometry_msgs/PointStamped closest drone centroid
  /radar/target_velocity   std_msgs/Float32           mean Doppler of best target
  /radar/drone_pose        geometry_msgs/PoseStamped  pose (compatible with monitor)

Run standalone (no radar hardware needed — subscribes to existing /radar/point_cloud):
  ros2 run mmwave_drone_detector radar_filter_node.py

Or pipe from detector:
  Terminal 1: ros2 launch mmwave_drone_detector detector.launch.py
  Terminal 2: ros2 run mmwave_drone_detector radar_filter_node.py
"""

import sys
import os
import math
import struct
import threading
import time

_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _dir not in sys.path:
    sys.path.insert(0, _dir)

import numpy as np
from collections import deque

import rclpy
from rclpy.node import Node
from rclpy.qos import (QoSProfile, QoSReliabilityPolicy,
                        QoSHistoryPolicy)

from sensor_msgs.msg import PointCloud2, PointField
from geometry_msgs.msg import PointStamped, PoseStamped, Quaternion
from std_msgs.msg import Float32
import struct as _struct

# Optional sklearn DBSCAN
try:
    from sklearn.cluster import DBSCAN as SklearnDBSCAN
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

# ── PointCloud2 helpers ───────────────────────────────────────────────────────

def parse_pointcloud2(msg: PointCloud2) -> np.ndarray:
    """
    Parse our PointCloud2 message into Nx5 numpy array [x,y,z,doppler,snr].
    Compatible with both our radar_driver output and standard xyz clouds.
    """
    fields = {f.name: f.offset for f in msg.fields}
    has_doppler = 'doppler' in fields
    has_snr     = 'snr' in fields

    data  = bytes(msg.data)
    step  = msg.point_step
    n     = msg.width * msg.height
    pts   = np.zeros((n, 5), dtype=np.float32)

    for i in range(n):
        base = i * step
        pts[i, 0] = _struct.unpack_from('<f', data, base + fields['x'])[0]
        pts[i, 1] = _struct.unpack_from('<f', data, base + fields['y'])[0]
        pts[i, 2] = _struct.unpack_from('<f', data, base + fields['z'])[0]
        if has_doppler:
            pts[i, 3] = _struct.unpack_from('<f', data, base + fields['doppler'])[0]
        if has_snr:
            pts[i, 4] = _struct.unpack_from('<f', data, base + fields['snr'])[0]

    # Remove NaN/Inf
    valid = np.all(np.isfinite(pts[:, :3]), axis=1)
    return pts[valid]


def make_pointcloud2(pts: np.ndarray, header) -> PointCloud2:
    """Create PointCloud2 from Nx3 or Nx5 numpy array."""
    msg = PointCloud2()
    msg.header = header
    msg.height = 1
    msg.width  = len(pts)
    msg.fields = [
        PointField(name='x',       offset=0,  datatype=PointField.FLOAT32, count=1),
        PointField(name='y',       offset=4,  datatype=PointField.FLOAT32, count=1),
        PointField(name='z',       offset=8,  datatype=PointField.FLOAT32, count=1),
        PointField(name='doppler', offset=12, datatype=PointField.FLOAT32, count=1),
        PointField(name='snr',     offset=16, datatype=PointField.FLOAT32, count=1),
    ]
    msg.is_bigendian = False
    msg.point_step   = 20
    msg.row_step     = 20 * len(pts)
    msg.is_dense     = True
    data = bytearray()
    for p in pts:
        x   = float(p[0]) if len(p) > 0 else 0.0
        y   = float(p[1]) if len(p) > 1 else 0.0
        z   = float(p[2]) if len(p) > 2 else 0.0
        dop = float(p[3]) if len(p) > 3 else 0.0
        snr = float(p[4]) if len(p) > 4 else 0.0
        data += _struct.pack('<fffff', x, y, z, dop, snr)
    msg.data = bytes(data)
    return msg


def yaw_to_quat(yaw: float) -> Quaternion:
    q = Quaternion()
    q.w = math.cos(yaw / 2)
    q.z = math.sin(yaw / 2)
    q.x = 0.0; q.y = 0.0
    return q


# ── DBSCAN fallback ───────────────────────────────────────────────────────────

def dbscan_numpy(xyz: np.ndarray, eps: float, min_samples: int) -> np.ndarray:
    """Pure numpy DBSCAN — O(n²), fine for n < 200 points."""
    n = len(xyz)
    labels = np.full(n, -1, dtype=int)
    cluster_id = 0
    visited = np.zeros(n, dtype=bool)

    # Pairwise distances
    diff = xyz[:, None, :] - xyz[None, :, :]  # n×n×3
    dists = np.sqrt((diff**2).sum(axis=2))     # n×n

    for i in range(n):
        if visited[i]:
            continue
        visited[i] = True
        nbrs = list(np.where(dists[i] <= eps)[0])
        if len(nbrs) < min_samples:
            continue
        labels[i] = cluster_id
        seed = list(nbrs)
        j = 0
        while j < len(seed):
            q = seed[j]
            if not visited[q]:
                visited[q] = True
                q_nbrs = list(np.where(dists[q] <= eps)[0])
                if len(q_nbrs) >= min_samples:
                    seed.extend(q_nbrs)
            if labels[q] == -1:
                labels[q] = cluster_id
            j += 1
        cluster_id += 1

    return labels


# ── Main node ─────────────────────────────────────────────────────────────────

class RadarFilterNode(Node):
    """
    Subscribes to /radar/point_cloud, applies:
      1. FOV + range + SNR + velocity filtering
      2. Frame buffering (concat N frames)
      3. Voxel downsampling
      4. DBSCAN clustering
      5. Best target selection (closest cluster)
    Publishes filtered cloud, target position and velocity.
    """

    def __init__(self):
        super().__init__('radar_filter_node')

        # ── Parameters ───────────────────────────────────────────────────────
        self.declare_parameter('azimuth_fov_deg',    60.0)
        self.declare_parameter('elevation_fov_deg',  60.0)
        self.declare_parameter('min_range',           0.3)
        self.declare_parameter('max_range',           20.0)
        self.declare_parameter('velocity_threshold',  0.15)
        self.declare_parameter('snr_threshold',       0.0)
        self.declare_parameter('concat_frames',       3)
        self.declare_parameter('voxel_size',          0.1)
        self.declare_parameter('dbscan_eps',          0.8)
        self.declare_parameter('dbscan_min_samples',  2)
        self.declare_parameter('sensor_frame_id',    'radar')
        self.declare_parameter('input_topic',        '/radar/point_cloud')

        self.az_fov   = math.radians(self.get_parameter('azimuth_fov_deg').value)
        self.el_fov   = math.radians(self.get_parameter('elevation_fov_deg').value)
        self.min_r    = self.get_parameter('min_range').value
        self.max_r    = self.get_parameter('max_range').value
        self.vel_thr  = self.get_parameter('velocity_threshold').value
        self.snr_thr  = self.get_parameter('snr_threshold').value
        self.n_frames = self.get_parameter('concat_frames').value
        self.vox_size = self.get_parameter('voxel_size').value
        self.eps      = self.get_parameter('dbscan_eps').value
        self.min_pts  = self.get_parameter('dbscan_min_samples').value
        self.frame_id = self.get_parameter('sensor_frame_id').value
        input_topic   = self.get_parameter('input_topic').value

        self.frame_buffer = deque(maxlen=self.n_frames)

        # ── QoS ───────────────────────────────────────────────────────────────
        sensor_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=5)
        reliable_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=10)

        # ── Subscribers ───────────────────────────────────────────────────────
        self.sub = self.create_subscription(
            PointCloud2, input_topic, self._cb, sensor_qos)

        # ── Publishers ────────────────────────────────────────────────────────
        self.pub_filtered = self.create_publisher(
            PointCloud2,   '/radar/filtered_pcl',    sensor_qos)
        self.pub_target   = self.create_publisher(
            PointStamped,  '/radar/best_target',     reliable_qos)
        self.pub_velocity = self.create_publisher(
            Float32,       '/radar/target_velocity', reliable_qos)
        self.pub_pose     = self.create_publisher(
            PoseStamped,   '/radar/drone_pose',      reliable_qos)

        # Stats
        self._frame_count   = 0
        self._detect_count  = 0
        self._last_log      = time.time()

        self.get_logger().info(
            f'RadarFilterNode ready — subscribed to {input_topic}')
        self.get_logger().info(
            f'DBSCAN via {"sklearn" if SKLEARN_AVAILABLE else "numpy fallback"}')

    # ── Callback ──────────────────────────────────────────────────────────────

    def _cb(self, msg: PointCloud2):
        try:
            # 1. Parse
            pts = parse_pointcloud2(msg)
            if pts is None or len(pts) == 0:
                return

            # 2. Filter
            pts = self._filter(pts)
            if len(pts) == 0:
                return

            # 3. Buffer + concatenate
            self.frame_buffer.append(pts)
            accumulated = np.vstack(self.frame_buffer)

            # 4. Voxel downsample
            accumulated = self._voxel_downsample(accumulated)

            # 5. Publish filtered cloud
            self.pub_filtered.publish(
                make_pointcloud2(accumulated, msg.header))

            # 6. Cluster
            clusters = self._cluster(accumulated)

            # 7. Select best (closest) cluster and publish
            if clusters:
                self._detect_count += 1
                best = min(clusters,
                           key=lambda c: float(np.linalg.norm(c['centroid'])))
                self._publish_target(msg, best)

            # Stats log
            self._frame_count += 1
            now = time.time()
            if now - self._last_log >= 5.0:
                fps = self._frame_count / max(1, now - self._last_log)
                self.get_logger().info(
                    f'fps={fps:.1f} | '
                    f'raw={len(pts)} | '
                    f'accumulated={len(accumulated)} | '
                    f'clusters={len(clusters)} | '
                    f'total_detections={self._detect_count}')
                if clusters:
                    c = best['centroid']
                    rng = float(np.linalg.norm(c))
                    self.get_logger().info(
                        f'  best: ({c[0]:+.2f},{c[1]:+.2f},{c[2]:+.2f})m '
                        f'range={rng:.2f}m '
                        f'vel={best["mean_doppler"]:+.2f}m/s '
                        f'pts={best["count"]}')
                self._frame_count = 0
                self._last_log    = now

        except Exception as e:
            self.get_logger().error(f'Filter error: {e}')
            import traceback
            self.get_logger().debug(traceback.format_exc())

    # ── Filter ────────────────────────────────────────────────────────────────

    def _filter(self, pts: np.ndarray) -> np.ndarray:
        x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]

        # Range artifact rejection (compressed format clipping values)
        dist = np.sqrt(x**2 + y**2 + z**2)
        mask = (dist > 0.26) & (dist < 20.0)

        # Physical range bounds
        mask &= (dist >= self.min_r) & (dist <= self.max_r)

        # Azimuth FOV
        az = np.arctan2(x, y)   # 0 = forward
        mask &= np.abs(az) <= self.az_fov

        # Elevation FOV
        el = np.arctan2(z, np.sqrt(x**2 + y**2))
        mask &= np.abs(el) <= self.el_fov

        # Velocity gate — only moving objects
        if pts.shape[1] > 3:
            mask &= np.abs(pts[:, 3]) >= self.vel_thr

        # SNR threshold
        if pts.shape[1] > 4:
            mask &= pts[:, 4] >= self.snr_thr

        return pts[mask]

    # ── Voxel downsample ──────────────────────────────────────────────────────

    def _voxel_downsample(self, pts: np.ndarray) -> np.ndarray:
        if len(pts) == 0:
            return pts
        keys = np.floor(pts[:, :3] / self.vox_size).astype(np.int32)
        # Use structured array trick for unique row detection
        keys_view = np.ascontiguousarray(keys).view(
            np.dtype((np.void, keys.dtype.itemsize * 3)))
        _, idx = np.unique(keys_view, return_index=True)
        return pts[idx]

    # ── DBSCAN ────────────────────────────────────────────────────────────────

    def _cluster(self, pts: np.ndarray) -> list:
        if len(pts) < self.min_pts:
            return []

        xyz = pts[:, :3]

        if SKLEARN_AVAILABLE:
            db = SklearnDBSCAN(
                eps=self.eps,
                min_samples=self.min_pts
            ).fit(xyz)
            labels = db.labels_
        else:
            labels = dbscan_numpy(xyz, self.eps, self.min_pts)

        clusters = []
        for lbl in set(labels):
            if lbl == -1:   # noise
                continue
            mask   = labels == lbl
            c_pts  = pts[mask]
            xyz_c  = c_pts[:, :3]
            centroid = xyz_c.mean(axis=0)
            mean_dop = float(c_pts[:, 3].mean()) if pts.shape[1] > 3 else 0.0
            mean_snr = float(c_pts[:, 4].mean()) if pts.shape[1] > 4 else 0.0
            clusters.append({
                'centroid':    centroid,
                'mean_doppler': mean_dop,
                'mean_snr':    mean_snr,
                'count':       int(mask.sum()),
                'std':         float(xyz_c.std(axis=0).mean()),
            })

        return clusters

    # ── Publish ───────────────────────────────────────────────────────────────

    def _publish_target(self, msg, cluster: dict):
        c   = cluster['centroid']
        vel = cluster['mean_doppler']
        stamp = msg.header.stamp

        # PointStamped
        ps = PointStamped()
        ps.header.stamp    = stamp
        ps.header.frame_id = self.frame_id
        ps.point.x = float(c[0])
        ps.point.y = float(c[1])
        ps.point.z = float(c[2])
        self.pub_target.publish(ps)

        # Float32 velocity
        self.pub_velocity.publish(Float32(data=vel))

        # PoseStamped (compatible with monitor_3d.py /radar/drone_pose)
        pm = PoseStamped()
        pm.header.stamp    = stamp
        pm.header.frame_id = self.frame_id
        pm.pose.position.x = float(c[0])
        pm.pose.position.y = float(c[1])
        pm.pose.position.z = float(c[2])
        yaw = math.atan2(float(c[0]), float(c[1]))
        pm.pose.orientation = yaw_to_quat(yaw)
        self.pub_pose.publish(pm)


# ── Entry point ───────────────────────────────────────────────────────────────

def main(args=None):
    rclpy.init(args=args)
    node = RadarFilterNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
