#!/usr/bin/env python3
"""SNR-aware radar filter node with cluster scoring and persistence.

Input:  /radar/point_cloud, fields x,y,z,doppler,snr[,noise]
Output: /radar/filtered_pcl, /radar/best_target, /radar/target_velocity, /radar/drone_pose

Coordinate convention:
  +X = radar boresight / forward range
  +Y = lateral
  +Z = vertical
"""

import sys, os, math, time
import struct as _struct
from collections import deque

_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _dir not in sys.path:
    sys.path.insert(0, _dir)

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
from sensor_msgs.msg import PointCloud2, PointField
from geometry_msgs.msg import PointStamped, PoseStamped, Quaternion
from std_msgs.msg import Float32

try:
    from sklearn.cluster import DBSCAN as SklearnDBSCAN
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

# Warn once if the numpy DBSCAN fallback is used with a large cloud.
_DBSCAN_NUMPY_POINT_WARN = 300


def parse_pointcloud2(msg: PointCloud2) -> np.ndarray:
    """Return Nx6 array: x, y, z, doppler, snr, noise.

    Uses a vectorised numpy path instead of a per-point Python loop for
    performance at high point rates / frame accumulation.
    """
    fields = {f.name: f.offset for f in msg.fields}
    if not all(k in fields for k in ('x', 'y', 'z')):
        return np.zeros((0, 6), dtype=np.float32)

    n = msg.width * msg.height
    if n == 0:
        return np.zeros((0, 6), dtype=np.float32)

    data = np.frombuffer(bytes(msg.data), dtype=np.uint8)
    step = msg.point_step
    pts = np.zeros((n, 6), dtype=np.float32)

    # Build a (n, step) byte view then slice each field at its byte offset.
    raw = data[:n * step].reshape(n, step)

    def _extract(field_name: str) -> np.ndarray:
        off = fields[field_name]
        # Each float32 is 4 bytes; pull the 4 columns and reinterpret.
        return raw[:, off:off + 4].copy().view(np.float32).reshape(n)

    pts[:, 0] = _extract('x')
    pts[:, 1] = _extract('y')
    pts[:, 2] = _extract('z')
    if 'doppler' in fields:
        pts[:, 3] = _extract('doppler')
    if 'snr' in fields:
        pts[:, 4] = _extract('snr')
    if 'noise' in fields:
        pts[:, 5] = _extract('noise')

    valid = np.all(np.isfinite(pts[:, :6]), axis=1)
    return pts[valid]


def make_pointcloud2(pts: np.ndarray, header) -> PointCloud2:
    msg = PointCloud2()
    msg.header = header
    msg.height = 1
    msg.width = len(pts)
    msg.fields = [
        PointField(name='x',       offset=0,  datatype=PointField.FLOAT32, count=1),
        PointField(name='y',       offset=4,  datatype=PointField.FLOAT32, count=1),
        PointField(name='z',       offset=8,  datatype=PointField.FLOAT32, count=1),
        PointField(name='doppler', offset=12, datatype=PointField.FLOAT32, count=1),
        PointField(name='snr',     offset=16, datatype=PointField.FLOAT32, count=1),
        PointField(name='noise',   offset=20, datatype=PointField.FLOAT32, count=1),
    ]
    msg.is_bigendian = False
    msg.point_step = 24
    msg.row_step = 24 * len(pts)
    msg.is_dense = True
    data = bytearray()
    for p in pts:
        vals = [0.0] * 6
        vals[:min(6, len(p))] = [float(v) for v in p[:min(6, len(p))]]
        data += _struct.pack('<ffffff', *vals)
    msg.data = bytes(data)
    return msg


def yaw_to_quat(yaw: float) -> Quaternion:
    q = Quaternion()
    q.w = math.cos(yaw / 2.0)
    q.z = math.sin(yaw / 2.0)
    q.x = 0.0
    q.y = 0.0
    return q


def dbscan_numpy(xyz: np.ndarray, eps: float, min_samples: int,
                 logger=None) -> np.ndarray:
    """Pure-numpy DBSCAN fallback.

    Builds an O(N²) distance matrix — fine for small clouds but degrades
    quickly beyond ~300 points.  A warning is emitted (once) when the point
    count exceeds _DBSCAN_NUMPY_POINT_WARN so the caller knows to install
    scikit-learn.
    """
    n = len(xyz)
    labels = np.full(n, -1, dtype=int)
    if n == 0:
        return labels

    if n > _DBSCAN_NUMPY_POINT_WARN and logger is not None:
        logger.warn(
            f'dbscan_numpy called with {n} points (>{_DBSCAN_NUMPY_POINT_WARN}). '
            f'Install scikit-learn for an efficient implementation.',
            throttle_duration_sec=30,
        )

    visited = np.zeros(n, dtype=bool)
    cluster_id = 0
    dists = np.sqrt(((xyz[:, None, :] - xyz[None, :, :]) ** 2).sum(axis=2))
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


class RadarFilterNode(Node):
    def __init__(self):
        super().__init__('radar_filter_node')

        self.declare_parameter('azimuth_fov_deg', 90.0)
        self.declare_parameter('elevation_fov_deg', 90.0)
        self.declare_parameter('min_range', 0.3)
        self.declare_parameter('max_range', 60.0)
        self.declare_parameter('velocity_threshold', 0.0)
        self.declare_parameter('snr_threshold', 0.0)
        self.declare_parameter('max_noise_db', 999.0)
        self.declare_parameter('concat_frames', 1)
        self.declare_parameter('voxel_size', 0.2)
        self.declare_parameter('dbscan_eps', 2.0)
        self.declare_parameter('dbscan_min_samples', 1)
        self.declare_parameter('sensor_frame_id', 'radar')
        self.declare_parameter('input_topic', '/radar/point_cloud')

        # Cluster scoring / persistence parameters
        self.declare_parameter('selection_mode', 'score')  # score, biggest, closest, farthest
        self.declare_parameter('persistence_frames', 3)
        self.declare_parameter('track_assoc_dist', 2.0)
        self.declare_parameter('publish_before_persistent', False)
        self.declare_parameter('min_cluster_snr', 0.0)
        self.declare_parameter('min_cluster_points', 1)
        self.declare_parameter('max_cluster_spread', 999.0)
        # Alpha for candidate centroid smoothing: new = alpha*new + (1-alpha)*old
        self.declare_parameter('position_alpha', 0.6)

        self.az_fov = math.radians(float(self.get_parameter('azimuth_fov_deg').value))
        self.el_fov = math.radians(float(self.get_parameter('elevation_fov_deg').value))
        self.min_r = float(self.get_parameter('min_range').value)
        self.max_r = float(self.get_parameter('max_range').value)
        self.vel_thr = float(self.get_parameter('velocity_threshold').value)
        self.snr_thr = float(self.get_parameter('snr_threshold').value)
        self.max_noise_db = float(self.get_parameter('max_noise_db').value)
        self.n_frames = int(self.get_parameter('concat_frames').value)
        self.vox_size = float(self.get_parameter('voxel_size').value)
        self.eps = float(self.get_parameter('dbscan_eps').value)
        self.min_pts = int(self.get_parameter('dbscan_min_samples').value)
        self.frame_id = str(self.get_parameter('sensor_frame_id').value)
        self.input_topic = str(self.get_parameter('input_topic').value)

        self.selection_mode = str(self.get_parameter('selection_mode').value)
        self.persistence_frames = int(self.get_parameter('persistence_frames').value)
        self.track_assoc_dist = float(self.get_parameter('track_assoc_dist').value)
        self.publish_before_persistent = bool(self.get_parameter('publish_before_persistent').value)
        self.min_cluster_snr = float(self.get_parameter('min_cluster_snr').value)
        self.min_cluster_points = int(self.get_parameter('min_cluster_points').value)
        self.max_cluster_spread = float(self.get_parameter('max_cluster_spread').value)
        self.position_alpha = float(self.get_parameter('position_alpha').value)

        self.frame_buffer = deque(maxlen=max(1, self.n_frames))
        self._candidate_centroid = None
        self._candidate_count = 0

        sensor_qos = QoSProfile(reliability=QoSReliabilityPolicy.BEST_EFFORT,
                                history=QoSHistoryPolicy.KEEP_LAST, depth=5)
        reliable_qos = QoSProfile(reliability=QoSReliabilityPolicy.RELIABLE,
                                  history=QoSHistoryPolicy.KEEP_LAST, depth=10)

        self.sub = self.create_subscription(PointCloud2, self.input_topic, self._cb, sensor_qos)
        self.pub_filtered = self.create_publisher(PointCloud2, '/radar/filtered_pcl', sensor_qos)
        self.pub_target = self.create_publisher(PointStamped, '/radar/best_target', reliable_qos)
        self.pub_velocity = self.create_publisher(Float32, '/radar/target_velocity', reliable_qos)
        self.pub_pose = self.create_publisher(PoseStamped, '/radar/drone_pose', reliable_qos)

        self._frame_count = 0
        self._detect_count = 0
        self._last_log = time.time()
        self.get_logger().info(f'RadarFilterNode ready — subscribed to {self.input_topic}')
        self.get_logger().info(f'DBSCAN via {"sklearn" if SKLEARN_AVAILABLE else "numpy fallback"}')
        self.get_logger().info(
            f'Filters: range={self.min_r}-{self.max_r}m, fov={math.degrees(self.az_fov):.1f}/{math.degrees(self.el_fov):.1f}deg, '
            f'snr>={self.snr_thr}, vel>={self.vel_thr}, selection={self.selection_mode}, persistence={self.persistence_frames}'
        )

    def _cb(self, msg: PointCloud2):
        try:
            raw_pts = parse_pointcloud2(msg)
            if raw_pts is None or len(raw_pts) == 0:
                return

            pts = self._filter(raw_pts)
            if len(pts) == 0:
                self._log_stats(raw_pts, pts, [])
                return

            self.frame_buffer.append(pts)
            accumulated = np.vstack(self.frame_buffer)
            accumulated = self._voxel_downsample(accumulated)
            self.pub_filtered.publish(make_pointcloud2(accumulated, msg.header))

            clusters = self._cluster(accumulated)
            clusters = [c for c in clusters if self._cluster_passes(c)]

            if clusters:
                best = self._select_cluster(clusters)
                if self._update_persistence(best):
                    self._detect_count += 1
                    self._publish_target(msg, best)
                elif self.publish_before_persistent:
                    self._publish_target(msg, best)

            self._log_stats(raw_pts, accumulated, clusters)
        except Exception as e:
            self.get_logger().error(f'Filter error: {e}')

    def _filter(self, pts: np.ndarray) -> np.ndarray:
        x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]
        dist = np.sqrt(x ** 2 + y ** 2 + z ** 2)
        mask = (dist > 0.05)
        mask &= (dist >= self.min_r) & (dist <= self.max_r)

        # +X is radar boresight/forward, +Y is lateral.
        az = np.arctan2(y, x)
        mask &= np.abs(az) <= self.az_fov

        el = np.arctan2(z, np.sqrt(x ** 2 + y ** 2))
        mask &= np.abs(el) <= self.el_fov

        if pts.shape[1] > 3 and self.vel_thr > 0.0:
            mask &= np.abs(pts[:, 3]) >= self.vel_thr

        if pts.shape[1] > 4 and self.snr_thr > 0.0:
            snr = pts[:, 4]
            if np.any(snr > 0.0):
                mask &= snr >= self.snr_thr

        if pts.shape[1] > 5 and self.max_noise_db < 999.0:
            noise = pts[:, 5]
            if np.any(noise > 0.0):
                mask &= noise <= self.max_noise_db

        return pts[mask]

    def _voxel_downsample(self, pts: np.ndarray) -> np.ndarray:
        if len(pts) == 0 or self.vox_size <= 0.0:
            return pts
        keys = np.floor(pts[:, :3] / self.vox_size).astype(np.int32)
        keys_view = np.ascontiguousarray(keys).view(np.dtype((np.void, keys.dtype.itemsize * 3)))
        _, idx = np.unique(keys_view, return_index=True)
        return pts[idx]

    def _cluster(self, pts: np.ndarray) -> list:
        if len(pts) < self.min_pts:
            return []
        xyz = pts[:, :3]
        if SKLEARN_AVAILABLE:
            labels = SklearnDBSCAN(eps=self.eps, min_samples=self.min_pts).fit(xyz).labels_
        else:
            labels = dbscan_numpy(xyz, self.eps, self.min_pts, logger=self.get_logger())

        clusters = []
        for lbl in set(labels):
            if lbl == -1:
                continue
            mask = labels == lbl
            c_pts = pts[mask]
            xyz_c = c_pts[:, :3]
            centroid = xyz_c.mean(axis=0)
            mean_doppler = float(c_pts[:, 3].mean()) if c_pts.shape[1] > 3 else 0.0
            snrs = c_pts[:, 4] if c_pts.shape[1] > 4 else np.array([])
            real_snrs = snrs[snrs > 0.0]
            noises = c_pts[:, 5] if c_pts.shape[1] > 5 else np.array([])
            real_noises = noises[noises > 0.0]
            std = float(xyz_c.std(axis=0).mean()) if len(xyz_c) > 1 else 0.0
            clusters.append({
                'centroid': centroid,
                'mean_doppler': mean_doppler,
                'mean_snr': float(real_snrs.mean()) if len(real_snrs) else 0.0,
                'mean_noise': float(real_noises.mean()) if len(real_noises) else 0.0,
                'count': int(mask.sum()),
                'std': std,
            })
        return clusters

    def _cluster_passes(self, c: dict) -> bool:
        if c['count'] < self.min_cluster_points:
            return False
        if c['mean_snr'] > 0.0 and c['mean_snr'] < self.min_cluster_snr:
            return False
        if c['std'] > self.max_cluster_spread:
            return False
        return True

    def _cluster_score(self, c: dict) -> float:
        rng = float(np.linalg.norm(c['centroid']))
        count_score = min(1.0, c['count'] / 5.0)
        snr_score = min(1.0, c['mean_snr'] / 20.0) if c['mean_snr'] > 0.0 else 0.0
        doppler_score = min(1.0, abs(c['mean_doppler']) / 2.0)
        spread_penalty = min(1.0, c['std'] / max(0.1, self.eps))
        range_score = min(1.0, rng / max(1.0, self.max_r))
        return 2.0 * count_score + 1.5 * snr_score + 1.0 * doppler_score + 0.2 * range_score - 1.0 * spread_penalty

    def _select_cluster(self, clusters: list) -> dict:
        mode = self.selection_mode.lower()
        if mode == 'biggest':
            return max(clusters, key=lambda c: c['count'])
        if mode == 'closest':
            return min(clusters, key=lambda c: float(np.linalg.norm(c['centroid'])))
        if mode == 'farthest':
            return max(clusters, key=lambda c: float(np.linalg.norm(c['centroid'])))
        return max(clusters, key=self._cluster_score)

    def _update_persistence(self, cluster: dict) -> bool:
        c = cluster['centroid']
        if self.persistence_frames <= 1:
            self._candidate_centroid = c.copy()
            self._candidate_count = 1
            return True

        if self._candidate_centroid is None:
            self._candidate_centroid = c.copy()
            self._candidate_count = 1
            return False

        d = float(np.linalg.norm(c - self._candidate_centroid))
        if d <= self.track_assoc_dist:
            self._candidate_count += 1
            # Smooth candidate position using the tunable position_alpha parameter.
            alpha = self.position_alpha
            self._candidate_centroid = alpha * c + (1.0 - alpha) * self._candidate_centroid
        else:
            self._candidate_centroid = c.copy()
            self._candidate_count = 1

        return self._candidate_count >= self.persistence_frames

    def _publish_target(self, msg, cluster: dict):
        c = cluster['centroid']
        vel = cluster['mean_doppler']
        stamp = msg.header.stamp

        ps = PointStamped()
        ps.header.stamp = stamp
        ps.header.frame_id = self.frame_id
        ps.point.x = float(c[0])
        ps.point.y = float(c[1])
        ps.point.z = float(c[2])
        self.pub_target.publish(ps)

        self.pub_velocity.publish(Float32(data=float(vel)))

        pm = PoseStamped()
        pm.header.stamp = stamp
        pm.header.frame_id = self.frame_id
        pm.pose.position.x = float(c[0])
        pm.pose.position.y = float(c[1])
        pm.pose.position.z = float(c[2])
        yaw = math.atan2(float(c[1]), float(c[0]))
        pm.pose.orientation = yaw_to_quat(yaw)
        self.pub_pose.publish(pm)

    def _log_stats(self, raw_pts, filtered_pts, clusters):
        self._frame_count += 1
        now = time.time()
        if now - self._last_log < 5.0:
            return
        fps = self._frame_count / max(1e-6, now - self._last_log)
        if len(raw_pts):
            dist = np.sqrt(raw_pts[:, 0] ** 2 + raw_pts[:, 1] ** 2 + raw_pts[:, 2] ** 2)
            raw_range = f'{float(dist.min()):.2f}-{float(dist.max()):.2f}m'
            snr_vals = raw_pts[:, 4]
            real_snr = snr_vals[snr_vals > 0]
            snr_txt = f'{float(real_snr.min()):.1f}-{float(real_snr.max()):.1f}dB' if len(real_snr) else 'none'
        else:
            raw_range = 'none'
            snr_txt = 'none'
        self.get_logger().info(
            f'fps={fps:.1f} | raw={len(raw_pts)} | filtered={len(filtered_pts)} | '
            f'raw_range={raw_range} | raw_snr={snr_txt} | clusters={len(clusters)} | '
            f'persistent={self._candidate_count}/{self.persistence_frames} | total_detections={self._detect_count}')
        if clusters:
            best = self._select_cluster(clusters)
            c = best['centroid']
            rng = float(np.linalg.norm(c))
            self.get_logger().info(
                f'  best[{self.selection_mode}]: ({c[0]:+.2f},{c[1]:+.2f},{c[2]:+.2f})m '
                f'range={rng:.2f}m vel={best["mean_doppler"]:+.2f}m/s '
                f'pts={best["count"]} snr={best["mean_snr"]:.1f} noise={best["mean_noise"]:.1f} '
                f'spread={best["std"]:.2f} score={self._cluster_score(best):.2f}')
        self._frame_count = 0
        self._last_log = now


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
