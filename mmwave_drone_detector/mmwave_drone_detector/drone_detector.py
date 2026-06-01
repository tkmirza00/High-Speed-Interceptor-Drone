#!/usr/bin/env python3
"""SNR-free drone detector for TI Long Range People Detection output.

This version expects radar_driver.py to convert TLV 1000 range/azimuth/elevation
measurements into Cartesian coordinates with:

    +X = radar boresight / forward range
    +Y = lateral left/right
    +Z = vertical

SNR is not used because this firmware path does not provide reliable per-point
SNR in the ROS point cloud.
"""

import math
import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
from mmwave_drone_detector.radar_driver import RadarFrame, RadarPoint

# Warn when the pure-numpy DBSCAN fallback is called with more than this
# many points — the O(N²) distance matrix becomes expensive beyond here.
_DBSCAN_NUMPY_POINT_WARN = 300


@dataclass
class DroneDetection:
    drone_id: int
    position: Tuple[float, float, float]
    velocity: Tuple[float, float, float]
    speed: float
    range: float
    azimuth_deg: float
    elevation_deg: float
    confidence: float
    point_count: int
    snr_mean: float
    doppler_mean: float
    is_approaching: bool
    timestamp: float


class DroneFilter:
    def __init__(self, min_snr_db=0.0, min_range_m=0.3, max_range_m=60.0,
                 max_azimuth_deg=90.0, min_height_m=-5.0, max_height_m=10.0,
                 min_doppler_ms=0.0):
        self.min_snr_db = min_snr_db
        self.min_range_m = min_range_m
        self.max_range_m = max_range_m
        self.max_azimuth_deg = max_azimuth_deg
        self.min_height_m = min_height_m
        self.max_height_m = max_height_m
        self.min_doppler_ms = min_doppler_ms

    def apply(self, points: List[RadarPoint]) -> List[RadarPoint]:
        """Return only the points that pass all configured gates."""
        filtered = []
        for pt in points:
            r = pt.range

            # Single consolidated range check — respects min_range_m fully,
            # no hardcoded floor that would silently override the parameter.
            if r <= 0.0 or r < self.min_range_m or r > self.max_range_m:
                continue

            if pt.z < self.min_height_m or pt.z > self.max_height_m:
                continue

            if abs(pt.azimuth_deg) > self.max_azimuth_deg:
                continue

            if abs(pt.doppler) < self.min_doppler_ms:
                continue

            if pt.snr_db > 0.0 and pt.snr_db < self.min_snr_db:
                continue

            filtered.append(pt)
        return filtered


class DBSCANClusterer:
    def __init__(self, eps=2.0, min_samples=1):
        self.eps = eps
        self.min_samples = min_samples

    def cluster(self, points: List[RadarPoint]) -> List[List[RadarPoint]]:
        if len(points) < self.min_samples:
            return []

        xyz = np.array([[p.x, p.y, p.z] for p in points], dtype=float)
        n = len(xyz)

        if n > _DBSCAN_NUMPY_POINT_WARN:
            print(
                f'[DBSCANClusterer] WARNING: clustering {n} points with the pure-numpy '
                f'O(N²) implementation (threshold={_DBSCAN_NUMPY_POINT_WARN}). '
                f'Consider installing scikit-learn for better performance.'
            )

        labels = np.full(n, -1, dtype=int)
        visited = np.zeros(n, dtype=bool)
        cluster_id = 0
        dists = np.sqrt(((xyz[:, None, :] - xyz[None, :, :]) ** 2).sum(axis=2))

        for i in range(n):
            if visited[i]:
                continue
            visited[i] = True
            nbrs = list(np.where(dists[i] <= self.eps)[0])
            if len(nbrs) < self.min_samples:
                continue
            labels[i] = cluster_id
            seed = list(nbrs)
            j = 0
            while j < len(seed):
                q = seed[j]
                if not visited[q]:
                    visited[q] = True
                    q_nbrs = list(np.where(dists[q] <= self.eps)[0])
                    if len(q_nbrs) >= self.min_samples:
                        seed.extend(q_nbrs)
                if labels[q] == -1:
                    labels[q] = cluster_id
                j += 1
            cluster_id += 1

        clusters = []
        for cid in range(cluster_id):
            idxs = np.where(labels == cid)[0]
            if len(idxs) >= self.min_samples:
                clusters.append([points[i] for i in idxs])
        return clusters


class DroneClassifier:
    def __init__(self, min_confidence=0.2, min_cluster_points=1):
        self.min_confidence = min_confidence
        self.min_cluster_points = min_cluster_points

    def score(self, cluster: List[RadarPoint]) -> float:
        if len(cluster) < self.min_cluster_points:
            return 0.0
        cluster_pts = np.array([[p.x, p.y, p.z] for p in cluster], dtype=float)
        dopplers = np.array([abs(p.doppler) for p in cluster], dtype=float)
        doppler_score = min(1.0, float(dopplers.mean()) / 3.0) if len(dopplers) else 0.0
        count_score = min(1.0, len(cluster) / 8.0)
        if len(cluster) > 1:
            compact_score = max(0.0, 1.0 - float(cluster_pts.std(axis=0).mean()) / 2.0)
        else:
            compact_score = 0.5
        return float(0.50 * doppler_score + 0.30 * count_score + 0.20 * compact_score)

    def classify(self, cluster: List[RadarPoint]) -> Optional[float]:
        conf = self.score(cluster)
        return conf if conf >= self.min_confidence else None


class DroneTracker:
    def __init__(self, pos_alpha=0.6, vel_alpha=0.4, max_age_s=0.8, max_assoc_dist=3.0):
        self.pos_alpha = pos_alpha
        self.vel_alpha = vel_alpha
        self.max_age_s = max_age_s
        self.max_assoc_dist = max_assoc_dist
        self._tracks: Dict[int, dict] = {}
        self._next_id = 0

    def _dist(self, a, b):
        return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))

    def update(self, detections: List[dict], timestamp: float) -> List[DroneDetection]:
        # Expire stale tracks.
        self._tracks = {
            tid: t for tid, t in self._tracks.items()
            if timestamp - t['last_seen'] < self.max_age_s
        }

        results = []
        used_tracks = set()

        for det in detections:
            raw_pos = det['position']
            best_tid = None
            best_dist = self.max_assoc_dist

            for tid, t in self._tracks.items():
                if tid in used_tracks:
                    continue
                d = self._dist(raw_pos, t['pos'])
                if d < best_dist:
                    best_dist = d
                    best_tid = tid

            if best_tid is not None:
                t = self._tracks[best_tid]
                dt = max(0.01, timestamp - t['last_seen'])
                raw_vel = tuple((raw_pos[i] - t['pos'][i]) / dt for i in range(3))
                pos = tuple(
                    self.pos_alpha * raw_pos[i] + (1.0 - self.pos_alpha) * t['pos'][i]
                    for i in range(3)
                )
                vel = tuple(
                    self.vel_alpha * raw_vel[i] + (1.0 - self.vel_alpha) * t['vel'][i]
                    for i in range(3)
                )
                t['pos'], t['vel'], t['last_seen'] = pos, vel, timestamp
                tid = best_tid
                used_tracks.add(tid)
            else:
                tid = self._next_id
                self._next_id += 1
                pos = raw_pos
                # Initialise velocity to zero — real velocity will be derived
                # from position deltas on the next association.  Seeding with
                # doppler_mean in the X slot was incorrect: doppler is radial
                # speed, not a Cartesian X-axis component.
                vel = (0.0, 0.0, 0.0)
                self._tracks[tid] = {'pos': pos, 'vel': vel, 'last_seen': timestamp}
                used_tracks.add(tid)

            pos = self._tracks[tid]['pos']
            vel = self._tracks[tid]['vel']
            speed = math.sqrt(sum(v * v for v in vel))
            rng = math.sqrt(sum(p * p for p in pos))

            # +X = boresight, +Y = lateral: azimuth is atan2(Y, X).
            az = math.degrees(math.atan2(pos[1], pos[0]))
            horiz = math.sqrt(pos[0] * pos[0] + pos[1] * pos[1])
            el = math.degrees(math.atan2(pos[2], horiz)) if horiz > 0 else 0.0

            results.append(DroneDetection(
                tid, pos, vel, speed, rng, az, el,
                det['confidence'], det['point_count'], det['snr_mean'],
                det['doppler_mean'], det['doppler_mean'] < 0, timestamp,
            ))
        return results


class DroneDetector:
    def __init__(self, min_snr_db=0.0, min_range_m=0.3, max_range_m=60.0,
                 max_azimuth_deg=90.0, min_height_m=-5.0, max_height_m=10.0,
                 min_doppler_ms=0.0, min_cluster_points=1, max_cluster_dist_m=2.0,
                 min_confidence=0.2, velocity_alpha=0.4, position_alpha=0.6):
        # Renamed from self.filter → self.point_filter to avoid shadowing
        # Python's built-in filter() function.
        self.point_filter = DroneFilter(
            min_snr_db, min_range_m, max_range_m,
            max_azimuth_deg, min_height_m, max_height_m, min_doppler_ms,
        )
        self.clusterer = DBSCANClusterer(max_cluster_dist_m, min_cluster_points)
        self.classifier = DroneClassifier(min_confidence, min_cluster_points)
        self.tracker = DroneTracker(pos_alpha=position_alpha, vel_alpha=velocity_alpha)
        self._filtered_points: List[RadarPoint] = []
        self._debug_count = 0

    def process(self, frame: RadarFrame) -> List[DroneDetection]:
        filtered = self.point_filter.apply(frame.points)
        self._filtered_points = filtered
        self._debug_count += 1

        if self._debug_count % 50 == 0:
            print(
                f'[DroneDetector] frame={frame.frame_number} '
                f'raw={len(frame.points)} filtered={len(filtered)}'
            )

        if not filtered:
            return []

        clusters = self.clusterer.cluster(filtered)
        if not clusters:
            return []

        raw_detections = []
        for cluster in clusters:
            conf = self.classifier.classify(cluster)
            if conf is None:
                continue
            cluster_pts = np.array([[p.x, p.y, p.z] for p in cluster], dtype=float)
            centroid = tuple(float(v) for v in cluster_pts.mean(axis=0))
            dopplers = [p.doppler for p in cluster]
            snrs = [p.snr_db for p in cluster if p.snr_db > 0.0]
            raw_detections.append({
                'position': centroid,
                'confidence': conf,
                'point_count': len(cluster),
                'snr_mean': float(np.mean(snrs)) if snrs else 0.0,
                'doppler_mean': float(np.mean(dopplers)) if dopplers else 0.0,
            })

        if not raw_detections:
            return []

        return self.tracker.update(raw_detections, frame.timestamp)

    @property
    def filtered_points(self) -> List[RadarPoint]:
        return self._filtered_points
