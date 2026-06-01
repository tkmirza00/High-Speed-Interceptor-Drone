#!/usr/bin/env python3
"""
Debug detector for IWR6843ISK long-range testing.

Purpose:
- Do NOT aggressively classify.
- Do NOT hide points with strict drone logic.
- Print why points are being rejected.
- Create simple detections from raw/filtered radar points.

Use this only for debugging.
"""

import math
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple

from mmwave_drone_detector.radar_driver import RadarFrame, RadarPoint


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


class DroneDetector:
    def __init__(self,
                 min_snr_db: float = 0.0,
                 min_range_m: float = 0.2,
                 max_range_m: float = 60.0,
                 max_azimuth_deg: float = 90.0,
                 min_height_m: float = -10.0,
                 max_height_m: float = 20.0,
                 min_doppler_ms: float = 0.0,
                 min_cluster_points: int = 1,
                 max_cluster_dist_m: float = 2.0,
                 min_confidence: float = 0.0,
                 velocity_alpha: float = 0.4,
                 position_alpha: float = 0.6):

        self.min_snr_db = min_snr_db
        self.min_range_m = min_range_m
        self.max_range_m = max_range_m
        self.max_azimuth_deg = max_azimuth_deg
        self.min_height_m = min_height_m
        self.max_height_m = max_height_m
        self.min_doppler_ms = min_doppler_ms

        self._filtered_points: List[RadarPoint] = []
        self._frame_counter = 0

    def process(self, frame: RadarFrame) -> List[DroneDetection]:
        self._frame_counter += 1

        raw = frame.points
        filtered = []

        reject = {
            "range": 0,
            "snr": 0,
            "height": 0,
            "azimuth": 0,
            "doppler": 0,
        }

        for pt in raw:
            r = pt.range

            # Enforce min_range_m only — no separate hardcoded floor that
            # silently overrides the user-supplied parameter.
            if r <= 0.0 or r > self.max_range_m or r < self.min_range_m:
                reject["range"] += 1
                continue

            if pt.snr_db < self.min_snr_db:
                reject["snr"] += 1
                continue

            if pt.z < self.min_height_m or pt.z > self.max_height_m:
                reject["height"] += 1
                continue

            if abs(pt.azimuth_deg) > self.max_azimuth_deg:
                reject["azimuth"] += 1
                continue

            if abs(pt.doppler) < self.min_doppler_ms:
                reject["doppler"] += 1
                continue

            filtered.append(pt)

        self._filtered_points = filtered

        if self._frame_counter % 10 == 0:
            if raw:
                ranges = [p.range for p in raw]
                snrs = [p.snr_db for p in raw]
                dopps = [p.doppler for p in raw]
                print(
                    f"[DEBUG DETECTOR] frame={frame.frame_number} "
                    f"raw={len(raw)} filtered={len(filtered)} "
                    f"r_min={min(ranges):.2f} r_max={max(ranges):.2f} "
                    f"snr_mean={np.mean(snrs):.2f} "
                    f"dopp_mean={np.mean(dopps):+.2f} "
                    f"reject={reject}"
                )
            else:
                print(
                    f"[DEBUG DETECTOR] frame={frame.frame_number} "
                    f"raw=0 filtered=0"
                )

        if not filtered:
            return []

        pts = np.array([[p.x, p.y, p.z] for p in filtered], dtype=float)
        dopplers = np.array([p.doppler for p in filtered], dtype=float)
        snrs = np.array([p.snr_db for p in filtered], dtype=float)

        centroid = pts.mean(axis=0)
        mean_doppler = float(dopplers.mean())
        mean_snr = float(snrs.mean())

        rng = float(np.linalg.norm(centroid))

        # Convention: +X = boresight/forward, +Y = lateral.
        # azimuth = angle of the centroid projected onto the XY plane,
        # measured from boresight (+X axis).  atan2(Y, X) is correct here.
        # The original code had the arguments swapped (atan2(X, Y)), which
        # would give the complement of the true azimuth.
        az = math.degrees(math.atan2(centroid[1], centroid[0]))
        el = math.degrees(
            math.atan2(centroid[2], math.sqrt(centroid[0] ** 2 + centroid[1] ** 2))
        )

        detection = DroneDetection(
            drone_id=0,
            position=(float(centroid[0]), float(centroid[1]), float(centroid[2])),
            velocity=(0.0, mean_doppler, 0.0),
            speed=abs(mean_doppler),
            range=rng,
            azimuth_deg=az,
            elevation_deg=el,
            confidence=1.0,
            point_count=len(filtered),
            snr_mean=mean_snr,
            doppler_mean=mean_doppler,
            is_approaching=mean_doppler < 0,
            timestamp=frame.timestamp,
        )

        return [detection]

    @property
    def filtered_points(self) -> List[RadarPoint]:
        return self._filtered_points
