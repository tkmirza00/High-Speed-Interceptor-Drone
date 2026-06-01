"""Frame transformation helpers for radar, body, and local ENU frames."""

import math
from typing import List, Tuple

from .config import RADAR_MOUNT_PITCH_DEG


def quat_rotate(q: List[float], v: List[float]) -> List[float]:
    """
    Rotate vector v by quaternion q=(x,y,z,w).

    Used to rotate a body-frame vector into local ENU using odometry orientation.
    """
    x, y, z, w = q
    vx, vy, vz = v

    tx = 2.0 * (y * vz - z * vy)
    ty = 2.0 * (z * vx - x * vz)
    tz = 2.0 * (x * vy - y * vx)

    vpx = vx + w * tx + (y * tz - z * ty)
    vpy = vy + w * ty + (z * tx - x * tz)
    vpz = vz + w * tz + (x * ty - y * tx)

    return [vpx, vpy, vpz]


def radar_to_body(x_r: float, y_r: float, z_r: float) -> Tuple[float, float, float]:
    """
    Convert radar-frame candidate to drone body/base_link FLU frame.

    Radar frame:
      +X_radar = radar boresight
      +Y_radar = lateral
      +Z_radar = vertical in radar frame

    Drone body/base_link FLU:
      +X_body = drone forward
      +Y_body = drone left
      +Z_body = drone up
    """
    theta = math.radians(RADAR_MOUNT_PITCH_DEG)

    x_b = math.cos(theta) * x_r + math.sin(theta) * z_r
    y_b = y_r
    z_b = -math.sin(theta) * x_r + math.cos(theta) * z_r

    return x_b, y_b, z_b


def body_to_local(q_body_to_local: List[float], v_body: List[float]) -> List[float]:
    return quat_rotate(q_body_to_local, v_body)
