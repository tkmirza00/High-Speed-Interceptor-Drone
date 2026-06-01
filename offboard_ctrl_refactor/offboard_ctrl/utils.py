"""Small math and ROS message helper functions."""

import math
from typing import List

from geometry_msgs.msg import TwistStamped


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def norm3(v: List[float]) -> float:
    return math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2])


def make_velocity_msg(vx: float, vy: float, vz: float) -> TwistStamped:
    msg = TwistStamped()
    msg.header.frame_id = "map"
    msg.twist.linear.x = float(vx)
    msg.twist.linear.y = float(vy)
    msg.twist.linear.z = float(vz)
    return msg
