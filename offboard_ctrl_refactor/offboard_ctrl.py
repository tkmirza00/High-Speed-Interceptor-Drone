#!/usr/bin/env python3
"""
Offboard Control — Expected-Gated Radar Candidate with Committed Target
======================================================================

This is the executable entry point. The node implementation is kept in
`offboard_ctrl/node.py` so the main file stays small.
"""

import rclpy

from offboard_ctrl.node import CommittedTargetRadarOffboard


def main(args=None):
    rclpy.init(args=args)
    node = CommittedTargetRadarOffboard()

    try:
        rclpy.spin(node)

    except KeyboardInterrupt:
        node.get_logger().info("Interrupted — sending zero velocity.")
        node._stop()

    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
