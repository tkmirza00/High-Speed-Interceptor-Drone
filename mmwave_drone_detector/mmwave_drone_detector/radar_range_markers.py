#!/usr/bin/env python3

import math
import rclpy
from rclpy.node import Node
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point


class RadarRangeMarkers(Node):
    def __init__(self):
        super().__init__('radar_range_markers')
        self.pub = self.create_publisher(MarkerArray, '/radar/range_markers', 10)
        self.timer = self.create_timer(1.0, self.publish_markers)

    def publish_markers(self):
        arr = MarkerArray()

        ranges = [1, 2, 5, 10, 15, 20, 30, 40, 50, 60]

        for idx, r in enumerate(ranges):
            ring = Marker()
            ring.header.frame_id = 'radar'
            ring.header.stamp = self.get_clock().now().to_msg()
            ring.ns = 'range_rings'
            ring.id = idx
            ring.type = Marker.LINE_STRIP
            ring.action = Marker.ADD
            ring.scale.x = 0.05
            ring.color.a = 0.8
            ring.color.r = 0.8
            ring.color.g = 0.8
            ring.color.b = 0.8

            # Your current parsed radar data uses +X as forward/range.
            # Ring lies in X-Y plane.
            for i in range(121):
                a = 2.0 * math.pi * i / 120.0
                p = Point()
                p.x = r * math.cos(a)
                p.y = r * math.sin(a)
                p.z = 0.0
                ring.points.append(p)

            arr.markers.append(ring)

            label = Marker()
            label.header.frame_id = 'radar'
            label.header.stamp = ring.header.stamp
            label.ns = 'range_labels'
            label.id = 100 + idx
            label.type = Marker.TEXT_VIEW_FACING
            label.action = Marker.ADD
            label.pose.position.x = float(r)
            label.pose.position.y = 0.0
            label.pose.position.z = 0.5
            label.scale.z = 0.6
            label.color.a = 1.0
            label.color.r = 1.0
            label.color.g = 1.0
            label.color.b = 1.0
            label.text = f'{r} m'

            arr.markers.append(label)

        self.pub.publish(arr)


def main(args=None):
    rclpy.init(args=args)
    node = RadarRangeMarkers()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
