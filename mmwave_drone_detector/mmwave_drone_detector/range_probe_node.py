#!/usr/bin/env python3

import math
import struct
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy


class RangeProbeNode(Node):
    def __init__(self):
        super().__init__('range_probe_node')
        self.declare_parameter('input_topic', '/radar/point_cloud')
        self.declare_parameter('min_snr', 0.0)
        self.declare_parameter('min_range', 0.2)
        self.declare_parameter('max_reasonable_range', 50.0)
        self.declare_parameter('print_top_n', 5)
        self.declare_parameter('sort_by', 'range')  # range, snr, abs_doppler

        self.input_topic = self.get_parameter('input_topic').value
        self.min_snr = float(self.get_parameter('min_snr').value)
        self.min_range = float(self.get_parameter('min_range').value)
        self.max_reasonable_range = float(self.get_parameter('max_reasonable_range').value)
        self.print_top_n = int(self.get_parameter('print_top_n').value)
        self.sort_by = str(self.get_parameter('sort_by').value)

        sensor_qos = QoSProfile(reliability=QoSReliabilityPolicy.BEST_EFFORT,
                                history=QoSHistoryPolicy.KEEP_LAST, depth=5)
        self.create_subscription(PointCloud2, self.input_topic, self.cloud_callback, sensor_qos)
        self.get_logger().info(f'Range probe listening to {self.input_topic}, min_snr={self.min_snr}')

    def parse_cloud(self, msg):
        fields = {f.name: f.offset for f in msg.fields}
        if not all(k in fields for k in ('x', 'y', 'z')):
            return []
        data = bytes(msg.data)
        points = []
        for i in range(msg.width * msg.height):
            base = i * msg.point_step
            x = struct.unpack_from('<f', data, base + fields['x'])[0]
            y = struct.unpack_from('<f', data, base + fields['y'])[0]
            z = struct.unpack_from('<f', data, base + fields['z'])[0]
            doppler = struct.unpack_from('<f', data, base + fields['doppler'])[0] if 'doppler' in fields else 0.0
            snr = struct.unpack_from('<f', data, base + fields['snr'])[0] if 'snr' in fields else 0.0
            noise = struct.unpack_from('<f', data, base + fields['noise'])[0] if 'noise' in fields else 0.0
            if not all(math.isfinite(v) for v in [x, y, z, doppler, snr, noise]):
                continue
            r = math.sqrt(x*x + y*y + z*z)
            if r < self.min_range or r > self.max_reasonable_range or snr < self.min_snr:
                continue
            # +X is boresight, +Y is lateral.
            az = math.degrees(math.atan2(y, x))
            el = math.degrees(math.atan2(z, math.sqrt(x*x + y*y)))
            points.append({
                'x': x, 'y': y, 'z': z, 'range': r,
                'snr': snr, 'noise': noise, 'doppler': doppler,
                'az': az, 'el': el,
            })
        return points

    def cloud_callback(self, msg):
        points = self.parse_cloud(msg)
        if not points:
            self.get_logger().info('No valid points')
            return
        if self.sort_by == 'snr':
            points_sorted = sorted(points, key=lambda p: p['snr'], reverse=True)
        elif self.sort_by == 'abs_doppler':
            points_sorted = sorted(points, key=lambda p: abs(p['doppler']), reverse=True)
        else:
            points_sorted = sorted(points, key=lambda p: p['range'], reverse=True)
        top = points_sorted[0]
        self.get_logger().info(
            f'TOP({self.sort_by}): r={top["range"]:.2f} m | SNR={top["snr"]:.1f} dB | '
            f'noise={top["noise"]:.1f} dB | Doppler={top["doppler"]:+.2f} m/s | '
            f'Az={top["az"]:+.1f} deg | El={top["el"]:+.1f} deg | Points={len(points)}'
        )
        for i, p in enumerate(points_sorted[:self.print_top_n]):
            self.get_logger().info(
                f'  #{i+1}: r={p["range"]:.2f} m, snr={p["snr"]:.1f}, noise={p["noise"]:.1f}, '
                f'dop={p["doppler"]:+.2f}, az={p["az"]:+.1f}, el={p["el"]:+.1f}, '
                f'x={p["x"]:+.2f}, y={p["y"]:+.2f}, z={p["z"]:+.2f}'
            )


def main(args=None):
    rclpy.init(args=args)
    node = RangeProbeNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
