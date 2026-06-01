"""Main ROS 2 node for committed-target radar offboard control."""

from typing import List, Optional

from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy

from geometry_msgs.msg import PointStamped, PoseArray, TwistStamped
from mavros_msgs.msg import State
from nav_msgs.msg import Odometry
from std_msgs.msg import Float32, String

from .config import (
    ARRIVAL_RADIUS,
    COARSE_TARGET_VELOCITY,
    COMMITTED_TARGET_REACHED_M,
    DECEL_DISTANCE,
    EXPECTED_TARGET_GATE_M,
    MAX_COMMITTED_TARGET_AGE_S,
    RADAR_ACQUIRE_RANGE_M,
    RADAR_GUIDED_MAX_SPEED,
    RADAR_GUIDED_MIN_SPEED,
    RADAR_MOUNT_PITCH_DEG,
    RADAR_TIMEOUT_S,
    RATE,
    STOP_RANGE_M,
    TARGET_X,
    TARGET_Y,
    TARGET_Z,
)
from .transforms import body_to_local, radar_to_body
from .utils import clamp, make_velocity_msg, norm3


class CommittedTargetRadarOffboard(Node):
    ST_IDLE = "IDLE"
    ST_COARSE = "COARSE_APPROACH"
    ST_RADAR = "RADAR_GUIDED_COMMITTED"
    ST_STOP = "STOP_DETECTED"
    ST_HOLD = "WAYPOINT_HOLD"

    def __init__(self):
        super().__init__("committed_target_radar_offboard")

        qos_best = QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT)
        qos_rel = QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE)

        # Publishers
        self.pub_vel = self.create_publisher(
            TwistStamped,
            "/mavros/setpoint_velocity/cmd_vel",
            qos_best,
        )

        self.pub_detected_body = self.create_publisher(
            PointStamped,
            "/interceptor/detected_object_body",
            qos_rel,
        )

        self.pub_detected_local = self.create_publisher(
            PointStamped,
            "/interceptor/detected_object_local",
            qos_rel,
        )

        self.pub_committed_local = self.create_publisher(
            PointStamped,
            "/interceptor/committed_target_local",
            qos_rel,
        )

        self.pub_phase = self.create_publisher(
            String,
            "/interceptor/phase",
            qos_rel,
        )

        self.pub_selected_range = self.create_publisher(
            Float32,
            "/interceptor/radar_selected_range",
            qos_rel,
        )

        self.pub_expected_error = self.create_publisher(
            Float32,
            "/interceptor/expected_error",
            qos_rel,
        )

        self.pub_committed_age = self.create_publisher(
            Float32,
            "/interceptor/committed_target_age",
            qos_rel,
        )

        self.pub_committed_dist = self.create_publisher(
            Float32,
            "/interceptor/committed_target_distance",
            qos_rel,
        )

        # Subscribers
        self.create_subscription(
            State,
            "/mavros/state",
            self._state_cb,
            qos_best,
        )

        self.create_subscription(
            Odometry,
            "/mavros/local_position/odom",
            self._odom_cb,
            qos_best,
        )

        self.create_subscription(
            PoseArray,
            "/radar/candidate_poses_filter",
            self._candidates_cb,
            qos_rel,
        )

        # Drone state
        self.state = self.ST_IDLE
        self.offboard_seen = False

        self.pos = [0.0, 0.0, 0.0]
        self.vel = [0.0, 0.0, 0.0]
        self.q_body_to_local = [0.0, 0.0, 0.0, 1.0]

        # Latest accepted candidate state
        self.selected_radar: Optional[List[float]] = None
        self.selected_body: Optional[List[float]] = None
        self.selected_local: Optional[List[float]] = None
        self.selected_range = float("inf")
        self.selected_expected_error = float("inf")
        self.last_candidate_time = 0.0
        self.raw_candidate_count = 0
        self.accepted_candidate_count = 0

        # Committed target state
        self.committed_target_local: Optional[List[float]] = None
        self.committed_target_body: Optional[List[float]] = None
        self.committed_target_range = float("inf")
        self.committed_target_expected_error = float("inf")
        self.committed_target_time = 0.0

        self.create_timer(1.0 / RATE, self._tick)

        self.get_logger().info(
            f"Ready. ApproxTargetENU=({TARGET_X:.1f}, {TARGET_Y:.1f}, {TARGET_Z:.1f}) m | "
            f"ExpectedGate={EXPECTED_TARGET_GATE_M:.1f} m | "
            f"AcquireRange={RADAR_ACQUIRE_RANGE_M:.1f} m | StopRange={STOP_RANGE_M:.1f} m | "
            f"CommitReached={COMMITTED_TARGET_REACHED_M:.1f} m | "
            f"MaxCommitAge={MAX_COMMITTED_TARGET_AGE_S:.1f} s | "
            f"RadarPitch={RADAR_MOUNT_PITCH_DEG:.1f} deg"
        )

    # ──────────────────────────────────────────────────────────────────────────
    # Callbacks
    # ──────────────────────────────────────────────────────────────────────────

    def _state_cb(self, msg: State):
        if not self.offboard_seen and msg.mode == "OFFBOARD":
            self.offboard_seen = True
            self.state = self.ST_COARSE
            self.get_logger().info("OFFBOARD confirmed — entering COARSE_APPROACH.")

    def _odom_cb(self, msg: Odometry):
        p = msg.pose.pose.position
        v = msg.twist.twist.linear
        q = msg.pose.pose.orientation

        self.pos = [float(p.x), float(p.y), float(p.z)]
        self.vel = [float(v.x), float(v.y), float(v.z)]
        self.q_body_to_local = [float(q.x), float(q.y), float(q.z), float(q.w)]

    def _candidates_cb(self, msg: PoseArray):
        self.raw_candidate_count = len(msg.poses)
        self.accepted_candidate_count = 0

        if not msg.poses:
            return

        accepted = []

        for pose in msg.poses:
            x_r = float(pose.position.x)
            y_r = float(pose.position.y)
            z_r = float(pose.position.z)

            radar_vec = [x_r, y_r, z_r]
            radar_range = norm3(radar_vec)

            body_vec = list(radar_to_body(x_r, y_r, z_r))
            local_offset = body_to_local(self.q_body_to_local, body_vec)
            local_pos = [
                self.pos[0] + local_offset[0],
                self.pos[1] + local_offset[1],
                self.pos[2] + local_offset[2],
            ]

            expected_error = norm3([
                local_pos[0] - TARGET_X,
                local_pos[1] - TARGET_Y,
                local_pos[2] - TARGET_Z,
            ])

            if expected_error <= EXPECTED_TARGET_GATE_M:
                accepted.append({
                    "radar": radar_vec,
                    "body": body_vec,
                    "local": local_pos,
                    "range": radar_range,
                    "expected_error": expected_error,
                })

        self.accepted_candidate_count = len(accepted)

        if not accepted:
            return

        # Candidate selection policy:
        # 1. Must pass expected-position gate.
        # 2. Choose shortest radar range among accepted candidates.
        best = min(accepted, key=lambda c: c["range"])

        self.selected_radar = best["radar"]
        self.selected_body = best["body"]
        self.selected_local = best["local"]
        self.selected_range = best["range"]
        self.selected_expected_error = best["expected_error"]
        self.last_candidate_time = self.get_clock().now().nanoseconds * 1e-9

        # Commit/update the target every time a newer accepted candidate arrives.
        self.committed_target_local = list(best["local"])
        self.committed_target_body = list(best["body"])
        self.committed_target_range = float(best["range"])
        self.committed_target_expected_error = float(best["expected_error"])
        self.committed_target_time = self.last_candidate_time

        self._publish_selected_candidate()
        self._publish_committed_target()

    # ──────────────────────────────────────────────────────────────────────────
    # Publishing helpers
    # ──────────────────────────────────────────────────────────────────────────

    def _publish_selected_candidate(self):
        if self.selected_body is None or self.selected_local is None:
            return

        now = self.get_clock().now().to_msg()

        body_msg = PointStamped()
        body_msg.header.stamp = now
        body_msg.header.frame_id = "base_link"
        body_msg.point.x = float(self.selected_body[0])
        body_msg.point.y = float(self.selected_body[1])
        body_msg.point.z = float(self.selected_body[2])
        self.pub_detected_body.publish(body_msg)

        local_msg = PointStamped()
        local_msg.header.stamp = now
        local_msg.header.frame_id = "map"
        local_msg.point.x = float(self.selected_local[0])
        local_msg.point.y = float(self.selected_local[1])
        local_msg.point.z = float(self.selected_local[2])
        self.pub_detected_local.publish(local_msg)

        self.pub_selected_range.publish(Float32(data=float(self.selected_range)))
        self.pub_expected_error.publish(Float32(data=float(self.selected_expected_error)))

    def _publish_committed_target(self):
        if self.committed_target_local is None:
            return

        now = self.get_clock().now().to_msg()

        msg = PointStamped()
        msg.header.stamp = now
        msg.header.frame_id = "map"
        msg.point.x = float(self.committed_target_local[0])
        msg.point.y = float(self.committed_target_local[1])
        msg.point.z = float(self.committed_target_local[2])
        self.pub_committed_local.publish(msg)

        self.pub_committed_age.publish(Float32(data=float(self._committed_target_age())))
        self.pub_committed_dist.publish(Float32(data=float(self._dist_to_committed_target())))

    def _publish_phase(self):
        self.pub_phase.publish(String(data=self.state))

    def _stop(self):
        self.pub_vel.publish(make_velocity_msg(0.0, 0.0, 0.0))

    # ──────────────────────────────────────────────────────────────────────────
    # State helpers
    # ──────────────────────────────────────────────────────────────────────────

    def _candidate_fresh(self) -> bool:
        now = self.get_clock().now().nanoseconds * 1e-9
        return (now - self.last_candidate_time) <= RADAR_TIMEOUT_S

    def _has_accepted_candidate(self) -> bool:
        return (
            self.selected_body is not None
            and self.selected_local is not None
            and self._candidate_fresh()
            and self.selected_range <= RADAR_ACQUIRE_RANGE_M
            and self.selected_expected_error <= EXPECTED_TARGET_GATE_M
        )

    def _committed_target_age(self) -> float:
        if self.committed_target_local is None:
            return float("inf")
        now = self.get_clock().now().nanoseconds * 1e-9
        return now - self.committed_target_time

    def _committed_target_too_old(self) -> bool:
        if self.committed_target_local is None:
            return True
        return self._committed_target_age() > MAX_COMMITTED_TARGET_AGE_S

    def _dist_to_committed_target(self) -> float:
        if self.committed_target_local is None:
            return float("inf")
        return norm3([
            self.committed_target_local[0] - self.pos[0],
            self.committed_target_local[1] - self.pos[1],
            self.committed_target_local[2] - self.pos[2],
        ])

    def _actual_speed(self) -> float:
        return norm3(self.vel)

    def _dist_to_goal(self) -> float:
        return norm3([
            TARGET_X - self.pos[0],
            TARGET_Y - self.pos[1],
            TARGET_Z - self.pos[2],
        ])

    def _unit_to_goal(self) -> List[float]:
        d = [
            TARGET_X - self.pos[0],
            TARGET_Y - self.pos[1],
            TARGET_Z - self.pos[2],
        ]
        n = norm3(d)

        if n < 1e-3:
            return [0.0, 0.0, 0.0]

        return [d[0] / n, d[1] / n, d[2] / n]

    def _coarse_desired_speed(self, dist: float) -> float:
        max_decel = 5.0
        v_stop = (max(0.0, 2.0 * max_decel * dist)) ** 0.5

        if dist < DECEL_DISTANCE:
            return min(COARSE_TARGET_VELOCITY, v_stop)

        return COARSE_TARGET_VELOCITY

    def _committed_speed_limit(self, dist_to_committed: float) -> float:
        if dist_to_committed <= COMMITTED_TARGET_REACHED_M:
            return 0.0

        span = max(1e-3, RADAR_ACQUIRE_RANGE_M - COMMITTED_TARGET_REACHED_M)
        frac = clamp(
            (dist_to_committed - COMMITTED_TARGET_REACHED_M) / span,
            0.0,
            1.0,
        )

        v = frac * RADAR_GUIDED_MAX_SPEED
        return clamp(v, RADAR_GUIDED_MIN_SPEED, RADAR_GUIDED_MAX_SPEED)

    def _committed_target_velocity_local(self) -> List[float]:
        if self.committed_target_local is None:
            return [0.0, 0.0, 0.0]

        err = [
            self.committed_target_local[0] - self.pos[0],
            self.committed_target_local[1] - self.pos[1],
            self.committed_target_local[2] - self.pos[2],
        ]

        dist = norm3(err)
        if dist < 1e-3:
            return [0.0, 0.0, 0.0]

        speed = self._committed_speed_limit(dist)

        return [
            err[0] / dist * speed,
            err[1] / dist * speed,
            err[2] / dist * speed,
        ]

    # ──────────────────────────────────────────────────────────────────────────
    # Main state machine
    # ──────────────────────────────────────────────────────────────────────────

    def _tick(self):
        self._publish_phase()
        self._publish_committed_target()

        if self.state == self.ST_IDLE:
            self._stop()
            return

        # Fresh radar range stop has highest priority.
        if self._candidate_fresh() and self.selected_range <= STOP_RANGE_M:
            if self.state != self.ST_STOP:
                self.get_logger().warn(
                    f"STOP: fresh accepted candidate inside {STOP_RANGE_M:.1f} m | "
                    f"radar_range={self.selected_range:.2f} m | "
                    f"expected_error={self.selected_expected_error:.2f} m | "
                    f"body={self.selected_body}"
                )
            self.state = self.ST_STOP

        # Reached committed target stop condition.
        dist_committed = self._dist_to_committed_target()
        if self.state == self.ST_RADAR and dist_committed <= COMMITTED_TARGET_REACHED_M:
            self.get_logger().warn(
                f"STOP: reached committed target | dist={dist_committed:.2f} m | "
                f"target={self.committed_target_local}"
            )
            self.state = self.ST_STOP

        # Commit timeout safety.
        if self.state == self.ST_RADAR and self._committed_target_too_old():
            self.get_logger().warn(
                f"STOP: committed target too old | age={self._committed_target_age():.2f} s | "
                f"limit={MAX_COMMITTED_TARGET_AGE_S:.2f} s"
            )
            self.state = self.ST_STOP

        # Switch from coarse to radar-guided when a gated candidate is available.
        if self.state == self.ST_COARSE and self._has_accepted_candidate():
            self.state = self.ST_RADAR
            self.get_logger().warn(
                f"RADAR ACQUIRED: committed target set | "
                f"radar_range={self.selected_range:.2f} m | "
                f"expected_error={self.selected_expected_error:.2f} m | "
                f"body={self.selected_body} | local={self.selected_local}"
            )

        if self.state == self.ST_COARSE:
            dist = self._dist_to_goal()
            uv = self._unit_to_goal()
            spd = self._coarse_desired_speed(dist)

            self.pub_vel.publish(make_velocity_msg(
                uv[0] * spd,
                uv[1] * spd,
                uv[2] * spd,
            ))

            self.get_logger().info(
                f"[COARSE] dist_to_expected={dist:.1f} m | "
                f"cmd={spd:.1f} m/s | actual={self._actual_speed():.1f} m/s | "
                f"raw_candidates={self.raw_candidate_count} | "
                f"accepted_candidates={self.accepted_candidate_count} | "
                f"selected_range={self.selected_range:.1f} m | "
                f"expected_error={self.selected_expected_error:.1f} m",
                throttle_duration_sec=0.5,
            )

            if dist < ARRIVAL_RADIUS:
                self.state = self.ST_HOLD
                self.get_logger().info(
                    "Approx target reached without radar stop — entering WAYPOINT_HOLD."
                )

            return

        if self.state == self.ST_RADAR:
            if self.committed_target_local is None:
                self._stop()
                self.get_logger().warn(
                    "[RADAR] no committed target available — stopping.",
                    throttle_duration_sec=0.5,
                )
                return

            v_local = self._committed_target_velocity_local()

            self.pub_vel.publish(make_velocity_msg(
                v_local[0],
                v_local[1],
                v_local[2],
            ))

            self._publish_selected_candidate()
            self._publish_committed_target()

            self.get_logger().info(
                f"[RADAR] moving to committed target | "
                f"dist_committed={self._dist_to_committed_target():.2f} m | "
                f"age={self._committed_target_age():.2f} s | "
                f"fresh_candidate={self._candidate_fresh()} | "
                f"radar_range={self.selected_range:.2f} m | "
                f"expected_error={self.selected_expected_error:.2f} m | "
                f"accepted={self.accepted_candidate_count}/{self.raw_candidate_count} | "
                f"cmd_local=({v_local[0]:+.2f}, {v_local[1]:+.2f}, {v_local[2]:+.2f}) m/s",
                throttle_duration_sec=0.3,
            )

            return

        if self.state == self.ST_STOP:
            self._stop()
            self._publish_selected_candidate()
            self._publish_committed_target()

            self.get_logger().info(
                f"[STOP] holding | "
                f"committed_dist={self._dist_to_committed_target():.2f} m | "
                f"committed_age={self._committed_target_age():.2f} s | "
                f"radar_range={self.selected_range:.2f} m | "
                f"body={self.selected_body} | "
                f"committed_local={self.committed_target_local}",
                throttle_duration_sec=0.5,
            )

            return

        if self.state == self.ST_HOLD:
            # Slow hold near approximate waypoint if no radar stop occurs.
            err = [
                TARGET_X - self.pos[0],
                TARGET_Y - self.pos[1],
                TARGET_Z - self.pos[2],
            ]

            kp = 0.5

            vx = clamp(kp * err[0], -2.0, 2.0)
            vy = clamp(kp * err[1], -2.0, 2.0)
            vz = clamp(kp * err[2], -2.0, 2.0)

            self.pub_vel.publish(make_velocity_msg(vx, vy, vz))

            self.get_logger().info(
                f"[HOLD] pos=({self.pos[0]:.1f}, {self.pos[1]:.1f}, {self.pos[2]:.1f}) | "
                f"err=({err[0]:+.1f}, {err[1]:+.1f}, {err[2]:+.1f})",
                throttle_duration_sec=0.5,
            )

            return
