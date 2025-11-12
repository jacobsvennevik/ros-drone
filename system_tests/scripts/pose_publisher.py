#!/usr/bin/env python3
"""Deterministic odometry publisher for system tests."""
from __future__ import annotations

import math
from typing import Optional

import rclpy
from nav_msgs.msg import Odometry
from rclpy.node import Node
from std_msgs.msg import Header


class CircularOdometryPublisher(Node):
    """Publish a slow, perfectly circular odometry trajectory."""

    def __init__(self) -> None:
        super().__init__("circular_odometry_publisher")
        self.declare_parameter("use_sim_time", False)
        self.declare_parameter("pose_topic", "/odom")
        self.declare_parameter("radius", 0.3)
        self.declare_parameter("linear_speed", 0.05)
        self.declare_parameter("angular_speed", 0.2)
        self.declare_parameter("publish_rate", 10.0)

        self._topic = (
            self.get_parameter("pose_topic").get_parameter_value().string_value or "/odom"
        )
        radius = self.get_parameter("radius").get_parameter_value().double_value or 0.3
        lin_speed = (
            self.get_parameter("linear_speed").get_parameter_value().double_value or 0.05
        )
        ang_speed = (
            self.get_parameter("angular_speed").get_parameter_value().double_value or 0.2
        )
        publish_rate = (
            self.get_parameter("publish_rate").get_parameter_value().double_value or 10.0
        )

        self._radius = float(max(radius, 0.05))
        self._angular_speed = float(max(ang_speed, 0.01))
        self._linear_speed = float(max(lin_speed, 0.01))
        self._dt = 1.0 / float(max(publish_rate, 1.0))

        self._theta: float = 0.0
        self._publisher = self.create_publisher(Odometry, self._topic, 10)
        self._timer = self.create_timer(self._dt, self._publish_once)
        self._last_stamp: Optional[float] = None

        self.get_logger().info(
            "Publishing odometry to %s at %.2f Hz (radius=%.2f m, "
            "linear_speed=%.2f m/s, angular_speed=%.2f rad/s).",
            self._topic,
            1.0 / self._dt,
            self._radius,
            self._linear_speed,
            self._angular_speed,
        )

    def _publish_once(self) -> None:
        self._theta = (self._theta + self._angular_speed * self._dt) % (2.0 * math.pi)
        x = self._radius * math.cos(self._theta)
        y = self._radius * math.sin(self._theta)
        yaw = (self._theta + math.pi / 2.0) % (2.0 * math.pi)

        half_yaw = yaw / 2.0
        qz = math.sin(half_yaw)
        qw = math.cos(half_yaw)

        now = self.get_clock().now()
        msg = Odometry()
        msg.header = Header()
        msg.header.stamp = now.to_msg()
        msg.header.frame_id = "odom"
        msg.child_frame_id = "base_link"

        msg.pose.pose.position.x = float(x)
        msg.pose.pose.position.y = float(y)
        msg.pose.pose.orientation.z = float(qz)
        msg.pose.pose.orientation.w = float(qw)

        msg.twist.twist.linear.x = float(self._linear_speed)
        msg.twist.twist.angular.z = float(self._angular_speed)

        self._publisher.publish(msg)


def main() -> None:
    rclpy.init()
    node = CircularOdometryPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()

