"""ROS 2 brain node that feeds pose observations into an SNNController."""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import rclpy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray

from hippocampus_core.controllers.place_cell_controller import (
    PlaceCellController,
    PlaceCellControllerConfig,
)
from hippocampus_core.env import Environment


class BrainNode(Node):
    """ROS 2 node that converts pose updates into controller actions."""

    def __init__(self) -> None:
        super().__init__("snn_brain_node")

        self.declare_parameter("controller_backend", "place_cells")
        self.declare_parameter("pose_topic", "/odom")
        self.declare_parameter("cmd_vel_topic", "/cmd_vel")
        self.declare_parameter("control_rate", 10.0)
        self.declare_parameter("max_linear", 0.3)
        self.declare_parameter("max_angular", 1.0)
        self.declare_parameter("log_every_n_cycles", 10)
        self.declare_parameter("arena_width", 1.0)
        self.declare_parameter("arena_height", 1.0)
        self.declare_parameter("rng_seed", 1234)

        self._controller_backend = (
            self.get_parameter("controller_backend").get_parameter_value().string_value
        )
        self._pose_topic = self.get_parameter("pose_topic").get_parameter_value().string_value
        self._cmd_vel_topic = (
            self.get_parameter("cmd_vel_topic").get_parameter_value().string_value
        )
        control_rate = self.get_parameter("control_rate").get_parameter_value().double_value
        self._control_rate = float(control_rate if control_rate > 0.0 else 10.0)
        self._max_linear = float(
            self.get_parameter("max_linear").get_parameter_value().double_value or 0.3
        )
        self._max_angular = float(
            self.get_parameter("max_angular").get_parameter_value().double_value or 1.0
        )
        log_every_n = self.get_parameter("log_every_n_cycles").get_parameter_value().integer_value
        self._log_every_n_cycles = max(int(log_every_n) if log_every_n else 10, 1)
        arena_width = self.get_parameter("arena_width").get_parameter_value().double_value
        arena_height = self.get_parameter("arena_height").get_parameter_value().double_value
        rng_seed = self.get_parameter("rng_seed").get_parameter_value().integer_value

        if self._controller_backend != "place_cells":
            self.get_logger().warning(
                "Controller backend '%s' not supported yet. Falling back to place_cells.",
                self._controller_backend,
            )

        self._environment = Environment(width=arena_width, height=arena_height)
        controller_config = PlaceCellControllerConfig()
        controller_rng = np.random.default_rng(int(rng_seed))
        self._controller = PlaceCellController(
            environment=self._environment,
            config=controller_config,
            rng=controller_rng,
        )

        self._pose_subscription = self.create_subscription(
            Odometry,
            self._pose_topic,
            self._pose_callback,
            10,
        )
        self._action_publisher = self.create_publisher(Float32MultiArray, "snn_action", 10)
        self._cmd_vel_publisher = self.create_publisher(Twist, self._cmd_vel_topic, 10)

        timer_period = 1.0 / self._control_rate if self._control_rate > 0.0 else 0.1
        self._timer = self.create_timer(timer_period, self._control_timer_callback)

        self._last_pose: Optional[Tuple[float, float]] = None
        self._last_time: Optional[float] = None
        self._step_count = 0
        self._warned_short_action = False

        self.get_logger().info(
            "Brain node ready. Subscribing to '%s', publishing actions on 'snn_action' "
            "and Twist on '%s'.",
            self._pose_topic,
            self._cmd_vel_topic,
        )

    def _pose_callback(self, msg: Odometry) -> None:
        position = msg.pose.pose.position
        self._last_pose = (float(position.x), float(position.y))

    def _control_timer_callback(self) -> None:
        if self._last_pose is None:
            return

        now = self.get_clock().now().nanoseconds / 1e9
        if self._last_time is None:
            dt = 1.0 / self._control_rate if self._control_rate > 0.0 else 0.1
        else:
            dt = max(now - self._last_time, 1e-6)
        self._last_time = now

        obs = np.asarray(self._last_pose, dtype=np.float32)
        action = self._controller.step(obs, dt)
        if action.size < 2:
            if not self._warned_short_action:
                self.get_logger().warning(
                    "Controller returned %d action elements; need at least 2 for cmd_vel.",
                    action.size,
                )
                self._warned_short_action = True
            return
        self._warned_short_action = False

        action_msg = Float32MultiArray()
        action_msg.data = [float(x) for x in action]
        self._action_publisher.publish(action_msg)

        twist = Twist()
        # Map controller action into a Twist command (see ROS 2 minimal publisher tutorial).
        linear_x = float(np.clip(action[0], -self._max_linear, self._max_linear))
        angular_z = float(np.clip(action[1], -self._max_angular, self._max_angular))
        twist.linear.x = linear_x
        twist.linear.y = 0.0
        twist.linear.z = 0.0
        twist.angular.x = 0.0
        twist.angular.y = 0.0
        twist.angular.z = angular_z
        self._cmd_vel_publisher.publish(twist)

        self._step_count += 1
        if self._step_count % self._log_every_n_cycles == 0:
            self.get_logger().info(
                "Cycle %d | action=%s | cmd_vel=(%.3f m/s, %.3f rad/s)",
                self._step_count,
                action,
                linear_x,
                angular_z,
            )


def main() -> None:
    rclpy.init()
    node = BrainNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
