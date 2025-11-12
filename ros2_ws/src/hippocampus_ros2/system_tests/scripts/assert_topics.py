#!/usr/bin/env python3
"""System test helper that validates topic activity and bounds."""
from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass
from typing import Optional

import rclpy
from geometry_msgs.msg import Twist
from rclpy.executors import SingleThreadedExecutor
from rclpy.logging import get_logger
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray


@dataclass
class TestConfig:
    snn_action_topic: str
    cmd_vel_topic: str
    timeout: float
    min_messages: int
    max_linear: float
    max_angular: float
    spin_period: float


class TopicAssertions(Node):
    """Subscribe to action / cmd_vel topics and enforce constraints."""

    def __init__(self, config: TestConfig) -> None:
        super().__init__("brain_system_test_assertions")
        self._config = config
        self._action_count = 0
        self._cmd_vel_count = 0
        self._fail_reason: Optional[str] = None
        self._result: Optional[bool] = None
        self._action_subscription = self.create_subscription(
            Float32MultiArray,
            self._config.snn_action_topic,
            self._handle_action,
            10,
        )
        self._cmd_vel_subscription = self.create_subscription(
            Twist,
            self._config.cmd_vel_topic,
            self._handle_cmd_vel,
            10,
        )
        self._deadline = self.get_clock().now() + rclpy.time.Duration(
            seconds=self._config.timeout
        )
        self._timer = self.create_timer(self._config.spin_period, self._evaluate)

    def _handle_action(self, msg: Float32MultiArray) -> None:
        self._action_count += 1
        if self._fail_reason:
            return
        if not msg.data:
            self._fail_reason = "snn_action message is empty"
        elif len(msg.data) < 2:
            self._fail_reason = f"snn_action has {len(msg.data)} elements, expected >= 2"

    def _handle_cmd_vel(self, msg: Twist) -> None:
        self._cmd_vel_count += 1
        if self._fail_reason:
            return
        if not self._is_within_limits(msg):
            self._fail_reason = (
                "cmd_vel out of bounds: "
                f"linear.x={msg.linear.x:.3f}, angular.z={msg.angular.z:.3f}"
            )

    def _is_within_limits(self, msg: Twist) -> bool:
        linear_mag = abs(float(msg.linear.x))
        angular_mag = abs(float(msg.angular.z))
        if math.isfinite(self._config.max_linear) and linear_mag > self._config.max_linear + 1e-6:
            return False
        if math.isfinite(self._config.max_angular) and angular_mag > self._config.max_angular + 1e-6:
            return False
        return True

    def _evaluate(self) -> None:
        if self._result is not None:
            return

        now = self.get_clock().now()
        if self._fail_reason:
            self.get_logger().error("Test failed: %s", self._fail_reason)
            self._result = False
            self._timer.cancel()
            return

        if (
            self._action_count >= self._config.min_messages
            and self._cmd_vel_count >= self._config.min_messages
        ):
            self.get_logger().info(
                "Test passed: received %d snn_action and %d cmd_vel messages.",
                self._action_count,
                self._cmd_vel_count,
            )
            self._result = True
            self._timer.cancel()
            return

        if now >= self._deadline:
            self._fail_reason = (
                f"Timed out after {self._config.timeout:.1f}s "
                f"waiting for {self._config.min_messages} messages "
                f"(snn_action={self._action_count}, cmd_vel={self._cmd_vel_count})."
            )
            self.get_logger().error("Test failed: %s", self._fail_reason)
            self._result = False
            self._timer.cancel()

    @property
    def result(self) -> Optional[bool]:
        return self._result

    @property
    def fail_reason(self) -> Optional[str]:
        return self._fail_reason


def parse_arguments(argv: list[str]) -> TestConfig:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snn-action-topic", default="/snn_action")
    parser.add_argument("--cmd-vel-topic", default="/cmd_vel")
    parser.add_argument("--timeout", type=float, default=20.0)
    parser.add_argument("--min-messages", type=int, default=5)
    parser.add_argument("--max-linear", type=float, default=0.3)
    parser.add_argument("--max-angular", type=float, default=1.0)
    parser.add_argument("--spin-period", type=float, default=0.5)
    args = parser.parse_args(argv)

    if args.min_messages <= 0:
        parser.error("--min-messages must be positive")
    if args.timeout <= 0.0:
        parser.error("--timeout must be positive")
    if args.spin_period <= 0.0:
        parser.error("--spin-period must be positive")

    return TestConfig(
        snn_action_topic=args.snn_action_topic,
        cmd_vel_topic=args.cmd_vel_topic,
        timeout=args.timeout,
        min_messages=args.min_messages,
        max_linear=args.max_linear,
        max_angular=args.max_angular,
        spin_period=args.spin_period,
    )


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_arguments(argv if argv is not None else sys.argv[1:])
    rclpy.init()
    node = TopicAssertions(args)
    spin_period = args.spin_period
    executor = SingleThreadedExecutor()
    executor.add_node(node)

    exit_code = 0
    try:
        while rclpy.ok() and node.result is None:
            executor.spin_once(timeout_sec=spin_period)
    except Exception as exc:  # pragma: no cover - defensive
        node.get_logger().exception("Unexpected error while spinning: %s", exc)
        exit_code = 1
    result = node.result
    fail_reason = node.fail_reason
    logger = get_logger("brain_system_test_assertions")
    finally:
        executor.remove_node(node)
        node.destroy_node()
        rclpy.shutdown()

    if exit_code != 0:
        return exit_code

    if result is True:
        logger.info("Topic assertions satisfied.")
        return 0

    if fail_reason:
        logger.error("Assertion check failed: %s", fail_reason)
    else:
        logger.error("Assertion check failed: unknown reason")
    return 1


if __name__ == "__main__":
    sys.exit(main())

