"""ROS 2 node for publishing mission goals."""
from __future__ import annotations

import rclpy
from rclpy.node import Node
from std_msgs.msg import Header
from builtin_interfaces.msg import Time

try:
    from hippocampus_ros2_msgs.msg import MissionGoal as MissionGoalMsg
    MSGS_AVAILABLE = True
except ImportError:
    MSGS_AVAILABLE = False
    # Fallback: use geometry_msgs for now
    from geometry_msgs.msg import Point
    MissionGoalMsg = None

from geometry_msgs.msg import Point


class MissionPublisher(Node):
    """ROS 2 node that publishes mission goals.

    This node can be used to:
    - Set goals programmatically
    - Accept goals from command line
    - Integrate with mission planners
    """

    def __init__(self) -> None:
        super().__init__("mission_publisher")

        self.declare_parameter("goal_topic", "/mission/goal")
        self.declare_parameter("goal_type", "point")  # "point", "node", "region"
        self.declare_parameter("goal_x", 0.9)
        self.declare_parameter("goal_y", 0.9)
        self.declare_parameter("goal_z", 0.0)
        self.declare_parameter("tolerance", 0.1)
        self.declare_parameter("node_id", 0)
        self.declare_parameter("region_radius", 0.5)
        self.declare_parameter("frame_id", "map")
        self.declare_parameter("timeout", 0.0)  # 0.0 = no timeout
        self.declare_parameter("publish_once", True)
        self.declare_parameter("publish_rate", 1.0)  # Hz

        goal_topic = self.get_parameter("goal_topic").get_parameter_value().string_value

        if MSGS_AVAILABLE:
            self._goal_publisher = self.create_publisher(MissionGoalMsg, goal_topic, 10)
        else:
            self.get_logger().warn(
                "hippocampus_ros2_msgs not available. Install and build the message package."
            )
            # Fallback: publish Point for now
            self._goal_publisher = self.create_publisher(Point, goal_topic, 10)

        publish_once = self.get_parameter("publish_once").get_parameter_value().bool_value
        publish_rate = self.get_parameter("publish_rate").get_parameter_value().double_value

        if publish_once:
            # Publish once on startup
            self._timer = self.create_timer(0.1, self._publish_once_callback)
            self._published = False
        else:
            # Publish periodically
            timer_period = 1.0 / publish_rate if publish_rate > 0.0 else 1.0
            self._timer = self.create_timer(timer_period, self._publish_periodic_callback)

        self.get_logger().info(
            f"Mission publisher ready. Publishing to '{goal_topic}' "
            f"({'once' if publish_once else f'at {publish_rate} Hz'})"
        )

    def _publish_once_callback(self) -> None:
        """Publish goal once."""
        if self._published:
            self._timer.cancel()
            return

        self._publish_goal()
        self._published = True
        self.get_logger().info("Mission goal published")

    def _publish_periodic_callback(self) -> None:
        """Publish goal periodically."""
        self._publish_goal()

    def _publish_goal(self) -> None:
        """Publish the configured goal."""
        if not MSGS_AVAILABLE:
            # Fallback: publish Point
            point = Point()
            point.x = float(self.get_parameter("goal_x").get_parameter_value().double_value)
            point.y = float(self.get_parameter("goal_y").get_parameter_value().double_value)
            point.z = float(self.get_parameter("goal_z").get_parameter_value().double_value)
            self._goal_publisher.publish(point)
            return

        goal_type_str = self.get_parameter("goal_type").get_parameter_value().string_value
        goal_x = self.get_parameter("goal_x").get_parameter_value().double_value
        goal_y = self.get_parameter("goal_y").get_parameter_value().double_value
        goal_z = self.get_parameter("goal_z").get_parameter_value().double_value
        tolerance = self.get_parameter("tolerance").get_parameter_value().double_value
        node_id = self.get_parameter("node_id").get_parameter_value().integer_value
        region_radius = self.get_parameter("region_radius").get_parameter_value().double_value
        frame_id = self.get_parameter("frame_id").get_parameter_value().string_value
        timeout = self.get_parameter("timeout").get_parameter_value().double_value

        msg = MissionGoalMsg()
        msg.stamp = self.get_clock().now().to_msg()
        msg.frame_id = frame_id
        msg.timeout = timeout

        # Set goal type
        if goal_type_str == "point":
            msg.goal_type = MissionGoalMsg.GOAL_TYPE_POINT
            msg.point_position = Point(x=float(goal_x), y=float(goal_y), z=float(goal_z))
            msg.point_tolerance = float(tolerance)
        elif goal_type_str == "node":
            msg.goal_type = MissionGoalMsg.GOAL_TYPE_NODE
            msg.node_id = node_id
        elif goal_type_str == "region":
            msg.goal_type = MissionGoalMsg.GOAL_TYPE_REGION
            msg.region_center = Point(x=float(goal_x), y=float(goal_y), z=float(goal_z))
            msg.region_radius = float(region_radius)
        else:
            self.get_logger().error(f"Unknown goal type: {goal_type_str}")
            return

        self._goal_publisher.publish(msg)
        self.get_logger().debug(f"Published {goal_type_str} goal: ({goal_x}, {goal_y}, {goal_z})")


def main() -> None:
    rclpy.init()
    node = MissionPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()

