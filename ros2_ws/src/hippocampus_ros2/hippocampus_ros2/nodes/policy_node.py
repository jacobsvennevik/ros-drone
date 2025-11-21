"""ROS 2 policy node integrating SpikingPolicyService with ROS topics."""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple
import math

import numpy as np
import rclpy
from geometry_msgs.msg import Point, Twist
from nav_msgs.msg import Odometry
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray, Header

from hippocampus_core.controllers.place_cell_controller import (
    PlaceCellController,
    PlaceCellControllerConfig,
)
try:
    from hippocampus_core.controllers.bat_navigation_controller import (
        BatNavigationController,
        BatNavigationControllerConfig,
    )
except ImportError:
    BatNavigationController = None
    BatNavigationControllerConfig = None
from hippocampus_core.env import Environment
from hippocampus_core.policy import (
    TopologyService,
    SpatialFeatureService,
    SpikingPolicyService,
    ActionArbitrationSafety,
    GraphNavigationService,
    RobotState,
    Mission,
    MissionGoal,
    GoalType,
    PointGoal,
    PolicySNN,
)

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


class PolicyNode(Node):
    """ROS 2 node that integrates the SNN Policy Service with ROS topics.

    This node:
    - Subscribes to `/odom` (robot pose)
    - Subscribes to `/mission/goal` (optional, for mission goals)
    - Publishes `/cmd_vel` (velocity commands)
    - Publishes `/policy/decision` (policy decisions)
    - Publishes `/policy/status` (diagnostics)
    """

    def __init__(self) -> None:
        super().__init__("policy_node")

        # Declare parameters
        self.declare_parameter("controller_backend", "place_cells")
        self.declare_parameter("pose_topic", "/odom")
        self.declare_parameter("cmd_vel_topic", "/cmd_vel")
        self.declare_parameter("mission_topic", "/mission/goal")
        self.declare_parameter("control_rate", 10.0)
        self.declare_parameter("max_linear", 0.3)
        self.declare_parameter("max_angular", 1.0)
        self.declare_parameter("max_vertical", 0.2)
        self.declare_parameter("arena_width", 1.0)
        self.declare_parameter("arena_height", 1.0)
        self.declare_parameter("rng_seed", 1234)
        self.declare_parameter("enable_hierarchical", False)
        self.declare_parameter("navigation_algorithm", "dijkstra")
        self.declare_parameter("use_snn", False)
        self.declare_parameter("snn_model_path", "")
        self.declare_parameter("snn_feature_dim", 44)
        self.declare_parameter("snn_hidden_dim", 64)
        self.declare_parameter("is_3d", False)
        self.declare_parameter("default_goal_x", 0.9)
        self.declare_parameter("default_goal_y", 0.9)
        self.declare_parameter("default_goal_z", 0.0)

        # Get parameters
        self._controller_backend = (
            self.get_parameter("controller_backend").get_parameter_value().string_value
        )
        self._pose_topic = self.get_parameter("pose_topic").get_parameter_value().string_value
        self._cmd_vel_topic = (
            self.get_parameter("cmd_vel_topic").get_parameter_value().string_value
        )
        self._mission_topic = (
            self.get_parameter("mission_topic").get_parameter_value().string_value
        )
        control_rate = self.get_parameter("control_rate").get_parameter_value().double_value
        self._control_rate = float(control_rate if control_rate > 0.0 else 10.0)
        self._max_linear = float(
            self.get_parameter("max_linear").get_parameter_value().double_value or 0.3
        )
        self._max_angular = float(
            self.get_parameter("max_angular").get_parameter_value().double_value or 1.0
        )
        self._max_vertical = float(
            self.get_parameter("max_vertical").get_parameter_value().double_value or 0.2
        )
        arena_width = self.get_parameter("arena_width").get_parameter_value().double_value
        arena_height = self.get_parameter("arena_height").get_parameter_value().double_value
        rng_seed = self.get_parameter("rng_seed").get_parameter_value().integer_value
        enable_hierarchical = (
            self.get_parameter("enable_hierarchical").get_parameter_value().bool_value
        )
        nav_algorithm = (
            self.get_parameter("navigation_algorithm").get_parameter_value().string_value
        )
        use_snn = self.get_parameter("use_snn").get_parameter_value().bool_value
        snn_model_path = (
            self.get_parameter("snn_model_path").get_parameter_value().string_value or ""
        )
        snn_feature_dim = (
            self.get_parameter("snn_feature_dim").get_parameter_value().integer_value or 44
        )
        snn_hidden_dim = (
            self.get_parameter("snn_hidden_dim").get_parameter_value().integer_value or 64
        )
        is_3d = self.get_parameter("is_3d").get_parameter_value().bool_value

        # Default goal
        default_goal_x = (
            self.get_parameter("default_goal_x").get_parameter_value().double_value or 0.9
        )
        default_goal_y = (
            self.get_parameter("default_goal_y").get_parameter_value().double_value or 0.9
        )
        default_goal_z = (
            self.get_parameter("default_goal_z").get_parameter_value().double_value or 0.0
        )

        # Initialize controller (for topology)
        controller_rng = np.random.default_rng(int(rng_seed))
        self._environment = Environment(width=arena_width, height=arena_height)
        
        if self._controller_backend == "bat_navigation" and BatNavigationController is not None:
            controller_config = BatNavigationControllerConfig(
                num_place_cells=80,
                hd_num_neurons=72,
                grid_size=(16, 16),
                calibration_interval=250,
                integration_window=None,  # Disable for real-time ROS operation
            )
            self._place_controller = BatNavigationController(
                environment=self._environment,
                config=controller_config,
                rng=controller_rng,
            )
            self.get_logger().info(
                "Using bat navigation controller (HD neurons=%d, grid size=%s).",
                controller_config.hd_num_neurons,
                controller_config.grid_size,
            )
        else:
            if self._controller_backend == "bat_navigation":
                self.get_logger().warning(
                    "Bat navigation requested but not available; falling back to place_cells."
                )
            controller_config = PlaceCellControllerConfig()
            self._place_controller = PlaceCellController(
                environment=self._environment,
                config=controller_config,
                rng=controller_rng,
            )

        # Initialize policy services
        self._topology_service = TopologyService()
        self._feature_service = SpatialFeatureService(
            self._topology_service,
            k_neighbors=12 if is_3d else 8,
            is_3d=is_3d,
        )

        # Initialize SNN model if requested
        snn_model = None
        if use_snn and TORCH_AVAILABLE:
            if snn_model_path:
                # Load from checkpoint (would need checkpoint loading logic)
                self.get_logger().warn("SNN checkpoint loading not yet implemented")
            else:
                # Create new model
                snn_model = PolicySNN(
                    feature_dim=snn_feature_dim,
                    hidden_dim=snn_hidden_dim,
                    output_dim=3 if is_3d else 2,
                )
                self.get_logger().info("Created new SNN model (randomly initialized)")

        # Initialize navigation service if hierarchical planning enabled
        navigation_service = None
        if enable_hierarchical:
            navigation_service = GraphNavigationService(algorithm=nav_algorithm)
            self.get_logger().info(f"Hierarchical planning enabled (algorithm: {nav_algorithm})")

        # Initialize policy service
        self._policy_service = SpikingPolicyService(
            self._feature_service,
            config={
                "max_linear": self._max_linear,
                "max_angular": self._max_angular,
                "max_vertical": self._max_vertical if is_3d else None,
            },
            snn_model=snn_model,
            navigation_service=navigation_service,
        )

        # Initialize safety arbitrator
        self._safety_arbitrator = ActionArbitrationSafety(
            max_linear=self._max_linear,
            max_angular=self._max_angular,
            max_vertical=self._max_vertical if is_3d else None,
        )

        # Default mission
        if is_3d:
            self._default_mission = Mission(
                goal=MissionGoal(
                    type=GoalType.POINT,
                    value=PointGoal(position=(default_goal_x, default_goal_y, default_goal_z)),
                )
            )
        else:
            self._default_mission = Mission(
                goal=MissionGoal(
                    type=GoalType.POINT,
                    value=PointGoal(position=(default_goal_x, default_goal_y)),
                )
            )
        self._current_mission = self._default_mission

        # ROS subscriptions
        self._pose_subscription = self.create_subscription(
            Odometry,
            self._pose_topic,
            self._pose_callback,
            10,
        )

        # ROS publishers
        self._cmd_vel_publisher = self.create_publisher(Twist, self._cmd_vel_topic, 10)
        self._action_publisher = self.create_publisher(Float32MultiArray, "policy_action", 10)
        
        # Policy message publishers (if available)
        try:
            from hippocampus_ros2_msgs.msg import PolicyDecision as PolicyDecisionMsg, PolicyStatus as PolicyStatusMsg
            self._decision_publisher = self.create_publisher(PolicyDecisionMsg, "policy/decision", 10)
            self._status_publisher = self.create_publisher(PolicyStatusMsg, "policy/status", 10)
            self._msgs_available = True
        except ImportError:
            self._decision_publisher = None
            self._status_publisher = None
            self._msgs_available = False
            self.get_logger().warn("hippocampus_ros2_msgs not available. Install message package for full functionality.")
        
        # Graph visualization
        self.declare_parameter("enable_viz", False)
        self.declare_parameter("viz_rate_hz", 2.0)
        self._viz_enabled = self.get_parameter("enable_viz").get_parameter_value().bool_value
        viz_rate_hz = self.get_parameter("viz_rate_hz").get_parameter_value().double_value or 2.0
        
        if self._viz_enabled:
            from visualization_msgs.msg import MarkerArray
            self._graph_marker_publisher = self.create_publisher(MarkerArray, "policy/graph", 10)
            self._waypoint_marker_publisher = self.create_publisher(MarkerArray, "policy/waypoint", 10)
            viz_timer_period = 1.0 / max(float(viz_rate_hz), 0.1)
            self._viz_timer = self.create_timer(viz_timer_period, self._publish_visualization)
        else:
            self._graph_marker_publisher = None
            self._waypoint_marker_publisher = None
            self._viz_timer = None

        # Control timer
        timer_period = 1.0 / self._control_rate if self._control_rate > 0.0 else 0.1
        self._timer = self.create_timer(timer_period, self._control_timer_callback)

        # State
        self._last_pose: Optional[Tuple[float, float, float]] = None  # (x, y, yaw) or (x, y, z, yaw, pitch) for 3D
        self._last_time: Optional[float] = None
        self._last_msg_timestamp = None  # For latency compensation
        self._prev_obs_position: Optional[np.ndarray] = None  # For latency compensation
        self._prev_obs_heading: Optional[float] = None  # For latency compensation
        self._step_count = 0
        self._topology_update_counter = 0

        self.get_logger().info(
            "Policy node ready. Subscribing to '%s', publishing on '%s'.",
            self._pose_topic,
            self._cmd_vel_topic,
        )

    def _pose_callback(self, msg: Odometry) -> None:
        """Handle odometry updates."""
        position = msg.pose.pose.position
        orientation = msg.pose.pose.orientation
        yaw = self._quat_to_yaw(
            orientation.x,
            orientation.y,
            orientation.z,
            orientation.w,
        )

        # Store pose (2D for now, can extend to 3D)
        self._last_pose = (float(position.x), float(position.y), yaw)
        # Store message timestamp for latency compensation
        self._last_msg_timestamp = msg.header.stamp

    def _control_timer_callback(self) -> None:
        """Control loop timer callback."""
        if self._last_pose is None:
            return

        # Update topology periodically
        self._topology_update_counter += 1
        if self._topology_update_counter % 10 == 0:
            self._topology_service.update_from_controller(self._place_controller)

        # Compute dt
        now = self.get_clock().now().nanoseconds / 1e9
        if self._last_time is None:
            dt = 1.0 / self._control_rate if self._control_rate > 0.0 else 0.1
        else:
            dt = max(now - self._last_time, 1e-6)
        self._last_time = now

        # Optional timestamp latency compensation
        # Apply correction if message is stale (latency > 20 ms)
        obs_position = np.array([self._last_pose[0], self._last_pose[1]])
        obs_heading = self._last_pose[2]
        
        if hasattr(self, '_last_msg_timestamp') and self._last_msg_timestamp:
            msg_time = self._last_msg_timestamp.sec + self._last_msg_timestamp.nanosec / 1e9
            latency = now - msg_time
            
            # Apply latency compensation if latency > 20 ms
            if latency > 0.02:  # 20 ms threshold
                # Estimate velocity from previous state
                if hasattr(self, '_prev_obs_position') and self._prev_obs_position is not None:
                    velocity = (obs_position - self._prev_obs_position) / dt if dt > 1e-6 else np.zeros(2)
                    # Apply correction: position += velocity * latency
                    obs_position = obs_position + velocity * latency
                
                if hasattr(self, '_prev_obs_heading') and self._prev_obs_heading is not None:
                    angular_velocity = (obs_heading - self._prev_obs_heading) / dt if dt > 1e-6 else 0.0
                    obs_heading = obs_heading + angular_velocity * latency
            
            self._prev_obs_position = obs_position.copy()
            self._prev_obs_heading = obs_heading

        # Update controller
        if self._controller_backend == "bat_navigation":
            # Bat controller requires [x, y, theta] observation
            obs = np.array([obs_position[0], obs_position[1], obs_heading])
            self._place_controller.step(obs, dt)
        else:
            # Legacy place cell controller uses [x, y] position
            position = np.array([self._last_pose[0], self._last_pose[1]])
            self._place_controller.step(position, dt)

        # Build robot state
        robot_state = RobotState(
            pose=self._last_pose,
            time=now,
        )

        # Build features
        features, local_context = self._feature_service.build_features(
            robot_state,
            self._current_mission,
        )

        # Make policy decision
        decision = self._policy_service.decide(
            features,
            local_context,
            dt,
            mission=self._current_mission,
        )

        # Filter through safety
        graph_snapshot = self._topology_service.get_graph_snapshot(robot_state.time)
        safe_cmd = self._safety_arbitrator.filter(
            decision,
            robot_state,
            graph_snapshot,
            self._current_mission,
        )

        # Publish action
        action_msg = Float32MultiArray()
        action_msg.data = list(safe_cmd.cmd)
        self._action_publisher.publish(action_msg)

        # Publish cmd_vel
        twist = Twist()
        if len(safe_cmd.cmd) >= 2:
            twist.linear.x = float(safe_cmd.cmd[0])
            twist.angular.z = float(safe_cmd.cmd[1])
            if len(safe_cmd.cmd) >= 3:
                twist.linear.z = float(safe_cmd.cmd[2])  # Vertical velocity for 3D
        self._cmd_vel_publisher.publish(twist)

        # Publish policy messages (if available)
        if self._msgs_available and self._decision_publisher:
            from hippocampus_ros2_msgs.msg import PolicyDecision as PolicyDecisionMsg
            decision_msg = PolicyDecisionMsg()
            decision_msg.stamp = self.get_clock().now().to_msg()
            decision_msg.linear_x = float(safe_cmd.cmd[0])
            decision_msg.angular_z = float(safe_cmd.cmd[1])
            if len(safe_cmd.cmd) >= 3:
                decision_msg.linear_z = float(safe_cmd.cmd[2])
            decision_msg.confidence = decision.confidence
            decision_msg.reason = decision.reason
            decision_msg.next_waypoint = decision.next_waypoint if decision.next_waypoint else -1
            self._decision_publisher.publish(decision_msg)

        # Publish status (periodically)
        if self._msgs_available and self._status_publisher and self._step_count % 10 == 0:
            from hippocampus_ros2_msgs.msg import PolicyStatus as PolicyStatusMsg
            status_msg = PolicyStatusMsg()
            status_msg.stamp = self.get_clock().now().to_msg()
            status_msg.is_active = True
            status_msg.using_snn = use_snn and self._policy_service._use_snn
            status_msg.hierarchical_enabled = enable_hierarchical
            status_msg.current_confidence = decision.confidence
            status_msg.current_reason = decision.reason
            status_msg.current_waypoint = decision.next_waypoint if decision.next_waypoint else -1
            status_msg.graph_nodes = len(graph_snapshot.V)
            status_msg.graph_edges = len(graph_snapshot.E)
            status_msg.graph_staleness_s = now - graph_snapshot.meta.last_updated
            status_msg.graph_stale = status_msg.graph_staleness_s > 5.0
            self._status_publisher.publish(status_msg)

        self._step_count += 1
        if self._step_count % 10 == 0:
            self.get_logger().info(
                "Cycle %d | action=%s | cmd_vel=(%.3f m/s, %.3f rad/s%s) | confidence=%.2f",
                self._step_count,
                safe_cmd.cmd,
                twist.linear.x,
                twist.angular.z,
                f", {twist.linear.z} m/s (vz)" if len(safe_cmd.cmd) >= 3 else "",
                decision.confidence,
            )

    def _publish_visualization(self) -> None:
        """Publish graph and waypoint visualization."""
        if not self._viz_enabled or self._graph_marker_publisher is None:
            return

        from visualization_msgs.msg import MarkerArray, Marker
        from geometry_msgs.msg import Point

        now = self.get_clock().now().to_msg()
        graph_snapshot = self._topology_service.get_graph_snapshot(
            self.get_clock().now().nanoseconds / 1e9
        )

        # Graph visualization
        graph_markers = MarkerArray()
        
        # Nodes
        for node in graph_snapshot.V:
            node_marker = Marker()
            node_marker.header.frame_id = "map"
            node_marker.header.stamp = now
            node_marker.ns = "graph_nodes"
            node_marker.id = node.node_id
            node_marker.type = Marker.SPHERE
            node_marker.action = Marker.ADD
            node_marker.pose.position.x = float(node.position[0])
            node_marker.pose.position.y = float(node.position[1])
            node_marker.pose.position.z = float(node.position[2]) if len(node.position) > 2 else 0.0
            node_marker.pose.orientation.w = 1.0
            node_marker.scale.x = 0.05
            node_marker.scale.y = 0.05
            node_marker.scale.z = 0.05
            node_marker.color.r = 0.2
            node_marker.color.g = 0.6
            node_marker.color.b = 1.0
            node_marker.color.a = 1.0
            graph_markers.markers.append(node_marker)

        # Edges
        edge_marker = Marker()
        edge_marker.header.frame_id = "map"
        edge_marker.header.stamp = now
        edge_marker.ns = "graph_edges"
        edge_marker.id = 0
        edge_marker.type = Marker.LINE_LIST
        edge_marker.action = Marker.ADD
        edge_marker.scale.x = 0.01
        edge_marker.color.r = 1.0
        edge_marker.color.g = 0.6
        edge_marker.color.b = 0.0
        edge_marker.color.a = 0.6

        for edge in graph_snapshot.E:
            u_node = next((n for n in graph_snapshot.V if n.node_id == edge.u), None)
            v_node = next((n for n in graph_snapshot.V if n.node_id == edge.v), None)
            if u_node and v_node:
                p1 = Point()
                p1.x = float(u_node.position[0])
                p1.y = float(u_node.position[1])
                p1.z = float(u_node.position[2]) if len(u_node.position) > 2 else 0.0
                p2 = Point()
                p2.x = float(v_node.position[0])
                p2.y = float(v_node.position[1])
                p2.z = float(v_node.position[2]) if len(v_node.position) > 2 else 0.0
                edge_marker.points.append(p1)
                edge_marker.points.append(p2)

        graph_markers.markers.append(edge_marker)
        self._graph_marker_publisher.publish(graph_markers)

        # Waypoint visualization
        if self._current_mission and self._waypoint_marker_publisher:
            waypoint_markers = MarkerArray()
            waypoint_marker = Marker()
            waypoint_marker.header.frame_id = "map"
            waypoint_marker.header.stamp = now
            waypoint_marker.ns = "waypoint"
            waypoint_marker.id = 0
            waypoint_marker.type = Marker.ARROW
            waypoint_marker.action = Marker.ADD
            
            goal = self._current_mission.goal
            if goal.type == GoalType.POINT:
                point_goal = goal.value
                waypoint_marker.pose.position.x = float(point_goal.position[0])
                waypoint_marker.pose.position.y = float(point_goal.position[1])
                waypoint_marker.pose.position.z = float(point_goal.position[2]) if len(point_goal.position) > 2 else 0.0
                waypoint_marker.pose.orientation.w = 1.0
                waypoint_marker.scale.x = 0.1
                waypoint_marker.scale.y = 0.05
                waypoint_marker.scale.z = 0.05
                waypoint_marker.color.r = 1.0
                waypoint_marker.color.g = 0.0
                waypoint_marker.color.b = 0.0
                waypoint_marker.color.a = 1.0
                waypoint_markers.markers.append(waypoint_marker)
            
            self._waypoint_marker_publisher.publish(waypoint_markers)

    @staticmethod
    def _quat_to_yaw(x: float, y: float, z: float, w: float) -> float:
        """Convert quaternion to yaw angle."""
        siny_cosp = 2.0 * (w * z + x * y)
        cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
        return math.atan2(siny_cosp, cosy_cosp)


def main() -> None:
    rclpy.init()
    node = PolicyNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()

