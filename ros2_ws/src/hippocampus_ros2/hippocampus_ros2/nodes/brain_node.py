"""ROS 2 brain node that feeds pose observations into an SNNController."""
from __future__ import annotations

from collections import deque
from pathlib import Path
from typing import Deque, Optional, Tuple
import math

import numpy as np
import rclpy
from geometry_msgs.msg import Point, Twist
from nav_msgs.msg import Odometry
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray
from visualization_msgs.msg import Marker, MarkerArray

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
try:
    from hippocampus_core.controllers.snntorch_controller import SnnTorchController
except ImportError:
    SnnTorchController = None
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
        self.declare_parameter("enable_viz", False)
        self.declare_parameter("viz_rate_hz", 2.0)
        self.declare_parameter("viz_frame_id", "map")
        self.declare_parameter("viz_trail_length", 200)
        self.declare_parameter("use_bag_replay", False)
        self.declare_parameter("model_path", "")
        self.declare_parameter("use_cpu", True)
        self.declare_parameter("model_kind", "state_dict")
        self.declare_parameter("torchscript_path", "")

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
        self._viz_enabled = (
            self.get_parameter("enable_viz").get_parameter_value().bool_value
        )
        viz_rate_hz = (
            self.get_parameter("viz_rate_hz").get_parameter_value().double_value or 2.0
        )
        self._viz_rate_hz = max(float(viz_rate_hz), 0.1)
        self._viz_frame_id = (
            self.get_parameter("viz_frame_id").get_parameter_value().string_value or "map"
        )
        trail_length = (
            self.get_parameter("viz_trail_length").get_parameter_value().integer_value or 200
        )
        self._viz_trail_length = max(int(trail_length), 1)
        self._use_bag_replay = (
            self.get_parameter("use_bag_replay").get_parameter_value().bool_value
        )
        model_path_param = (
            self.get_parameter("model_path").get_parameter_value().string_value or ""
        )
        use_cpu_param = (
            self.get_parameter("use_cpu").get_parameter_value().bool_value
        )
        model_kind_param = (
            self.get_parameter("model_kind").get_parameter_value().string_value or "state_dict"
        )
        torchscript_path_param = (
            self.get_parameter("torchscript_path").get_parameter_value().string_value or ""
        )

        self._requested_backend = self._controller_backend
        self._backend_in_use = self._controller_backend
        self._environment: Optional[Environment] = None
        self._controller = None

        if self._requested_backend == "snntorch":
            self._controller = self._try_create_snntorch_controller(
                model_path=model_path_param,
                use_cpu=bool(use_cpu_param),
                model_kind=model_kind_param,
                torchscript_path=torchscript_path_param,
            )
            if self._controller is None:
                self.get_logger().warning(
                    "Falling back to place_cells backend after snnTorch initialisation failed."
                )
                self._backend_in_use = "place_cells"
            else:
                self._backend_in_use = "snntorch"

        if self._controller is None and self._requested_backend == "bat_navigation":
            self._controller = self._try_create_bat_controller(
                arena_width=arena_width,
                arena_height=arena_height,
                rng_seed=int(rng_seed),
            )
            if self._controller is None:
                self.get_logger().warning(
                    "Falling back to place_cells backend after bat navigation controller initialisation failed."
                )
                self._backend_in_use = "place_cells"
            else:
                self._backend_in_use = "bat_navigation"

        if self._controller is None:
            controller_rng = np.random.default_rng(int(rng_seed))
            self._environment = Environment(width=arena_width, height=arena_height)
            controller_config = PlaceCellControllerConfig()
            self._controller = PlaceCellController(
                environment=self._environment,
                config=controller_config,
                rng=controller_rng,
            )
            self._backend_in_use = "place_cells"

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
        self._last_heading: Optional[float] = None
        self._last_time: Optional[float] = None
        self._last_msg_timestamp = None  # For latency compensation
        self._prev_obs_position: Optional[np.ndarray] = None  # For latency compensation
        self._prev_obs_heading: Optional[float] = None  # For latency compensation
        self._step_count = 0
        self._warned_short_action = False
        self._pose_history: Deque[Tuple[float, float]] = deque(maxlen=self._viz_trail_length)
        self._marker_publisher = (
            self.create_publisher(MarkerArray, "brain_markers", 10) if self._viz_enabled else None
        )
        self._viz_timer = (
            self.create_timer(1.0 / self._viz_rate_hz, self._publish_markers)
            if self._viz_enabled
            else None
        )

        self.get_logger().info(
            "Brain node ready (backend=%s). Subscribing to '%s', publishing actions on 'snn_action' "
            "and Twist on '%s'.",
            self._backend_in_use,
            self._pose_topic,
            self._cmd_vel_topic,
        )
        if self._use_bag_replay:
            self.get_logger().info(
                "Bag replay mode enabled. Expecting external ros2 bag play to publish '%s'.",
                self._pose_topic,
            )

    def _try_create_snntorch_controller(
        self,
        *,
        model_path: str,
        use_cpu: bool,
        model_kind: str,
        torchscript_path: str,
    ):
        if SnnTorchController is None:
            self.get_logger().error("snnTorch backend requested but snnTorch is not installed.")
            return None

        if not model_path:
            self.get_logger().error("Parameter 'model_path' must point to a trained snnTorch checkpoint.")
            return None

        checkpoint_path = Path(model_path).expanduser()
        if not checkpoint_path.exists():
            self.get_logger().error("snnTorch checkpoint not found at '%s'.", checkpoint_path)
            return None

        model_kind_normalised = (model_kind or "state_dict").lower()
        if model_kind_normalised not in {"state_dict", "torchscript"}:
            self.get_logger().error(
                "Unknown model_kind '%s'. Expected 'state_dict' or 'torchscript'.",
                model_kind,
            )
            return None

        script_path: Optional[Path] = None
        if model_kind_normalised == "torchscript":
            if torchscript_path:
                script_path = Path(torchscript_path).expanduser()
            else:
                script_path = checkpoint_path.with_suffix(".ts")
            if not script_path.exists():
                self.get_logger().error(
                    "TorchScript model requested but module not found at '%s'.",
                    script_path,
                )
                return None

        device = "cpu"
        if not use_cpu:
            try:
                import torch

                if torch.cuda.is_available():
                    device = "cuda"
                elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
                    device = "mps"
                else:
                    self.get_logger().warning(
                        "Accelerated inference requested but no GPU/MPS backend is available; using CPU."
                    )
            except ImportError:
                self.get_logger().warning(
                    "PyTorch not available to detect accelerators; defaulting to CPU for snnTorch inference."
                )

        try:
            controller = SnnTorchController.from_checkpoint(
                checkpoint_path,
                device=device,
                model_kind=model_kind_normalised,  # type: ignore[arg-type]
                torchscript_path=script_path,
            )
        except Exception as exc:  # pragma: no cover - defensive against runtime loader issues
            self.get_logger().error(
                "Failed to initialise snnTorch controller from '%s': %s",
                checkpoint_path,
                exc,
            )
            return None

        metadata = getattr(controller, "metadata", {})
        summary_parts = [
            f"device={device}",
            f"time_steps={controller.config.time_steps}",
            f"kind={model_kind_normalised}",
        ]
        val_loss = metadata.get("val_loss")
        if isinstance(val_loss, (int, float)):
            summary_parts.append(f"val_loss={val_loss:.6f}")
        val_mse = metadata.get("val_mse_actual")
        if isinstance(val_mse, (int, float)):
            summary_parts.append(f"val_mse={val_mse:.6f}")

        if script_path is not None:
            summary_parts.append(f"torchscript={script_path}")

        self.get_logger().info(
            "Loaded snnTorch controller from '%s' (%s).",
            checkpoint_path,
            ", ".join(summary_parts),
        )
        return controller

    def _try_create_bat_controller(
        self,
        *,
        arena_width: float,
        arena_height: float,
        rng_seed: int,
    ):
        """Create a BatNavigationController instance."""
        if BatNavigationController is None:
            self.get_logger().error(
                "Bat navigation backend requested but BatNavigationController is not available."
            )
            return None

        try:
            controller_rng = np.random.default_rng(rng_seed)
            self._environment = Environment(width=arena_width, height=arena_height)
            controller_config = BatNavigationControllerConfig(
                num_place_cells=80,
                hd_num_neurons=72,
                grid_size=(16, 16),
                calibration_interval=250,
                integration_window=None,  # Disable for real-time ROS operation
            )
            controller = BatNavigationController(
                environment=self._environment,
                config=controller_config,
                rng=controller_rng,
            )
            self.get_logger().info(
                "Loaded bat navigation controller (HD neurons=%d, grid size=%s, place cells=%d).",
                controller_config.hd_num_neurons,
                controller_config.grid_size,
                controller_config.num_place_cells,
            )
            return controller
        except Exception as exc:
            self.get_logger().error(
                "Failed to initialise bat navigation controller: %s", exc
            )
            return None

    def _pose_callback(self, msg: Odometry) -> None:
        position = msg.pose.pose.position
        self._last_pose = (float(position.x), float(position.y))
        orientation = msg.pose.pose.orientation
        self._last_heading = self._quat_to_yaw(
            orientation.x,
            orientation.y,
            orientation.z,
            orientation.w,
        )
        # Store message timestamp for latency compensation
        self._last_msg_timestamp = msg.header.stamp

    def _control_timer_callback(self) -> None:
        if self._last_pose is None or self._last_heading is None:
            return

        now = self.get_clock().now().nanoseconds / 1e9
        if self._last_time is None:
            dt = 1.0 / self._control_rate if self._control_rate > 0.0 else 0.1
        else:
            dt = max(now - self._last_time, 1e-6)
        self._last_time = now

        # Optional timestamp latency compensation
        # Apply correction if message is stale (latency > 20 ms)
        obs_position = np.array([self._last_pose[0], self._last_pose[1]])
        obs_heading = self._last_heading
        
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

        # Construct observation based on controller backend
        if self._backend_in_use == "bat_navigation":
            # Bat controller requires [x, y, theta] observation
            obs = np.asarray(
                [
                    obs_position[0],
                    obs_position[1],
                    obs_heading,
                ],
                dtype=np.float32,
            )
        else:
            # Legacy controllers (snntorch, place_cells) use [x, y, cos(theta), sin(theta)]
            obs = np.asarray(
                [
                    self._last_pose[0],
                    self._last_pose[1],
                    math.cos(self._last_heading),
                    math.sin(self._last_heading),
                ],
                dtype=np.float32,
            )
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
        if self._viz_enabled:
            self._pose_history.append(self._last_pose)

    def _publish_markers(self) -> None:
        if (
            not self._viz_enabled
            or self._marker_publisher is None
            or self._backend_in_use not in ("place_cells", "bat_navigation")
        ):
            return

        now = self.get_clock().now().to_msg()
        marker_array = MarkerArray()

        positions = self._controller.place_cell_positions
        sigma = getattr(self._controller.config, "sigma", 0.1)
        graph = self._controller.get_graph()
        frame_id = self._viz_frame_id

        for idx, pos in enumerate(positions):
            center_marker = Marker()
            center_marker.header.frame_id = frame_id
            center_marker.header.stamp = now
            center_marker.ns = "pc_centers"
            center_marker.id = idx
            center_marker.type = Marker.SPHERE
            center_marker.action = Marker.ADD
            center_marker.pose.position.x = float(pos[0])
            center_marker.pose.position.y = float(pos[1])
            center_marker.pose.position.z = 0.0
            center_marker.pose.orientation.w = 1.0
            center_marker.scale.x = 0.05
            center_marker.scale.y = 0.05
            center_marker.scale.z = 0.05
            center_marker.color.r = 0.1
            center_marker.color.g = 0.4
            center_marker.color.b = 0.9
            center_marker.color.a = 1.0
            marker_array.markers.append(center_marker)

            field_marker = Marker()
            field_marker.header.frame_id = frame_id
            field_marker.header.stamp = now
            field_marker.ns = "pc_fields"
            field_marker.id = idx
            field_marker.type = Marker.CYLINDER
            field_marker.action = Marker.ADD
            field_marker.pose.position.x = float(pos[0])
            field_marker.pose.position.y = float(pos[1])
            field_marker.pose.position.z = -0.01
            field_marker.pose.orientation.w = 1.0
            diameter = 2.0 * float(sigma)
            field_marker.scale.x = diameter
            field_marker.scale.y = diameter
            field_marker.scale.z = 0.02
            field_marker.color.r = 0.1
            field_marker.color.g = 0.8
            field_marker.color.b = 0.3
            field_marker.color.a = 0.25
            marker_array.markers.append(field_marker)

        edge_marker = Marker()
        edge_marker.header.frame_id = frame_id
        edge_marker.header.stamp = now
        edge_marker.ns = "graph_edges"
        edge_marker.id = 0
        edge_marker.type = Marker.LINE_LIST
        edge_marker.action = Marker.ADD
        edge_marker.scale.x = 0.01
        edge_marker.color.r = 1.0
        edge_marker.color.g = 0.6
        edge_marker.color.b = 0.0
        edge_marker.color.a = 0.8

        for i, j in graph.graph.edges():
            start = positions[i]
            end = positions[j]
            edge_marker.points.append(self._make_point(start))
            edge_marker.points.append(self._make_point(end))
        marker_array.markers.append(edge_marker)

        if self._pose_history and len(self._pose_history) > 1:
            trail_marker = Marker()
            trail_marker.header.frame_id = frame_id
            trail_marker.header.stamp = now
            trail_marker.ns = "agent_trail"
            trail_marker.id = 0
            trail_marker.type = Marker.LINE_STRIP
            trail_marker.action = Marker.ADD
            trail_marker.scale.x = 0.01
            trail_marker.color.r = 0.9
            trail_marker.color.g = 0.1
            trail_marker.color.b = 0.2
            trail_marker.color.a = 0.8
            for pose in self._pose_history:
                trail_marker.points.append(self._make_point(pose))
            marker_array.markers.append(trail_marker)

        self._marker_publisher.publish(marker_array)

    @staticmethod
    def _make_point(position: Tuple[float, float]):
        point = Point()
        point.x = float(position[0])
        point.y = float(position[1])
        point.z = 0.0
        return point

    @staticmethod
    def _quat_to_yaw(x: float, y: float, z: float, w: float) -> float:
        siny_cosp = 2.0 * (w * z + x * y)
        cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
        return math.atan2(siny_cosp, cosy_cosp)


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
