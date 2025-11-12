from __future__ import annotations

import math
import time
from pathlib import Path
from typing import List

import launch_testing
import pytest
import rclpy
from geometry_msgs.msg import Twist
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_testing.actions import ReadyToTest
from nav_msgs.msg import Odometry
from std_msgs.msg import Float32MultiArray


def _write_test_params(model_path: Path) -> Path:
    params_content = (
        "hippocampus_ros2:\n"
        "  ros__parameters:\n"
        "    controller_backend: snntorch\n"
        f"    model_path: \"{model_path.resolve()}\"\n"
        "    use_cpu: true\n"
        "    control_rate: 5.0\n"
        "    max_linear: 0.3\n"
        "    max_angular: 1.0\n"
    )
    params_file = Path(__file__).with_name("brain_launch_test_params.yaml")
    params_file.write_text(params_content, encoding="utf-8")
    return params_file


@pytest.mark.launch_test
def generate_test_description():
    launch_path = PythonLaunchDescriptionSource(
        str(
            Path(get_package_share_directory("hippocampus_ros2"))
            / "launch"
            / "brain.launch.py"
        )
    )

    repo_root = Path(__file__).resolve().parents[4]
    model_path = repo_root / "models" / "snn_controller.pt"
    if not model_path.exists():
        raise FileNotFoundError(
            "Expected snnTorch checkpoint at models/snn_controller.pt. "
            "Run experiments/train_snntorch_policy.py before executing tests.",
        )
    params_file = _write_test_params(model_path)

    brain_launch = IncludeLaunchDescription(
        launch_path,
        launch_arguments={
            "params_file": str(params_file),
        }.items(),
    )

    ld = LaunchDescription([brain_launch, ReadyToTest()])
    return ld, {"params_file": params_file}


class _MessageCollector:
    def __init__(self) -> None:
        self.messages: List = []

    def __call__(self, msg) -> None:
        self.messages.append(msg)


class TestBrainNodeOutputs:
    @classmethod
    def setup_class(cls) -> None:
        rclpy.init()

    @classmethod
    def teardown_class(cls) -> None:
        rclpy.shutdown()

    def setup_method(self) -> None:
        self.node = rclpy.create_node("brain_launch_test")
        self.snn_collector = _MessageCollector()
        self.cmd_collector = _MessageCollector()
        self.node.create_subscription(
            Float32MultiArray,
            "snn_action",
            self.snn_collector,
            10,
        )
        self.node.create_subscription(
            Twist,
            "/cmd_vel",
            self.cmd_collector,
            10,
        )
        self.odom_pub = self.node.create_publisher(Odometry, "/odom", 10)

    def teardown_method(self) -> None:
        self.node.destroy_node()

    def test_brain_emits_actions(self) -> None:
        max_linear = 0.3
        max_angular = 1.0
        deadline = time.time() + 8.0
        theta = 0.0

        while time.time() < deadline:
            msg = Odometry()
            msg.pose.pose.position.x = 0.4 * math.cos(theta)
            msg.pose.pose.position.y = 0.4 * math.sin(theta)
            msg.pose.pose.orientation.w = 1.0
            self.odom_pub.publish(msg)
            rclpy.spin_once(self.node, timeout_sec=0.1)
            time.sleep(0.05)
            theta += 0.2

        # Allow final callbacks to run
        for _ in range(5):
            rclpy.spin_once(self.node, timeout_sec=0.1)

        assert len(self.snn_collector.messages) >= 5, "No SNN actions received."
        assert len(self.cmd_collector.messages) >= 5, "No cmd_vel messages received."

        for msg in self.snn_collector.messages:
            assert len(msg.data) >= 2, "Expected at least 2 action dimensions."

        for twist in self.cmd_collector.messages:
            assert abs(twist.linear.x) <= max_linear + 1e-6
            assert abs(twist.angular.z) <= max_angular + 1e-6


@launch_testing.post_shutdown_test()
class TestBrainNodeShutdown:
    def test_processes_exit_cleanly(self, proc_info):
        proc_info.assertWaitForShutdown()
        proc_info.assertExitCodes()

    def test_cleanup_params_file(self, proc_info, params_file):
        params_path = Path(params_file)
        if params_path.exists():
            params_path.unlink()

