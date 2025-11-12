"""Launch BrainNode with ros2_tracing instrumentation for timing analysis."""
from __future__ import annotations

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, EmitEvent, TimerAction
from launch.events import Shutdown
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare

try:
    from tracetools_launch.action import Trace
except ImportError as exc:  # pragma: no cover - optional dependency
    raise ImportError(
        "Tracing launch requires the 'ros2_tracing' stack (tracetools_launch). "
        "Install it on Linux with: sudo apt install ros-${ROS_DISTRO}-ros2-tracing"
    ) from exc


def generate_launch_description() -> LaunchDescription:
    pkg_share = FindPackageShare("hippocampus_ros2")
    default_params = PathJoinSubstitution([pkg_share, "config", "brain.yaml"])

    params_file_arg = DeclareLaunchArgument(
        "params_file",
        default_value=default_params,
        description="Path to the BrainNode parameter file.",
    )
    use_sim_time_arg = DeclareLaunchArgument(
        "use_sim_time",
        default_value="false",
        description="Set true when running with simulated time.",
    )
    pose_topic_arg = DeclareLaunchArgument(
        "pose_topic",
        default_value="/odom",
        description="Pose topic remapping for BrainNode input.",
    )
    cmd_vel_topic_arg = DeclareLaunchArgument(
        "cmd_vel_topic",
        default_value="/cmd_vel",
        description="Command velocity topic published by BrainNode.",
    )
    use_bag_replay_arg = DeclareLaunchArgument(
        "use_bag_replay",
        default_value="false",
        description="Set true when playing back recorded bags instead of a live simulator.",
    )
    trace_output_arg = DeclareLaunchArgument(
        "trace_output",
        default_value="traces",
        description="Directory to store traced events (relative to launch working directory).",
    )
    trace_session_arg = DeclareLaunchArgument(
        "trace_session",
        default_value="hippocampus_trace",
        description="Name of the tracing session (visible in trace metadata).",
    )
    trace_duration_arg = DeclareLaunchArgument(
        "trace_duration",
        default_value="15.0",
        description="Seconds to run tracing before shutting down.",
    )

    trace_action = Trace(
        session_name=LaunchConfiguration("trace_session"),
        events_ust=["ros2:*"],
        path=LaunchConfiguration("trace_output"),
    )

    brain_node = Node(
        package="hippocampus_ros2",
        executable="snn_brain_node",
        name="snn_brain_node",
        output="screen",
        parameters=[
            LaunchConfiguration("params_file"),
            {
                "use_sim_time": LaunchConfiguration("use_sim_time"),
                "pose_topic": LaunchConfiguration("pose_topic"),
                "cmd_vel_topic": LaunchConfiguration("cmd_vel_topic"),
                "use_bag_replay": LaunchConfiguration("use_bag_replay"),
            },
        ],
    )

    shutdown_timer = TimerAction(
        period=LaunchConfiguration("trace_duration"),
        actions=[EmitEvent(event=Shutdown(reason="Tracing duration elapsed"))],
    )

    return LaunchDescription(
        [
            params_file_arg,
            use_sim_time_arg,
            pose_topic_arg,
            cmd_vel_topic_arg,
            use_bag_replay_arg,
            trace_output_arg,
            trace_session_arg,
            trace_duration_arg,
            trace_action,
            brain_node,
            shutdown_timer,
        ]
    )

