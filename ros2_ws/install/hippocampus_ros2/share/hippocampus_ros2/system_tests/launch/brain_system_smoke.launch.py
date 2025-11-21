"""Launch a minimal Gazebo scenario and verify BrainNode system behavior."""
from __future__ import annotations

from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    EmitEvent,
    IncludeLaunchDescription,
    RegisterEventHandler,
    TimerAction,
)
from launch.event_handlers import OnProcessExit
from launch.events import Shutdown
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch.actions import ExecuteProcess
from launch_ros.substitutions import FindPackageShare


def generate_launch_description() -> LaunchDescription:
    pkg_share = FindPackageShare("hippocampus_ros2")

    world_path = PathJoinSubstitution([pkg_share, "system_tests", "worlds", "minimal.world"])
    brain_launch = PathJoinSubstitution([pkg_share, "launch", "brain.launch.py"])
    pose_script = PathJoinSubstitution([pkg_share, "system_tests", "scripts", "pose_publisher.py"])
    assert_script = PathJoinSubstitution([pkg_share, "system_tests", "scripts", "assert_topics.py"])
    params_file = PathJoinSubstitution([pkg_share, "config", "brain.yaml"])

    gui_arg = DeclareLaunchArgument(
        "gui",
        default_value="false",
        description="Enable Gazebo GUI (defaults to headless for CI).",
    )
    timeout_arg = DeclareLaunchArgument(
        "timeout",
        default_value="20.0",
        description="Assertion timeout in seconds.",
    )
    min_messages_arg = DeclareLaunchArgument(
        "min_messages",
        default_value="5",
        description="Minimum number of messages expected on each topic.",
    )
    max_linear_arg = DeclareLaunchArgument(
        "max_linear",
        default_value="0.3",
        description="Maximum allowed |cmd_vel.linear.x|.",
    )
    max_angular_arg = DeclareLaunchArgument(
        "max_angular",
        default_value="1.0",
        description="Maximum allowed |cmd_vel.angular.z|.",
    )
    snn_topic_arg = DeclareLaunchArgument(
        "snn_action_topic",
        default_value="/snn_action",
        description="Expected snn_action topic.",
    )
    cmd_vel_topic_arg = DeclareLaunchArgument(
        "cmd_vel_topic",
        default_value="/cmd_vel",
        description="Expected cmd_vel topic.",
    )

    gazebo_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([FindPackageShare("gazebo_ros"), "launch", "gazebo.launch.py"])
        ),
        launch_arguments={
            "world": world_path,
            "gui": LaunchConfiguration("gui"),
            "verbose": "false",
        }.items(),
    )

    brain_launch_include = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(brain_launch),
        launch_arguments={
            "use_sim_time": "true",
            "params_file": params_file,
            "pose_topic": "/odom",
            "cmd_vel_topic": LaunchConfiguration("cmd_vel_topic"),
        }.items(),
    )

    pose_publisher_proc = ExecuteProcess(
        cmd=[
            "python3",
            pose_script,
            "--ros-args",
            "-p",
            "pose_topic:=/odom",
            "-p",
            "publish_rate:=10.0",
            "-p",
            "linear_speed:=0.05",
            "-p",
            "angular_speed:=0.2",
            "-p",
            "use_sim_time:=true",
        ],
        output="screen",
    )

    assert_process = ExecuteProcess(
        cmd=[
            "python3",
            assert_script,
            "--snn-action-topic",
            LaunchConfiguration("snn_action_topic"),
            "--cmd-vel-topic",
            LaunchConfiguration("cmd_vel_topic"),
            "--timeout",
            LaunchConfiguration("timeout"),
            "--min-messages",
            LaunchConfiguration("min_messages"),
            "--max-linear",
            LaunchConfiguration("max_linear"),
            "--max-angular",
            LaunchConfiguration("max_angular"),
        ],
        output="screen",
    )

    shutdown_on_assert = RegisterEventHandler(
        OnProcessExit(
            target_action=assert_process,
            on_exit=[EmitEvent(event=Shutdown(reason="Topic assertions finished"))],
        )
    )

    return LaunchDescription(
        [
            gui_arg,
            timeout_arg,
            min_messages_arg,
            max_linear_arg,
            max_angular_arg,
            snn_topic_arg,
            cmd_vel_topic_arg,
            gazebo_launch,
            brain_launch_include,
            pose_publisher_proc,
            TimerAction(period=2.0, actions=[assert_process]),
            shutdown_on_assert,
        ]
    )

