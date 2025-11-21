from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description() -> LaunchDescription:
    pkg_share = FindPackageShare("hippocampus_ros2")
    default_params = PathJoinSubstitution([pkg_share, "config", "brain.yaml"])

    use_sim_time_arg = DeclareLaunchArgument(
        "use_sim_time",
        default_value="false",
        description="Set to true when running against simulated time (e.g., Gazebo).",
    )
    params_file_arg = DeclareLaunchArgument(
        "params_file",
        default_value=default_params,
        description="Path to a YAML file with BrainNode parameters.",
    )
    pose_topic_arg = DeclareLaunchArgument(
        "pose_topic",
        default_value="/odom",
        description="Remap target for pose subscription.",
    )
    cmd_vel_topic_arg = DeclareLaunchArgument(
        "cmd_vel_topic",
        default_value="/cmd_vel",
        description="Remap target for published Twist commands.",
    )
    use_bag_replay_arg = DeclareLaunchArgument(
        "use_bag_replay",
        default_value="false",
        description="Set true when running alongside ros2 bag replay rather than a live simulator.",
    )

    brain_node = Node(
        package="hippocampus_ros2",
        executable="snn_brain_node",
        name="snn_brain_node",
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

    return LaunchDescription(
        [
            use_sim_time_arg,
            params_file_arg,
            pose_topic_arg,
            cmd_vel_topic_arg,
            use_bag_replay_arg,
            brain_node,
        ]
    )

