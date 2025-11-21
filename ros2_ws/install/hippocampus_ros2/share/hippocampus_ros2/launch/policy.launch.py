"""Launch file for policy node."""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description() -> LaunchDescription:
    pkg_share = FindPackageShare("hippocampus_ros2")
    default_params = PathJoinSubstitution([pkg_share, "config", "policy.yaml"])

    use_sim_time_arg = DeclareLaunchArgument(
        "use_sim_time",
        default_value="false",
        description="Set to true when running against simulated time (e.g., Gazebo).",
    )
    params_file_arg = DeclareLaunchArgument(
        "params_file",
        default_value=default_params,
        description="Path to a YAML file with PolicyNode parameters.",
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
    mission_topic_arg = DeclareLaunchArgument(
        "mission_topic",
        default_value="/mission/goal",
        description="Topic for mission goal updates.",
    )

    policy_node = Node(
        package="hippocampus_ros2",
        executable="policy_node",
        name="policy_node",
        parameters=[
            LaunchConfiguration("params_file"),
            {
                "use_sim_time": LaunchConfiguration("use_sim_time"),
                "pose_topic": LaunchConfiguration("pose_topic"),
                "cmd_vel_topic": LaunchConfiguration("cmd_vel_topic"),
                "mission_topic": LaunchConfiguration("mission_topic"),
            },
        ],
    )

    return LaunchDescription(
        [
            use_sim_time_arg,
            params_file_arg,
            pose_topic_arg,
            cmd_vel_topic_arg,
            mission_topic_arg,
            policy_node,
        ]
    )

