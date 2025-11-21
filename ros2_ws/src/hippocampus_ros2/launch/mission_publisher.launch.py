"""Launch file for mission publisher node."""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    goal_type_arg = DeclareLaunchArgument(
        "goal_type",
        default_value="point",
        description="Goal type: 'point', 'node', or 'region'",
    )
    goal_x_arg = DeclareLaunchArgument(
        "goal_x",
        default_value="0.9",
        description="Goal X position (for point/region goals)",
    )
    goal_y_arg = DeclareLaunchArgument(
        "goal_y",
        default_value="0.9",
        description="Goal Y position (for point/region goals)",
    )
    goal_z_arg = DeclareLaunchArgument(
        "goal_z",
        default_value="0.0",
        description="Goal Z position (for point/region goals)",
    )
    tolerance_arg = DeclareLaunchArgument(
        "tolerance",
        default_value="0.1",
        description="Distance tolerance for point goals (m)",
    )
    node_id_arg = DeclareLaunchArgument(
        "node_id",
        default_value="0",
        description="Node ID (for node goals)",
    )
    region_radius_arg = DeclareLaunchArgument(
        "region_radius",
        default_value="0.5",
        description="Region radius (for region goals, m)",
    )
    frame_id_arg = DeclareLaunchArgument(
        "frame_id",
        default_value="map",
        description="Coordinate frame ID",
    )
    timeout_arg = DeclareLaunchArgument(
        "timeout",
        default_value="0.0",
        description="Goal timeout in seconds (0.0 = no timeout)",
    )
    publish_once_arg = DeclareLaunchArgument(
        "publish_once",
        default_value="true",
        description="Publish goal once (true) or periodically (false)",
    )
    publish_rate_arg = DeclareLaunchArgument(
        "publish_rate",
        default_value="1.0",
        description="Publish rate in Hz (when publish_once=false)",
    )

    mission_publisher_node = Node(
        package="hippocampus_ros2",
        executable="mission_publisher",
        name="mission_publisher",
        parameters=[
            {
                "goal_type": LaunchConfiguration("goal_type"),
                "goal_x": LaunchConfiguration("goal_x"),
                "goal_y": LaunchConfiguration("goal_y"),
                "goal_z": LaunchConfiguration("goal_z"),
                "tolerance": LaunchConfiguration("tolerance"),
                "node_id": LaunchConfiguration("node_id"),
                "region_radius": LaunchConfiguration("region_radius"),
                "frame_id": LaunchConfiguration("frame_id"),
                "timeout": LaunchConfiguration("timeout"),
                "publish_once": LaunchConfiguration("publish_once"),
                "publish_rate": LaunchConfiguration("publish_rate"),
            },
        ],
    )

    return LaunchDescription(
        [
            goal_type_arg,
            goal_x_arg,
            goal_y_arg,
            goal_z_arg,
            tolerance_arg,
            node_id_arg,
            region_radius_arg,
            frame_id_arg,
            timeout_arg,
            publish_once_arg,
            publish_rate_arg,
            mission_publisher_node,
        ]
    )

