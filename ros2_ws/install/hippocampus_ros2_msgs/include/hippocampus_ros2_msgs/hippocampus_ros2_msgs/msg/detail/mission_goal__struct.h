// generated from rosidl_generator_c/resource/idl__struct.h.em
// with input from hippocampus_ros2_msgs:msg/MissionGoal.idl
// generated code does not contain a copyright notice

#ifndef HIPPOCAMPUS_ROS2_MSGS__MSG__DETAIL__MISSION_GOAL__STRUCT_H_
#define HIPPOCAMPUS_ROS2_MSGS__MSG__DETAIL__MISSION_GOAL__STRUCT_H_

#ifdef __cplusplus
extern "C"
{
#endif

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>


// Constants defined in the message

/// Constant 'GOAL_TYPE_POINT'.
/**
  * Goal type
 */
enum
{
  hippocampus_ros2_msgs__msg__MissionGoal__GOAL_TYPE_POINT = 0
};

/// Constant 'GOAL_TYPE_NODE'.
enum
{
  hippocampus_ros2_msgs__msg__MissionGoal__GOAL_TYPE_NODE = 1
};

/// Constant 'GOAL_TYPE_REGION'.
enum
{
  hippocampus_ros2_msgs__msg__MissionGoal__GOAL_TYPE_REGION = 2
};

// Include directives for member types
// Member 'point_position'
// Member 'region_center'
#include "geometry_msgs/msg/detail/point__struct.h"
// Member 'frame_id'
#include "rosidl_runtime_c/string.h"
// Member 'stamp'
#include "builtin_interfaces/msg/detail/time__struct.h"

/// Struct defined in msg/MissionGoal in the package hippocampus_ros2_msgs.
/**
  * Mission goal message
  * Represents a navigation goal for the policy system
 */
typedef struct hippocampus_ros2_msgs__msg__MissionGoal
{
  /// Type of goal (POINT, NODE, REGION)
  uint8_t goal_type;
  /// Point goal (for GOAL_TYPE_POINT)
  geometry_msgs__msg__Point point_position;
  /// Distance tolerance (m)
  double point_tolerance;
  /// Node goal (for GOAL_TYPE_NODE)
  /// Graph node ID
  int32_t node_id;
  /// Region goal (for GOAL_TYPE_REGION)
  geometry_msgs__msg__Point region_center;
  /// Region radius (m)
  double region_radius;
  /// Goal metadata
  /// Coordinate frame
  rosidl_runtime_c__String frame_id;
  /// Timeout in seconds (0.0 = no timeout)
  double timeout;
  /// Goal reached flag
  bool is_reached;
  /// Timestamp
  builtin_interfaces__msg__Time stamp;
} hippocampus_ros2_msgs__msg__MissionGoal;

// Struct for a sequence of hippocampus_ros2_msgs__msg__MissionGoal.
typedef struct hippocampus_ros2_msgs__msg__MissionGoal__Sequence
{
  hippocampus_ros2_msgs__msg__MissionGoal * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} hippocampus_ros2_msgs__msg__MissionGoal__Sequence;

#ifdef __cplusplus
}
#endif

#endif  // HIPPOCAMPUS_ROS2_MSGS__MSG__DETAIL__MISSION_GOAL__STRUCT_H_
