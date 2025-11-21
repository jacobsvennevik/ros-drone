// generated from rosidl_generator_c/resource/idl__struct.h.em
// with input from hippocampus_ros2_msgs:msg/PolicyDecision.idl
// generated code does not contain a copyright notice

#ifndef HIPPOCAMPUS_ROS2_MSGS__MSG__DETAIL__POLICY_DECISION__STRUCT_H_
#define HIPPOCAMPUS_ROS2_MSGS__MSG__DETAIL__POLICY_DECISION__STRUCT_H_

#ifdef __cplusplus
extern "C"
{
#endif

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>


// Constants defined in the message

// Include directives for member types
// Member 'reason'
#include "rosidl_runtime_c/string.h"
// Member 'stamp'
#include "builtin_interfaces/msg/detail/time__struct.h"

/// Struct defined in msg/PolicyDecision in the package hippocampus_ros2_msgs.
/**
  * Policy decision message
  * Represents a decision made by the SNN Policy Service
 */
typedef struct hippocampus_ros2_msgs__msg__PolicyDecision
{
  /// Action proposal
  /// Linear velocity (m/s)
  double linear_x;
  /// Angular velocity (rad/s)
  double angular_z;
  /// Vertical velocity (m/s) for 3D, 0.0 for 2D
  double linear_z;
  /// Decision metadata
  /// Confidence score [0.0, 1.0]
  double confidence;
  /// Decision reason ("heuristic", "snn", etc.)
  rosidl_runtime_c__String reason;
  /// Next waypoint node ID (-1 if none)
  int32_t next_waypoint;
  /// Timestamp
  builtin_interfaces__msg__Time stamp;
} hippocampus_ros2_msgs__msg__PolicyDecision;

// Struct for a sequence of hippocampus_ros2_msgs__msg__PolicyDecision.
typedef struct hippocampus_ros2_msgs__msg__PolicyDecision__Sequence
{
  hippocampus_ros2_msgs__msg__PolicyDecision * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} hippocampus_ros2_msgs__msg__PolicyDecision__Sequence;

#ifdef __cplusplus
}
#endif

#endif  // HIPPOCAMPUS_ROS2_MSGS__MSG__DETAIL__POLICY_DECISION__STRUCT_H_
