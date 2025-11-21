// generated from rosidl_generator_c/resource/idl__struct.h.em
// with input from hippocampus_ros2_msgs:msg/PolicyStatus.idl
// generated code does not contain a copyright notice

#ifndef HIPPOCAMPUS_ROS2_MSGS__MSG__DETAIL__POLICY_STATUS__STRUCT_H_
#define HIPPOCAMPUS_ROS2_MSGS__MSG__DETAIL__POLICY_STATUS__STRUCT_H_

#ifdef __cplusplus
extern "C"
{
#endif

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>


// Constants defined in the message

// Include directives for member types
// Member 'current_reason'
#include "rosidl_runtime_c/string.h"
// Member 'stamp'
#include "builtin_interfaces/msg/detail/time__struct.h"

/// Struct defined in msg/PolicyStatus in the package hippocampus_ros2_msgs.
/**
  * Policy status message
  * Diagnostics and status information for the policy system
 */
typedef struct hippocampus_ros2_msgs__msg__PolicyStatus
{
  /// Status flags
  /// Policy is active and making decisions
  bool is_active;
  /// Topological graph is stale
  bool graph_stale;
  /// Currently using SNN (vs heuristic)
  bool using_snn;
  /// Hierarchical planning enabled
  bool hierarchical_enabled;
  /// Performance metrics
  /// Feature computation time (ms)
  double feature_compute_time_ms;
  /// Policy decision time (ms)
  double policy_decision_time_ms;
  /// Safety filtering time (ms)
  double safety_filter_time_ms;
  /// Total latency (ms)
  double total_latency_ms;
  /// Graph information
  /// Number of nodes in graph
  uint32_t graph_nodes;
  /// Number of edges in graph
  uint32_t graph_edges;
  /// Time since last graph update (seconds)
  double graph_staleness_s;
  /// Current state
  /// Current decision confidence
  double current_confidence;
  /// Current decision reason
  rosidl_runtime_c__String current_reason;
  /// Current waypoint node ID (-1 if none)
  int32_t current_waypoint;
  /// Timestamp
  builtin_interfaces__msg__Time stamp;
} hippocampus_ros2_msgs__msg__PolicyStatus;

// Struct for a sequence of hippocampus_ros2_msgs__msg__PolicyStatus.
typedef struct hippocampus_ros2_msgs__msg__PolicyStatus__Sequence
{
  hippocampus_ros2_msgs__msg__PolicyStatus * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} hippocampus_ros2_msgs__msg__PolicyStatus__Sequence;

#ifdef __cplusplus
}
#endif

#endif  // HIPPOCAMPUS_ROS2_MSGS__MSG__DETAIL__POLICY_STATUS__STRUCT_H_
