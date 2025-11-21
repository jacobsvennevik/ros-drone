// generated from rosidl_generator_c/resource/idl__struct.h.em
// with input from hippocampus_ros2_msgs:msg/GraphSnapshot.idl
// generated code does not contain a copyright notice

#ifndef HIPPOCAMPUS_ROS2_MSGS__MSG__DETAIL__GRAPH_SNAPSHOT__STRUCT_H_
#define HIPPOCAMPUS_ROS2_MSGS__MSG__DETAIL__GRAPH_SNAPSHOT__STRUCT_H_

#ifdef __cplusplus
extern "C"
{
#endif

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>


// Constants defined in the message

// Include directives for member types
// Member 'frame_id'
#include "rosidl_runtime_c/string.h"
// Member 'stamp'
// Member 'last_updated'
#include "builtin_interfaces/msg/detail/time__struct.h"
// Member 'nodes'
#include "hippocampus_ros2_msgs/msg/detail/graph_node__struct.h"
// Member 'edges'
#include "hippocampus_ros2_msgs/msg/detail/graph_edge__struct.h"

/// Struct defined in msg/GraphSnapshot in the package hippocampus_ros2_msgs.
/**
  * Graph snapshot message
  * Represents a snapshot of the topological graph
 */
typedef struct hippocampus_ros2_msgs__msg__GraphSnapshot
{
  /// Graph metadata
  /// Graph epoch/version ID
  uint32_t epoch_id;
  /// Coordinate frame
  rosidl_runtime_c__String frame_id;
  builtin_interfaces__msg__Time stamp;
  builtin_interfaces__msg__Time last_updated;
  /// Graph update rate (Hz)
  double update_rate;
  /// Staleness warning flag
  bool staleness_warning;
  /// Nodes
  hippocampus_ros2_msgs__msg__GraphNode__Sequence nodes;
  /// Edges
  hippocampus_ros2_msgs__msg__GraphEdge__Sequence edges;
} hippocampus_ros2_msgs__msg__GraphSnapshot;

// Struct for a sequence of hippocampus_ros2_msgs__msg__GraphSnapshot.
typedef struct hippocampus_ros2_msgs__msg__GraphSnapshot__Sequence
{
  hippocampus_ros2_msgs__msg__GraphSnapshot * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} hippocampus_ros2_msgs__msg__GraphSnapshot__Sequence;

#ifdef __cplusplus
}
#endif

#endif  // HIPPOCAMPUS_ROS2_MSGS__MSG__DETAIL__GRAPH_SNAPSHOT__STRUCT_H_
