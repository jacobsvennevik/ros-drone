// generated from rosidl_generator_c/resource/idl__struct.h.em
// with input from hippocampus_ros2_msgs:msg/GraphNode.idl
// generated code does not contain a copyright notice

#ifndef HIPPOCAMPUS_ROS2_MSGS__MSG__DETAIL__GRAPH_NODE__STRUCT_H_
#define HIPPOCAMPUS_ROS2_MSGS__MSG__DETAIL__GRAPH_NODE__STRUCT_H_

#ifdef __cplusplus
extern "C"
{
#endif

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>


// Constants defined in the message

// Include directives for member types
// Member 'position'
#include "geometry_msgs/msg/detail/point__struct.h"
// Member 'normal'
#include "geometry_msgs/msg/detail/vector3__struct.h"
// Member 'tags'
#include "rosidl_runtime_c/string.h"

/// Struct defined in msg/GraphNode in the package hippocampus_ros2_msgs.
/**
  * Graph node message
 */
typedef struct hippocampus_ros2_msgs__msg__GraphNode
{
  int32_t node_id;
  geometry_msgs__msg__Point position;
  /// Surface normal (for 3D)
  geometry_msgs__msg__Vector3 normal;
  /// Node degree
  uint32_t degree;
  rosidl_runtime_c__String__Sequence tags;
} hippocampus_ros2_msgs__msg__GraphNode;

// Struct for a sequence of hippocampus_ros2_msgs__msg__GraphNode.
typedef struct hippocampus_ros2_msgs__msg__GraphNode__Sequence
{
  hippocampus_ros2_msgs__msg__GraphNode * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} hippocampus_ros2_msgs__msg__GraphNode__Sequence;

#ifdef __cplusplus
}
#endif

#endif  // HIPPOCAMPUS_ROS2_MSGS__MSG__DETAIL__GRAPH_NODE__STRUCT_H_
