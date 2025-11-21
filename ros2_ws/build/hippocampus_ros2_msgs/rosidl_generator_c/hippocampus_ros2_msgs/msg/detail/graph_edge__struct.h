// generated from rosidl_generator_c/resource/idl__struct.h.em
// with input from hippocampus_ros2_msgs:msg/GraphEdge.idl
// generated code does not contain a copyright notice

#ifndef HIPPOCAMPUS_ROS2_MSGS__MSG__DETAIL__GRAPH_EDGE__STRUCT_H_
#define HIPPOCAMPUS_ROS2_MSGS__MSG__DETAIL__GRAPH_EDGE__STRUCT_H_

#ifdef __cplusplus
extern "C"
{
#endif

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>


// Constants defined in the message

/// Struct defined in msg/GraphEdge in the package hippocampus_ros2_msgs.
/**
  * Graph edge message
 */
typedef struct hippocampus_ros2_msgs__msg__GraphEdge
{
  /// Source node ID
  int32_t u;
  /// Target node ID
  int32_t v;
  /// Edge length (m)
  double length;
  /// Is edge traversable
  bool traversable;
} hippocampus_ros2_msgs__msg__GraphEdge;

// Struct for a sequence of hippocampus_ros2_msgs__msg__GraphEdge.
typedef struct hippocampus_ros2_msgs__msg__GraphEdge__Sequence
{
  hippocampus_ros2_msgs__msg__GraphEdge * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} hippocampus_ros2_msgs__msg__GraphEdge__Sequence;

#ifdef __cplusplus
}
#endif

#endif  // HIPPOCAMPUS_ROS2_MSGS__MSG__DETAIL__GRAPH_EDGE__STRUCT_H_
