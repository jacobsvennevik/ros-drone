// generated from rosidl_generator_c/resource/idl__functions.h.em
// with input from hippocampus_ros2_msgs:msg/GraphSnapshot.idl
// generated code does not contain a copyright notice

#ifndef HIPPOCAMPUS_ROS2_MSGS__MSG__DETAIL__GRAPH_SNAPSHOT__FUNCTIONS_H_
#define HIPPOCAMPUS_ROS2_MSGS__MSG__DETAIL__GRAPH_SNAPSHOT__FUNCTIONS_H_

#ifdef __cplusplus
extern "C"
{
#endif

#include <stdbool.h>
#include <stdlib.h>

#include "rosidl_runtime_c/visibility_control.h"
#include "hippocampus_ros2_msgs/msg/rosidl_generator_c__visibility_control.h"

#include "hippocampus_ros2_msgs/msg/detail/graph_snapshot__struct.h"

/// Initialize msg/GraphSnapshot message.
/**
 * If the init function is called twice for the same message without
 * calling fini inbetween previously allocated memory will be leaked.
 * \param[in,out] msg The previously allocated message pointer.
 * Fields without a default value will not be initialized by this function.
 * You might want to call memset(msg, 0, sizeof(
 * hippocampus_ros2_msgs__msg__GraphSnapshot
 * )) before or use
 * hippocampus_ros2_msgs__msg__GraphSnapshot__create()
 * to allocate and initialize the message.
 * \return true if initialization was successful, otherwise false
 */
ROSIDL_GENERATOR_C_PUBLIC_hippocampus_ros2_msgs
bool
hippocampus_ros2_msgs__msg__GraphSnapshot__init(hippocampus_ros2_msgs__msg__GraphSnapshot * msg);

/// Finalize msg/GraphSnapshot message.
/**
 * \param[in,out] msg The allocated message pointer.
 */
ROSIDL_GENERATOR_C_PUBLIC_hippocampus_ros2_msgs
void
hippocampus_ros2_msgs__msg__GraphSnapshot__fini(hippocampus_ros2_msgs__msg__GraphSnapshot * msg);

/// Create msg/GraphSnapshot message.
/**
 * It allocates the memory for the message, sets the memory to zero, and
 * calls
 * hippocampus_ros2_msgs__msg__GraphSnapshot__init().
 * \return The pointer to the initialized message if successful,
 * otherwise NULL
 */
ROSIDL_GENERATOR_C_PUBLIC_hippocampus_ros2_msgs
hippocampus_ros2_msgs__msg__GraphSnapshot *
hippocampus_ros2_msgs__msg__GraphSnapshot__create();

/// Destroy msg/GraphSnapshot message.
/**
 * It calls
 * hippocampus_ros2_msgs__msg__GraphSnapshot__fini()
 * and frees the memory of the message.
 * \param[in,out] msg The allocated message pointer.
 */
ROSIDL_GENERATOR_C_PUBLIC_hippocampus_ros2_msgs
void
hippocampus_ros2_msgs__msg__GraphSnapshot__destroy(hippocampus_ros2_msgs__msg__GraphSnapshot * msg);

/// Check for msg/GraphSnapshot message equality.
/**
 * \param[in] lhs The message on the left hand size of the equality operator.
 * \param[in] rhs The message on the right hand size of the equality operator.
 * \return true if messages are equal, otherwise false.
 */
ROSIDL_GENERATOR_C_PUBLIC_hippocampus_ros2_msgs
bool
hippocampus_ros2_msgs__msg__GraphSnapshot__are_equal(const hippocampus_ros2_msgs__msg__GraphSnapshot * lhs, const hippocampus_ros2_msgs__msg__GraphSnapshot * rhs);

/// Copy a msg/GraphSnapshot message.
/**
 * This functions performs a deep copy, as opposed to the shallow copy that
 * plain assignment yields.
 *
 * \param[in] input The source message pointer.
 * \param[out] output The target message pointer, which must
 *   have been initialized before calling this function.
 * \return true if successful, or false if either pointer is null
 *   or memory allocation fails.
 */
ROSIDL_GENERATOR_C_PUBLIC_hippocampus_ros2_msgs
bool
hippocampus_ros2_msgs__msg__GraphSnapshot__copy(
  const hippocampus_ros2_msgs__msg__GraphSnapshot * input,
  hippocampus_ros2_msgs__msg__GraphSnapshot * output);

/// Initialize array of msg/GraphSnapshot messages.
/**
 * It allocates the memory for the number of elements and calls
 * hippocampus_ros2_msgs__msg__GraphSnapshot__init()
 * for each element of the array.
 * \param[in,out] array The allocated array pointer.
 * \param[in] size The size / capacity of the array.
 * \return true if initialization was successful, otherwise false
 * If the array pointer is valid and the size is zero it is guaranteed
 # to return true.
 */
ROSIDL_GENERATOR_C_PUBLIC_hippocampus_ros2_msgs
bool
hippocampus_ros2_msgs__msg__GraphSnapshot__Sequence__init(hippocampus_ros2_msgs__msg__GraphSnapshot__Sequence * array, size_t size);

/// Finalize array of msg/GraphSnapshot messages.
/**
 * It calls
 * hippocampus_ros2_msgs__msg__GraphSnapshot__fini()
 * for each element of the array and frees the memory for the number of
 * elements.
 * \param[in,out] array The initialized array pointer.
 */
ROSIDL_GENERATOR_C_PUBLIC_hippocampus_ros2_msgs
void
hippocampus_ros2_msgs__msg__GraphSnapshot__Sequence__fini(hippocampus_ros2_msgs__msg__GraphSnapshot__Sequence * array);

/// Create array of msg/GraphSnapshot messages.
/**
 * It allocates the memory for the array and calls
 * hippocampus_ros2_msgs__msg__GraphSnapshot__Sequence__init().
 * \param[in] size The size / capacity of the array.
 * \return The pointer to the initialized array if successful, otherwise NULL
 */
ROSIDL_GENERATOR_C_PUBLIC_hippocampus_ros2_msgs
hippocampus_ros2_msgs__msg__GraphSnapshot__Sequence *
hippocampus_ros2_msgs__msg__GraphSnapshot__Sequence__create(size_t size);

/// Destroy array of msg/GraphSnapshot messages.
/**
 * It calls
 * hippocampus_ros2_msgs__msg__GraphSnapshot__Sequence__fini()
 * on the array,
 * and frees the memory of the array.
 * \param[in,out] array The initialized array pointer.
 */
ROSIDL_GENERATOR_C_PUBLIC_hippocampus_ros2_msgs
void
hippocampus_ros2_msgs__msg__GraphSnapshot__Sequence__destroy(hippocampus_ros2_msgs__msg__GraphSnapshot__Sequence * array);

/// Check for msg/GraphSnapshot message array equality.
/**
 * \param[in] lhs The message array on the left hand size of the equality operator.
 * \param[in] rhs The message array on the right hand size of the equality operator.
 * \return true if message arrays are equal in size and content, otherwise false.
 */
ROSIDL_GENERATOR_C_PUBLIC_hippocampus_ros2_msgs
bool
hippocampus_ros2_msgs__msg__GraphSnapshot__Sequence__are_equal(const hippocampus_ros2_msgs__msg__GraphSnapshot__Sequence * lhs, const hippocampus_ros2_msgs__msg__GraphSnapshot__Sequence * rhs);

/// Copy an array of msg/GraphSnapshot messages.
/**
 * This functions performs a deep copy, as opposed to the shallow copy that
 * plain assignment yields.
 *
 * \param[in] input The source array pointer.
 * \param[out] output The target array pointer, which must
 *   have been initialized before calling this function.
 * \return true if successful, or false if either pointer
 *   is null or memory allocation fails.
 */
ROSIDL_GENERATOR_C_PUBLIC_hippocampus_ros2_msgs
bool
hippocampus_ros2_msgs__msg__GraphSnapshot__Sequence__copy(
  const hippocampus_ros2_msgs__msg__GraphSnapshot__Sequence * input,
  hippocampus_ros2_msgs__msg__GraphSnapshot__Sequence * output);

#ifdef __cplusplus
}
#endif

#endif  // HIPPOCAMPUS_ROS2_MSGS__MSG__DETAIL__GRAPH_SNAPSHOT__FUNCTIONS_H_
