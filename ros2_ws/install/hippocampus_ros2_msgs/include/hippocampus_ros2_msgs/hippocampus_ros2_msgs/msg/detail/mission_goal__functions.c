// generated from rosidl_generator_c/resource/idl__functions.c.em
// with input from hippocampus_ros2_msgs:msg/MissionGoal.idl
// generated code does not contain a copyright notice
#include "hippocampus_ros2_msgs/msg/detail/mission_goal__functions.h"

#include <assert.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>

#include "rcutils/allocator.h"


// Include directives for member types
// Member `point_position`
// Member `region_center`
#include "geometry_msgs/msg/detail/point__functions.h"
// Member `frame_id`
#include "rosidl_runtime_c/string_functions.h"
// Member `stamp`
#include "builtin_interfaces/msg/detail/time__functions.h"

bool
hippocampus_ros2_msgs__msg__MissionGoal__init(hippocampus_ros2_msgs__msg__MissionGoal * msg)
{
  if (!msg) {
    return false;
  }
  // goal_type
  // point_position
  if (!geometry_msgs__msg__Point__init(&msg->point_position)) {
    hippocampus_ros2_msgs__msg__MissionGoal__fini(msg);
    return false;
  }
  // point_tolerance
  // node_id
  // region_center
  if (!geometry_msgs__msg__Point__init(&msg->region_center)) {
    hippocampus_ros2_msgs__msg__MissionGoal__fini(msg);
    return false;
  }
  // region_radius
  // frame_id
  if (!rosidl_runtime_c__String__init(&msg->frame_id)) {
    hippocampus_ros2_msgs__msg__MissionGoal__fini(msg);
    return false;
  }
  // timeout
  // is_reached
  // stamp
  if (!builtin_interfaces__msg__Time__init(&msg->stamp)) {
    hippocampus_ros2_msgs__msg__MissionGoal__fini(msg);
    return false;
  }
  return true;
}

void
hippocampus_ros2_msgs__msg__MissionGoal__fini(hippocampus_ros2_msgs__msg__MissionGoal * msg)
{
  if (!msg) {
    return;
  }
  // goal_type
  // point_position
  geometry_msgs__msg__Point__fini(&msg->point_position);
  // point_tolerance
  // node_id
  // region_center
  geometry_msgs__msg__Point__fini(&msg->region_center);
  // region_radius
  // frame_id
  rosidl_runtime_c__String__fini(&msg->frame_id);
  // timeout
  // is_reached
  // stamp
  builtin_interfaces__msg__Time__fini(&msg->stamp);
}

bool
hippocampus_ros2_msgs__msg__MissionGoal__are_equal(const hippocampus_ros2_msgs__msg__MissionGoal * lhs, const hippocampus_ros2_msgs__msg__MissionGoal * rhs)
{
  if (!lhs || !rhs) {
    return false;
  }
  // goal_type
  if (lhs->goal_type != rhs->goal_type) {
    return false;
  }
  // point_position
  if (!geometry_msgs__msg__Point__are_equal(
      &(lhs->point_position), &(rhs->point_position)))
  {
    return false;
  }
  // point_tolerance
  if (lhs->point_tolerance != rhs->point_tolerance) {
    return false;
  }
  // node_id
  if (lhs->node_id != rhs->node_id) {
    return false;
  }
  // region_center
  if (!geometry_msgs__msg__Point__are_equal(
      &(lhs->region_center), &(rhs->region_center)))
  {
    return false;
  }
  // region_radius
  if (lhs->region_radius != rhs->region_radius) {
    return false;
  }
  // frame_id
  if (!rosidl_runtime_c__String__are_equal(
      &(lhs->frame_id), &(rhs->frame_id)))
  {
    return false;
  }
  // timeout
  if (lhs->timeout != rhs->timeout) {
    return false;
  }
  // is_reached
  if (lhs->is_reached != rhs->is_reached) {
    return false;
  }
  // stamp
  if (!builtin_interfaces__msg__Time__are_equal(
      &(lhs->stamp), &(rhs->stamp)))
  {
    return false;
  }
  return true;
}

bool
hippocampus_ros2_msgs__msg__MissionGoal__copy(
  const hippocampus_ros2_msgs__msg__MissionGoal * input,
  hippocampus_ros2_msgs__msg__MissionGoal * output)
{
  if (!input || !output) {
    return false;
  }
  // goal_type
  output->goal_type = input->goal_type;
  // point_position
  if (!geometry_msgs__msg__Point__copy(
      &(input->point_position), &(output->point_position)))
  {
    return false;
  }
  // point_tolerance
  output->point_tolerance = input->point_tolerance;
  // node_id
  output->node_id = input->node_id;
  // region_center
  if (!geometry_msgs__msg__Point__copy(
      &(input->region_center), &(output->region_center)))
  {
    return false;
  }
  // region_radius
  output->region_radius = input->region_radius;
  // frame_id
  if (!rosidl_runtime_c__String__copy(
      &(input->frame_id), &(output->frame_id)))
  {
    return false;
  }
  // timeout
  output->timeout = input->timeout;
  // is_reached
  output->is_reached = input->is_reached;
  // stamp
  if (!builtin_interfaces__msg__Time__copy(
      &(input->stamp), &(output->stamp)))
  {
    return false;
  }
  return true;
}

hippocampus_ros2_msgs__msg__MissionGoal *
hippocampus_ros2_msgs__msg__MissionGoal__create()
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  hippocampus_ros2_msgs__msg__MissionGoal * msg = (hippocampus_ros2_msgs__msg__MissionGoal *)allocator.allocate(sizeof(hippocampus_ros2_msgs__msg__MissionGoal), allocator.state);
  if (!msg) {
    return NULL;
  }
  memset(msg, 0, sizeof(hippocampus_ros2_msgs__msg__MissionGoal));
  bool success = hippocampus_ros2_msgs__msg__MissionGoal__init(msg);
  if (!success) {
    allocator.deallocate(msg, allocator.state);
    return NULL;
  }
  return msg;
}

void
hippocampus_ros2_msgs__msg__MissionGoal__destroy(hippocampus_ros2_msgs__msg__MissionGoal * msg)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  if (msg) {
    hippocampus_ros2_msgs__msg__MissionGoal__fini(msg);
  }
  allocator.deallocate(msg, allocator.state);
}


bool
hippocampus_ros2_msgs__msg__MissionGoal__Sequence__init(hippocampus_ros2_msgs__msg__MissionGoal__Sequence * array, size_t size)
{
  if (!array) {
    return false;
  }
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  hippocampus_ros2_msgs__msg__MissionGoal * data = NULL;

  if (size) {
    data = (hippocampus_ros2_msgs__msg__MissionGoal *)allocator.zero_allocate(size, sizeof(hippocampus_ros2_msgs__msg__MissionGoal), allocator.state);
    if (!data) {
      return false;
    }
    // initialize all array elements
    size_t i;
    for (i = 0; i < size; ++i) {
      bool success = hippocampus_ros2_msgs__msg__MissionGoal__init(&data[i]);
      if (!success) {
        break;
      }
    }
    if (i < size) {
      // if initialization failed finalize the already initialized array elements
      for (; i > 0; --i) {
        hippocampus_ros2_msgs__msg__MissionGoal__fini(&data[i - 1]);
      }
      allocator.deallocate(data, allocator.state);
      return false;
    }
  }
  array->data = data;
  array->size = size;
  array->capacity = size;
  return true;
}

void
hippocampus_ros2_msgs__msg__MissionGoal__Sequence__fini(hippocampus_ros2_msgs__msg__MissionGoal__Sequence * array)
{
  if (!array) {
    return;
  }
  rcutils_allocator_t allocator = rcutils_get_default_allocator();

  if (array->data) {
    // ensure that data and capacity values are consistent
    assert(array->capacity > 0);
    // finalize all array elements
    for (size_t i = 0; i < array->capacity; ++i) {
      hippocampus_ros2_msgs__msg__MissionGoal__fini(&array->data[i]);
    }
    allocator.deallocate(array->data, allocator.state);
    array->data = NULL;
    array->size = 0;
    array->capacity = 0;
  } else {
    // ensure that data, size, and capacity values are consistent
    assert(0 == array->size);
    assert(0 == array->capacity);
  }
}

hippocampus_ros2_msgs__msg__MissionGoal__Sequence *
hippocampus_ros2_msgs__msg__MissionGoal__Sequence__create(size_t size)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  hippocampus_ros2_msgs__msg__MissionGoal__Sequence * array = (hippocampus_ros2_msgs__msg__MissionGoal__Sequence *)allocator.allocate(sizeof(hippocampus_ros2_msgs__msg__MissionGoal__Sequence), allocator.state);
  if (!array) {
    return NULL;
  }
  bool success = hippocampus_ros2_msgs__msg__MissionGoal__Sequence__init(array, size);
  if (!success) {
    allocator.deallocate(array, allocator.state);
    return NULL;
  }
  return array;
}

void
hippocampus_ros2_msgs__msg__MissionGoal__Sequence__destroy(hippocampus_ros2_msgs__msg__MissionGoal__Sequence * array)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  if (array) {
    hippocampus_ros2_msgs__msg__MissionGoal__Sequence__fini(array);
  }
  allocator.deallocate(array, allocator.state);
}

bool
hippocampus_ros2_msgs__msg__MissionGoal__Sequence__are_equal(const hippocampus_ros2_msgs__msg__MissionGoal__Sequence * lhs, const hippocampus_ros2_msgs__msg__MissionGoal__Sequence * rhs)
{
  if (!lhs || !rhs) {
    return false;
  }
  if (lhs->size != rhs->size) {
    return false;
  }
  for (size_t i = 0; i < lhs->size; ++i) {
    if (!hippocampus_ros2_msgs__msg__MissionGoal__are_equal(&(lhs->data[i]), &(rhs->data[i]))) {
      return false;
    }
  }
  return true;
}

bool
hippocampus_ros2_msgs__msg__MissionGoal__Sequence__copy(
  const hippocampus_ros2_msgs__msg__MissionGoal__Sequence * input,
  hippocampus_ros2_msgs__msg__MissionGoal__Sequence * output)
{
  if (!input || !output) {
    return false;
  }
  if (output->capacity < input->size) {
    const size_t allocation_size =
      input->size * sizeof(hippocampus_ros2_msgs__msg__MissionGoal);
    rcutils_allocator_t allocator = rcutils_get_default_allocator();
    hippocampus_ros2_msgs__msg__MissionGoal * data =
      (hippocampus_ros2_msgs__msg__MissionGoal *)allocator.reallocate(
      output->data, allocation_size, allocator.state);
    if (!data) {
      return false;
    }
    // If reallocation succeeded, memory may or may not have been moved
    // to fulfill the allocation request, invalidating output->data.
    output->data = data;
    for (size_t i = output->capacity; i < input->size; ++i) {
      if (!hippocampus_ros2_msgs__msg__MissionGoal__init(&output->data[i])) {
        // If initialization of any new item fails, roll back
        // all previously initialized items. Existing items
        // in output are to be left unmodified.
        for (; i-- > output->capacity; ) {
          hippocampus_ros2_msgs__msg__MissionGoal__fini(&output->data[i]);
        }
        return false;
      }
    }
    output->capacity = input->size;
  }
  output->size = input->size;
  for (size_t i = 0; i < input->size; ++i) {
    if (!hippocampus_ros2_msgs__msg__MissionGoal__copy(
        &(input->data[i]), &(output->data[i])))
    {
      return false;
    }
  }
  return true;
}
