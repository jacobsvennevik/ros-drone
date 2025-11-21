// generated from rosidl_generator_c/resource/idl__functions.c.em
// with input from hippocampus_ros2_msgs:msg/PolicyDecision.idl
// generated code does not contain a copyright notice
#include "hippocampus_ros2_msgs/msg/detail/policy_decision__functions.h"

#include <assert.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>

#include "rcutils/allocator.h"


// Include directives for member types
// Member `reason`
#include "rosidl_runtime_c/string_functions.h"
// Member `stamp`
#include "builtin_interfaces/msg/detail/time__functions.h"

bool
hippocampus_ros2_msgs__msg__PolicyDecision__init(hippocampus_ros2_msgs__msg__PolicyDecision * msg)
{
  if (!msg) {
    return false;
  }
  // linear_x
  // angular_z
  // linear_z
  // confidence
  // reason
  if (!rosidl_runtime_c__String__init(&msg->reason)) {
    hippocampus_ros2_msgs__msg__PolicyDecision__fini(msg);
    return false;
  }
  // next_waypoint
  // stamp
  if (!builtin_interfaces__msg__Time__init(&msg->stamp)) {
    hippocampus_ros2_msgs__msg__PolicyDecision__fini(msg);
    return false;
  }
  return true;
}

void
hippocampus_ros2_msgs__msg__PolicyDecision__fini(hippocampus_ros2_msgs__msg__PolicyDecision * msg)
{
  if (!msg) {
    return;
  }
  // linear_x
  // angular_z
  // linear_z
  // confidence
  // reason
  rosidl_runtime_c__String__fini(&msg->reason);
  // next_waypoint
  // stamp
  builtin_interfaces__msg__Time__fini(&msg->stamp);
}

bool
hippocampus_ros2_msgs__msg__PolicyDecision__are_equal(const hippocampus_ros2_msgs__msg__PolicyDecision * lhs, const hippocampus_ros2_msgs__msg__PolicyDecision * rhs)
{
  if (!lhs || !rhs) {
    return false;
  }
  // linear_x
  if (lhs->linear_x != rhs->linear_x) {
    return false;
  }
  // angular_z
  if (lhs->angular_z != rhs->angular_z) {
    return false;
  }
  // linear_z
  if (lhs->linear_z != rhs->linear_z) {
    return false;
  }
  // confidence
  if (lhs->confidence != rhs->confidence) {
    return false;
  }
  // reason
  if (!rosidl_runtime_c__String__are_equal(
      &(lhs->reason), &(rhs->reason)))
  {
    return false;
  }
  // next_waypoint
  if (lhs->next_waypoint != rhs->next_waypoint) {
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
hippocampus_ros2_msgs__msg__PolicyDecision__copy(
  const hippocampus_ros2_msgs__msg__PolicyDecision * input,
  hippocampus_ros2_msgs__msg__PolicyDecision * output)
{
  if (!input || !output) {
    return false;
  }
  // linear_x
  output->linear_x = input->linear_x;
  // angular_z
  output->angular_z = input->angular_z;
  // linear_z
  output->linear_z = input->linear_z;
  // confidence
  output->confidence = input->confidence;
  // reason
  if (!rosidl_runtime_c__String__copy(
      &(input->reason), &(output->reason)))
  {
    return false;
  }
  // next_waypoint
  output->next_waypoint = input->next_waypoint;
  // stamp
  if (!builtin_interfaces__msg__Time__copy(
      &(input->stamp), &(output->stamp)))
  {
    return false;
  }
  return true;
}

hippocampus_ros2_msgs__msg__PolicyDecision *
hippocampus_ros2_msgs__msg__PolicyDecision__create()
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  hippocampus_ros2_msgs__msg__PolicyDecision * msg = (hippocampus_ros2_msgs__msg__PolicyDecision *)allocator.allocate(sizeof(hippocampus_ros2_msgs__msg__PolicyDecision), allocator.state);
  if (!msg) {
    return NULL;
  }
  memset(msg, 0, sizeof(hippocampus_ros2_msgs__msg__PolicyDecision));
  bool success = hippocampus_ros2_msgs__msg__PolicyDecision__init(msg);
  if (!success) {
    allocator.deallocate(msg, allocator.state);
    return NULL;
  }
  return msg;
}

void
hippocampus_ros2_msgs__msg__PolicyDecision__destroy(hippocampus_ros2_msgs__msg__PolicyDecision * msg)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  if (msg) {
    hippocampus_ros2_msgs__msg__PolicyDecision__fini(msg);
  }
  allocator.deallocate(msg, allocator.state);
}


bool
hippocampus_ros2_msgs__msg__PolicyDecision__Sequence__init(hippocampus_ros2_msgs__msg__PolicyDecision__Sequence * array, size_t size)
{
  if (!array) {
    return false;
  }
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  hippocampus_ros2_msgs__msg__PolicyDecision * data = NULL;

  if (size) {
    data = (hippocampus_ros2_msgs__msg__PolicyDecision *)allocator.zero_allocate(size, sizeof(hippocampus_ros2_msgs__msg__PolicyDecision), allocator.state);
    if (!data) {
      return false;
    }
    // initialize all array elements
    size_t i;
    for (i = 0; i < size; ++i) {
      bool success = hippocampus_ros2_msgs__msg__PolicyDecision__init(&data[i]);
      if (!success) {
        break;
      }
    }
    if (i < size) {
      // if initialization failed finalize the already initialized array elements
      for (; i > 0; --i) {
        hippocampus_ros2_msgs__msg__PolicyDecision__fini(&data[i - 1]);
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
hippocampus_ros2_msgs__msg__PolicyDecision__Sequence__fini(hippocampus_ros2_msgs__msg__PolicyDecision__Sequence * array)
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
      hippocampus_ros2_msgs__msg__PolicyDecision__fini(&array->data[i]);
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

hippocampus_ros2_msgs__msg__PolicyDecision__Sequence *
hippocampus_ros2_msgs__msg__PolicyDecision__Sequence__create(size_t size)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  hippocampus_ros2_msgs__msg__PolicyDecision__Sequence * array = (hippocampus_ros2_msgs__msg__PolicyDecision__Sequence *)allocator.allocate(sizeof(hippocampus_ros2_msgs__msg__PolicyDecision__Sequence), allocator.state);
  if (!array) {
    return NULL;
  }
  bool success = hippocampus_ros2_msgs__msg__PolicyDecision__Sequence__init(array, size);
  if (!success) {
    allocator.deallocate(array, allocator.state);
    return NULL;
  }
  return array;
}

void
hippocampus_ros2_msgs__msg__PolicyDecision__Sequence__destroy(hippocampus_ros2_msgs__msg__PolicyDecision__Sequence * array)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  if (array) {
    hippocampus_ros2_msgs__msg__PolicyDecision__Sequence__fini(array);
  }
  allocator.deallocate(array, allocator.state);
}

bool
hippocampus_ros2_msgs__msg__PolicyDecision__Sequence__are_equal(const hippocampus_ros2_msgs__msg__PolicyDecision__Sequence * lhs, const hippocampus_ros2_msgs__msg__PolicyDecision__Sequence * rhs)
{
  if (!lhs || !rhs) {
    return false;
  }
  if (lhs->size != rhs->size) {
    return false;
  }
  for (size_t i = 0; i < lhs->size; ++i) {
    if (!hippocampus_ros2_msgs__msg__PolicyDecision__are_equal(&(lhs->data[i]), &(rhs->data[i]))) {
      return false;
    }
  }
  return true;
}

bool
hippocampus_ros2_msgs__msg__PolicyDecision__Sequence__copy(
  const hippocampus_ros2_msgs__msg__PolicyDecision__Sequence * input,
  hippocampus_ros2_msgs__msg__PolicyDecision__Sequence * output)
{
  if (!input || !output) {
    return false;
  }
  if (output->capacity < input->size) {
    const size_t allocation_size =
      input->size * sizeof(hippocampus_ros2_msgs__msg__PolicyDecision);
    rcutils_allocator_t allocator = rcutils_get_default_allocator();
    hippocampus_ros2_msgs__msg__PolicyDecision * data =
      (hippocampus_ros2_msgs__msg__PolicyDecision *)allocator.reallocate(
      output->data, allocation_size, allocator.state);
    if (!data) {
      return false;
    }
    // If reallocation succeeded, memory may or may not have been moved
    // to fulfill the allocation request, invalidating output->data.
    output->data = data;
    for (size_t i = output->capacity; i < input->size; ++i) {
      if (!hippocampus_ros2_msgs__msg__PolicyDecision__init(&output->data[i])) {
        // If initialization of any new item fails, roll back
        // all previously initialized items. Existing items
        // in output are to be left unmodified.
        for (; i-- > output->capacity; ) {
          hippocampus_ros2_msgs__msg__PolicyDecision__fini(&output->data[i]);
        }
        return false;
      }
    }
    output->capacity = input->size;
  }
  output->size = input->size;
  for (size_t i = 0; i < input->size; ++i) {
    if (!hippocampus_ros2_msgs__msg__PolicyDecision__copy(
        &(input->data[i]), &(output->data[i])))
    {
      return false;
    }
  }
  return true;
}
