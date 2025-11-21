// generated from rosidl_generator_c/resource/idl__functions.c.em
// with input from hippocampus_ros2_msgs:msg/GraphSnapshot.idl
// generated code does not contain a copyright notice
#include "hippocampus_ros2_msgs/msg/detail/graph_snapshot__functions.h"

#include <assert.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>

#include "rcutils/allocator.h"


// Include directives for member types
// Member `frame_id`
#include "rosidl_runtime_c/string_functions.h"
// Member `stamp`
// Member `last_updated`
#include "builtin_interfaces/msg/detail/time__functions.h"
// Member `nodes`
#include "hippocampus_ros2_msgs/msg/detail/graph_node__functions.h"
// Member `edges`
#include "hippocampus_ros2_msgs/msg/detail/graph_edge__functions.h"

bool
hippocampus_ros2_msgs__msg__GraphSnapshot__init(hippocampus_ros2_msgs__msg__GraphSnapshot * msg)
{
  if (!msg) {
    return false;
  }
  // epoch_id
  // frame_id
  if (!rosidl_runtime_c__String__init(&msg->frame_id)) {
    hippocampus_ros2_msgs__msg__GraphSnapshot__fini(msg);
    return false;
  }
  // stamp
  if (!builtin_interfaces__msg__Time__init(&msg->stamp)) {
    hippocampus_ros2_msgs__msg__GraphSnapshot__fini(msg);
    return false;
  }
  // last_updated
  if (!builtin_interfaces__msg__Time__init(&msg->last_updated)) {
    hippocampus_ros2_msgs__msg__GraphSnapshot__fini(msg);
    return false;
  }
  // update_rate
  // staleness_warning
  // nodes
  if (!hippocampus_ros2_msgs__msg__GraphNode__Sequence__init(&msg->nodes, 0)) {
    hippocampus_ros2_msgs__msg__GraphSnapshot__fini(msg);
    return false;
  }
  // edges
  if (!hippocampus_ros2_msgs__msg__GraphEdge__Sequence__init(&msg->edges, 0)) {
    hippocampus_ros2_msgs__msg__GraphSnapshot__fini(msg);
    return false;
  }
  return true;
}

void
hippocampus_ros2_msgs__msg__GraphSnapshot__fini(hippocampus_ros2_msgs__msg__GraphSnapshot * msg)
{
  if (!msg) {
    return;
  }
  // epoch_id
  // frame_id
  rosidl_runtime_c__String__fini(&msg->frame_id);
  // stamp
  builtin_interfaces__msg__Time__fini(&msg->stamp);
  // last_updated
  builtin_interfaces__msg__Time__fini(&msg->last_updated);
  // update_rate
  // staleness_warning
  // nodes
  hippocampus_ros2_msgs__msg__GraphNode__Sequence__fini(&msg->nodes);
  // edges
  hippocampus_ros2_msgs__msg__GraphEdge__Sequence__fini(&msg->edges);
}

bool
hippocampus_ros2_msgs__msg__GraphSnapshot__are_equal(const hippocampus_ros2_msgs__msg__GraphSnapshot * lhs, const hippocampus_ros2_msgs__msg__GraphSnapshot * rhs)
{
  if (!lhs || !rhs) {
    return false;
  }
  // epoch_id
  if (lhs->epoch_id != rhs->epoch_id) {
    return false;
  }
  // frame_id
  if (!rosidl_runtime_c__String__are_equal(
      &(lhs->frame_id), &(rhs->frame_id)))
  {
    return false;
  }
  // stamp
  if (!builtin_interfaces__msg__Time__are_equal(
      &(lhs->stamp), &(rhs->stamp)))
  {
    return false;
  }
  // last_updated
  if (!builtin_interfaces__msg__Time__are_equal(
      &(lhs->last_updated), &(rhs->last_updated)))
  {
    return false;
  }
  // update_rate
  if (lhs->update_rate != rhs->update_rate) {
    return false;
  }
  // staleness_warning
  if (lhs->staleness_warning != rhs->staleness_warning) {
    return false;
  }
  // nodes
  if (!hippocampus_ros2_msgs__msg__GraphNode__Sequence__are_equal(
      &(lhs->nodes), &(rhs->nodes)))
  {
    return false;
  }
  // edges
  if (!hippocampus_ros2_msgs__msg__GraphEdge__Sequence__are_equal(
      &(lhs->edges), &(rhs->edges)))
  {
    return false;
  }
  return true;
}

bool
hippocampus_ros2_msgs__msg__GraphSnapshot__copy(
  const hippocampus_ros2_msgs__msg__GraphSnapshot * input,
  hippocampus_ros2_msgs__msg__GraphSnapshot * output)
{
  if (!input || !output) {
    return false;
  }
  // epoch_id
  output->epoch_id = input->epoch_id;
  // frame_id
  if (!rosidl_runtime_c__String__copy(
      &(input->frame_id), &(output->frame_id)))
  {
    return false;
  }
  // stamp
  if (!builtin_interfaces__msg__Time__copy(
      &(input->stamp), &(output->stamp)))
  {
    return false;
  }
  // last_updated
  if (!builtin_interfaces__msg__Time__copy(
      &(input->last_updated), &(output->last_updated)))
  {
    return false;
  }
  // update_rate
  output->update_rate = input->update_rate;
  // staleness_warning
  output->staleness_warning = input->staleness_warning;
  // nodes
  if (!hippocampus_ros2_msgs__msg__GraphNode__Sequence__copy(
      &(input->nodes), &(output->nodes)))
  {
    return false;
  }
  // edges
  if (!hippocampus_ros2_msgs__msg__GraphEdge__Sequence__copy(
      &(input->edges), &(output->edges)))
  {
    return false;
  }
  return true;
}

hippocampus_ros2_msgs__msg__GraphSnapshot *
hippocampus_ros2_msgs__msg__GraphSnapshot__create()
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  hippocampus_ros2_msgs__msg__GraphSnapshot * msg = (hippocampus_ros2_msgs__msg__GraphSnapshot *)allocator.allocate(sizeof(hippocampus_ros2_msgs__msg__GraphSnapshot), allocator.state);
  if (!msg) {
    return NULL;
  }
  memset(msg, 0, sizeof(hippocampus_ros2_msgs__msg__GraphSnapshot));
  bool success = hippocampus_ros2_msgs__msg__GraphSnapshot__init(msg);
  if (!success) {
    allocator.deallocate(msg, allocator.state);
    return NULL;
  }
  return msg;
}

void
hippocampus_ros2_msgs__msg__GraphSnapshot__destroy(hippocampus_ros2_msgs__msg__GraphSnapshot * msg)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  if (msg) {
    hippocampus_ros2_msgs__msg__GraphSnapshot__fini(msg);
  }
  allocator.deallocate(msg, allocator.state);
}


bool
hippocampus_ros2_msgs__msg__GraphSnapshot__Sequence__init(hippocampus_ros2_msgs__msg__GraphSnapshot__Sequence * array, size_t size)
{
  if (!array) {
    return false;
  }
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  hippocampus_ros2_msgs__msg__GraphSnapshot * data = NULL;

  if (size) {
    data = (hippocampus_ros2_msgs__msg__GraphSnapshot *)allocator.zero_allocate(size, sizeof(hippocampus_ros2_msgs__msg__GraphSnapshot), allocator.state);
    if (!data) {
      return false;
    }
    // initialize all array elements
    size_t i;
    for (i = 0; i < size; ++i) {
      bool success = hippocampus_ros2_msgs__msg__GraphSnapshot__init(&data[i]);
      if (!success) {
        break;
      }
    }
    if (i < size) {
      // if initialization failed finalize the already initialized array elements
      for (; i > 0; --i) {
        hippocampus_ros2_msgs__msg__GraphSnapshot__fini(&data[i - 1]);
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
hippocampus_ros2_msgs__msg__GraphSnapshot__Sequence__fini(hippocampus_ros2_msgs__msg__GraphSnapshot__Sequence * array)
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
      hippocampus_ros2_msgs__msg__GraphSnapshot__fini(&array->data[i]);
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

hippocampus_ros2_msgs__msg__GraphSnapshot__Sequence *
hippocampus_ros2_msgs__msg__GraphSnapshot__Sequence__create(size_t size)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  hippocampus_ros2_msgs__msg__GraphSnapshot__Sequence * array = (hippocampus_ros2_msgs__msg__GraphSnapshot__Sequence *)allocator.allocate(sizeof(hippocampus_ros2_msgs__msg__GraphSnapshot__Sequence), allocator.state);
  if (!array) {
    return NULL;
  }
  bool success = hippocampus_ros2_msgs__msg__GraphSnapshot__Sequence__init(array, size);
  if (!success) {
    allocator.deallocate(array, allocator.state);
    return NULL;
  }
  return array;
}

void
hippocampus_ros2_msgs__msg__GraphSnapshot__Sequence__destroy(hippocampus_ros2_msgs__msg__GraphSnapshot__Sequence * array)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  if (array) {
    hippocampus_ros2_msgs__msg__GraphSnapshot__Sequence__fini(array);
  }
  allocator.deallocate(array, allocator.state);
}

bool
hippocampus_ros2_msgs__msg__GraphSnapshot__Sequence__are_equal(const hippocampus_ros2_msgs__msg__GraphSnapshot__Sequence * lhs, const hippocampus_ros2_msgs__msg__GraphSnapshot__Sequence * rhs)
{
  if (!lhs || !rhs) {
    return false;
  }
  if (lhs->size != rhs->size) {
    return false;
  }
  for (size_t i = 0; i < lhs->size; ++i) {
    if (!hippocampus_ros2_msgs__msg__GraphSnapshot__are_equal(&(lhs->data[i]), &(rhs->data[i]))) {
      return false;
    }
  }
  return true;
}

bool
hippocampus_ros2_msgs__msg__GraphSnapshot__Sequence__copy(
  const hippocampus_ros2_msgs__msg__GraphSnapshot__Sequence * input,
  hippocampus_ros2_msgs__msg__GraphSnapshot__Sequence * output)
{
  if (!input || !output) {
    return false;
  }
  if (output->capacity < input->size) {
    const size_t allocation_size =
      input->size * sizeof(hippocampus_ros2_msgs__msg__GraphSnapshot);
    rcutils_allocator_t allocator = rcutils_get_default_allocator();
    hippocampus_ros2_msgs__msg__GraphSnapshot * data =
      (hippocampus_ros2_msgs__msg__GraphSnapshot *)allocator.reallocate(
      output->data, allocation_size, allocator.state);
    if (!data) {
      return false;
    }
    // If reallocation succeeded, memory may or may not have been moved
    // to fulfill the allocation request, invalidating output->data.
    output->data = data;
    for (size_t i = output->capacity; i < input->size; ++i) {
      if (!hippocampus_ros2_msgs__msg__GraphSnapshot__init(&output->data[i])) {
        // If initialization of any new item fails, roll back
        // all previously initialized items. Existing items
        // in output are to be left unmodified.
        for (; i-- > output->capacity; ) {
          hippocampus_ros2_msgs__msg__GraphSnapshot__fini(&output->data[i]);
        }
        return false;
      }
    }
    output->capacity = input->size;
  }
  output->size = input->size;
  for (size_t i = 0; i < input->size; ++i) {
    if (!hippocampus_ros2_msgs__msg__GraphSnapshot__copy(
        &(input->data[i]), &(output->data[i])))
    {
      return false;
    }
  }
  return true;
}
