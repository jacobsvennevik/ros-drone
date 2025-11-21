// generated from rosidl_generator_c/resource/idl__functions.c.em
// with input from hippocampus_ros2_msgs:msg/GraphNode.idl
// generated code does not contain a copyright notice
#include "hippocampus_ros2_msgs/msg/detail/graph_node__functions.h"

#include <assert.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>

#include "rcutils/allocator.h"


// Include directives for member types
// Member `position`
#include "geometry_msgs/msg/detail/point__functions.h"
// Member `normal`
#include "geometry_msgs/msg/detail/vector3__functions.h"
// Member `tags`
#include "rosidl_runtime_c/string_functions.h"

bool
hippocampus_ros2_msgs__msg__GraphNode__init(hippocampus_ros2_msgs__msg__GraphNode * msg)
{
  if (!msg) {
    return false;
  }
  // node_id
  // position
  if (!geometry_msgs__msg__Point__init(&msg->position)) {
    hippocampus_ros2_msgs__msg__GraphNode__fini(msg);
    return false;
  }
  // normal
  if (!geometry_msgs__msg__Vector3__init(&msg->normal)) {
    hippocampus_ros2_msgs__msg__GraphNode__fini(msg);
    return false;
  }
  // degree
  // tags
  if (!rosidl_runtime_c__String__Sequence__init(&msg->tags, 0)) {
    hippocampus_ros2_msgs__msg__GraphNode__fini(msg);
    return false;
  }
  return true;
}

void
hippocampus_ros2_msgs__msg__GraphNode__fini(hippocampus_ros2_msgs__msg__GraphNode * msg)
{
  if (!msg) {
    return;
  }
  // node_id
  // position
  geometry_msgs__msg__Point__fini(&msg->position);
  // normal
  geometry_msgs__msg__Vector3__fini(&msg->normal);
  // degree
  // tags
  rosidl_runtime_c__String__Sequence__fini(&msg->tags);
}

bool
hippocampus_ros2_msgs__msg__GraphNode__are_equal(const hippocampus_ros2_msgs__msg__GraphNode * lhs, const hippocampus_ros2_msgs__msg__GraphNode * rhs)
{
  if (!lhs || !rhs) {
    return false;
  }
  // node_id
  if (lhs->node_id != rhs->node_id) {
    return false;
  }
  // position
  if (!geometry_msgs__msg__Point__are_equal(
      &(lhs->position), &(rhs->position)))
  {
    return false;
  }
  // normal
  if (!geometry_msgs__msg__Vector3__are_equal(
      &(lhs->normal), &(rhs->normal)))
  {
    return false;
  }
  // degree
  if (lhs->degree != rhs->degree) {
    return false;
  }
  // tags
  if (!rosidl_runtime_c__String__Sequence__are_equal(
      &(lhs->tags), &(rhs->tags)))
  {
    return false;
  }
  return true;
}

bool
hippocampus_ros2_msgs__msg__GraphNode__copy(
  const hippocampus_ros2_msgs__msg__GraphNode * input,
  hippocampus_ros2_msgs__msg__GraphNode * output)
{
  if (!input || !output) {
    return false;
  }
  // node_id
  output->node_id = input->node_id;
  // position
  if (!geometry_msgs__msg__Point__copy(
      &(input->position), &(output->position)))
  {
    return false;
  }
  // normal
  if (!geometry_msgs__msg__Vector3__copy(
      &(input->normal), &(output->normal)))
  {
    return false;
  }
  // degree
  output->degree = input->degree;
  // tags
  if (!rosidl_runtime_c__String__Sequence__copy(
      &(input->tags), &(output->tags)))
  {
    return false;
  }
  return true;
}

hippocampus_ros2_msgs__msg__GraphNode *
hippocampus_ros2_msgs__msg__GraphNode__create()
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  hippocampus_ros2_msgs__msg__GraphNode * msg = (hippocampus_ros2_msgs__msg__GraphNode *)allocator.allocate(sizeof(hippocampus_ros2_msgs__msg__GraphNode), allocator.state);
  if (!msg) {
    return NULL;
  }
  memset(msg, 0, sizeof(hippocampus_ros2_msgs__msg__GraphNode));
  bool success = hippocampus_ros2_msgs__msg__GraphNode__init(msg);
  if (!success) {
    allocator.deallocate(msg, allocator.state);
    return NULL;
  }
  return msg;
}

void
hippocampus_ros2_msgs__msg__GraphNode__destroy(hippocampus_ros2_msgs__msg__GraphNode * msg)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  if (msg) {
    hippocampus_ros2_msgs__msg__GraphNode__fini(msg);
  }
  allocator.deallocate(msg, allocator.state);
}


bool
hippocampus_ros2_msgs__msg__GraphNode__Sequence__init(hippocampus_ros2_msgs__msg__GraphNode__Sequence * array, size_t size)
{
  if (!array) {
    return false;
  }
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  hippocampus_ros2_msgs__msg__GraphNode * data = NULL;

  if (size) {
    data = (hippocampus_ros2_msgs__msg__GraphNode *)allocator.zero_allocate(size, sizeof(hippocampus_ros2_msgs__msg__GraphNode), allocator.state);
    if (!data) {
      return false;
    }
    // initialize all array elements
    size_t i;
    for (i = 0; i < size; ++i) {
      bool success = hippocampus_ros2_msgs__msg__GraphNode__init(&data[i]);
      if (!success) {
        break;
      }
    }
    if (i < size) {
      // if initialization failed finalize the already initialized array elements
      for (; i > 0; --i) {
        hippocampus_ros2_msgs__msg__GraphNode__fini(&data[i - 1]);
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
hippocampus_ros2_msgs__msg__GraphNode__Sequence__fini(hippocampus_ros2_msgs__msg__GraphNode__Sequence * array)
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
      hippocampus_ros2_msgs__msg__GraphNode__fini(&array->data[i]);
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

hippocampus_ros2_msgs__msg__GraphNode__Sequence *
hippocampus_ros2_msgs__msg__GraphNode__Sequence__create(size_t size)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  hippocampus_ros2_msgs__msg__GraphNode__Sequence * array = (hippocampus_ros2_msgs__msg__GraphNode__Sequence *)allocator.allocate(sizeof(hippocampus_ros2_msgs__msg__GraphNode__Sequence), allocator.state);
  if (!array) {
    return NULL;
  }
  bool success = hippocampus_ros2_msgs__msg__GraphNode__Sequence__init(array, size);
  if (!success) {
    allocator.deallocate(array, allocator.state);
    return NULL;
  }
  return array;
}

void
hippocampus_ros2_msgs__msg__GraphNode__Sequence__destroy(hippocampus_ros2_msgs__msg__GraphNode__Sequence * array)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  if (array) {
    hippocampus_ros2_msgs__msg__GraphNode__Sequence__fini(array);
  }
  allocator.deallocate(array, allocator.state);
}

bool
hippocampus_ros2_msgs__msg__GraphNode__Sequence__are_equal(const hippocampus_ros2_msgs__msg__GraphNode__Sequence * lhs, const hippocampus_ros2_msgs__msg__GraphNode__Sequence * rhs)
{
  if (!lhs || !rhs) {
    return false;
  }
  if (lhs->size != rhs->size) {
    return false;
  }
  for (size_t i = 0; i < lhs->size; ++i) {
    if (!hippocampus_ros2_msgs__msg__GraphNode__are_equal(&(lhs->data[i]), &(rhs->data[i]))) {
      return false;
    }
  }
  return true;
}

bool
hippocampus_ros2_msgs__msg__GraphNode__Sequence__copy(
  const hippocampus_ros2_msgs__msg__GraphNode__Sequence * input,
  hippocampus_ros2_msgs__msg__GraphNode__Sequence * output)
{
  if (!input || !output) {
    return false;
  }
  if (output->capacity < input->size) {
    const size_t allocation_size =
      input->size * sizeof(hippocampus_ros2_msgs__msg__GraphNode);
    rcutils_allocator_t allocator = rcutils_get_default_allocator();
    hippocampus_ros2_msgs__msg__GraphNode * data =
      (hippocampus_ros2_msgs__msg__GraphNode *)allocator.reallocate(
      output->data, allocation_size, allocator.state);
    if (!data) {
      return false;
    }
    // If reallocation succeeded, memory may or may not have been moved
    // to fulfill the allocation request, invalidating output->data.
    output->data = data;
    for (size_t i = output->capacity; i < input->size; ++i) {
      if (!hippocampus_ros2_msgs__msg__GraphNode__init(&output->data[i])) {
        // If initialization of any new item fails, roll back
        // all previously initialized items. Existing items
        // in output are to be left unmodified.
        for (; i-- > output->capacity; ) {
          hippocampus_ros2_msgs__msg__GraphNode__fini(&output->data[i]);
        }
        return false;
      }
    }
    output->capacity = input->size;
  }
  output->size = input->size;
  for (size_t i = 0; i < input->size; ++i) {
    if (!hippocampus_ros2_msgs__msg__GraphNode__copy(
        &(input->data[i]), &(output->data[i])))
    {
      return false;
    }
  }
  return true;
}
