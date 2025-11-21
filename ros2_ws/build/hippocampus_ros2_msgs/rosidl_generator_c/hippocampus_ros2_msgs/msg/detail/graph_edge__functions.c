// generated from rosidl_generator_c/resource/idl__functions.c.em
// with input from hippocampus_ros2_msgs:msg/GraphEdge.idl
// generated code does not contain a copyright notice
#include "hippocampus_ros2_msgs/msg/detail/graph_edge__functions.h"

#include <assert.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>

#include "rcutils/allocator.h"


bool
hippocampus_ros2_msgs__msg__GraphEdge__init(hippocampus_ros2_msgs__msg__GraphEdge * msg)
{
  if (!msg) {
    return false;
  }
  // u
  // v
  // length
  // traversable
  return true;
}

void
hippocampus_ros2_msgs__msg__GraphEdge__fini(hippocampus_ros2_msgs__msg__GraphEdge * msg)
{
  if (!msg) {
    return;
  }
  // u
  // v
  // length
  // traversable
}

bool
hippocampus_ros2_msgs__msg__GraphEdge__are_equal(const hippocampus_ros2_msgs__msg__GraphEdge * lhs, const hippocampus_ros2_msgs__msg__GraphEdge * rhs)
{
  if (!lhs || !rhs) {
    return false;
  }
  // u
  if (lhs->u != rhs->u) {
    return false;
  }
  // v
  if (lhs->v != rhs->v) {
    return false;
  }
  // length
  if (lhs->length != rhs->length) {
    return false;
  }
  // traversable
  if (lhs->traversable != rhs->traversable) {
    return false;
  }
  return true;
}

bool
hippocampus_ros2_msgs__msg__GraphEdge__copy(
  const hippocampus_ros2_msgs__msg__GraphEdge * input,
  hippocampus_ros2_msgs__msg__GraphEdge * output)
{
  if (!input || !output) {
    return false;
  }
  // u
  output->u = input->u;
  // v
  output->v = input->v;
  // length
  output->length = input->length;
  // traversable
  output->traversable = input->traversable;
  return true;
}

hippocampus_ros2_msgs__msg__GraphEdge *
hippocampus_ros2_msgs__msg__GraphEdge__create()
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  hippocampus_ros2_msgs__msg__GraphEdge * msg = (hippocampus_ros2_msgs__msg__GraphEdge *)allocator.allocate(sizeof(hippocampus_ros2_msgs__msg__GraphEdge), allocator.state);
  if (!msg) {
    return NULL;
  }
  memset(msg, 0, sizeof(hippocampus_ros2_msgs__msg__GraphEdge));
  bool success = hippocampus_ros2_msgs__msg__GraphEdge__init(msg);
  if (!success) {
    allocator.deallocate(msg, allocator.state);
    return NULL;
  }
  return msg;
}

void
hippocampus_ros2_msgs__msg__GraphEdge__destroy(hippocampus_ros2_msgs__msg__GraphEdge * msg)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  if (msg) {
    hippocampus_ros2_msgs__msg__GraphEdge__fini(msg);
  }
  allocator.deallocate(msg, allocator.state);
}


bool
hippocampus_ros2_msgs__msg__GraphEdge__Sequence__init(hippocampus_ros2_msgs__msg__GraphEdge__Sequence * array, size_t size)
{
  if (!array) {
    return false;
  }
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  hippocampus_ros2_msgs__msg__GraphEdge * data = NULL;

  if (size) {
    data = (hippocampus_ros2_msgs__msg__GraphEdge *)allocator.zero_allocate(size, sizeof(hippocampus_ros2_msgs__msg__GraphEdge), allocator.state);
    if (!data) {
      return false;
    }
    // initialize all array elements
    size_t i;
    for (i = 0; i < size; ++i) {
      bool success = hippocampus_ros2_msgs__msg__GraphEdge__init(&data[i]);
      if (!success) {
        break;
      }
    }
    if (i < size) {
      // if initialization failed finalize the already initialized array elements
      for (; i > 0; --i) {
        hippocampus_ros2_msgs__msg__GraphEdge__fini(&data[i - 1]);
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
hippocampus_ros2_msgs__msg__GraphEdge__Sequence__fini(hippocampus_ros2_msgs__msg__GraphEdge__Sequence * array)
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
      hippocampus_ros2_msgs__msg__GraphEdge__fini(&array->data[i]);
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

hippocampus_ros2_msgs__msg__GraphEdge__Sequence *
hippocampus_ros2_msgs__msg__GraphEdge__Sequence__create(size_t size)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  hippocampus_ros2_msgs__msg__GraphEdge__Sequence * array = (hippocampus_ros2_msgs__msg__GraphEdge__Sequence *)allocator.allocate(sizeof(hippocampus_ros2_msgs__msg__GraphEdge__Sequence), allocator.state);
  if (!array) {
    return NULL;
  }
  bool success = hippocampus_ros2_msgs__msg__GraphEdge__Sequence__init(array, size);
  if (!success) {
    allocator.deallocate(array, allocator.state);
    return NULL;
  }
  return array;
}

void
hippocampus_ros2_msgs__msg__GraphEdge__Sequence__destroy(hippocampus_ros2_msgs__msg__GraphEdge__Sequence * array)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  if (array) {
    hippocampus_ros2_msgs__msg__GraphEdge__Sequence__fini(array);
  }
  allocator.deallocate(array, allocator.state);
}

bool
hippocampus_ros2_msgs__msg__GraphEdge__Sequence__are_equal(const hippocampus_ros2_msgs__msg__GraphEdge__Sequence * lhs, const hippocampus_ros2_msgs__msg__GraphEdge__Sequence * rhs)
{
  if (!lhs || !rhs) {
    return false;
  }
  if (lhs->size != rhs->size) {
    return false;
  }
  for (size_t i = 0; i < lhs->size; ++i) {
    if (!hippocampus_ros2_msgs__msg__GraphEdge__are_equal(&(lhs->data[i]), &(rhs->data[i]))) {
      return false;
    }
  }
  return true;
}

bool
hippocampus_ros2_msgs__msg__GraphEdge__Sequence__copy(
  const hippocampus_ros2_msgs__msg__GraphEdge__Sequence * input,
  hippocampus_ros2_msgs__msg__GraphEdge__Sequence * output)
{
  if (!input || !output) {
    return false;
  }
  if (output->capacity < input->size) {
    const size_t allocation_size =
      input->size * sizeof(hippocampus_ros2_msgs__msg__GraphEdge);
    rcutils_allocator_t allocator = rcutils_get_default_allocator();
    hippocampus_ros2_msgs__msg__GraphEdge * data =
      (hippocampus_ros2_msgs__msg__GraphEdge *)allocator.reallocate(
      output->data, allocation_size, allocator.state);
    if (!data) {
      return false;
    }
    // If reallocation succeeded, memory may or may not have been moved
    // to fulfill the allocation request, invalidating output->data.
    output->data = data;
    for (size_t i = output->capacity; i < input->size; ++i) {
      if (!hippocampus_ros2_msgs__msg__GraphEdge__init(&output->data[i])) {
        // If initialization of any new item fails, roll back
        // all previously initialized items. Existing items
        // in output are to be left unmodified.
        for (; i-- > output->capacity; ) {
          hippocampus_ros2_msgs__msg__GraphEdge__fini(&output->data[i]);
        }
        return false;
      }
    }
    output->capacity = input->size;
  }
  output->size = input->size;
  for (size_t i = 0; i < input->size; ++i) {
    if (!hippocampus_ros2_msgs__msg__GraphEdge__copy(
        &(input->data[i]), &(output->data[i])))
    {
      return false;
    }
  }
  return true;
}
