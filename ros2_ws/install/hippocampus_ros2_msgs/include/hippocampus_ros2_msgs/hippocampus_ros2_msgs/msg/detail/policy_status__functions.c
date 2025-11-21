// generated from rosidl_generator_c/resource/idl__functions.c.em
// with input from hippocampus_ros2_msgs:msg/PolicyStatus.idl
// generated code does not contain a copyright notice
#include "hippocampus_ros2_msgs/msg/detail/policy_status__functions.h"

#include <assert.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>

#include "rcutils/allocator.h"


// Include directives for member types
// Member `current_reason`
#include "rosidl_runtime_c/string_functions.h"
// Member `stamp`
#include "builtin_interfaces/msg/detail/time__functions.h"

bool
hippocampus_ros2_msgs__msg__PolicyStatus__init(hippocampus_ros2_msgs__msg__PolicyStatus * msg)
{
  if (!msg) {
    return false;
  }
  // is_active
  // graph_stale
  // using_snn
  // hierarchical_enabled
  // feature_compute_time_ms
  // policy_decision_time_ms
  // safety_filter_time_ms
  // total_latency_ms
  // graph_nodes
  // graph_edges
  // graph_staleness_s
  // current_confidence
  // current_reason
  if (!rosidl_runtime_c__String__init(&msg->current_reason)) {
    hippocampus_ros2_msgs__msg__PolicyStatus__fini(msg);
    return false;
  }
  // current_waypoint
  // stamp
  if (!builtin_interfaces__msg__Time__init(&msg->stamp)) {
    hippocampus_ros2_msgs__msg__PolicyStatus__fini(msg);
    return false;
  }
  return true;
}

void
hippocampus_ros2_msgs__msg__PolicyStatus__fini(hippocampus_ros2_msgs__msg__PolicyStatus * msg)
{
  if (!msg) {
    return;
  }
  // is_active
  // graph_stale
  // using_snn
  // hierarchical_enabled
  // feature_compute_time_ms
  // policy_decision_time_ms
  // safety_filter_time_ms
  // total_latency_ms
  // graph_nodes
  // graph_edges
  // graph_staleness_s
  // current_confidence
  // current_reason
  rosidl_runtime_c__String__fini(&msg->current_reason);
  // current_waypoint
  // stamp
  builtin_interfaces__msg__Time__fini(&msg->stamp);
}

bool
hippocampus_ros2_msgs__msg__PolicyStatus__are_equal(const hippocampus_ros2_msgs__msg__PolicyStatus * lhs, const hippocampus_ros2_msgs__msg__PolicyStatus * rhs)
{
  if (!lhs || !rhs) {
    return false;
  }
  // is_active
  if (lhs->is_active != rhs->is_active) {
    return false;
  }
  // graph_stale
  if (lhs->graph_stale != rhs->graph_stale) {
    return false;
  }
  // using_snn
  if (lhs->using_snn != rhs->using_snn) {
    return false;
  }
  // hierarchical_enabled
  if (lhs->hierarchical_enabled != rhs->hierarchical_enabled) {
    return false;
  }
  // feature_compute_time_ms
  if (lhs->feature_compute_time_ms != rhs->feature_compute_time_ms) {
    return false;
  }
  // policy_decision_time_ms
  if (lhs->policy_decision_time_ms != rhs->policy_decision_time_ms) {
    return false;
  }
  // safety_filter_time_ms
  if (lhs->safety_filter_time_ms != rhs->safety_filter_time_ms) {
    return false;
  }
  // total_latency_ms
  if (lhs->total_latency_ms != rhs->total_latency_ms) {
    return false;
  }
  // graph_nodes
  if (lhs->graph_nodes != rhs->graph_nodes) {
    return false;
  }
  // graph_edges
  if (lhs->graph_edges != rhs->graph_edges) {
    return false;
  }
  // graph_staleness_s
  if (lhs->graph_staleness_s != rhs->graph_staleness_s) {
    return false;
  }
  // current_confidence
  if (lhs->current_confidence != rhs->current_confidence) {
    return false;
  }
  // current_reason
  if (!rosidl_runtime_c__String__are_equal(
      &(lhs->current_reason), &(rhs->current_reason)))
  {
    return false;
  }
  // current_waypoint
  if (lhs->current_waypoint != rhs->current_waypoint) {
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
hippocampus_ros2_msgs__msg__PolicyStatus__copy(
  const hippocampus_ros2_msgs__msg__PolicyStatus * input,
  hippocampus_ros2_msgs__msg__PolicyStatus * output)
{
  if (!input || !output) {
    return false;
  }
  // is_active
  output->is_active = input->is_active;
  // graph_stale
  output->graph_stale = input->graph_stale;
  // using_snn
  output->using_snn = input->using_snn;
  // hierarchical_enabled
  output->hierarchical_enabled = input->hierarchical_enabled;
  // feature_compute_time_ms
  output->feature_compute_time_ms = input->feature_compute_time_ms;
  // policy_decision_time_ms
  output->policy_decision_time_ms = input->policy_decision_time_ms;
  // safety_filter_time_ms
  output->safety_filter_time_ms = input->safety_filter_time_ms;
  // total_latency_ms
  output->total_latency_ms = input->total_latency_ms;
  // graph_nodes
  output->graph_nodes = input->graph_nodes;
  // graph_edges
  output->graph_edges = input->graph_edges;
  // graph_staleness_s
  output->graph_staleness_s = input->graph_staleness_s;
  // current_confidence
  output->current_confidence = input->current_confidence;
  // current_reason
  if (!rosidl_runtime_c__String__copy(
      &(input->current_reason), &(output->current_reason)))
  {
    return false;
  }
  // current_waypoint
  output->current_waypoint = input->current_waypoint;
  // stamp
  if (!builtin_interfaces__msg__Time__copy(
      &(input->stamp), &(output->stamp)))
  {
    return false;
  }
  return true;
}

hippocampus_ros2_msgs__msg__PolicyStatus *
hippocampus_ros2_msgs__msg__PolicyStatus__create()
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  hippocampus_ros2_msgs__msg__PolicyStatus * msg = (hippocampus_ros2_msgs__msg__PolicyStatus *)allocator.allocate(sizeof(hippocampus_ros2_msgs__msg__PolicyStatus), allocator.state);
  if (!msg) {
    return NULL;
  }
  memset(msg, 0, sizeof(hippocampus_ros2_msgs__msg__PolicyStatus));
  bool success = hippocampus_ros2_msgs__msg__PolicyStatus__init(msg);
  if (!success) {
    allocator.deallocate(msg, allocator.state);
    return NULL;
  }
  return msg;
}

void
hippocampus_ros2_msgs__msg__PolicyStatus__destroy(hippocampus_ros2_msgs__msg__PolicyStatus * msg)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  if (msg) {
    hippocampus_ros2_msgs__msg__PolicyStatus__fini(msg);
  }
  allocator.deallocate(msg, allocator.state);
}


bool
hippocampus_ros2_msgs__msg__PolicyStatus__Sequence__init(hippocampus_ros2_msgs__msg__PolicyStatus__Sequence * array, size_t size)
{
  if (!array) {
    return false;
  }
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  hippocampus_ros2_msgs__msg__PolicyStatus * data = NULL;

  if (size) {
    data = (hippocampus_ros2_msgs__msg__PolicyStatus *)allocator.zero_allocate(size, sizeof(hippocampus_ros2_msgs__msg__PolicyStatus), allocator.state);
    if (!data) {
      return false;
    }
    // initialize all array elements
    size_t i;
    for (i = 0; i < size; ++i) {
      bool success = hippocampus_ros2_msgs__msg__PolicyStatus__init(&data[i]);
      if (!success) {
        break;
      }
    }
    if (i < size) {
      // if initialization failed finalize the already initialized array elements
      for (; i > 0; --i) {
        hippocampus_ros2_msgs__msg__PolicyStatus__fini(&data[i - 1]);
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
hippocampus_ros2_msgs__msg__PolicyStatus__Sequence__fini(hippocampus_ros2_msgs__msg__PolicyStatus__Sequence * array)
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
      hippocampus_ros2_msgs__msg__PolicyStatus__fini(&array->data[i]);
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

hippocampus_ros2_msgs__msg__PolicyStatus__Sequence *
hippocampus_ros2_msgs__msg__PolicyStatus__Sequence__create(size_t size)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  hippocampus_ros2_msgs__msg__PolicyStatus__Sequence * array = (hippocampus_ros2_msgs__msg__PolicyStatus__Sequence *)allocator.allocate(sizeof(hippocampus_ros2_msgs__msg__PolicyStatus__Sequence), allocator.state);
  if (!array) {
    return NULL;
  }
  bool success = hippocampus_ros2_msgs__msg__PolicyStatus__Sequence__init(array, size);
  if (!success) {
    allocator.deallocate(array, allocator.state);
    return NULL;
  }
  return array;
}

void
hippocampus_ros2_msgs__msg__PolicyStatus__Sequence__destroy(hippocampus_ros2_msgs__msg__PolicyStatus__Sequence * array)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  if (array) {
    hippocampus_ros2_msgs__msg__PolicyStatus__Sequence__fini(array);
  }
  allocator.deallocate(array, allocator.state);
}

bool
hippocampus_ros2_msgs__msg__PolicyStatus__Sequence__are_equal(const hippocampus_ros2_msgs__msg__PolicyStatus__Sequence * lhs, const hippocampus_ros2_msgs__msg__PolicyStatus__Sequence * rhs)
{
  if (!lhs || !rhs) {
    return false;
  }
  if (lhs->size != rhs->size) {
    return false;
  }
  for (size_t i = 0; i < lhs->size; ++i) {
    if (!hippocampus_ros2_msgs__msg__PolicyStatus__are_equal(&(lhs->data[i]), &(rhs->data[i]))) {
      return false;
    }
  }
  return true;
}

bool
hippocampus_ros2_msgs__msg__PolicyStatus__Sequence__copy(
  const hippocampus_ros2_msgs__msg__PolicyStatus__Sequence * input,
  hippocampus_ros2_msgs__msg__PolicyStatus__Sequence * output)
{
  if (!input || !output) {
    return false;
  }
  if (output->capacity < input->size) {
    const size_t allocation_size =
      input->size * sizeof(hippocampus_ros2_msgs__msg__PolicyStatus);
    rcutils_allocator_t allocator = rcutils_get_default_allocator();
    hippocampus_ros2_msgs__msg__PolicyStatus * data =
      (hippocampus_ros2_msgs__msg__PolicyStatus *)allocator.reallocate(
      output->data, allocation_size, allocator.state);
    if (!data) {
      return false;
    }
    // If reallocation succeeded, memory may or may not have been moved
    // to fulfill the allocation request, invalidating output->data.
    output->data = data;
    for (size_t i = output->capacity; i < input->size; ++i) {
      if (!hippocampus_ros2_msgs__msg__PolicyStatus__init(&output->data[i])) {
        // If initialization of any new item fails, roll back
        // all previously initialized items. Existing items
        // in output are to be left unmodified.
        for (; i-- > output->capacity; ) {
          hippocampus_ros2_msgs__msg__PolicyStatus__fini(&output->data[i]);
        }
        return false;
      }
    }
    output->capacity = input->size;
  }
  output->size = input->size;
  for (size_t i = 0; i < input->size; ++i) {
    if (!hippocampus_ros2_msgs__msg__PolicyStatus__copy(
        &(input->data[i]), &(output->data[i])))
    {
      return false;
    }
  }
  return true;
}
