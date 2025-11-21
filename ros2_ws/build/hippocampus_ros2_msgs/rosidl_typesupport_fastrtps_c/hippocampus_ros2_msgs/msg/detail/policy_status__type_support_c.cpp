// generated from rosidl_typesupport_fastrtps_c/resource/idl__type_support_c.cpp.em
// with input from hippocampus_ros2_msgs:msg/PolicyStatus.idl
// generated code does not contain a copyright notice
#include "hippocampus_ros2_msgs/msg/detail/policy_status__rosidl_typesupport_fastrtps_c.h"


#include <cassert>
#include <limits>
#include <string>
#include "rosidl_typesupport_fastrtps_c/identifier.h"
#include "rosidl_typesupport_fastrtps_c/wstring_conversion.hpp"
#include "rosidl_typesupport_fastrtps_cpp/message_type_support.h"
#include "hippocampus_ros2_msgs/msg/rosidl_typesupport_fastrtps_c__visibility_control.h"
#include "hippocampus_ros2_msgs/msg/detail/policy_status__struct.h"
#include "hippocampus_ros2_msgs/msg/detail/policy_status__functions.h"
#include "fastcdr/Cdr.h"

#ifndef _WIN32
# pragma GCC diagnostic push
# pragma GCC diagnostic ignored "-Wunused-parameter"
# ifdef __clang__
#  pragma clang diagnostic ignored "-Wdeprecated-register"
#  pragma clang diagnostic ignored "-Wreturn-type-c-linkage"
# endif
#endif
#ifndef _WIN32
# pragma GCC diagnostic pop
#endif

// includes and forward declarations of message dependencies and their conversion functions

#if defined(__cplusplus)
extern "C"
{
#endif

#include "builtin_interfaces/msg/detail/time__functions.h"  // stamp
#include "rosidl_runtime_c/string.h"  // current_reason
#include "rosidl_runtime_c/string_functions.h"  // current_reason

// forward declare type support functions
ROSIDL_TYPESUPPORT_FASTRTPS_C_IMPORT_hippocampus_ros2_msgs
size_t get_serialized_size_builtin_interfaces__msg__Time(
  const void * untyped_ros_message,
  size_t current_alignment);

ROSIDL_TYPESUPPORT_FASTRTPS_C_IMPORT_hippocampus_ros2_msgs
size_t max_serialized_size_builtin_interfaces__msg__Time(
  bool & full_bounded,
  bool & is_plain,
  size_t current_alignment);

ROSIDL_TYPESUPPORT_FASTRTPS_C_IMPORT_hippocampus_ros2_msgs
const rosidl_message_type_support_t *
  ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_fastrtps_c, builtin_interfaces, msg, Time)();


using _PolicyStatus__ros_msg_type = hippocampus_ros2_msgs__msg__PolicyStatus;

static bool _PolicyStatus__cdr_serialize(
  const void * untyped_ros_message,
  eprosima::fastcdr::Cdr & cdr)
{
  if (!untyped_ros_message) {
    fprintf(stderr, "ros message handle is null\n");
    return false;
  }
  const _PolicyStatus__ros_msg_type * ros_message = static_cast<const _PolicyStatus__ros_msg_type *>(untyped_ros_message);
  // Field name: is_active
  {
    cdr << (ros_message->is_active ? true : false);
  }

  // Field name: graph_stale
  {
    cdr << (ros_message->graph_stale ? true : false);
  }

  // Field name: using_snn
  {
    cdr << (ros_message->using_snn ? true : false);
  }

  // Field name: hierarchical_enabled
  {
    cdr << (ros_message->hierarchical_enabled ? true : false);
  }

  // Field name: feature_compute_time_ms
  {
    cdr << ros_message->feature_compute_time_ms;
  }

  // Field name: policy_decision_time_ms
  {
    cdr << ros_message->policy_decision_time_ms;
  }

  // Field name: safety_filter_time_ms
  {
    cdr << ros_message->safety_filter_time_ms;
  }

  // Field name: total_latency_ms
  {
    cdr << ros_message->total_latency_ms;
  }

  // Field name: graph_nodes
  {
    cdr << ros_message->graph_nodes;
  }

  // Field name: graph_edges
  {
    cdr << ros_message->graph_edges;
  }

  // Field name: graph_staleness_s
  {
    cdr << ros_message->graph_staleness_s;
  }

  // Field name: current_confidence
  {
    cdr << ros_message->current_confidence;
  }

  // Field name: current_reason
  {
    const rosidl_runtime_c__String * str = &ros_message->current_reason;
    if (str->capacity == 0 || str->capacity <= str->size) {
      fprintf(stderr, "string capacity not greater than size\n");
      return false;
    }
    if (str->data[str->size] != '\0') {
      fprintf(stderr, "string not null-terminated\n");
      return false;
    }
    cdr << str->data;
  }

  // Field name: current_waypoint
  {
    cdr << ros_message->current_waypoint;
  }

  // Field name: stamp
  {
    const message_type_support_callbacks_t * callbacks =
      static_cast<const message_type_support_callbacks_t *>(
      ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(
        rosidl_typesupport_fastrtps_c, builtin_interfaces, msg, Time
      )()->data);
    if (!callbacks->cdr_serialize(
        &ros_message->stamp, cdr))
    {
      return false;
    }
  }

  return true;
}

static bool _PolicyStatus__cdr_deserialize(
  eprosima::fastcdr::Cdr & cdr,
  void * untyped_ros_message)
{
  if (!untyped_ros_message) {
    fprintf(stderr, "ros message handle is null\n");
    return false;
  }
  _PolicyStatus__ros_msg_type * ros_message = static_cast<_PolicyStatus__ros_msg_type *>(untyped_ros_message);
  // Field name: is_active
  {
    uint8_t tmp;
    cdr >> tmp;
    ros_message->is_active = tmp ? true : false;
  }

  // Field name: graph_stale
  {
    uint8_t tmp;
    cdr >> tmp;
    ros_message->graph_stale = tmp ? true : false;
  }

  // Field name: using_snn
  {
    uint8_t tmp;
    cdr >> tmp;
    ros_message->using_snn = tmp ? true : false;
  }

  // Field name: hierarchical_enabled
  {
    uint8_t tmp;
    cdr >> tmp;
    ros_message->hierarchical_enabled = tmp ? true : false;
  }

  // Field name: feature_compute_time_ms
  {
    cdr >> ros_message->feature_compute_time_ms;
  }

  // Field name: policy_decision_time_ms
  {
    cdr >> ros_message->policy_decision_time_ms;
  }

  // Field name: safety_filter_time_ms
  {
    cdr >> ros_message->safety_filter_time_ms;
  }

  // Field name: total_latency_ms
  {
    cdr >> ros_message->total_latency_ms;
  }

  // Field name: graph_nodes
  {
    cdr >> ros_message->graph_nodes;
  }

  // Field name: graph_edges
  {
    cdr >> ros_message->graph_edges;
  }

  // Field name: graph_staleness_s
  {
    cdr >> ros_message->graph_staleness_s;
  }

  // Field name: current_confidence
  {
    cdr >> ros_message->current_confidence;
  }

  // Field name: current_reason
  {
    std::string tmp;
    cdr >> tmp;
    if (!ros_message->current_reason.data) {
      rosidl_runtime_c__String__init(&ros_message->current_reason);
    }
    bool succeeded = rosidl_runtime_c__String__assign(
      &ros_message->current_reason,
      tmp.c_str());
    if (!succeeded) {
      fprintf(stderr, "failed to assign string into field 'current_reason'\n");
      return false;
    }
  }

  // Field name: current_waypoint
  {
    cdr >> ros_message->current_waypoint;
  }

  // Field name: stamp
  {
    const message_type_support_callbacks_t * callbacks =
      static_cast<const message_type_support_callbacks_t *>(
      ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(
        rosidl_typesupport_fastrtps_c, builtin_interfaces, msg, Time
      )()->data);
    if (!callbacks->cdr_deserialize(
        cdr, &ros_message->stamp))
    {
      return false;
    }
  }

  return true;
}  // NOLINT(readability/fn_size)

ROSIDL_TYPESUPPORT_FASTRTPS_C_PUBLIC_hippocampus_ros2_msgs
size_t get_serialized_size_hippocampus_ros2_msgs__msg__PolicyStatus(
  const void * untyped_ros_message,
  size_t current_alignment)
{
  const _PolicyStatus__ros_msg_type * ros_message = static_cast<const _PolicyStatus__ros_msg_type *>(untyped_ros_message);
  (void)ros_message;
  size_t initial_alignment = current_alignment;

  const size_t padding = 4;
  const size_t wchar_size = 4;
  (void)padding;
  (void)wchar_size;

  // field.name is_active
  {
    size_t item_size = sizeof(ros_message->is_active);
    current_alignment += item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // field.name graph_stale
  {
    size_t item_size = sizeof(ros_message->graph_stale);
    current_alignment += item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // field.name using_snn
  {
    size_t item_size = sizeof(ros_message->using_snn);
    current_alignment += item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // field.name hierarchical_enabled
  {
    size_t item_size = sizeof(ros_message->hierarchical_enabled);
    current_alignment += item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // field.name feature_compute_time_ms
  {
    size_t item_size = sizeof(ros_message->feature_compute_time_ms);
    current_alignment += item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // field.name policy_decision_time_ms
  {
    size_t item_size = sizeof(ros_message->policy_decision_time_ms);
    current_alignment += item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // field.name safety_filter_time_ms
  {
    size_t item_size = sizeof(ros_message->safety_filter_time_ms);
    current_alignment += item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // field.name total_latency_ms
  {
    size_t item_size = sizeof(ros_message->total_latency_ms);
    current_alignment += item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // field.name graph_nodes
  {
    size_t item_size = sizeof(ros_message->graph_nodes);
    current_alignment += item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // field.name graph_edges
  {
    size_t item_size = sizeof(ros_message->graph_edges);
    current_alignment += item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // field.name graph_staleness_s
  {
    size_t item_size = sizeof(ros_message->graph_staleness_s);
    current_alignment += item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // field.name current_confidence
  {
    size_t item_size = sizeof(ros_message->current_confidence);
    current_alignment += item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // field.name current_reason
  current_alignment += padding +
    eprosima::fastcdr::Cdr::alignment(current_alignment, padding) +
    (ros_message->current_reason.size + 1);
  // field.name current_waypoint
  {
    size_t item_size = sizeof(ros_message->current_waypoint);
    current_alignment += item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // field.name stamp

  current_alignment += get_serialized_size_builtin_interfaces__msg__Time(
    &(ros_message->stamp), current_alignment);

  return current_alignment - initial_alignment;
}

static uint32_t _PolicyStatus__get_serialized_size(const void * untyped_ros_message)
{
  return static_cast<uint32_t>(
    get_serialized_size_hippocampus_ros2_msgs__msg__PolicyStatus(
      untyped_ros_message, 0));
}

ROSIDL_TYPESUPPORT_FASTRTPS_C_PUBLIC_hippocampus_ros2_msgs
size_t max_serialized_size_hippocampus_ros2_msgs__msg__PolicyStatus(
  bool & full_bounded,
  bool & is_plain,
  size_t current_alignment)
{
  size_t initial_alignment = current_alignment;

  const size_t padding = 4;
  const size_t wchar_size = 4;
  size_t last_member_size = 0;
  (void)last_member_size;
  (void)padding;
  (void)wchar_size;

  full_bounded = true;
  is_plain = true;

  // member: is_active
  {
    size_t array_size = 1;

    last_member_size = array_size * sizeof(uint8_t);
    current_alignment += array_size * sizeof(uint8_t);
  }
  // member: graph_stale
  {
    size_t array_size = 1;

    last_member_size = array_size * sizeof(uint8_t);
    current_alignment += array_size * sizeof(uint8_t);
  }
  // member: using_snn
  {
    size_t array_size = 1;

    last_member_size = array_size * sizeof(uint8_t);
    current_alignment += array_size * sizeof(uint8_t);
  }
  // member: hierarchical_enabled
  {
    size_t array_size = 1;

    last_member_size = array_size * sizeof(uint8_t);
    current_alignment += array_size * sizeof(uint8_t);
  }
  // member: feature_compute_time_ms
  {
    size_t array_size = 1;

    last_member_size = array_size * sizeof(uint64_t);
    current_alignment += array_size * sizeof(uint64_t) +
      eprosima::fastcdr::Cdr::alignment(current_alignment, sizeof(uint64_t));
  }
  // member: policy_decision_time_ms
  {
    size_t array_size = 1;

    last_member_size = array_size * sizeof(uint64_t);
    current_alignment += array_size * sizeof(uint64_t) +
      eprosima::fastcdr::Cdr::alignment(current_alignment, sizeof(uint64_t));
  }
  // member: safety_filter_time_ms
  {
    size_t array_size = 1;

    last_member_size = array_size * sizeof(uint64_t);
    current_alignment += array_size * sizeof(uint64_t) +
      eprosima::fastcdr::Cdr::alignment(current_alignment, sizeof(uint64_t));
  }
  // member: total_latency_ms
  {
    size_t array_size = 1;

    last_member_size = array_size * sizeof(uint64_t);
    current_alignment += array_size * sizeof(uint64_t) +
      eprosima::fastcdr::Cdr::alignment(current_alignment, sizeof(uint64_t));
  }
  // member: graph_nodes
  {
    size_t array_size = 1;

    last_member_size = array_size * sizeof(uint32_t);
    current_alignment += array_size * sizeof(uint32_t) +
      eprosima::fastcdr::Cdr::alignment(current_alignment, sizeof(uint32_t));
  }
  // member: graph_edges
  {
    size_t array_size = 1;

    last_member_size = array_size * sizeof(uint32_t);
    current_alignment += array_size * sizeof(uint32_t) +
      eprosima::fastcdr::Cdr::alignment(current_alignment, sizeof(uint32_t));
  }
  // member: graph_staleness_s
  {
    size_t array_size = 1;

    last_member_size = array_size * sizeof(uint64_t);
    current_alignment += array_size * sizeof(uint64_t) +
      eprosima::fastcdr::Cdr::alignment(current_alignment, sizeof(uint64_t));
  }
  // member: current_confidence
  {
    size_t array_size = 1;

    last_member_size = array_size * sizeof(uint64_t);
    current_alignment += array_size * sizeof(uint64_t) +
      eprosima::fastcdr::Cdr::alignment(current_alignment, sizeof(uint64_t));
  }
  // member: current_reason
  {
    size_t array_size = 1;

    full_bounded = false;
    is_plain = false;
    for (size_t index = 0; index < array_size; ++index) {
      current_alignment += padding +
        eprosima::fastcdr::Cdr::alignment(current_alignment, padding) +
        1;
    }
  }
  // member: current_waypoint
  {
    size_t array_size = 1;

    last_member_size = array_size * sizeof(uint32_t);
    current_alignment += array_size * sizeof(uint32_t) +
      eprosima::fastcdr::Cdr::alignment(current_alignment, sizeof(uint32_t));
  }
  // member: stamp
  {
    size_t array_size = 1;


    last_member_size = 0;
    for (size_t index = 0; index < array_size; ++index) {
      bool inner_full_bounded;
      bool inner_is_plain;
      size_t inner_size;
      inner_size =
        max_serialized_size_builtin_interfaces__msg__Time(
        inner_full_bounded, inner_is_plain, current_alignment);
      last_member_size += inner_size;
      current_alignment += inner_size;
      full_bounded &= inner_full_bounded;
      is_plain &= inner_is_plain;
    }
  }

  size_t ret_val = current_alignment - initial_alignment;
  if (is_plain) {
    // All members are plain, and type is not empty.
    // We still need to check that the in-memory alignment
    // is the same as the CDR mandated alignment.
    using DataType = hippocampus_ros2_msgs__msg__PolicyStatus;
    is_plain =
      (
      offsetof(DataType, stamp) +
      last_member_size
      ) == ret_val;
  }

  return ret_val;
}

static size_t _PolicyStatus__max_serialized_size(char & bounds_info)
{
  bool full_bounded;
  bool is_plain;
  size_t ret_val;

  ret_val = max_serialized_size_hippocampus_ros2_msgs__msg__PolicyStatus(
    full_bounded, is_plain, 0);

  bounds_info =
    is_plain ? ROSIDL_TYPESUPPORT_FASTRTPS_PLAIN_TYPE :
    full_bounded ? ROSIDL_TYPESUPPORT_FASTRTPS_BOUNDED_TYPE : ROSIDL_TYPESUPPORT_FASTRTPS_UNBOUNDED_TYPE;
  return ret_val;
}


static message_type_support_callbacks_t __callbacks_PolicyStatus = {
  "hippocampus_ros2_msgs::msg",
  "PolicyStatus",
  _PolicyStatus__cdr_serialize,
  _PolicyStatus__cdr_deserialize,
  _PolicyStatus__get_serialized_size,
  _PolicyStatus__max_serialized_size
};

static rosidl_message_type_support_t _PolicyStatus__type_support = {
  rosidl_typesupport_fastrtps_c__identifier,
  &__callbacks_PolicyStatus,
  get_message_typesupport_handle_function,
};

const rosidl_message_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_fastrtps_c, hippocampus_ros2_msgs, msg, PolicyStatus)() {
  return &_PolicyStatus__type_support;
}

#if defined(__cplusplus)
}
#endif
