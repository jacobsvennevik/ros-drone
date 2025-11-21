// generated from rosidl_typesupport_fastrtps_cpp/resource/idl__type_support.cpp.em
// with input from hippocampus_ros2_msgs:msg/PolicyStatus.idl
// generated code does not contain a copyright notice
#include "hippocampus_ros2_msgs/msg/detail/policy_status__rosidl_typesupport_fastrtps_cpp.hpp"
#include "hippocampus_ros2_msgs/msg/detail/policy_status__struct.hpp"

#include <limits>
#include <stdexcept>
#include <string>
#include "rosidl_typesupport_cpp/message_type_support.hpp"
#include "rosidl_typesupport_fastrtps_cpp/identifier.hpp"
#include "rosidl_typesupport_fastrtps_cpp/message_type_support.h"
#include "rosidl_typesupport_fastrtps_cpp/message_type_support_decl.hpp"
#include "rosidl_typesupport_fastrtps_cpp/wstring_conversion.hpp"
#include "fastcdr/Cdr.h"


// forward declaration of message dependencies and their conversion functions
namespace builtin_interfaces
{
namespace msg
{
namespace typesupport_fastrtps_cpp
{
bool cdr_serialize(
  const builtin_interfaces::msg::Time &,
  eprosima::fastcdr::Cdr &);
bool cdr_deserialize(
  eprosima::fastcdr::Cdr &,
  builtin_interfaces::msg::Time &);
size_t get_serialized_size(
  const builtin_interfaces::msg::Time &,
  size_t current_alignment);
size_t
max_serialized_size_Time(
  bool & full_bounded,
  bool & is_plain,
  size_t current_alignment);
}  // namespace typesupport_fastrtps_cpp
}  // namespace msg
}  // namespace builtin_interfaces


namespace hippocampus_ros2_msgs
{

namespace msg
{

namespace typesupport_fastrtps_cpp
{

bool
ROSIDL_TYPESUPPORT_FASTRTPS_CPP_PUBLIC_hippocampus_ros2_msgs
cdr_serialize(
  const hippocampus_ros2_msgs::msg::PolicyStatus & ros_message,
  eprosima::fastcdr::Cdr & cdr)
{
  // Member: is_active
  cdr << (ros_message.is_active ? true : false);
  // Member: graph_stale
  cdr << (ros_message.graph_stale ? true : false);
  // Member: using_snn
  cdr << (ros_message.using_snn ? true : false);
  // Member: hierarchical_enabled
  cdr << (ros_message.hierarchical_enabled ? true : false);
  // Member: feature_compute_time_ms
  cdr << ros_message.feature_compute_time_ms;
  // Member: policy_decision_time_ms
  cdr << ros_message.policy_decision_time_ms;
  // Member: safety_filter_time_ms
  cdr << ros_message.safety_filter_time_ms;
  // Member: total_latency_ms
  cdr << ros_message.total_latency_ms;
  // Member: graph_nodes
  cdr << ros_message.graph_nodes;
  // Member: graph_edges
  cdr << ros_message.graph_edges;
  // Member: graph_staleness_s
  cdr << ros_message.graph_staleness_s;
  // Member: current_confidence
  cdr << ros_message.current_confidence;
  // Member: current_reason
  cdr << ros_message.current_reason;
  // Member: current_waypoint
  cdr << ros_message.current_waypoint;
  // Member: stamp
  builtin_interfaces::msg::typesupport_fastrtps_cpp::cdr_serialize(
    ros_message.stamp,
    cdr);
  return true;
}

bool
ROSIDL_TYPESUPPORT_FASTRTPS_CPP_PUBLIC_hippocampus_ros2_msgs
cdr_deserialize(
  eprosima::fastcdr::Cdr & cdr,
  hippocampus_ros2_msgs::msg::PolicyStatus & ros_message)
{
  // Member: is_active
  {
    uint8_t tmp;
    cdr >> tmp;
    ros_message.is_active = tmp ? true : false;
  }

  // Member: graph_stale
  {
    uint8_t tmp;
    cdr >> tmp;
    ros_message.graph_stale = tmp ? true : false;
  }

  // Member: using_snn
  {
    uint8_t tmp;
    cdr >> tmp;
    ros_message.using_snn = tmp ? true : false;
  }

  // Member: hierarchical_enabled
  {
    uint8_t tmp;
    cdr >> tmp;
    ros_message.hierarchical_enabled = tmp ? true : false;
  }

  // Member: feature_compute_time_ms
  cdr >> ros_message.feature_compute_time_ms;

  // Member: policy_decision_time_ms
  cdr >> ros_message.policy_decision_time_ms;

  // Member: safety_filter_time_ms
  cdr >> ros_message.safety_filter_time_ms;

  // Member: total_latency_ms
  cdr >> ros_message.total_latency_ms;

  // Member: graph_nodes
  cdr >> ros_message.graph_nodes;

  // Member: graph_edges
  cdr >> ros_message.graph_edges;

  // Member: graph_staleness_s
  cdr >> ros_message.graph_staleness_s;

  // Member: current_confidence
  cdr >> ros_message.current_confidence;

  // Member: current_reason
  cdr >> ros_message.current_reason;

  // Member: current_waypoint
  cdr >> ros_message.current_waypoint;

  // Member: stamp
  builtin_interfaces::msg::typesupport_fastrtps_cpp::cdr_deserialize(
    cdr, ros_message.stamp);

  return true;
}  // NOLINT(readability/fn_size)

size_t
ROSIDL_TYPESUPPORT_FASTRTPS_CPP_PUBLIC_hippocampus_ros2_msgs
get_serialized_size(
  const hippocampus_ros2_msgs::msg::PolicyStatus & ros_message,
  size_t current_alignment)
{
  size_t initial_alignment = current_alignment;

  const size_t padding = 4;
  const size_t wchar_size = 4;
  (void)padding;
  (void)wchar_size;

  // Member: is_active
  {
    size_t item_size = sizeof(ros_message.is_active);
    current_alignment += item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // Member: graph_stale
  {
    size_t item_size = sizeof(ros_message.graph_stale);
    current_alignment += item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // Member: using_snn
  {
    size_t item_size = sizeof(ros_message.using_snn);
    current_alignment += item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // Member: hierarchical_enabled
  {
    size_t item_size = sizeof(ros_message.hierarchical_enabled);
    current_alignment += item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // Member: feature_compute_time_ms
  {
    size_t item_size = sizeof(ros_message.feature_compute_time_ms);
    current_alignment += item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // Member: policy_decision_time_ms
  {
    size_t item_size = sizeof(ros_message.policy_decision_time_ms);
    current_alignment += item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // Member: safety_filter_time_ms
  {
    size_t item_size = sizeof(ros_message.safety_filter_time_ms);
    current_alignment += item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // Member: total_latency_ms
  {
    size_t item_size = sizeof(ros_message.total_latency_ms);
    current_alignment += item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // Member: graph_nodes
  {
    size_t item_size = sizeof(ros_message.graph_nodes);
    current_alignment += item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // Member: graph_edges
  {
    size_t item_size = sizeof(ros_message.graph_edges);
    current_alignment += item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // Member: graph_staleness_s
  {
    size_t item_size = sizeof(ros_message.graph_staleness_s);
    current_alignment += item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // Member: current_confidence
  {
    size_t item_size = sizeof(ros_message.current_confidence);
    current_alignment += item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // Member: current_reason
  current_alignment += padding +
    eprosima::fastcdr::Cdr::alignment(current_alignment, padding) +
    (ros_message.current_reason.size() + 1);
  // Member: current_waypoint
  {
    size_t item_size = sizeof(ros_message.current_waypoint);
    current_alignment += item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // Member: stamp

  current_alignment +=
    builtin_interfaces::msg::typesupport_fastrtps_cpp::get_serialized_size(
    ros_message.stamp, current_alignment);

  return current_alignment - initial_alignment;
}

size_t
ROSIDL_TYPESUPPORT_FASTRTPS_CPP_PUBLIC_hippocampus_ros2_msgs
max_serialized_size_PolicyStatus(
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


  // Member: is_active
  {
    size_t array_size = 1;

    last_member_size = array_size * sizeof(uint8_t);
    current_alignment += array_size * sizeof(uint8_t);
  }

  // Member: graph_stale
  {
    size_t array_size = 1;

    last_member_size = array_size * sizeof(uint8_t);
    current_alignment += array_size * sizeof(uint8_t);
  }

  // Member: using_snn
  {
    size_t array_size = 1;

    last_member_size = array_size * sizeof(uint8_t);
    current_alignment += array_size * sizeof(uint8_t);
  }

  // Member: hierarchical_enabled
  {
    size_t array_size = 1;

    last_member_size = array_size * sizeof(uint8_t);
    current_alignment += array_size * sizeof(uint8_t);
  }

  // Member: feature_compute_time_ms
  {
    size_t array_size = 1;

    last_member_size = array_size * sizeof(uint64_t);
    current_alignment += array_size * sizeof(uint64_t) +
      eprosima::fastcdr::Cdr::alignment(current_alignment, sizeof(uint64_t));
  }

  // Member: policy_decision_time_ms
  {
    size_t array_size = 1;

    last_member_size = array_size * sizeof(uint64_t);
    current_alignment += array_size * sizeof(uint64_t) +
      eprosima::fastcdr::Cdr::alignment(current_alignment, sizeof(uint64_t));
  }

  // Member: safety_filter_time_ms
  {
    size_t array_size = 1;

    last_member_size = array_size * sizeof(uint64_t);
    current_alignment += array_size * sizeof(uint64_t) +
      eprosima::fastcdr::Cdr::alignment(current_alignment, sizeof(uint64_t));
  }

  // Member: total_latency_ms
  {
    size_t array_size = 1;

    last_member_size = array_size * sizeof(uint64_t);
    current_alignment += array_size * sizeof(uint64_t) +
      eprosima::fastcdr::Cdr::alignment(current_alignment, sizeof(uint64_t));
  }

  // Member: graph_nodes
  {
    size_t array_size = 1;

    last_member_size = array_size * sizeof(uint32_t);
    current_alignment += array_size * sizeof(uint32_t) +
      eprosima::fastcdr::Cdr::alignment(current_alignment, sizeof(uint32_t));
  }

  // Member: graph_edges
  {
    size_t array_size = 1;

    last_member_size = array_size * sizeof(uint32_t);
    current_alignment += array_size * sizeof(uint32_t) +
      eprosima::fastcdr::Cdr::alignment(current_alignment, sizeof(uint32_t));
  }

  // Member: graph_staleness_s
  {
    size_t array_size = 1;

    last_member_size = array_size * sizeof(uint64_t);
    current_alignment += array_size * sizeof(uint64_t) +
      eprosima::fastcdr::Cdr::alignment(current_alignment, sizeof(uint64_t));
  }

  // Member: current_confidence
  {
    size_t array_size = 1;

    last_member_size = array_size * sizeof(uint64_t);
    current_alignment += array_size * sizeof(uint64_t) +
      eprosima::fastcdr::Cdr::alignment(current_alignment, sizeof(uint64_t));
  }

  // Member: current_reason
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

  // Member: current_waypoint
  {
    size_t array_size = 1;

    last_member_size = array_size * sizeof(uint32_t);
    current_alignment += array_size * sizeof(uint32_t) +
      eprosima::fastcdr::Cdr::alignment(current_alignment, sizeof(uint32_t));
  }

  // Member: stamp
  {
    size_t array_size = 1;


    last_member_size = 0;
    for (size_t index = 0; index < array_size; ++index) {
      bool inner_full_bounded;
      bool inner_is_plain;
      size_t inner_size =
        builtin_interfaces::msg::typesupport_fastrtps_cpp::max_serialized_size_Time(
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
    using DataType = hippocampus_ros2_msgs::msg::PolicyStatus;
    is_plain =
      (
      offsetof(DataType, stamp) +
      last_member_size
      ) == ret_val;
  }

  return ret_val;
}

static bool _PolicyStatus__cdr_serialize(
  const void * untyped_ros_message,
  eprosima::fastcdr::Cdr & cdr)
{
  auto typed_message =
    static_cast<const hippocampus_ros2_msgs::msg::PolicyStatus *>(
    untyped_ros_message);
  return cdr_serialize(*typed_message, cdr);
}

static bool _PolicyStatus__cdr_deserialize(
  eprosima::fastcdr::Cdr & cdr,
  void * untyped_ros_message)
{
  auto typed_message =
    static_cast<hippocampus_ros2_msgs::msg::PolicyStatus *>(
    untyped_ros_message);
  return cdr_deserialize(cdr, *typed_message);
}

static uint32_t _PolicyStatus__get_serialized_size(
  const void * untyped_ros_message)
{
  auto typed_message =
    static_cast<const hippocampus_ros2_msgs::msg::PolicyStatus *>(
    untyped_ros_message);
  return static_cast<uint32_t>(get_serialized_size(*typed_message, 0));
}

static size_t _PolicyStatus__max_serialized_size(char & bounds_info)
{
  bool full_bounded;
  bool is_plain;
  size_t ret_val;

  ret_val = max_serialized_size_PolicyStatus(full_bounded, is_plain, 0);

  bounds_info =
    is_plain ? ROSIDL_TYPESUPPORT_FASTRTPS_PLAIN_TYPE :
    full_bounded ? ROSIDL_TYPESUPPORT_FASTRTPS_BOUNDED_TYPE : ROSIDL_TYPESUPPORT_FASTRTPS_UNBOUNDED_TYPE;
  return ret_val;
}

static message_type_support_callbacks_t _PolicyStatus__callbacks = {
  "hippocampus_ros2_msgs::msg",
  "PolicyStatus",
  _PolicyStatus__cdr_serialize,
  _PolicyStatus__cdr_deserialize,
  _PolicyStatus__get_serialized_size,
  _PolicyStatus__max_serialized_size
};

static rosidl_message_type_support_t _PolicyStatus__handle = {
  rosidl_typesupport_fastrtps_cpp::typesupport_identifier,
  &_PolicyStatus__callbacks,
  get_message_typesupport_handle_function,
};

}  // namespace typesupport_fastrtps_cpp

}  // namespace msg

}  // namespace hippocampus_ros2_msgs

namespace rosidl_typesupport_fastrtps_cpp
{

template<>
ROSIDL_TYPESUPPORT_FASTRTPS_CPP_EXPORT_hippocampus_ros2_msgs
const rosidl_message_type_support_t *
get_message_type_support_handle<hippocampus_ros2_msgs::msg::PolicyStatus>()
{
  return &hippocampus_ros2_msgs::msg::typesupport_fastrtps_cpp::_PolicyStatus__handle;
}

}  // namespace rosidl_typesupport_fastrtps_cpp

#ifdef __cplusplus
extern "C"
{
#endif

const rosidl_message_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_fastrtps_cpp, hippocampus_ros2_msgs, msg, PolicyStatus)() {
  return &hippocampus_ros2_msgs::msg::typesupport_fastrtps_cpp::_PolicyStatus__handle;
}

#ifdef __cplusplus
}
#endif
