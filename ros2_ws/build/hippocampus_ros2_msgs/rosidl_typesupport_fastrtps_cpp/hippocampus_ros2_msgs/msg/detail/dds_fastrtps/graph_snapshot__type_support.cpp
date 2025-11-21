// generated from rosidl_typesupport_fastrtps_cpp/resource/idl__type_support.cpp.em
// with input from hippocampus_ros2_msgs:msg/GraphSnapshot.idl
// generated code does not contain a copyright notice
#include "hippocampus_ros2_msgs/msg/detail/graph_snapshot__rosidl_typesupport_fastrtps_cpp.hpp"
#include "hippocampus_ros2_msgs/msg/detail/graph_snapshot__struct.hpp"

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

// functions for builtin_interfaces::msg::Time already declared above

namespace hippocampus_ros2_msgs
{
namespace msg
{
namespace typesupport_fastrtps_cpp
{
bool cdr_serialize(
  const hippocampus_ros2_msgs::msg::GraphNode &,
  eprosima::fastcdr::Cdr &);
bool cdr_deserialize(
  eprosima::fastcdr::Cdr &,
  hippocampus_ros2_msgs::msg::GraphNode &);
size_t get_serialized_size(
  const hippocampus_ros2_msgs::msg::GraphNode &,
  size_t current_alignment);
size_t
max_serialized_size_GraphNode(
  bool & full_bounded,
  bool & is_plain,
  size_t current_alignment);
}  // namespace typesupport_fastrtps_cpp
}  // namespace msg
}  // namespace hippocampus_ros2_msgs

namespace hippocampus_ros2_msgs
{
namespace msg
{
namespace typesupport_fastrtps_cpp
{
bool cdr_serialize(
  const hippocampus_ros2_msgs::msg::GraphEdge &,
  eprosima::fastcdr::Cdr &);
bool cdr_deserialize(
  eprosima::fastcdr::Cdr &,
  hippocampus_ros2_msgs::msg::GraphEdge &);
size_t get_serialized_size(
  const hippocampus_ros2_msgs::msg::GraphEdge &,
  size_t current_alignment);
size_t
max_serialized_size_GraphEdge(
  bool & full_bounded,
  bool & is_plain,
  size_t current_alignment);
}  // namespace typesupport_fastrtps_cpp
}  // namespace msg
}  // namespace hippocampus_ros2_msgs


namespace hippocampus_ros2_msgs
{

namespace msg
{

namespace typesupport_fastrtps_cpp
{

bool
ROSIDL_TYPESUPPORT_FASTRTPS_CPP_PUBLIC_hippocampus_ros2_msgs
cdr_serialize(
  const hippocampus_ros2_msgs::msg::GraphSnapshot & ros_message,
  eprosima::fastcdr::Cdr & cdr)
{
  // Member: epoch_id
  cdr << ros_message.epoch_id;
  // Member: frame_id
  cdr << ros_message.frame_id;
  // Member: stamp
  builtin_interfaces::msg::typesupport_fastrtps_cpp::cdr_serialize(
    ros_message.stamp,
    cdr);
  // Member: last_updated
  builtin_interfaces::msg::typesupport_fastrtps_cpp::cdr_serialize(
    ros_message.last_updated,
    cdr);
  // Member: update_rate
  cdr << ros_message.update_rate;
  // Member: staleness_warning
  cdr << (ros_message.staleness_warning ? true : false);
  // Member: nodes
  {
    size_t size = ros_message.nodes.size();
    cdr << static_cast<uint32_t>(size);
    for (size_t i = 0; i < size; i++) {
      hippocampus_ros2_msgs::msg::typesupport_fastrtps_cpp::cdr_serialize(
        ros_message.nodes[i],
        cdr);
    }
  }
  // Member: edges
  {
    size_t size = ros_message.edges.size();
    cdr << static_cast<uint32_t>(size);
    for (size_t i = 0; i < size; i++) {
      hippocampus_ros2_msgs::msg::typesupport_fastrtps_cpp::cdr_serialize(
        ros_message.edges[i],
        cdr);
    }
  }
  return true;
}

bool
ROSIDL_TYPESUPPORT_FASTRTPS_CPP_PUBLIC_hippocampus_ros2_msgs
cdr_deserialize(
  eprosima::fastcdr::Cdr & cdr,
  hippocampus_ros2_msgs::msg::GraphSnapshot & ros_message)
{
  // Member: epoch_id
  cdr >> ros_message.epoch_id;

  // Member: frame_id
  cdr >> ros_message.frame_id;

  // Member: stamp
  builtin_interfaces::msg::typesupport_fastrtps_cpp::cdr_deserialize(
    cdr, ros_message.stamp);

  // Member: last_updated
  builtin_interfaces::msg::typesupport_fastrtps_cpp::cdr_deserialize(
    cdr, ros_message.last_updated);

  // Member: update_rate
  cdr >> ros_message.update_rate;

  // Member: staleness_warning
  {
    uint8_t tmp;
    cdr >> tmp;
    ros_message.staleness_warning = tmp ? true : false;
  }

  // Member: nodes
  {
    uint32_t cdrSize;
    cdr >> cdrSize;
    size_t size = static_cast<size_t>(cdrSize);

    // Check there are at least 'size' remaining bytes in the CDR stream before resizing
    auto old_state = cdr.getState();
    bool correct_size = cdr.jump(size);
    cdr.setState(old_state);
    if (!correct_size) {
      fprintf(stderr, "sequence size exceeds remaining buffer\n");
      return false;
    }

    ros_message.nodes.resize(size);
    for (size_t i = 0; i < size; i++) {
      hippocampus_ros2_msgs::msg::typesupport_fastrtps_cpp::cdr_deserialize(
        cdr, ros_message.nodes[i]);
    }
  }

  // Member: edges
  {
    uint32_t cdrSize;
    cdr >> cdrSize;
    size_t size = static_cast<size_t>(cdrSize);

    // Check there are at least 'size' remaining bytes in the CDR stream before resizing
    auto old_state = cdr.getState();
    bool correct_size = cdr.jump(size);
    cdr.setState(old_state);
    if (!correct_size) {
      fprintf(stderr, "sequence size exceeds remaining buffer\n");
      return false;
    }

    ros_message.edges.resize(size);
    for (size_t i = 0; i < size; i++) {
      hippocampus_ros2_msgs::msg::typesupport_fastrtps_cpp::cdr_deserialize(
        cdr, ros_message.edges[i]);
    }
  }

  return true;
}  // NOLINT(readability/fn_size)

size_t
ROSIDL_TYPESUPPORT_FASTRTPS_CPP_PUBLIC_hippocampus_ros2_msgs
get_serialized_size(
  const hippocampus_ros2_msgs::msg::GraphSnapshot & ros_message,
  size_t current_alignment)
{
  size_t initial_alignment = current_alignment;

  const size_t padding = 4;
  const size_t wchar_size = 4;
  (void)padding;
  (void)wchar_size;

  // Member: epoch_id
  {
    size_t item_size = sizeof(ros_message.epoch_id);
    current_alignment += item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // Member: frame_id
  current_alignment += padding +
    eprosima::fastcdr::Cdr::alignment(current_alignment, padding) +
    (ros_message.frame_id.size() + 1);
  // Member: stamp

  current_alignment +=
    builtin_interfaces::msg::typesupport_fastrtps_cpp::get_serialized_size(
    ros_message.stamp, current_alignment);
  // Member: last_updated

  current_alignment +=
    builtin_interfaces::msg::typesupport_fastrtps_cpp::get_serialized_size(
    ros_message.last_updated, current_alignment);
  // Member: update_rate
  {
    size_t item_size = sizeof(ros_message.update_rate);
    current_alignment += item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // Member: staleness_warning
  {
    size_t item_size = sizeof(ros_message.staleness_warning);
    current_alignment += item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // Member: nodes
  {
    size_t array_size = ros_message.nodes.size();

    current_alignment += padding +
      eprosima::fastcdr::Cdr::alignment(current_alignment, padding);

    for (size_t index = 0; index < array_size; ++index) {
      current_alignment +=
        hippocampus_ros2_msgs::msg::typesupport_fastrtps_cpp::get_serialized_size(
        ros_message.nodes[index], current_alignment);
    }
  }
  // Member: edges
  {
    size_t array_size = ros_message.edges.size();

    current_alignment += padding +
      eprosima::fastcdr::Cdr::alignment(current_alignment, padding);

    for (size_t index = 0; index < array_size; ++index) {
      current_alignment +=
        hippocampus_ros2_msgs::msg::typesupport_fastrtps_cpp::get_serialized_size(
        ros_message.edges[index], current_alignment);
    }
  }

  return current_alignment - initial_alignment;
}

size_t
ROSIDL_TYPESUPPORT_FASTRTPS_CPP_PUBLIC_hippocampus_ros2_msgs
max_serialized_size_GraphSnapshot(
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


  // Member: epoch_id
  {
    size_t array_size = 1;

    last_member_size = array_size * sizeof(uint32_t);
    current_alignment += array_size * sizeof(uint32_t) +
      eprosima::fastcdr::Cdr::alignment(current_alignment, sizeof(uint32_t));
  }

  // Member: frame_id
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

  // Member: last_updated
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

  // Member: update_rate
  {
    size_t array_size = 1;

    last_member_size = array_size * sizeof(uint64_t);
    current_alignment += array_size * sizeof(uint64_t) +
      eprosima::fastcdr::Cdr::alignment(current_alignment, sizeof(uint64_t));
  }

  // Member: staleness_warning
  {
    size_t array_size = 1;

    last_member_size = array_size * sizeof(uint8_t);
    current_alignment += array_size * sizeof(uint8_t);
  }

  // Member: nodes
  {
    size_t array_size = 0;
    full_bounded = false;
    is_plain = false;
    current_alignment += padding +
      eprosima::fastcdr::Cdr::alignment(current_alignment, padding);


    last_member_size = 0;
    for (size_t index = 0; index < array_size; ++index) {
      bool inner_full_bounded;
      bool inner_is_plain;
      size_t inner_size =
        hippocampus_ros2_msgs::msg::typesupport_fastrtps_cpp::max_serialized_size_GraphNode(
        inner_full_bounded, inner_is_plain, current_alignment);
      last_member_size += inner_size;
      current_alignment += inner_size;
      full_bounded &= inner_full_bounded;
      is_plain &= inner_is_plain;
    }
  }

  // Member: edges
  {
    size_t array_size = 0;
    full_bounded = false;
    is_plain = false;
    current_alignment += padding +
      eprosima::fastcdr::Cdr::alignment(current_alignment, padding);


    last_member_size = 0;
    for (size_t index = 0; index < array_size; ++index) {
      bool inner_full_bounded;
      bool inner_is_plain;
      size_t inner_size =
        hippocampus_ros2_msgs::msg::typesupport_fastrtps_cpp::max_serialized_size_GraphEdge(
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
    using DataType = hippocampus_ros2_msgs::msg::GraphSnapshot;
    is_plain =
      (
      offsetof(DataType, edges) +
      last_member_size
      ) == ret_val;
  }

  return ret_val;
}

static bool _GraphSnapshot__cdr_serialize(
  const void * untyped_ros_message,
  eprosima::fastcdr::Cdr & cdr)
{
  auto typed_message =
    static_cast<const hippocampus_ros2_msgs::msg::GraphSnapshot *>(
    untyped_ros_message);
  return cdr_serialize(*typed_message, cdr);
}

static bool _GraphSnapshot__cdr_deserialize(
  eprosima::fastcdr::Cdr & cdr,
  void * untyped_ros_message)
{
  auto typed_message =
    static_cast<hippocampus_ros2_msgs::msg::GraphSnapshot *>(
    untyped_ros_message);
  return cdr_deserialize(cdr, *typed_message);
}

static uint32_t _GraphSnapshot__get_serialized_size(
  const void * untyped_ros_message)
{
  auto typed_message =
    static_cast<const hippocampus_ros2_msgs::msg::GraphSnapshot *>(
    untyped_ros_message);
  return static_cast<uint32_t>(get_serialized_size(*typed_message, 0));
}

static size_t _GraphSnapshot__max_serialized_size(char & bounds_info)
{
  bool full_bounded;
  bool is_plain;
  size_t ret_val;

  ret_val = max_serialized_size_GraphSnapshot(full_bounded, is_plain, 0);

  bounds_info =
    is_plain ? ROSIDL_TYPESUPPORT_FASTRTPS_PLAIN_TYPE :
    full_bounded ? ROSIDL_TYPESUPPORT_FASTRTPS_BOUNDED_TYPE : ROSIDL_TYPESUPPORT_FASTRTPS_UNBOUNDED_TYPE;
  return ret_val;
}

static message_type_support_callbacks_t _GraphSnapshot__callbacks = {
  "hippocampus_ros2_msgs::msg",
  "GraphSnapshot",
  _GraphSnapshot__cdr_serialize,
  _GraphSnapshot__cdr_deserialize,
  _GraphSnapshot__get_serialized_size,
  _GraphSnapshot__max_serialized_size
};

static rosidl_message_type_support_t _GraphSnapshot__handle = {
  rosidl_typesupport_fastrtps_cpp::typesupport_identifier,
  &_GraphSnapshot__callbacks,
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
get_message_type_support_handle<hippocampus_ros2_msgs::msg::GraphSnapshot>()
{
  return &hippocampus_ros2_msgs::msg::typesupport_fastrtps_cpp::_GraphSnapshot__handle;
}

}  // namespace rosidl_typesupport_fastrtps_cpp

#ifdef __cplusplus
extern "C"
{
#endif

const rosidl_message_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_fastrtps_cpp, hippocampus_ros2_msgs, msg, GraphSnapshot)() {
  return &hippocampus_ros2_msgs::msg::typesupport_fastrtps_cpp::_GraphSnapshot__handle;
}

#ifdef __cplusplus
}
#endif
