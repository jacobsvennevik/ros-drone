// generated from rosidl_typesupport_introspection_cpp/resource/idl__type_support.cpp.em
// with input from hippocampus_ros2_msgs:msg/GraphSnapshot.idl
// generated code does not contain a copyright notice

#include "array"
#include "cstddef"
#include "string"
#include "vector"
#include "rosidl_runtime_c/message_type_support_struct.h"
#include "rosidl_typesupport_cpp/message_type_support.hpp"
#include "rosidl_typesupport_interface/macros.h"
#include "hippocampus_ros2_msgs/msg/detail/graph_snapshot__struct.hpp"
#include "rosidl_typesupport_introspection_cpp/field_types.hpp"
#include "rosidl_typesupport_introspection_cpp/identifier.hpp"
#include "rosidl_typesupport_introspection_cpp/message_introspection.hpp"
#include "rosidl_typesupport_introspection_cpp/message_type_support_decl.hpp"
#include "rosidl_typesupport_introspection_cpp/visibility_control.h"

namespace hippocampus_ros2_msgs
{

namespace msg
{

namespace rosidl_typesupport_introspection_cpp
{

void GraphSnapshot_init_function(
  void * message_memory, rosidl_runtime_cpp::MessageInitialization _init)
{
  new (message_memory) hippocampus_ros2_msgs::msg::GraphSnapshot(_init);
}

void GraphSnapshot_fini_function(void * message_memory)
{
  auto typed_message = static_cast<hippocampus_ros2_msgs::msg::GraphSnapshot *>(message_memory);
  typed_message->~GraphSnapshot();
}

size_t size_function__GraphSnapshot__nodes(const void * untyped_member)
{
  const auto * member = reinterpret_cast<const std::vector<hippocampus_ros2_msgs::msg::GraphNode> *>(untyped_member);
  return member->size();
}

const void * get_const_function__GraphSnapshot__nodes(const void * untyped_member, size_t index)
{
  const auto & member =
    *reinterpret_cast<const std::vector<hippocampus_ros2_msgs::msg::GraphNode> *>(untyped_member);
  return &member[index];
}

void * get_function__GraphSnapshot__nodes(void * untyped_member, size_t index)
{
  auto & member =
    *reinterpret_cast<std::vector<hippocampus_ros2_msgs::msg::GraphNode> *>(untyped_member);
  return &member[index];
}

void fetch_function__GraphSnapshot__nodes(
  const void * untyped_member, size_t index, void * untyped_value)
{
  const auto & item = *reinterpret_cast<const hippocampus_ros2_msgs::msg::GraphNode *>(
    get_const_function__GraphSnapshot__nodes(untyped_member, index));
  auto & value = *reinterpret_cast<hippocampus_ros2_msgs::msg::GraphNode *>(untyped_value);
  value = item;
}

void assign_function__GraphSnapshot__nodes(
  void * untyped_member, size_t index, const void * untyped_value)
{
  auto & item = *reinterpret_cast<hippocampus_ros2_msgs::msg::GraphNode *>(
    get_function__GraphSnapshot__nodes(untyped_member, index));
  const auto & value = *reinterpret_cast<const hippocampus_ros2_msgs::msg::GraphNode *>(untyped_value);
  item = value;
}

void resize_function__GraphSnapshot__nodes(void * untyped_member, size_t size)
{
  auto * member =
    reinterpret_cast<std::vector<hippocampus_ros2_msgs::msg::GraphNode> *>(untyped_member);
  member->resize(size);
}

size_t size_function__GraphSnapshot__edges(const void * untyped_member)
{
  const auto * member = reinterpret_cast<const std::vector<hippocampus_ros2_msgs::msg::GraphEdge> *>(untyped_member);
  return member->size();
}

const void * get_const_function__GraphSnapshot__edges(const void * untyped_member, size_t index)
{
  const auto & member =
    *reinterpret_cast<const std::vector<hippocampus_ros2_msgs::msg::GraphEdge> *>(untyped_member);
  return &member[index];
}

void * get_function__GraphSnapshot__edges(void * untyped_member, size_t index)
{
  auto & member =
    *reinterpret_cast<std::vector<hippocampus_ros2_msgs::msg::GraphEdge> *>(untyped_member);
  return &member[index];
}

void fetch_function__GraphSnapshot__edges(
  const void * untyped_member, size_t index, void * untyped_value)
{
  const auto & item = *reinterpret_cast<const hippocampus_ros2_msgs::msg::GraphEdge *>(
    get_const_function__GraphSnapshot__edges(untyped_member, index));
  auto & value = *reinterpret_cast<hippocampus_ros2_msgs::msg::GraphEdge *>(untyped_value);
  value = item;
}

void assign_function__GraphSnapshot__edges(
  void * untyped_member, size_t index, const void * untyped_value)
{
  auto & item = *reinterpret_cast<hippocampus_ros2_msgs::msg::GraphEdge *>(
    get_function__GraphSnapshot__edges(untyped_member, index));
  const auto & value = *reinterpret_cast<const hippocampus_ros2_msgs::msg::GraphEdge *>(untyped_value);
  item = value;
}

void resize_function__GraphSnapshot__edges(void * untyped_member, size_t size)
{
  auto * member =
    reinterpret_cast<std::vector<hippocampus_ros2_msgs::msg::GraphEdge> *>(untyped_member);
  member->resize(size);
}

static const ::rosidl_typesupport_introspection_cpp::MessageMember GraphSnapshot_message_member_array[8] = {
  {
    "epoch_id",  // name
    ::rosidl_typesupport_introspection_cpp::ROS_TYPE_UINT32,  // type
    0,  // upper bound of string
    nullptr,  // members of sub message
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(hippocampus_ros2_msgs::msg::GraphSnapshot, epoch_id),  // bytes offset in struct
    nullptr,  // default value
    nullptr,  // size() function pointer
    nullptr,  // get_const(index) function pointer
    nullptr,  // get(index) function pointer
    nullptr,  // fetch(index, &value) function pointer
    nullptr,  // assign(index, value) function pointer
    nullptr  // resize(index) function pointer
  },
  {
    "frame_id",  // name
    ::rosidl_typesupport_introspection_cpp::ROS_TYPE_STRING,  // type
    0,  // upper bound of string
    nullptr,  // members of sub message
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(hippocampus_ros2_msgs::msg::GraphSnapshot, frame_id),  // bytes offset in struct
    nullptr,  // default value
    nullptr,  // size() function pointer
    nullptr,  // get_const(index) function pointer
    nullptr,  // get(index) function pointer
    nullptr,  // fetch(index, &value) function pointer
    nullptr,  // assign(index, value) function pointer
    nullptr  // resize(index) function pointer
  },
  {
    "stamp",  // name
    ::rosidl_typesupport_introspection_cpp::ROS_TYPE_MESSAGE,  // type
    0,  // upper bound of string
    ::rosidl_typesupport_introspection_cpp::get_message_type_support_handle<builtin_interfaces::msg::Time>(),  // members of sub message
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(hippocampus_ros2_msgs::msg::GraphSnapshot, stamp),  // bytes offset in struct
    nullptr,  // default value
    nullptr,  // size() function pointer
    nullptr,  // get_const(index) function pointer
    nullptr,  // get(index) function pointer
    nullptr,  // fetch(index, &value) function pointer
    nullptr,  // assign(index, value) function pointer
    nullptr  // resize(index) function pointer
  },
  {
    "last_updated",  // name
    ::rosidl_typesupport_introspection_cpp::ROS_TYPE_MESSAGE,  // type
    0,  // upper bound of string
    ::rosidl_typesupport_introspection_cpp::get_message_type_support_handle<builtin_interfaces::msg::Time>(),  // members of sub message
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(hippocampus_ros2_msgs::msg::GraphSnapshot, last_updated),  // bytes offset in struct
    nullptr,  // default value
    nullptr,  // size() function pointer
    nullptr,  // get_const(index) function pointer
    nullptr,  // get(index) function pointer
    nullptr,  // fetch(index, &value) function pointer
    nullptr,  // assign(index, value) function pointer
    nullptr  // resize(index) function pointer
  },
  {
    "update_rate",  // name
    ::rosidl_typesupport_introspection_cpp::ROS_TYPE_DOUBLE,  // type
    0,  // upper bound of string
    nullptr,  // members of sub message
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(hippocampus_ros2_msgs::msg::GraphSnapshot, update_rate),  // bytes offset in struct
    nullptr,  // default value
    nullptr,  // size() function pointer
    nullptr,  // get_const(index) function pointer
    nullptr,  // get(index) function pointer
    nullptr,  // fetch(index, &value) function pointer
    nullptr,  // assign(index, value) function pointer
    nullptr  // resize(index) function pointer
  },
  {
    "staleness_warning",  // name
    ::rosidl_typesupport_introspection_cpp::ROS_TYPE_BOOLEAN,  // type
    0,  // upper bound of string
    nullptr,  // members of sub message
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(hippocampus_ros2_msgs::msg::GraphSnapshot, staleness_warning),  // bytes offset in struct
    nullptr,  // default value
    nullptr,  // size() function pointer
    nullptr,  // get_const(index) function pointer
    nullptr,  // get(index) function pointer
    nullptr,  // fetch(index, &value) function pointer
    nullptr,  // assign(index, value) function pointer
    nullptr  // resize(index) function pointer
  },
  {
    "nodes",  // name
    ::rosidl_typesupport_introspection_cpp::ROS_TYPE_MESSAGE,  // type
    0,  // upper bound of string
    ::rosidl_typesupport_introspection_cpp::get_message_type_support_handle<hippocampus_ros2_msgs::msg::GraphNode>(),  // members of sub message
    true,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(hippocampus_ros2_msgs::msg::GraphSnapshot, nodes),  // bytes offset in struct
    nullptr,  // default value
    size_function__GraphSnapshot__nodes,  // size() function pointer
    get_const_function__GraphSnapshot__nodes,  // get_const(index) function pointer
    get_function__GraphSnapshot__nodes,  // get(index) function pointer
    fetch_function__GraphSnapshot__nodes,  // fetch(index, &value) function pointer
    assign_function__GraphSnapshot__nodes,  // assign(index, value) function pointer
    resize_function__GraphSnapshot__nodes  // resize(index) function pointer
  },
  {
    "edges",  // name
    ::rosidl_typesupport_introspection_cpp::ROS_TYPE_MESSAGE,  // type
    0,  // upper bound of string
    ::rosidl_typesupport_introspection_cpp::get_message_type_support_handle<hippocampus_ros2_msgs::msg::GraphEdge>(),  // members of sub message
    true,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(hippocampus_ros2_msgs::msg::GraphSnapshot, edges),  // bytes offset in struct
    nullptr,  // default value
    size_function__GraphSnapshot__edges,  // size() function pointer
    get_const_function__GraphSnapshot__edges,  // get_const(index) function pointer
    get_function__GraphSnapshot__edges,  // get(index) function pointer
    fetch_function__GraphSnapshot__edges,  // fetch(index, &value) function pointer
    assign_function__GraphSnapshot__edges,  // assign(index, value) function pointer
    resize_function__GraphSnapshot__edges  // resize(index) function pointer
  }
};

static const ::rosidl_typesupport_introspection_cpp::MessageMembers GraphSnapshot_message_members = {
  "hippocampus_ros2_msgs::msg",  // message namespace
  "GraphSnapshot",  // message name
  8,  // number of fields
  sizeof(hippocampus_ros2_msgs::msg::GraphSnapshot),
  GraphSnapshot_message_member_array,  // message members
  GraphSnapshot_init_function,  // function to initialize message memory (memory has to be allocated)
  GraphSnapshot_fini_function  // function to terminate message instance (will not free memory)
};

static const rosidl_message_type_support_t GraphSnapshot_message_type_support_handle = {
  ::rosidl_typesupport_introspection_cpp::typesupport_identifier,
  &GraphSnapshot_message_members,
  get_message_typesupport_handle_function,
};

}  // namespace rosidl_typesupport_introspection_cpp

}  // namespace msg

}  // namespace hippocampus_ros2_msgs


namespace rosidl_typesupport_introspection_cpp
{

template<>
ROSIDL_TYPESUPPORT_INTROSPECTION_CPP_PUBLIC
const rosidl_message_type_support_t *
get_message_type_support_handle<hippocampus_ros2_msgs::msg::GraphSnapshot>()
{
  return &::hippocampus_ros2_msgs::msg::rosidl_typesupport_introspection_cpp::GraphSnapshot_message_type_support_handle;
}

}  // namespace rosidl_typesupport_introspection_cpp

#ifdef __cplusplus
extern "C"
{
#endif

ROSIDL_TYPESUPPORT_INTROSPECTION_CPP_PUBLIC
const rosidl_message_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_cpp, hippocampus_ros2_msgs, msg, GraphSnapshot)() {
  return &::hippocampus_ros2_msgs::msg::rosidl_typesupport_introspection_cpp::GraphSnapshot_message_type_support_handle;
}

#ifdef __cplusplus
}
#endif
