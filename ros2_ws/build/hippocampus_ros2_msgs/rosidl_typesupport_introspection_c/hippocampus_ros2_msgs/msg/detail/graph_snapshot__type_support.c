// generated from rosidl_typesupport_introspection_c/resource/idl__type_support.c.em
// with input from hippocampus_ros2_msgs:msg/GraphSnapshot.idl
// generated code does not contain a copyright notice

#include <stddef.h>
#include "hippocampus_ros2_msgs/msg/detail/graph_snapshot__rosidl_typesupport_introspection_c.h"
#include "hippocampus_ros2_msgs/msg/rosidl_typesupport_introspection_c__visibility_control.h"
#include "rosidl_typesupport_introspection_c/field_types.h"
#include "rosidl_typesupport_introspection_c/identifier.h"
#include "rosidl_typesupport_introspection_c/message_introspection.h"
#include "hippocampus_ros2_msgs/msg/detail/graph_snapshot__functions.h"
#include "hippocampus_ros2_msgs/msg/detail/graph_snapshot__struct.h"


// Include directives for member types
// Member `frame_id`
#include "rosidl_runtime_c/string_functions.h"
// Member `stamp`
// Member `last_updated`
#include "builtin_interfaces/msg/time.h"
// Member `stamp`
// Member `last_updated`
#include "builtin_interfaces/msg/detail/time__rosidl_typesupport_introspection_c.h"
// Member `nodes`
#include "hippocampus_ros2_msgs/msg/graph_node.h"
// Member `nodes`
#include "hippocampus_ros2_msgs/msg/detail/graph_node__rosidl_typesupport_introspection_c.h"
// Member `edges`
#include "hippocampus_ros2_msgs/msg/graph_edge.h"
// Member `edges`
#include "hippocampus_ros2_msgs/msg/detail/graph_edge__rosidl_typesupport_introspection_c.h"

#ifdef __cplusplus
extern "C"
{
#endif

void hippocampus_ros2_msgs__msg__GraphSnapshot__rosidl_typesupport_introspection_c__GraphSnapshot_init_function(
  void * message_memory, enum rosidl_runtime_c__message_initialization _init)
{
  // TODO(karsten1987): initializers are not yet implemented for typesupport c
  // see https://github.com/ros2/ros2/issues/397
  (void) _init;
  hippocampus_ros2_msgs__msg__GraphSnapshot__init(message_memory);
}

void hippocampus_ros2_msgs__msg__GraphSnapshot__rosidl_typesupport_introspection_c__GraphSnapshot_fini_function(void * message_memory)
{
  hippocampus_ros2_msgs__msg__GraphSnapshot__fini(message_memory);
}

size_t hippocampus_ros2_msgs__msg__GraphSnapshot__rosidl_typesupport_introspection_c__size_function__GraphSnapshot__nodes(
  const void * untyped_member)
{
  const hippocampus_ros2_msgs__msg__GraphNode__Sequence * member =
    (const hippocampus_ros2_msgs__msg__GraphNode__Sequence *)(untyped_member);
  return member->size;
}

const void * hippocampus_ros2_msgs__msg__GraphSnapshot__rosidl_typesupport_introspection_c__get_const_function__GraphSnapshot__nodes(
  const void * untyped_member, size_t index)
{
  const hippocampus_ros2_msgs__msg__GraphNode__Sequence * member =
    (const hippocampus_ros2_msgs__msg__GraphNode__Sequence *)(untyped_member);
  return &member->data[index];
}

void * hippocampus_ros2_msgs__msg__GraphSnapshot__rosidl_typesupport_introspection_c__get_function__GraphSnapshot__nodes(
  void * untyped_member, size_t index)
{
  hippocampus_ros2_msgs__msg__GraphNode__Sequence * member =
    (hippocampus_ros2_msgs__msg__GraphNode__Sequence *)(untyped_member);
  return &member->data[index];
}

void hippocampus_ros2_msgs__msg__GraphSnapshot__rosidl_typesupport_introspection_c__fetch_function__GraphSnapshot__nodes(
  const void * untyped_member, size_t index, void * untyped_value)
{
  const hippocampus_ros2_msgs__msg__GraphNode * item =
    ((const hippocampus_ros2_msgs__msg__GraphNode *)
    hippocampus_ros2_msgs__msg__GraphSnapshot__rosidl_typesupport_introspection_c__get_const_function__GraphSnapshot__nodes(untyped_member, index));
  hippocampus_ros2_msgs__msg__GraphNode * value =
    (hippocampus_ros2_msgs__msg__GraphNode *)(untyped_value);
  *value = *item;
}

void hippocampus_ros2_msgs__msg__GraphSnapshot__rosidl_typesupport_introspection_c__assign_function__GraphSnapshot__nodes(
  void * untyped_member, size_t index, const void * untyped_value)
{
  hippocampus_ros2_msgs__msg__GraphNode * item =
    ((hippocampus_ros2_msgs__msg__GraphNode *)
    hippocampus_ros2_msgs__msg__GraphSnapshot__rosidl_typesupport_introspection_c__get_function__GraphSnapshot__nodes(untyped_member, index));
  const hippocampus_ros2_msgs__msg__GraphNode * value =
    (const hippocampus_ros2_msgs__msg__GraphNode *)(untyped_value);
  *item = *value;
}

bool hippocampus_ros2_msgs__msg__GraphSnapshot__rosidl_typesupport_introspection_c__resize_function__GraphSnapshot__nodes(
  void * untyped_member, size_t size)
{
  hippocampus_ros2_msgs__msg__GraphNode__Sequence * member =
    (hippocampus_ros2_msgs__msg__GraphNode__Sequence *)(untyped_member);
  hippocampus_ros2_msgs__msg__GraphNode__Sequence__fini(member);
  return hippocampus_ros2_msgs__msg__GraphNode__Sequence__init(member, size);
}

size_t hippocampus_ros2_msgs__msg__GraphSnapshot__rosidl_typesupport_introspection_c__size_function__GraphSnapshot__edges(
  const void * untyped_member)
{
  const hippocampus_ros2_msgs__msg__GraphEdge__Sequence * member =
    (const hippocampus_ros2_msgs__msg__GraphEdge__Sequence *)(untyped_member);
  return member->size;
}

const void * hippocampus_ros2_msgs__msg__GraphSnapshot__rosidl_typesupport_introspection_c__get_const_function__GraphSnapshot__edges(
  const void * untyped_member, size_t index)
{
  const hippocampus_ros2_msgs__msg__GraphEdge__Sequence * member =
    (const hippocampus_ros2_msgs__msg__GraphEdge__Sequence *)(untyped_member);
  return &member->data[index];
}

void * hippocampus_ros2_msgs__msg__GraphSnapshot__rosidl_typesupport_introspection_c__get_function__GraphSnapshot__edges(
  void * untyped_member, size_t index)
{
  hippocampus_ros2_msgs__msg__GraphEdge__Sequence * member =
    (hippocampus_ros2_msgs__msg__GraphEdge__Sequence *)(untyped_member);
  return &member->data[index];
}

void hippocampus_ros2_msgs__msg__GraphSnapshot__rosidl_typesupport_introspection_c__fetch_function__GraphSnapshot__edges(
  const void * untyped_member, size_t index, void * untyped_value)
{
  const hippocampus_ros2_msgs__msg__GraphEdge * item =
    ((const hippocampus_ros2_msgs__msg__GraphEdge *)
    hippocampus_ros2_msgs__msg__GraphSnapshot__rosidl_typesupport_introspection_c__get_const_function__GraphSnapshot__edges(untyped_member, index));
  hippocampus_ros2_msgs__msg__GraphEdge * value =
    (hippocampus_ros2_msgs__msg__GraphEdge *)(untyped_value);
  *value = *item;
}

void hippocampus_ros2_msgs__msg__GraphSnapshot__rosidl_typesupport_introspection_c__assign_function__GraphSnapshot__edges(
  void * untyped_member, size_t index, const void * untyped_value)
{
  hippocampus_ros2_msgs__msg__GraphEdge * item =
    ((hippocampus_ros2_msgs__msg__GraphEdge *)
    hippocampus_ros2_msgs__msg__GraphSnapshot__rosidl_typesupport_introspection_c__get_function__GraphSnapshot__edges(untyped_member, index));
  const hippocampus_ros2_msgs__msg__GraphEdge * value =
    (const hippocampus_ros2_msgs__msg__GraphEdge *)(untyped_value);
  *item = *value;
}

bool hippocampus_ros2_msgs__msg__GraphSnapshot__rosidl_typesupport_introspection_c__resize_function__GraphSnapshot__edges(
  void * untyped_member, size_t size)
{
  hippocampus_ros2_msgs__msg__GraphEdge__Sequence * member =
    (hippocampus_ros2_msgs__msg__GraphEdge__Sequence *)(untyped_member);
  hippocampus_ros2_msgs__msg__GraphEdge__Sequence__fini(member);
  return hippocampus_ros2_msgs__msg__GraphEdge__Sequence__init(member, size);
}

static rosidl_typesupport_introspection_c__MessageMember hippocampus_ros2_msgs__msg__GraphSnapshot__rosidl_typesupport_introspection_c__GraphSnapshot_message_member_array[8] = {
  {
    "epoch_id",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_UINT32,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(hippocampus_ros2_msgs__msg__GraphSnapshot, epoch_id),  // bytes offset in struct
    NULL,  // default value
    NULL,  // size() function pointer
    NULL,  // get_const(index) function pointer
    NULL,  // get(index) function pointer
    NULL,  // fetch(index, &value) function pointer
    NULL,  // assign(index, value) function pointer
    NULL  // resize(index) function pointer
  },
  {
    "frame_id",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_STRING,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(hippocampus_ros2_msgs__msg__GraphSnapshot, frame_id),  // bytes offset in struct
    NULL,  // default value
    NULL,  // size() function pointer
    NULL,  // get_const(index) function pointer
    NULL,  // get(index) function pointer
    NULL,  // fetch(index, &value) function pointer
    NULL,  // assign(index, value) function pointer
    NULL  // resize(index) function pointer
  },
  {
    "stamp",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_MESSAGE,  // type
    0,  // upper bound of string
    NULL,  // members of sub message (initialized later)
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(hippocampus_ros2_msgs__msg__GraphSnapshot, stamp),  // bytes offset in struct
    NULL,  // default value
    NULL,  // size() function pointer
    NULL,  // get_const(index) function pointer
    NULL,  // get(index) function pointer
    NULL,  // fetch(index, &value) function pointer
    NULL,  // assign(index, value) function pointer
    NULL  // resize(index) function pointer
  },
  {
    "last_updated",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_MESSAGE,  // type
    0,  // upper bound of string
    NULL,  // members of sub message (initialized later)
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(hippocampus_ros2_msgs__msg__GraphSnapshot, last_updated),  // bytes offset in struct
    NULL,  // default value
    NULL,  // size() function pointer
    NULL,  // get_const(index) function pointer
    NULL,  // get(index) function pointer
    NULL,  // fetch(index, &value) function pointer
    NULL,  // assign(index, value) function pointer
    NULL  // resize(index) function pointer
  },
  {
    "update_rate",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_DOUBLE,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(hippocampus_ros2_msgs__msg__GraphSnapshot, update_rate),  // bytes offset in struct
    NULL,  // default value
    NULL,  // size() function pointer
    NULL,  // get_const(index) function pointer
    NULL,  // get(index) function pointer
    NULL,  // fetch(index, &value) function pointer
    NULL,  // assign(index, value) function pointer
    NULL  // resize(index) function pointer
  },
  {
    "staleness_warning",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_BOOLEAN,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(hippocampus_ros2_msgs__msg__GraphSnapshot, staleness_warning),  // bytes offset in struct
    NULL,  // default value
    NULL,  // size() function pointer
    NULL,  // get_const(index) function pointer
    NULL,  // get(index) function pointer
    NULL,  // fetch(index, &value) function pointer
    NULL,  // assign(index, value) function pointer
    NULL  // resize(index) function pointer
  },
  {
    "nodes",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_MESSAGE,  // type
    0,  // upper bound of string
    NULL,  // members of sub message (initialized later)
    true,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(hippocampus_ros2_msgs__msg__GraphSnapshot, nodes),  // bytes offset in struct
    NULL,  // default value
    hippocampus_ros2_msgs__msg__GraphSnapshot__rosidl_typesupport_introspection_c__size_function__GraphSnapshot__nodes,  // size() function pointer
    hippocampus_ros2_msgs__msg__GraphSnapshot__rosidl_typesupport_introspection_c__get_const_function__GraphSnapshot__nodes,  // get_const(index) function pointer
    hippocampus_ros2_msgs__msg__GraphSnapshot__rosidl_typesupport_introspection_c__get_function__GraphSnapshot__nodes,  // get(index) function pointer
    hippocampus_ros2_msgs__msg__GraphSnapshot__rosidl_typesupport_introspection_c__fetch_function__GraphSnapshot__nodes,  // fetch(index, &value) function pointer
    hippocampus_ros2_msgs__msg__GraphSnapshot__rosidl_typesupport_introspection_c__assign_function__GraphSnapshot__nodes,  // assign(index, value) function pointer
    hippocampus_ros2_msgs__msg__GraphSnapshot__rosidl_typesupport_introspection_c__resize_function__GraphSnapshot__nodes  // resize(index) function pointer
  },
  {
    "edges",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_MESSAGE,  // type
    0,  // upper bound of string
    NULL,  // members of sub message (initialized later)
    true,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(hippocampus_ros2_msgs__msg__GraphSnapshot, edges),  // bytes offset in struct
    NULL,  // default value
    hippocampus_ros2_msgs__msg__GraphSnapshot__rosidl_typesupport_introspection_c__size_function__GraphSnapshot__edges,  // size() function pointer
    hippocampus_ros2_msgs__msg__GraphSnapshot__rosidl_typesupport_introspection_c__get_const_function__GraphSnapshot__edges,  // get_const(index) function pointer
    hippocampus_ros2_msgs__msg__GraphSnapshot__rosidl_typesupport_introspection_c__get_function__GraphSnapshot__edges,  // get(index) function pointer
    hippocampus_ros2_msgs__msg__GraphSnapshot__rosidl_typesupport_introspection_c__fetch_function__GraphSnapshot__edges,  // fetch(index, &value) function pointer
    hippocampus_ros2_msgs__msg__GraphSnapshot__rosidl_typesupport_introspection_c__assign_function__GraphSnapshot__edges,  // assign(index, value) function pointer
    hippocampus_ros2_msgs__msg__GraphSnapshot__rosidl_typesupport_introspection_c__resize_function__GraphSnapshot__edges  // resize(index) function pointer
  }
};

static const rosidl_typesupport_introspection_c__MessageMembers hippocampus_ros2_msgs__msg__GraphSnapshot__rosidl_typesupport_introspection_c__GraphSnapshot_message_members = {
  "hippocampus_ros2_msgs__msg",  // message namespace
  "GraphSnapshot",  // message name
  8,  // number of fields
  sizeof(hippocampus_ros2_msgs__msg__GraphSnapshot),
  hippocampus_ros2_msgs__msg__GraphSnapshot__rosidl_typesupport_introspection_c__GraphSnapshot_message_member_array,  // message members
  hippocampus_ros2_msgs__msg__GraphSnapshot__rosidl_typesupport_introspection_c__GraphSnapshot_init_function,  // function to initialize message memory (memory has to be allocated)
  hippocampus_ros2_msgs__msg__GraphSnapshot__rosidl_typesupport_introspection_c__GraphSnapshot_fini_function  // function to terminate message instance (will not free memory)
};

// this is not const since it must be initialized on first access
// since C does not allow non-integral compile-time constants
static rosidl_message_type_support_t hippocampus_ros2_msgs__msg__GraphSnapshot__rosidl_typesupport_introspection_c__GraphSnapshot_message_type_support_handle = {
  0,
  &hippocampus_ros2_msgs__msg__GraphSnapshot__rosidl_typesupport_introspection_c__GraphSnapshot_message_members,
  get_message_typesupport_handle_function,
};

ROSIDL_TYPESUPPORT_INTROSPECTION_C_EXPORT_hippocampus_ros2_msgs
const rosidl_message_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, hippocampus_ros2_msgs, msg, GraphSnapshot)() {
  hippocampus_ros2_msgs__msg__GraphSnapshot__rosidl_typesupport_introspection_c__GraphSnapshot_message_member_array[2].members_ =
    ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, builtin_interfaces, msg, Time)();
  hippocampus_ros2_msgs__msg__GraphSnapshot__rosidl_typesupport_introspection_c__GraphSnapshot_message_member_array[3].members_ =
    ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, builtin_interfaces, msg, Time)();
  hippocampus_ros2_msgs__msg__GraphSnapshot__rosidl_typesupport_introspection_c__GraphSnapshot_message_member_array[6].members_ =
    ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, hippocampus_ros2_msgs, msg, GraphNode)();
  hippocampus_ros2_msgs__msg__GraphSnapshot__rosidl_typesupport_introspection_c__GraphSnapshot_message_member_array[7].members_ =
    ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, hippocampus_ros2_msgs, msg, GraphEdge)();
  if (!hippocampus_ros2_msgs__msg__GraphSnapshot__rosidl_typesupport_introspection_c__GraphSnapshot_message_type_support_handle.typesupport_identifier) {
    hippocampus_ros2_msgs__msg__GraphSnapshot__rosidl_typesupport_introspection_c__GraphSnapshot_message_type_support_handle.typesupport_identifier =
      rosidl_typesupport_introspection_c__identifier;
  }
  return &hippocampus_ros2_msgs__msg__GraphSnapshot__rosidl_typesupport_introspection_c__GraphSnapshot_message_type_support_handle;
}
#ifdef __cplusplus
}
#endif
