// generated from rosidl_typesupport_introspection_c/resource/idl__type_support.c.em
// with input from hippocampus_ros2_msgs:msg/GraphNode.idl
// generated code does not contain a copyright notice

#include <stddef.h>
#include "hippocampus_ros2_msgs/msg/detail/graph_node__rosidl_typesupport_introspection_c.h"
#include "hippocampus_ros2_msgs/msg/rosidl_typesupport_introspection_c__visibility_control.h"
#include "rosidl_typesupport_introspection_c/field_types.h"
#include "rosidl_typesupport_introspection_c/identifier.h"
#include "rosidl_typesupport_introspection_c/message_introspection.h"
#include "hippocampus_ros2_msgs/msg/detail/graph_node__functions.h"
#include "hippocampus_ros2_msgs/msg/detail/graph_node__struct.h"


// Include directives for member types
// Member `position`
#include "geometry_msgs/msg/point.h"
// Member `position`
#include "geometry_msgs/msg/detail/point__rosidl_typesupport_introspection_c.h"
// Member `normal`
#include "geometry_msgs/msg/vector3.h"
// Member `normal`
#include "geometry_msgs/msg/detail/vector3__rosidl_typesupport_introspection_c.h"
// Member `tags`
#include "rosidl_runtime_c/string_functions.h"

#ifdef __cplusplus
extern "C"
{
#endif

void hippocampus_ros2_msgs__msg__GraphNode__rosidl_typesupport_introspection_c__GraphNode_init_function(
  void * message_memory, enum rosidl_runtime_c__message_initialization _init)
{
  // TODO(karsten1987): initializers are not yet implemented for typesupport c
  // see https://github.com/ros2/ros2/issues/397
  (void) _init;
  hippocampus_ros2_msgs__msg__GraphNode__init(message_memory);
}

void hippocampus_ros2_msgs__msg__GraphNode__rosidl_typesupport_introspection_c__GraphNode_fini_function(void * message_memory)
{
  hippocampus_ros2_msgs__msg__GraphNode__fini(message_memory);
}

size_t hippocampus_ros2_msgs__msg__GraphNode__rosidl_typesupport_introspection_c__size_function__GraphNode__tags(
  const void * untyped_member)
{
  const rosidl_runtime_c__String__Sequence * member =
    (const rosidl_runtime_c__String__Sequence *)(untyped_member);
  return member->size;
}

const void * hippocampus_ros2_msgs__msg__GraphNode__rosidl_typesupport_introspection_c__get_const_function__GraphNode__tags(
  const void * untyped_member, size_t index)
{
  const rosidl_runtime_c__String__Sequence * member =
    (const rosidl_runtime_c__String__Sequence *)(untyped_member);
  return &member->data[index];
}

void * hippocampus_ros2_msgs__msg__GraphNode__rosidl_typesupport_introspection_c__get_function__GraphNode__tags(
  void * untyped_member, size_t index)
{
  rosidl_runtime_c__String__Sequence * member =
    (rosidl_runtime_c__String__Sequence *)(untyped_member);
  return &member->data[index];
}

void hippocampus_ros2_msgs__msg__GraphNode__rosidl_typesupport_introspection_c__fetch_function__GraphNode__tags(
  const void * untyped_member, size_t index, void * untyped_value)
{
  const rosidl_runtime_c__String * item =
    ((const rosidl_runtime_c__String *)
    hippocampus_ros2_msgs__msg__GraphNode__rosidl_typesupport_introspection_c__get_const_function__GraphNode__tags(untyped_member, index));
  rosidl_runtime_c__String * value =
    (rosidl_runtime_c__String *)(untyped_value);
  *value = *item;
}

void hippocampus_ros2_msgs__msg__GraphNode__rosidl_typesupport_introspection_c__assign_function__GraphNode__tags(
  void * untyped_member, size_t index, const void * untyped_value)
{
  rosidl_runtime_c__String * item =
    ((rosidl_runtime_c__String *)
    hippocampus_ros2_msgs__msg__GraphNode__rosidl_typesupport_introspection_c__get_function__GraphNode__tags(untyped_member, index));
  const rosidl_runtime_c__String * value =
    (const rosidl_runtime_c__String *)(untyped_value);
  *item = *value;
}

bool hippocampus_ros2_msgs__msg__GraphNode__rosidl_typesupport_introspection_c__resize_function__GraphNode__tags(
  void * untyped_member, size_t size)
{
  rosidl_runtime_c__String__Sequence * member =
    (rosidl_runtime_c__String__Sequence *)(untyped_member);
  rosidl_runtime_c__String__Sequence__fini(member);
  return rosidl_runtime_c__String__Sequence__init(member, size);
}

static rosidl_typesupport_introspection_c__MessageMember hippocampus_ros2_msgs__msg__GraphNode__rosidl_typesupport_introspection_c__GraphNode_message_member_array[5] = {
  {
    "node_id",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_INT32,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(hippocampus_ros2_msgs__msg__GraphNode, node_id),  // bytes offset in struct
    NULL,  // default value
    NULL,  // size() function pointer
    NULL,  // get_const(index) function pointer
    NULL,  // get(index) function pointer
    NULL,  // fetch(index, &value) function pointer
    NULL,  // assign(index, value) function pointer
    NULL  // resize(index) function pointer
  },
  {
    "position",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_MESSAGE,  // type
    0,  // upper bound of string
    NULL,  // members of sub message (initialized later)
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(hippocampus_ros2_msgs__msg__GraphNode, position),  // bytes offset in struct
    NULL,  // default value
    NULL,  // size() function pointer
    NULL,  // get_const(index) function pointer
    NULL,  // get(index) function pointer
    NULL,  // fetch(index, &value) function pointer
    NULL,  // assign(index, value) function pointer
    NULL  // resize(index) function pointer
  },
  {
    "normal",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_MESSAGE,  // type
    0,  // upper bound of string
    NULL,  // members of sub message (initialized later)
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(hippocampus_ros2_msgs__msg__GraphNode, normal),  // bytes offset in struct
    NULL,  // default value
    NULL,  // size() function pointer
    NULL,  // get_const(index) function pointer
    NULL,  // get(index) function pointer
    NULL,  // fetch(index, &value) function pointer
    NULL,  // assign(index, value) function pointer
    NULL  // resize(index) function pointer
  },
  {
    "degree",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_UINT32,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(hippocampus_ros2_msgs__msg__GraphNode, degree),  // bytes offset in struct
    NULL,  // default value
    NULL,  // size() function pointer
    NULL,  // get_const(index) function pointer
    NULL,  // get(index) function pointer
    NULL,  // fetch(index, &value) function pointer
    NULL,  // assign(index, value) function pointer
    NULL  // resize(index) function pointer
  },
  {
    "tags",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_STRING,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    true,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(hippocampus_ros2_msgs__msg__GraphNode, tags),  // bytes offset in struct
    NULL,  // default value
    hippocampus_ros2_msgs__msg__GraphNode__rosidl_typesupport_introspection_c__size_function__GraphNode__tags,  // size() function pointer
    hippocampus_ros2_msgs__msg__GraphNode__rosidl_typesupport_introspection_c__get_const_function__GraphNode__tags,  // get_const(index) function pointer
    hippocampus_ros2_msgs__msg__GraphNode__rosidl_typesupport_introspection_c__get_function__GraphNode__tags,  // get(index) function pointer
    hippocampus_ros2_msgs__msg__GraphNode__rosidl_typesupport_introspection_c__fetch_function__GraphNode__tags,  // fetch(index, &value) function pointer
    hippocampus_ros2_msgs__msg__GraphNode__rosidl_typesupport_introspection_c__assign_function__GraphNode__tags,  // assign(index, value) function pointer
    hippocampus_ros2_msgs__msg__GraphNode__rosidl_typesupport_introspection_c__resize_function__GraphNode__tags  // resize(index) function pointer
  }
};

static const rosidl_typesupport_introspection_c__MessageMembers hippocampus_ros2_msgs__msg__GraphNode__rosidl_typesupport_introspection_c__GraphNode_message_members = {
  "hippocampus_ros2_msgs__msg",  // message namespace
  "GraphNode",  // message name
  5,  // number of fields
  sizeof(hippocampus_ros2_msgs__msg__GraphNode),
  hippocampus_ros2_msgs__msg__GraphNode__rosidl_typesupport_introspection_c__GraphNode_message_member_array,  // message members
  hippocampus_ros2_msgs__msg__GraphNode__rosidl_typesupport_introspection_c__GraphNode_init_function,  // function to initialize message memory (memory has to be allocated)
  hippocampus_ros2_msgs__msg__GraphNode__rosidl_typesupport_introspection_c__GraphNode_fini_function  // function to terminate message instance (will not free memory)
};

// this is not const since it must be initialized on first access
// since C does not allow non-integral compile-time constants
static rosidl_message_type_support_t hippocampus_ros2_msgs__msg__GraphNode__rosidl_typesupport_introspection_c__GraphNode_message_type_support_handle = {
  0,
  &hippocampus_ros2_msgs__msg__GraphNode__rosidl_typesupport_introspection_c__GraphNode_message_members,
  get_message_typesupport_handle_function,
};

ROSIDL_TYPESUPPORT_INTROSPECTION_C_EXPORT_hippocampus_ros2_msgs
const rosidl_message_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, hippocampus_ros2_msgs, msg, GraphNode)() {
  hippocampus_ros2_msgs__msg__GraphNode__rosidl_typesupport_introspection_c__GraphNode_message_member_array[1].members_ =
    ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, geometry_msgs, msg, Point)();
  hippocampus_ros2_msgs__msg__GraphNode__rosidl_typesupport_introspection_c__GraphNode_message_member_array[2].members_ =
    ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, geometry_msgs, msg, Vector3)();
  if (!hippocampus_ros2_msgs__msg__GraphNode__rosidl_typesupport_introspection_c__GraphNode_message_type_support_handle.typesupport_identifier) {
    hippocampus_ros2_msgs__msg__GraphNode__rosidl_typesupport_introspection_c__GraphNode_message_type_support_handle.typesupport_identifier =
      rosidl_typesupport_introspection_c__identifier;
  }
  return &hippocampus_ros2_msgs__msg__GraphNode__rosidl_typesupport_introspection_c__GraphNode_message_type_support_handle;
}
#ifdef __cplusplus
}
#endif
