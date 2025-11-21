// generated from rosidl_typesupport_introspection_c/resource/idl__type_support.c.em
// with input from hippocampus_ros2_msgs:msg/GraphEdge.idl
// generated code does not contain a copyright notice

#include <stddef.h>
#include "hippocampus_ros2_msgs/msg/detail/graph_edge__rosidl_typesupport_introspection_c.h"
#include "hippocampus_ros2_msgs/msg/rosidl_typesupport_introspection_c__visibility_control.h"
#include "rosidl_typesupport_introspection_c/field_types.h"
#include "rosidl_typesupport_introspection_c/identifier.h"
#include "rosidl_typesupport_introspection_c/message_introspection.h"
#include "hippocampus_ros2_msgs/msg/detail/graph_edge__functions.h"
#include "hippocampus_ros2_msgs/msg/detail/graph_edge__struct.h"


#ifdef __cplusplus
extern "C"
{
#endif

void hippocampus_ros2_msgs__msg__GraphEdge__rosidl_typesupport_introspection_c__GraphEdge_init_function(
  void * message_memory, enum rosidl_runtime_c__message_initialization _init)
{
  // TODO(karsten1987): initializers are not yet implemented for typesupport c
  // see https://github.com/ros2/ros2/issues/397
  (void) _init;
  hippocampus_ros2_msgs__msg__GraphEdge__init(message_memory);
}

void hippocampus_ros2_msgs__msg__GraphEdge__rosidl_typesupport_introspection_c__GraphEdge_fini_function(void * message_memory)
{
  hippocampus_ros2_msgs__msg__GraphEdge__fini(message_memory);
}

static rosidl_typesupport_introspection_c__MessageMember hippocampus_ros2_msgs__msg__GraphEdge__rosidl_typesupport_introspection_c__GraphEdge_message_member_array[4] = {
  {
    "u",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_INT32,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(hippocampus_ros2_msgs__msg__GraphEdge, u),  // bytes offset in struct
    NULL,  // default value
    NULL,  // size() function pointer
    NULL,  // get_const(index) function pointer
    NULL,  // get(index) function pointer
    NULL,  // fetch(index, &value) function pointer
    NULL,  // assign(index, value) function pointer
    NULL  // resize(index) function pointer
  },
  {
    "v",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_INT32,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(hippocampus_ros2_msgs__msg__GraphEdge, v),  // bytes offset in struct
    NULL,  // default value
    NULL,  // size() function pointer
    NULL,  // get_const(index) function pointer
    NULL,  // get(index) function pointer
    NULL,  // fetch(index, &value) function pointer
    NULL,  // assign(index, value) function pointer
    NULL  // resize(index) function pointer
  },
  {
    "length",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_DOUBLE,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(hippocampus_ros2_msgs__msg__GraphEdge, length),  // bytes offset in struct
    NULL,  // default value
    NULL,  // size() function pointer
    NULL,  // get_const(index) function pointer
    NULL,  // get(index) function pointer
    NULL,  // fetch(index, &value) function pointer
    NULL,  // assign(index, value) function pointer
    NULL  // resize(index) function pointer
  },
  {
    "traversable",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_BOOLEAN,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(hippocampus_ros2_msgs__msg__GraphEdge, traversable),  // bytes offset in struct
    NULL,  // default value
    NULL,  // size() function pointer
    NULL,  // get_const(index) function pointer
    NULL,  // get(index) function pointer
    NULL,  // fetch(index, &value) function pointer
    NULL,  // assign(index, value) function pointer
    NULL  // resize(index) function pointer
  }
};

static const rosidl_typesupport_introspection_c__MessageMembers hippocampus_ros2_msgs__msg__GraphEdge__rosidl_typesupport_introspection_c__GraphEdge_message_members = {
  "hippocampus_ros2_msgs__msg",  // message namespace
  "GraphEdge",  // message name
  4,  // number of fields
  sizeof(hippocampus_ros2_msgs__msg__GraphEdge),
  hippocampus_ros2_msgs__msg__GraphEdge__rosidl_typesupport_introspection_c__GraphEdge_message_member_array,  // message members
  hippocampus_ros2_msgs__msg__GraphEdge__rosidl_typesupport_introspection_c__GraphEdge_init_function,  // function to initialize message memory (memory has to be allocated)
  hippocampus_ros2_msgs__msg__GraphEdge__rosidl_typesupport_introspection_c__GraphEdge_fini_function  // function to terminate message instance (will not free memory)
};

// this is not const since it must be initialized on first access
// since C does not allow non-integral compile-time constants
static rosidl_message_type_support_t hippocampus_ros2_msgs__msg__GraphEdge__rosidl_typesupport_introspection_c__GraphEdge_message_type_support_handle = {
  0,
  &hippocampus_ros2_msgs__msg__GraphEdge__rosidl_typesupport_introspection_c__GraphEdge_message_members,
  get_message_typesupport_handle_function,
};

ROSIDL_TYPESUPPORT_INTROSPECTION_C_EXPORT_hippocampus_ros2_msgs
const rosidl_message_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, hippocampus_ros2_msgs, msg, GraphEdge)() {
  if (!hippocampus_ros2_msgs__msg__GraphEdge__rosidl_typesupport_introspection_c__GraphEdge_message_type_support_handle.typesupport_identifier) {
    hippocampus_ros2_msgs__msg__GraphEdge__rosidl_typesupport_introspection_c__GraphEdge_message_type_support_handle.typesupport_identifier =
      rosidl_typesupport_introspection_c__identifier;
  }
  return &hippocampus_ros2_msgs__msg__GraphEdge__rosidl_typesupport_introspection_c__GraphEdge_message_type_support_handle;
}
#ifdef __cplusplus
}
#endif
