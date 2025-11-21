// generated from rosidl_generator_py/resource/_idl_support.c.em
// with input from hippocampus_ros2_msgs:msg/MissionGoal.idl
// generated code does not contain a copyright notice
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <Python.h>
#include <stdbool.h>
#ifndef _WIN32
# pragma GCC diagnostic push
# pragma GCC diagnostic ignored "-Wunused-function"
#endif
#include "numpy/ndarrayobject.h"
#ifndef _WIN32
# pragma GCC diagnostic pop
#endif
#include "rosidl_runtime_c/visibility_control.h"
#include "hippocampus_ros2_msgs/msg/detail/mission_goal__struct.h"
#include "hippocampus_ros2_msgs/msg/detail/mission_goal__functions.h"

#include "rosidl_runtime_c/string.h"
#include "rosidl_runtime_c/string_functions.h"

ROSIDL_GENERATOR_C_IMPORT
bool geometry_msgs__msg__point__convert_from_py(PyObject * _pymsg, void * _ros_message);
ROSIDL_GENERATOR_C_IMPORT
PyObject * geometry_msgs__msg__point__convert_to_py(void * raw_ros_message);
ROSIDL_GENERATOR_C_IMPORT
bool geometry_msgs__msg__point__convert_from_py(PyObject * _pymsg, void * _ros_message);
ROSIDL_GENERATOR_C_IMPORT
PyObject * geometry_msgs__msg__point__convert_to_py(void * raw_ros_message);
ROSIDL_GENERATOR_C_IMPORT
bool builtin_interfaces__msg__time__convert_from_py(PyObject * _pymsg, void * _ros_message);
ROSIDL_GENERATOR_C_IMPORT
PyObject * builtin_interfaces__msg__time__convert_to_py(void * raw_ros_message);

ROSIDL_GENERATOR_C_EXPORT
bool hippocampus_ros2_msgs__msg__mission_goal__convert_from_py(PyObject * _pymsg, void * _ros_message)
{
  // check that the passed message is of the expected Python class
  {
    char full_classname_dest[52];
    {
      char * class_name = NULL;
      char * module_name = NULL;
      {
        PyObject * class_attr = PyObject_GetAttrString(_pymsg, "__class__");
        if (class_attr) {
          PyObject * name_attr = PyObject_GetAttrString(class_attr, "__name__");
          if (name_attr) {
            class_name = (char *)PyUnicode_1BYTE_DATA(name_attr);
            Py_DECREF(name_attr);
          }
          PyObject * module_attr = PyObject_GetAttrString(class_attr, "__module__");
          if (module_attr) {
            module_name = (char *)PyUnicode_1BYTE_DATA(module_attr);
            Py_DECREF(module_attr);
          }
          Py_DECREF(class_attr);
        }
      }
      if (!class_name || !module_name) {
        return false;
      }
      snprintf(full_classname_dest, sizeof(full_classname_dest), "%s.%s", module_name, class_name);
    }
    assert(strncmp("hippocampus_ros2_msgs.msg._mission_goal.MissionGoal", full_classname_dest, 51) == 0);
  }
  hippocampus_ros2_msgs__msg__MissionGoal * ros_message = _ros_message;
  {  // goal_type
    PyObject * field = PyObject_GetAttrString(_pymsg, "goal_type");
    if (!field) {
      return false;
    }
    assert(PyLong_Check(field));
    ros_message->goal_type = (uint8_t)PyLong_AsUnsignedLong(field);
    Py_DECREF(field);
  }
  {  // point_position
    PyObject * field = PyObject_GetAttrString(_pymsg, "point_position");
    if (!field) {
      return false;
    }
    if (!geometry_msgs__msg__point__convert_from_py(field, &ros_message->point_position)) {
      Py_DECREF(field);
      return false;
    }
    Py_DECREF(field);
  }
  {  // point_tolerance
    PyObject * field = PyObject_GetAttrString(_pymsg, "point_tolerance");
    if (!field) {
      return false;
    }
    assert(PyFloat_Check(field));
    ros_message->point_tolerance = PyFloat_AS_DOUBLE(field);
    Py_DECREF(field);
  }
  {  // node_id
    PyObject * field = PyObject_GetAttrString(_pymsg, "node_id");
    if (!field) {
      return false;
    }
    assert(PyLong_Check(field));
    ros_message->node_id = (int32_t)PyLong_AsLong(field);
    Py_DECREF(field);
  }
  {  // region_center
    PyObject * field = PyObject_GetAttrString(_pymsg, "region_center");
    if (!field) {
      return false;
    }
    if (!geometry_msgs__msg__point__convert_from_py(field, &ros_message->region_center)) {
      Py_DECREF(field);
      return false;
    }
    Py_DECREF(field);
  }
  {  // region_radius
    PyObject * field = PyObject_GetAttrString(_pymsg, "region_radius");
    if (!field) {
      return false;
    }
    assert(PyFloat_Check(field));
    ros_message->region_radius = PyFloat_AS_DOUBLE(field);
    Py_DECREF(field);
  }
  {  // frame_id
    PyObject * field = PyObject_GetAttrString(_pymsg, "frame_id");
    if (!field) {
      return false;
    }
    assert(PyUnicode_Check(field));
    PyObject * encoded_field = PyUnicode_AsUTF8String(field);
    if (!encoded_field) {
      Py_DECREF(field);
      return false;
    }
    rosidl_runtime_c__String__assign(&ros_message->frame_id, PyBytes_AS_STRING(encoded_field));
    Py_DECREF(encoded_field);
    Py_DECREF(field);
  }
  {  // timeout
    PyObject * field = PyObject_GetAttrString(_pymsg, "timeout");
    if (!field) {
      return false;
    }
    assert(PyFloat_Check(field));
    ros_message->timeout = PyFloat_AS_DOUBLE(field);
    Py_DECREF(field);
  }
  {  // is_reached
    PyObject * field = PyObject_GetAttrString(_pymsg, "is_reached");
    if (!field) {
      return false;
    }
    assert(PyBool_Check(field));
    ros_message->is_reached = (Py_True == field);
    Py_DECREF(field);
  }
  {  // stamp
    PyObject * field = PyObject_GetAttrString(_pymsg, "stamp");
    if (!field) {
      return false;
    }
    if (!builtin_interfaces__msg__time__convert_from_py(field, &ros_message->stamp)) {
      Py_DECREF(field);
      return false;
    }
    Py_DECREF(field);
  }

  return true;
}

ROSIDL_GENERATOR_C_EXPORT
PyObject * hippocampus_ros2_msgs__msg__mission_goal__convert_to_py(void * raw_ros_message)
{
  /* NOTE(esteve): Call constructor of MissionGoal */
  PyObject * _pymessage = NULL;
  {
    PyObject * pymessage_module = PyImport_ImportModule("hippocampus_ros2_msgs.msg._mission_goal");
    assert(pymessage_module);
    PyObject * pymessage_class = PyObject_GetAttrString(pymessage_module, "MissionGoal");
    assert(pymessage_class);
    Py_DECREF(pymessage_module);
    _pymessage = PyObject_CallObject(pymessage_class, NULL);
    Py_DECREF(pymessage_class);
    if (!_pymessage) {
      return NULL;
    }
  }
  hippocampus_ros2_msgs__msg__MissionGoal * ros_message = (hippocampus_ros2_msgs__msg__MissionGoal *)raw_ros_message;
  {  // goal_type
    PyObject * field = NULL;
    field = PyLong_FromUnsignedLong(ros_message->goal_type);
    {
      int rc = PyObject_SetAttrString(_pymessage, "goal_type", field);
      Py_DECREF(field);
      if (rc) {
        return NULL;
      }
    }
  }
  {  // point_position
    PyObject * field = NULL;
    field = geometry_msgs__msg__point__convert_to_py(&ros_message->point_position);
    if (!field) {
      return NULL;
    }
    {
      int rc = PyObject_SetAttrString(_pymessage, "point_position", field);
      Py_DECREF(field);
      if (rc) {
        return NULL;
      }
    }
  }
  {  // point_tolerance
    PyObject * field = NULL;
    field = PyFloat_FromDouble(ros_message->point_tolerance);
    {
      int rc = PyObject_SetAttrString(_pymessage, "point_tolerance", field);
      Py_DECREF(field);
      if (rc) {
        return NULL;
      }
    }
  }
  {  // node_id
    PyObject * field = NULL;
    field = PyLong_FromLong(ros_message->node_id);
    {
      int rc = PyObject_SetAttrString(_pymessage, "node_id", field);
      Py_DECREF(field);
      if (rc) {
        return NULL;
      }
    }
  }
  {  // region_center
    PyObject * field = NULL;
    field = geometry_msgs__msg__point__convert_to_py(&ros_message->region_center);
    if (!field) {
      return NULL;
    }
    {
      int rc = PyObject_SetAttrString(_pymessage, "region_center", field);
      Py_DECREF(field);
      if (rc) {
        return NULL;
      }
    }
  }
  {  // region_radius
    PyObject * field = NULL;
    field = PyFloat_FromDouble(ros_message->region_radius);
    {
      int rc = PyObject_SetAttrString(_pymessage, "region_radius", field);
      Py_DECREF(field);
      if (rc) {
        return NULL;
      }
    }
  }
  {  // frame_id
    PyObject * field = NULL;
    field = PyUnicode_DecodeUTF8(
      ros_message->frame_id.data,
      strlen(ros_message->frame_id.data),
      "replace");
    if (!field) {
      return NULL;
    }
    {
      int rc = PyObject_SetAttrString(_pymessage, "frame_id", field);
      Py_DECREF(field);
      if (rc) {
        return NULL;
      }
    }
  }
  {  // timeout
    PyObject * field = NULL;
    field = PyFloat_FromDouble(ros_message->timeout);
    {
      int rc = PyObject_SetAttrString(_pymessage, "timeout", field);
      Py_DECREF(field);
      if (rc) {
        return NULL;
      }
    }
  }
  {  // is_reached
    PyObject * field = NULL;
    field = PyBool_FromLong(ros_message->is_reached ? 1 : 0);
    {
      int rc = PyObject_SetAttrString(_pymessage, "is_reached", field);
      Py_DECREF(field);
      if (rc) {
        return NULL;
      }
    }
  }
  {  // stamp
    PyObject * field = NULL;
    field = builtin_interfaces__msg__time__convert_to_py(&ros_message->stamp);
    if (!field) {
      return NULL;
    }
    {
      int rc = PyObject_SetAttrString(_pymessage, "stamp", field);
      Py_DECREF(field);
      if (rc) {
        return NULL;
      }
    }
  }

  // ownership of _pymessage is transferred to the caller
  return _pymessage;
}
