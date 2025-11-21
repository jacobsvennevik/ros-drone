// generated from rosidl_generator_py/resource/_idl_support.c.em
// with input from hippocampus_ros2_msgs:msg/PolicyStatus.idl
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
#include "hippocampus_ros2_msgs/msg/detail/policy_status__struct.h"
#include "hippocampus_ros2_msgs/msg/detail/policy_status__functions.h"

#include "rosidl_runtime_c/string.h"
#include "rosidl_runtime_c/string_functions.h"

ROSIDL_GENERATOR_C_IMPORT
bool builtin_interfaces__msg__time__convert_from_py(PyObject * _pymsg, void * _ros_message);
ROSIDL_GENERATOR_C_IMPORT
PyObject * builtin_interfaces__msg__time__convert_to_py(void * raw_ros_message);

ROSIDL_GENERATOR_C_EXPORT
bool hippocampus_ros2_msgs__msg__policy_status__convert_from_py(PyObject * _pymsg, void * _ros_message)
{
  // check that the passed message is of the expected Python class
  {
    char full_classname_dest[54];
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
    assert(strncmp("hippocampus_ros2_msgs.msg._policy_status.PolicyStatus", full_classname_dest, 53) == 0);
  }
  hippocampus_ros2_msgs__msg__PolicyStatus * ros_message = _ros_message;
  {  // is_active
    PyObject * field = PyObject_GetAttrString(_pymsg, "is_active");
    if (!field) {
      return false;
    }
    assert(PyBool_Check(field));
    ros_message->is_active = (Py_True == field);
    Py_DECREF(field);
  }
  {  // graph_stale
    PyObject * field = PyObject_GetAttrString(_pymsg, "graph_stale");
    if (!field) {
      return false;
    }
    assert(PyBool_Check(field));
    ros_message->graph_stale = (Py_True == field);
    Py_DECREF(field);
  }
  {  // using_snn
    PyObject * field = PyObject_GetAttrString(_pymsg, "using_snn");
    if (!field) {
      return false;
    }
    assert(PyBool_Check(field));
    ros_message->using_snn = (Py_True == field);
    Py_DECREF(field);
  }
  {  // hierarchical_enabled
    PyObject * field = PyObject_GetAttrString(_pymsg, "hierarchical_enabled");
    if (!field) {
      return false;
    }
    assert(PyBool_Check(field));
    ros_message->hierarchical_enabled = (Py_True == field);
    Py_DECREF(field);
  }
  {  // feature_compute_time_ms
    PyObject * field = PyObject_GetAttrString(_pymsg, "feature_compute_time_ms");
    if (!field) {
      return false;
    }
    assert(PyFloat_Check(field));
    ros_message->feature_compute_time_ms = PyFloat_AS_DOUBLE(field);
    Py_DECREF(field);
  }
  {  // policy_decision_time_ms
    PyObject * field = PyObject_GetAttrString(_pymsg, "policy_decision_time_ms");
    if (!field) {
      return false;
    }
    assert(PyFloat_Check(field));
    ros_message->policy_decision_time_ms = PyFloat_AS_DOUBLE(field);
    Py_DECREF(field);
  }
  {  // safety_filter_time_ms
    PyObject * field = PyObject_GetAttrString(_pymsg, "safety_filter_time_ms");
    if (!field) {
      return false;
    }
    assert(PyFloat_Check(field));
    ros_message->safety_filter_time_ms = PyFloat_AS_DOUBLE(field);
    Py_DECREF(field);
  }
  {  // total_latency_ms
    PyObject * field = PyObject_GetAttrString(_pymsg, "total_latency_ms");
    if (!field) {
      return false;
    }
    assert(PyFloat_Check(field));
    ros_message->total_latency_ms = PyFloat_AS_DOUBLE(field);
    Py_DECREF(field);
  }
  {  // graph_nodes
    PyObject * field = PyObject_GetAttrString(_pymsg, "graph_nodes");
    if (!field) {
      return false;
    }
    assert(PyLong_Check(field));
    ros_message->graph_nodes = PyLong_AsUnsignedLong(field);
    Py_DECREF(field);
  }
  {  // graph_edges
    PyObject * field = PyObject_GetAttrString(_pymsg, "graph_edges");
    if (!field) {
      return false;
    }
    assert(PyLong_Check(field));
    ros_message->graph_edges = PyLong_AsUnsignedLong(field);
    Py_DECREF(field);
  }
  {  // graph_staleness_s
    PyObject * field = PyObject_GetAttrString(_pymsg, "graph_staleness_s");
    if (!field) {
      return false;
    }
    assert(PyFloat_Check(field));
    ros_message->graph_staleness_s = PyFloat_AS_DOUBLE(field);
    Py_DECREF(field);
  }
  {  // current_confidence
    PyObject * field = PyObject_GetAttrString(_pymsg, "current_confidence");
    if (!field) {
      return false;
    }
    assert(PyFloat_Check(field));
    ros_message->current_confidence = PyFloat_AS_DOUBLE(field);
    Py_DECREF(field);
  }
  {  // current_reason
    PyObject * field = PyObject_GetAttrString(_pymsg, "current_reason");
    if (!field) {
      return false;
    }
    assert(PyUnicode_Check(field));
    PyObject * encoded_field = PyUnicode_AsUTF8String(field);
    if (!encoded_field) {
      Py_DECREF(field);
      return false;
    }
    rosidl_runtime_c__String__assign(&ros_message->current_reason, PyBytes_AS_STRING(encoded_field));
    Py_DECREF(encoded_field);
    Py_DECREF(field);
  }
  {  // current_waypoint
    PyObject * field = PyObject_GetAttrString(_pymsg, "current_waypoint");
    if (!field) {
      return false;
    }
    assert(PyLong_Check(field));
    ros_message->current_waypoint = (int32_t)PyLong_AsLong(field);
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
PyObject * hippocampus_ros2_msgs__msg__policy_status__convert_to_py(void * raw_ros_message)
{
  /* NOTE(esteve): Call constructor of PolicyStatus */
  PyObject * _pymessage = NULL;
  {
    PyObject * pymessage_module = PyImport_ImportModule("hippocampus_ros2_msgs.msg._policy_status");
    assert(pymessage_module);
    PyObject * pymessage_class = PyObject_GetAttrString(pymessage_module, "PolicyStatus");
    assert(pymessage_class);
    Py_DECREF(pymessage_module);
    _pymessage = PyObject_CallObject(pymessage_class, NULL);
    Py_DECREF(pymessage_class);
    if (!_pymessage) {
      return NULL;
    }
  }
  hippocampus_ros2_msgs__msg__PolicyStatus * ros_message = (hippocampus_ros2_msgs__msg__PolicyStatus *)raw_ros_message;
  {  // is_active
    PyObject * field = NULL;
    field = PyBool_FromLong(ros_message->is_active ? 1 : 0);
    {
      int rc = PyObject_SetAttrString(_pymessage, "is_active", field);
      Py_DECREF(field);
      if (rc) {
        return NULL;
      }
    }
  }
  {  // graph_stale
    PyObject * field = NULL;
    field = PyBool_FromLong(ros_message->graph_stale ? 1 : 0);
    {
      int rc = PyObject_SetAttrString(_pymessage, "graph_stale", field);
      Py_DECREF(field);
      if (rc) {
        return NULL;
      }
    }
  }
  {  // using_snn
    PyObject * field = NULL;
    field = PyBool_FromLong(ros_message->using_snn ? 1 : 0);
    {
      int rc = PyObject_SetAttrString(_pymessage, "using_snn", field);
      Py_DECREF(field);
      if (rc) {
        return NULL;
      }
    }
  }
  {  // hierarchical_enabled
    PyObject * field = NULL;
    field = PyBool_FromLong(ros_message->hierarchical_enabled ? 1 : 0);
    {
      int rc = PyObject_SetAttrString(_pymessage, "hierarchical_enabled", field);
      Py_DECREF(field);
      if (rc) {
        return NULL;
      }
    }
  }
  {  // feature_compute_time_ms
    PyObject * field = NULL;
    field = PyFloat_FromDouble(ros_message->feature_compute_time_ms);
    {
      int rc = PyObject_SetAttrString(_pymessage, "feature_compute_time_ms", field);
      Py_DECREF(field);
      if (rc) {
        return NULL;
      }
    }
  }
  {  // policy_decision_time_ms
    PyObject * field = NULL;
    field = PyFloat_FromDouble(ros_message->policy_decision_time_ms);
    {
      int rc = PyObject_SetAttrString(_pymessage, "policy_decision_time_ms", field);
      Py_DECREF(field);
      if (rc) {
        return NULL;
      }
    }
  }
  {  // safety_filter_time_ms
    PyObject * field = NULL;
    field = PyFloat_FromDouble(ros_message->safety_filter_time_ms);
    {
      int rc = PyObject_SetAttrString(_pymessage, "safety_filter_time_ms", field);
      Py_DECREF(field);
      if (rc) {
        return NULL;
      }
    }
  }
  {  // total_latency_ms
    PyObject * field = NULL;
    field = PyFloat_FromDouble(ros_message->total_latency_ms);
    {
      int rc = PyObject_SetAttrString(_pymessage, "total_latency_ms", field);
      Py_DECREF(field);
      if (rc) {
        return NULL;
      }
    }
  }
  {  // graph_nodes
    PyObject * field = NULL;
    field = PyLong_FromUnsignedLong(ros_message->graph_nodes);
    {
      int rc = PyObject_SetAttrString(_pymessage, "graph_nodes", field);
      Py_DECREF(field);
      if (rc) {
        return NULL;
      }
    }
  }
  {  // graph_edges
    PyObject * field = NULL;
    field = PyLong_FromUnsignedLong(ros_message->graph_edges);
    {
      int rc = PyObject_SetAttrString(_pymessage, "graph_edges", field);
      Py_DECREF(field);
      if (rc) {
        return NULL;
      }
    }
  }
  {  // graph_staleness_s
    PyObject * field = NULL;
    field = PyFloat_FromDouble(ros_message->graph_staleness_s);
    {
      int rc = PyObject_SetAttrString(_pymessage, "graph_staleness_s", field);
      Py_DECREF(field);
      if (rc) {
        return NULL;
      }
    }
  }
  {  // current_confidence
    PyObject * field = NULL;
    field = PyFloat_FromDouble(ros_message->current_confidence);
    {
      int rc = PyObject_SetAttrString(_pymessage, "current_confidence", field);
      Py_DECREF(field);
      if (rc) {
        return NULL;
      }
    }
  }
  {  // current_reason
    PyObject * field = NULL;
    field = PyUnicode_DecodeUTF8(
      ros_message->current_reason.data,
      strlen(ros_message->current_reason.data),
      "replace");
    if (!field) {
      return NULL;
    }
    {
      int rc = PyObject_SetAttrString(_pymessage, "current_reason", field);
      Py_DECREF(field);
      if (rc) {
        return NULL;
      }
    }
  }
  {  // current_waypoint
    PyObject * field = NULL;
    field = PyLong_FromLong(ros_message->current_waypoint);
    {
      int rc = PyObject_SetAttrString(_pymessage, "current_waypoint", field);
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
