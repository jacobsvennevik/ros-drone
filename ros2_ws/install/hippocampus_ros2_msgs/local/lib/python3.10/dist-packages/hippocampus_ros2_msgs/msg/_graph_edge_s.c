// generated from rosidl_generator_py/resource/_idl_support.c.em
// with input from hippocampus_ros2_msgs:msg/GraphEdge.idl
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
#include "hippocampus_ros2_msgs/msg/detail/graph_edge__struct.h"
#include "hippocampus_ros2_msgs/msg/detail/graph_edge__functions.h"


ROSIDL_GENERATOR_C_EXPORT
bool hippocampus_ros2_msgs__msg__graph_edge__convert_from_py(PyObject * _pymsg, void * _ros_message)
{
  // check that the passed message is of the expected Python class
  {
    char full_classname_dest[48];
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
    assert(strncmp("hippocampus_ros2_msgs.msg._graph_edge.GraphEdge", full_classname_dest, 47) == 0);
  }
  hippocampus_ros2_msgs__msg__GraphEdge * ros_message = _ros_message;
  {  // u
    PyObject * field = PyObject_GetAttrString(_pymsg, "u");
    if (!field) {
      return false;
    }
    assert(PyLong_Check(field));
    ros_message->u = (int32_t)PyLong_AsLong(field);
    Py_DECREF(field);
  }
  {  // v
    PyObject * field = PyObject_GetAttrString(_pymsg, "v");
    if (!field) {
      return false;
    }
    assert(PyLong_Check(field));
    ros_message->v = (int32_t)PyLong_AsLong(field);
    Py_DECREF(field);
  }
  {  // length
    PyObject * field = PyObject_GetAttrString(_pymsg, "length");
    if (!field) {
      return false;
    }
    assert(PyFloat_Check(field));
    ros_message->length = PyFloat_AS_DOUBLE(field);
    Py_DECREF(field);
  }
  {  // traversable
    PyObject * field = PyObject_GetAttrString(_pymsg, "traversable");
    if (!field) {
      return false;
    }
    assert(PyBool_Check(field));
    ros_message->traversable = (Py_True == field);
    Py_DECREF(field);
  }

  return true;
}

ROSIDL_GENERATOR_C_EXPORT
PyObject * hippocampus_ros2_msgs__msg__graph_edge__convert_to_py(void * raw_ros_message)
{
  /* NOTE(esteve): Call constructor of GraphEdge */
  PyObject * _pymessage = NULL;
  {
    PyObject * pymessage_module = PyImport_ImportModule("hippocampus_ros2_msgs.msg._graph_edge");
    assert(pymessage_module);
    PyObject * pymessage_class = PyObject_GetAttrString(pymessage_module, "GraphEdge");
    assert(pymessage_class);
    Py_DECREF(pymessage_module);
    _pymessage = PyObject_CallObject(pymessage_class, NULL);
    Py_DECREF(pymessage_class);
    if (!_pymessage) {
      return NULL;
    }
  }
  hippocampus_ros2_msgs__msg__GraphEdge * ros_message = (hippocampus_ros2_msgs__msg__GraphEdge *)raw_ros_message;
  {  // u
    PyObject * field = NULL;
    field = PyLong_FromLong(ros_message->u);
    {
      int rc = PyObject_SetAttrString(_pymessage, "u", field);
      Py_DECREF(field);
      if (rc) {
        return NULL;
      }
    }
  }
  {  // v
    PyObject * field = NULL;
    field = PyLong_FromLong(ros_message->v);
    {
      int rc = PyObject_SetAttrString(_pymessage, "v", field);
      Py_DECREF(field);
      if (rc) {
        return NULL;
      }
    }
  }
  {  // length
    PyObject * field = NULL;
    field = PyFloat_FromDouble(ros_message->length);
    {
      int rc = PyObject_SetAttrString(_pymessage, "length", field);
      Py_DECREF(field);
      if (rc) {
        return NULL;
      }
    }
  }
  {  // traversable
    PyObject * field = NULL;
    field = PyBool_FromLong(ros_message->traversable ? 1 : 0);
    {
      int rc = PyObject_SetAttrString(_pymessage, "traversable", field);
      Py_DECREF(field);
      if (rc) {
        return NULL;
      }
    }
  }

  // ownership of _pymessage is transferred to the caller
  return _pymessage;
}
