// generated from rosidl_generator_py/resource/_idl_support.c.em
// with input from hippocampus_ros2_msgs:msg/GraphSnapshot.idl
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
#include "hippocampus_ros2_msgs/msg/detail/graph_snapshot__struct.h"
#include "hippocampus_ros2_msgs/msg/detail/graph_snapshot__functions.h"

#include "rosidl_runtime_c/string.h"
#include "rosidl_runtime_c/string_functions.h"

#include "rosidl_runtime_c/primitives_sequence.h"
#include "rosidl_runtime_c/primitives_sequence_functions.h"

// Nested array functions includes
#include "hippocampus_ros2_msgs/msg/detail/graph_edge__functions.h"
#include "hippocampus_ros2_msgs/msg/detail/graph_node__functions.h"
// end nested array functions include
ROSIDL_GENERATOR_C_IMPORT
bool builtin_interfaces__msg__time__convert_from_py(PyObject * _pymsg, void * _ros_message);
ROSIDL_GENERATOR_C_IMPORT
PyObject * builtin_interfaces__msg__time__convert_to_py(void * raw_ros_message);
ROSIDL_GENERATOR_C_IMPORT
bool builtin_interfaces__msg__time__convert_from_py(PyObject * _pymsg, void * _ros_message);
ROSIDL_GENERATOR_C_IMPORT
PyObject * builtin_interfaces__msg__time__convert_to_py(void * raw_ros_message);
bool hippocampus_ros2_msgs__msg__graph_node__convert_from_py(PyObject * _pymsg, void * _ros_message);
PyObject * hippocampus_ros2_msgs__msg__graph_node__convert_to_py(void * raw_ros_message);
bool hippocampus_ros2_msgs__msg__graph_edge__convert_from_py(PyObject * _pymsg, void * _ros_message);
PyObject * hippocampus_ros2_msgs__msg__graph_edge__convert_to_py(void * raw_ros_message);

ROSIDL_GENERATOR_C_EXPORT
bool hippocampus_ros2_msgs__msg__graph_snapshot__convert_from_py(PyObject * _pymsg, void * _ros_message)
{
  // check that the passed message is of the expected Python class
  {
    char full_classname_dest[56];
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
    assert(strncmp("hippocampus_ros2_msgs.msg._graph_snapshot.GraphSnapshot", full_classname_dest, 55) == 0);
  }
  hippocampus_ros2_msgs__msg__GraphSnapshot * ros_message = _ros_message;
  {  // epoch_id
    PyObject * field = PyObject_GetAttrString(_pymsg, "epoch_id");
    if (!field) {
      return false;
    }
    assert(PyLong_Check(field));
    ros_message->epoch_id = PyLong_AsUnsignedLong(field);
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
  {  // last_updated
    PyObject * field = PyObject_GetAttrString(_pymsg, "last_updated");
    if (!field) {
      return false;
    }
    if (!builtin_interfaces__msg__time__convert_from_py(field, &ros_message->last_updated)) {
      Py_DECREF(field);
      return false;
    }
    Py_DECREF(field);
  }
  {  // update_rate
    PyObject * field = PyObject_GetAttrString(_pymsg, "update_rate");
    if (!field) {
      return false;
    }
    assert(PyFloat_Check(field));
    ros_message->update_rate = PyFloat_AS_DOUBLE(field);
    Py_DECREF(field);
  }
  {  // staleness_warning
    PyObject * field = PyObject_GetAttrString(_pymsg, "staleness_warning");
    if (!field) {
      return false;
    }
    assert(PyBool_Check(field));
    ros_message->staleness_warning = (Py_True == field);
    Py_DECREF(field);
  }
  {  // nodes
    PyObject * field = PyObject_GetAttrString(_pymsg, "nodes");
    if (!field) {
      return false;
    }
    PyObject * seq_field = PySequence_Fast(field, "expected a sequence in 'nodes'");
    if (!seq_field) {
      Py_DECREF(field);
      return false;
    }
    Py_ssize_t size = PySequence_Size(field);
    if (-1 == size) {
      Py_DECREF(seq_field);
      Py_DECREF(field);
      return false;
    }
    if (!hippocampus_ros2_msgs__msg__GraphNode__Sequence__init(&(ros_message->nodes), size)) {
      PyErr_SetString(PyExc_RuntimeError, "unable to create hippocampus_ros2_msgs__msg__GraphNode__Sequence ros_message");
      Py_DECREF(seq_field);
      Py_DECREF(field);
      return false;
    }
    hippocampus_ros2_msgs__msg__GraphNode * dest = ros_message->nodes.data;
    for (Py_ssize_t i = 0; i < size; ++i) {
      if (!hippocampus_ros2_msgs__msg__graph_node__convert_from_py(PySequence_Fast_GET_ITEM(seq_field, i), &dest[i])) {
        Py_DECREF(seq_field);
        Py_DECREF(field);
        return false;
      }
    }
    Py_DECREF(seq_field);
    Py_DECREF(field);
  }
  {  // edges
    PyObject * field = PyObject_GetAttrString(_pymsg, "edges");
    if (!field) {
      return false;
    }
    PyObject * seq_field = PySequence_Fast(field, "expected a sequence in 'edges'");
    if (!seq_field) {
      Py_DECREF(field);
      return false;
    }
    Py_ssize_t size = PySequence_Size(field);
    if (-1 == size) {
      Py_DECREF(seq_field);
      Py_DECREF(field);
      return false;
    }
    if (!hippocampus_ros2_msgs__msg__GraphEdge__Sequence__init(&(ros_message->edges), size)) {
      PyErr_SetString(PyExc_RuntimeError, "unable to create hippocampus_ros2_msgs__msg__GraphEdge__Sequence ros_message");
      Py_DECREF(seq_field);
      Py_DECREF(field);
      return false;
    }
    hippocampus_ros2_msgs__msg__GraphEdge * dest = ros_message->edges.data;
    for (Py_ssize_t i = 0; i < size; ++i) {
      if (!hippocampus_ros2_msgs__msg__graph_edge__convert_from_py(PySequence_Fast_GET_ITEM(seq_field, i), &dest[i])) {
        Py_DECREF(seq_field);
        Py_DECREF(field);
        return false;
      }
    }
    Py_DECREF(seq_field);
    Py_DECREF(field);
  }

  return true;
}

ROSIDL_GENERATOR_C_EXPORT
PyObject * hippocampus_ros2_msgs__msg__graph_snapshot__convert_to_py(void * raw_ros_message)
{
  /* NOTE(esteve): Call constructor of GraphSnapshot */
  PyObject * _pymessage = NULL;
  {
    PyObject * pymessage_module = PyImport_ImportModule("hippocampus_ros2_msgs.msg._graph_snapshot");
    assert(pymessage_module);
    PyObject * pymessage_class = PyObject_GetAttrString(pymessage_module, "GraphSnapshot");
    assert(pymessage_class);
    Py_DECREF(pymessage_module);
    _pymessage = PyObject_CallObject(pymessage_class, NULL);
    Py_DECREF(pymessage_class);
    if (!_pymessage) {
      return NULL;
    }
  }
  hippocampus_ros2_msgs__msg__GraphSnapshot * ros_message = (hippocampus_ros2_msgs__msg__GraphSnapshot *)raw_ros_message;
  {  // epoch_id
    PyObject * field = NULL;
    field = PyLong_FromUnsignedLong(ros_message->epoch_id);
    {
      int rc = PyObject_SetAttrString(_pymessage, "epoch_id", field);
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
  {  // last_updated
    PyObject * field = NULL;
    field = builtin_interfaces__msg__time__convert_to_py(&ros_message->last_updated);
    if (!field) {
      return NULL;
    }
    {
      int rc = PyObject_SetAttrString(_pymessage, "last_updated", field);
      Py_DECREF(field);
      if (rc) {
        return NULL;
      }
    }
  }
  {  // update_rate
    PyObject * field = NULL;
    field = PyFloat_FromDouble(ros_message->update_rate);
    {
      int rc = PyObject_SetAttrString(_pymessage, "update_rate", field);
      Py_DECREF(field);
      if (rc) {
        return NULL;
      }
    }
  }
  {  // staleness_warning
    PyObject * field = NULL;
    field = PyBool_FromLong(ros_message->staleness_warning ? 1 : 0);
    {
      int rc = PyObject_SetAttrString(_pymessage, "staleness_warning", field);
      Py_DECREF(field);
      if (rc) {
        return NULL;
      }
    }
  }
  {  // nodes
    PyObject * field = NULL;
    size_t size = ros_message->nodes.size;
    field = PyList_New(size);
    if (!field) {
      return NULL;
    }
    hippocampus_ros2_msgs__msg__GraphNode * item;
    for (size_t i = 0; i < size; ++i) {
      item = &(ros_message->nodes.data[i]);
      PyObject * pyitem = hippocampus_ros2_msgs__msg__graph_node__convert_to_py(item);
      if (!pyitem) {
        Py_DECREF(field);
        return NULL;
      }
      int rc = PyList_SetItem(field, i, pyitem);
      (void)rc;
      assert(rc == 0);
    }
    assert(PySequence_Check(field));
    {
      int rc = PyObject_SetAttrString(_pymessage, "nodes", field);
      Py_DECREF(field);
      if (rc) {
        return NULL;
      }
    }
  }
  {  // edges
    PyObject * field = NULL;
    size_t size = ros_message->edges.size;
    field = PyList_New(size);
    if (!field) {
      return NULL;
    }
    hippocampus_ros2_msgs__msg__GraphEdge * item;
    for (size_t i = 0; i < size; ++i) {
      item = &(ros_message->edges.data[i]);
      PyObject * pyitem = hippocampus_ros2_msgs__msg__graph_edge__convert_to_py(item);
      if (!pyitem) {
        Py_DECREF(field);
        return NULL;
      }
      int rc = PyList_SetItem(field, i, pyitem);
      (void)rc;
      assert(rc == 0);
    }
    assert(PySequence_Check(field));
    {
      int rc = PyObject_SetAttrString(_pymessage, "edges", field);
      Py_DECREF(field);
      if (rc) {
        return NULL;
      }
    }
  }

  // ownership of _pymessage is transferred to the caller
  return _pymessage;
}
