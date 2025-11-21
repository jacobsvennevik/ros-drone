# generated from rosidl_generator_py/resource/_idl.py.em
# with input from hippocampus_ros2_msgs:msg/GraphSnapshot.idl
# generated code does not contain a copyright notice


# Import statements for member types

import builtins  # noqa: E402, I100

import math  # noqa: E402, I100

import rosidl_parser.definition  # noqa: E402, I100


class Metaclass_GraphSnapshot(type):
    """Metaclass of message 'GraphSnapshot'."""

    _CREATE_ROS_MESSAGE = None
    _CONVERT_FROM_PY = None
    _CONVERT_TO_PY = None
    _DESTROY_ROS_MESSAGE = None
    _TYPE_SUPPORT = None

    __constants = {
    }

    @classmethod
    def __import_type_support__(cls):
        try:
            from rosidl_generator_py import import_type_support
            module = import_type_support('hippocampus_ros2_msgs')
        except ImportError:
            import logging
            import traceback
            logger = logging.getLogger(
                'hippocampus_ros2_msgs.msg.GraphSnapshot')
            logger.debug(
                'Failed to import needed modules for type support:\n' +
                traceback.format_exc())
        else:
            cls._CREATE_ROS_MESSAGE = module.create_ros_message_msg__msg__graph_snapshot
            cls._CONVERT_FROM_PY = module.convert_from_py_msg__msg__graph_snapshot
            cls._CONVERT_TO_PY = module.convert_to_py_msg__msg__graph_snapshot
            cls._TYPE_SUPPORT = module.type_support_msg__msg__graph_snapshot
            cls._DESTROY_ROS_MESSAGE = module.destroy_ros_message_msg__msg__graph_snapshot

            from builtin_interfaces.msg import Time
            if Time.__class__._TYPE_SUPPORT is None:
                Time.__class__.__import_type_support__()

            from hippocampus_ros2_msgs.msg import GraphEdge
            if GraphEdge.__class__._TYPE_SUPPORT is None:
                GraphEdge.__class__.__import_type_support__()

            from hippocampus_ros2_msgs.msg import GraphNode
            if GraphNode.__class__._TYPE_SUPPORT is None:
                GraphNode.__class__.__import_type_support__()

    @classmethod
    def __prepare__(cls, name, bases, **kwargs):
        # list constant names here so that they appear in the help text of
        # the message class under "Data and other attributes defined here:"
        # as well as populate each message instance
        return {
        }


class GraphSnapshot(metaclass=Metaclass_GraphSnapshot):
    """Message class 'GraphSnapshot'."""

    __slots__ = [
        '_epoch_id',
        '_frame_id',
        '_stamp',
        '_last_updated',
        '_update_rate',
        '_staleness_warning',
        '_nodes',
        '_edges',
    ]

    _fields_and_field_types = {
        'epoch_id': 'uint32',
        'frame_id': 'string',
        'stamp': 'builtin_interfaces/Time',
        'last_updated': 'builtin_interfaces/Time',
        'update_rate': 'double',
        'staleness_warning': 'boolean',
        'nodes': 'sequence<hippocampus_ros2_msgs/GraphNode>',
        'edges': 'sequence<hippocampus_ros2_msgs/GraphEdge>',
    }

    SLOT_TYPES = (
        rosidl_parser.definition.BasicType('uint32'),  # noqa: E501
        rosidl_parser.definition.UnboundedString(),  # noqa: E501
        rosidl_parser.definition.NamespacedType(['builtin_interfaces', 'msg'], 'Time'),  # noqa: E501
        rosidl_parser.definition.NamespacedType(['builtin_interfaces', 'msg'], 'Time'),  # noqa: E501
        rosidl_parser.definition.BasicType('double'),  # noqa: E501
        rosidl_parser.definition.BasicType('boolean'),  # noqa: E501
        rosidl_parser.definition.UnboundedSequence(rosidl_parser.definition.NamespacedType(['hippocampus_ros2_msgs', 'msg'], 'GraphNode')),  # noqa: E501
        rosidl_parser.definition.UnboundedSequence(rosidl_parser.definition.NamespacedType(['hippocampus_ros2_msgs', 'msg'], 'GraphEdge')),  # noqa: E501
    )

    def __init__(self, **kwargs):
        assert all('_' + key in self.__slots__ for key in kwargs.keys()), \
            'Invalid arguments passed to constructor: %s' % \
            ', '.join(sorted(k for k in kwargs.keys() if '_' + k not in self.__slots__))
        self.epoch_id = kwargs.get('epoch_id', int())
        self.frame_id = kwargs.get('frame_id', str())
        from builtin_interfaces.msg import Time
        self.stamp = kwargs.get('stamp', Time())
        from builtin_interfaces.msg import Time
        self.last_updated = kwargs.get('last_updated', Time())
        self.update_rate = kwargs.get('update_rate', float())
        self.staleness_warning = kwargs.get('staleness_warning', bool())
        self.nodes = kwargs.get('nodes', [])
        self.edges = kwargs.get('edges', [])

    def __repr__(self):
        typename = self.__class__.__module__.split('.')
        typename.pop()
        typename.append(self.__class__.__name__)
        args = []
        for s, t in zip(self.__slots__, self.SLOT_TYPES):
            field = getattr(self, s)
            fieldstr = repr(field)
            # We use Python array type for fields that can be directly stored
            # in them, and "normal" sequences for everything else.  If it is
            # a type that we store in an array, strip off the 'array' portion.
            if (
                isinstance(t, rosidl_parser.definition.AbstractSequence) and
                isinstance(t.value_type, rosidl_parser.definition.BasicType) and
                t.value_type.typename in ['float', 'double', 'int8', 'uint8', 'int16', 'uint16', 'int32', 'uint32', 'int64', 'uint64']
            ):
                if len(field) == 0:
                    fieldstr = '[]'
                else:
                    assert fieldstr.startswith('array(')
                    prefix = "array('X', "
                    suffix = ')'
                    fieldstr = fieldstr[len(prefix):-len(suffix)]
            args.append(s[1:] + '=' + fieldstr)
        return '%s(%s)' % ('.'.join(typename), ', '.join(args))

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return False
        if self.epoch_id != other.epoch_id:
            return False
        if self.frame_id != other.frame_id:
            return False
        if self.stamp != other.stamp:
            return False
        if self.last_updated != other.last_updated:
            return False
        if self.update_rate != other.update_rate:
            return False
        if self.staleness_warning != other.staleness_warning:
            return False
        if self.nodes != other.nodes:
            return False
        if self.edges != other.edges:
            return False
        return True

    @classmethod
    def get_fields_and_field_types(cls):
        from copy import copy
        return copy(cls._fields_and_field_types)

    @builtins.property
    def epoch_id(self):
        """Message field 'epoch_id'."""
        return self._epoch_id

    @epoch_id.setter
    def epoch_id(self, value):
        if __debug__:
            assert \
                isinstance(value, int), \
                "The 'epoch_id' field must be of type 'int'"
            assert value >= 0 and value < 4294967296, \
                "The 'epoch_id' field must be an unsigned integer in [0, 4294967295]"
        self._epoch_id = value

    @builtins.property
    def frame_id(self):
        """Message field 'frame_id'."""
        return self._frame_id

    @frame_id.setter
    def frame_id(self, value):
        if __debug__:
            assert \
                isinstance(value, str), \
                "The 'frame_id' field must be of type 'str'"
        self._frame_id = value

    @builtins.property
    def stamp(self):
        """Message field 'stamp'."""
        return self._stamp

    @stamp.setter
    def stamp(self, value):
        if __debug__:
            from builtin_interfaces.msg import Time
            assert \
                isinstance(value, Time), \
                "The 'stamp' field must be a sub message of type 'Time'"
        self._stamp = value

    @builtins.property
    def last_updated(self):
        """Message field 'last_updated'."""
        return self._last_updated

    @last_updated.setter
    def last_updated(self, value):
        if __debug__:
            from builtin_interfaces.msg import Time
            assert \
                isinstance(value, Time), \
                "The 'last_updated' field must be a sub message of type 'Time'"
        self._last_updated = value

    @builtins.property
    def update_rate(self):
        """Message field 'update_rate'."""
        return self._update_rate

    @update_rate.setter
    def update_rate(self, value):
        if __debug__:
            assert \
                isinstance(value, float), \
                "The 'update_rate' field must be of type 'float'"
            assert not (value < -1.7976931348623157e+308 or value > 1.7976931348623157e+308) or math.isinf(value), \
                "The 'update_rate' field must be a double in [-1.7976931348623157e+308, 1.7976931348623157e+308]"
        self._update_rate = value

    @builtins.property
    def staleness_warning(self):
        """Message field 'staleness_warning'."""
        return self._staleness_warning

    @staleness_warning.setter
    def staleness_warning(self, value):
        if __debug__:
            assert \
                isinstance(value, bool), \
                "The 'staleness_warning' field must be of type 'bool'"
        self._staleness_warning = value

    @builtins.property
    def nodes(self):
        """Message field 'nodes'."""
        return self._nodes

    @nodes.setter
    def nodes(self, value):
        if __debug__:
            from hippocampus_ros2_msgs.msg import GraphNode
            from collections.abc import Sequence
            from collections.abc import Set
            from collections import UserList
            from collections import UserString
            assert \
                ((isinstance(value, Sequence) or
                  isinstance(value, Set) or
                  isinstance(value, UserList)) and
                 not isinstance(value, str) and
                 not isinstance(value, UserString) and
                 all(isinstance(v, GraphNode) for v in value) and
                 True), \
                "The 'nodes' field must be a set or sequence and each value of type 'GraphNode'"
        self._nodes = value

    @builtins.property
    def edges(self):
        """Message field 'edges'."""
        return self._edges

    @edges.setter
    def edges(self, value):
        if __debug__:
            from hippocampus_ros2_msgs.msg import GraphEdge
            from collections.abc import Sequence
            from collections.abc import Set
            from collections import UserList
            from collections import UserString
            assert \
                ((isinstance(value, Sequence) or
                  isinstance(value, Set) or
                  isinstance(value, UserList)) and
                 not isinstance(value, str) and
                 not isinstance(value, UserString) and
                 all(isinstance(v, GraphEdge) for v in value) and
                 True), \
                "The 'edges' field must be a set or sequence and each value of type 'GraphEdge'"
        self._edges = value
