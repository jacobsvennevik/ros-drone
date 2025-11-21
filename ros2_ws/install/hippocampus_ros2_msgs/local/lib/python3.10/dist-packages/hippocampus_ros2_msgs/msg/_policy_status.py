# generated from rosidl_generator_py/resource/_idl.py.em
# with input from hippocampus_ros2_msgs:msg/PolicyStatus.idl
# generated code does not contain a copyright notice


# Import statements for member types

import builtins  # noqa: E402, I100

import math  # noqa: E402, I100

import rosidl_parser.definition  # noqa: E402, I100


class Metaclass_PolicyStatus(type):
    """Metaclass of message 'PolicyStatus'."""

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
                'hippocampus_ros2_msgs.msg.PolicyStatus')
            logger.debug(
                'Failed to import needed modules for type support:\n' +
                traceback.format_exc())
        else:
            cls._CREATE_ROS_MESSAGE = module.create_ros_message_msg__msg__policy_status
            cls._CONVERT_FROM_PY = module.convert_from_py_msg__msg__policy_status
            cls._CONVERT_TO_PY = module.convert_to_py_msg__msg__policy_status
            cls._TYPE_SUPPORT = module.type_support_msg__msg__policy_status
            cls._DESTROY_ROS_MESSAGE = module.destroy_ros_message_msg__msg__policy_status

            from builtin_interfaces.msg import Time
            if Time.__class__._TYPE_SUPPORT is None:
                Time.__class__.__import_type_support__()

    @classmethod
    def __prepare__(cls, name, bases, **kwargs):
        # list constant names here so that they appear in the help text of
        # the message class under "Data and other attributes defined here:"
        # as well as populate each message instance
        return {
        }


class PolicyStatus(metaclass=Metaclass_PolicyStatus):
    """Message class 'PolicyStatus'."""

    __slots__ = [
        '_is_active',
        '_graph_stale',
        '_using_snn',
        '_hierarchical_enabled',
        '_feature_compute_time_ms',
        '_policy_decision_time_ms',
        '_safety_filter_time_ms',
        '_total_latency_ms',
        '_graph_nodes',
        '_graph_edges',
        '_graph_staleness_s',
        '_current_confidence',
        '_current_reason',
        '_current_waypoint',
        '_stamp',
    ]

    _fields_and_field_types = {
        'is_active': 'boolean',
        'graph_stale': 'boolean',
        'using_snn': 'boolean',
        'hierarchical_enabled': 'boolean',
        'feature_compute_time_ms': 'double',
        'policy_decision_time_ms': 'double',
        'safety_filter_time_ms': 'double',
        'total_latency_ms': 'double',
        'graph_nodes': 'uint32',
        'graph_edges': 'uint32',
        'graph_staleness_s': 'double',
        'current_confidence': 'double',
        'current_reason': 'string',
        'current_waypoint': 'int32',
        'stamp': 'builtin_interfaces/Time',
    }

    SLOT_TYPES = (
        rosidl_parser.definition.BasicType('boolean'),  # noqa: E501
        rosidl_parser.definition.BasicType('boolean'),  # noqa: E501
        rosidl_parser.definition.BasicType('boolean'),  # noqa: E501
        rosidl_parser.definition.BasicType('boolean'),  # noqa: E501
        rosidl_parser.definition.BasicType('double'),  # noqa: E501
        rosidl_parser.definition.BasicType('double'),  # noqa: E501
        rosidl_parser.definition.BasicType('double'),  # noqa: E501
        rosidl_parser.definition.BasicType('double'),  # noqa: E501
        rosidl_parser.definition.BasicType('uint32'),  # noqa: E501
        rosidl_parser.definition.BasicType('uint32'),  # noqa: E501
        rosidl_parser.definition.BasicType('double'),  # noqa: E501
        rosidl_parser.definition.BasicType('double'),  # noqa: E501
        rosidl_parser.definition.UnboundedString(),  # noqa: E501
        rosidl_parser.definition.BasicType('int32'),  # noqa: E501
        rosidl_parser.definition.NamespacedType(['builtin_interfaces', 'msg'], 'Time'),  # noqa: E501
    )

    def __init__(self, **kwargs):
        assert all('_' + key in self.__slots__ for key in kwargs.keys()), \
            'Invalid arguments passed to constructor: %s' % \
            ', '.join(sorted(k for k in kwargs.keys() if '_' + k not in self.__slots__))
        self.is_active = kwargs.get('is_active', bool())
        self.graph_stale = kwargs.get('graph_stale', bool())
        self.using_snn = kwargs.get('using_snn', bool())
        self.hierarchical_enabled = kwargs.get('hierarchical_enabled', bool())
        self.feature_compute_time_ms = kwargs.get('feature_compute_time_ms', float())
        self.policy_decision_time_ms = kwargs.get('policy_decision_time_ms', float())
        self.safety_filter_time_ms = kwargs.get('safety_filter_time_ms', float())
        self.total_latency_ms = kwargs.get('total_latency_ms', float())
        self.graph_nodes = kwargs.get('graph_nodes', int())
        self.graph_edges = kwargs.get('graph_edges', int())
        self.graph_staleness_s = kwargs.get('graph_staleness_s', float())
        self.current_confidence = kwargs.get('current_confidence', float())
        self.current_reason = kwargs.get('current_reason', str())
        self.current_waypoint = kwargs.get('current_waypoint', int())
        from builtin_interfaces.msg import Time
        self.stamp = kwargs.get('stamp', Time())

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
        if self.is_active != other.is_active:
            return False
        if self.graph_stale != other.graph_stale:
            return False
        if self.using_snn != other.using_snn:
            return False
        if self.hierarchical_enabled != other.hierarchical_enabled:
            return False
        if self.feature_compute_time_ms != other.feature_compute_time_ms:
            return False
        if self.policy_decision_time_ms != other.policy_decision_time_ms:
            return False
        if self.safety_filter_time_ms != other.safety_filter_time_ms:
            return False
        if self.total_latency_ms != other.total_latency_ms:
            return False
        if self.graph_nodes != other.graph_nodes:
            return False
        if self.graph_edges != other.graph_edges:
            return False
        if self.graph_staleness_s != other.graph_staleness_s:
            return False
        if self.current_confidence != other.current_confidence:
            return False
        if self.current_reason != other.current_reason:
            return False
        if self.current_waypoint != other.current_waypoint:
            return False
        if self.stamp != other.stamp:
            return False
        return True

    @classmethod
    def get_fields_and_field_types(cls):
        from copy import copy
        return copy(cls._fields_and_field_types)

    @builtins.property
    def is_active(self):
        """Message field 'is_active'."""
        return self._is_active

    @is_active.setter
    def is_active(self, value):
        if __debug__:
            assert \
                isinstance(value, bool), \
                "The 'is_active' field must be of type 'bool'"
        self._is_active = value

    @builtins.property
    def graph_stale(self):
        """Message field 'graph_stale'."""
        return self._graph_stale

    @graph_stale.setter
    def graph_stale(self, value):
        if __debug__:
            assert \
                isinstance(value, bool), \
                "The 'graph_stale' field must be of type 'bool'"
        self._graph_stale = value

    @builtins.property
    def using_snn(self):
        """Message field 'using_snn'."""
        return self._using_snn

    @using_snn.setter
    def using_snn(self, value):
        if __debug__:
            assert \
                isinstance(value, bool), \
                "The 'using_snn' field must be of type 'bool'"
        self._using_snn = value

    @builtins.property
    def hierarchical_enabled(self):
        """Message field 'hierarchical_enabled'."""
        return self._hierarchical_enabled

    @hierarchical_enabled.setter
    def hierarchical_enabled(self, value):
        if __debug__:
            assert \
                isinstance(value, bool), \
                "The 'hierarchical_enabled' field must be of type 'bool'"
        self._hierarchical_enabled = value

    @builtins.property
    def feature_compute_time_ms(self):
        """Message field 'feature_compute_time_ms'."""
        return self._feature_compute_time_ms

    @feature_compute_time_ms.setter
    def feature_compute_time_ms(self, value):
        if __debug__:
            assert \
                isinstance(value, float), \
                "The 'feature_compute_time_ms' field must be of type 'float'"
            assert not (value < -1.7976931348623157e+308 or value > 1.7976931348623157e+308) or math.isinf(value), \
                "The 'feature_compute_time_ms' field must be a double in [-1.7976931348623157e+308, 1.7976931348623157e+308]"
        self._feature_compute_time_ms = value

    @builtins.property
    def policy_decision_time_ms(self):
        """Message field 'policy_decision_time_ms'."""
        return self._policy_decision_time_ms

    @policy_decision_time_ms.setter
    def policy_decision_time_ms(self, value):
        if __debug__:
            assert \
                isinstance(value, float), \
                "The 'policy_decision_time_ms' field must be of type 'float'"
            assert not (value < -1.7976931348623157e+308 or value > 1.7976931348623157e+308) or math.isinf(value), \
                "The 'policy_decision_time_ms' field must be a double in [-1.7976931348623157e+308, 1.7976931348623157e+308]"
        self._policy_decision_time_ms = value

    @builtins.property
    def safety_filter_time_ms(self):
        """Message field 'safety_filter_time_ms'."""
        return self._safety_filter_time_ms

    @safety_filter_time_ms.setter
    def safety_filter_time_ms(self, value):
        if __debug__:
            assert \
                isinstance(value, float), \
                "The 'safety_filter_time_ms' field must be of type 'float'"
            assert not (value < -1.7976931348623157e+308 or value > 1.7976931348623157e+308) or math.isinf(value), \
                "The 'safety_filter_time_ms' field must be a double in [-1.7976931348623157e+308, 1.7976931348623157e+308]"
        self._safety_filter_time_ms = value

    @builtins.property
    def total_latency_ms(self):
        """Message field 'total_latency_ms'."""
        return self._total_latency_ms

    @total_latency_ms.setter
    def total_latency_ms(self, value):
        if __debug__:
            assert \
                isinstance(value, float), \
                "The 'total_latency_ms' field must be of type 'float'"
            assert not (value < -1.7976931348623157e+308 or value > 1.7976931348623157e+308) or math.isinf(value), \
                "The 'total_latency_ms' field must be a double in [-1.7976931348623157e+308, 1.7976931348623157e+308]"
        self._total_latency_ms = value

    @builtins.property
    def graph_nodes(self):
        """Message field 'graph_nodes'."""
        return self._graph_nodes

    @graph_nodes.setter
    def graph_nodes(self, value):
        if __debug__:
            assert \
                isinstance(value, int), \
                "The 'graph_nodes' field must be of type 'int'"
            assert value >= 0 and value < 4294967296, \
                "The 'graph_nodes' field must be an unsigned integer in [0, 4294967295]"
        self._graph_nodes = value

    @builtins.property
    def graph_edges(self):
        """Message field 'graph_edges'."""
        return self._graph_edges

    @graph_edges.setter
    def graph_edges(self, value):
        if __debug__:
            assert \
                isinstance(value, int), \
                "The 'graph_edges' field must be of type 'int'"
            assert value >= 0 and value < 4294967296, \
                "The 'graph_edges' field must be an unsigned integer in [0, 4294967295]"
        self._graph_edges = value

    @builtins.property
    def graph_staleness_s(self):
        """Message field 'graph_staleness_s'."""
        return self._graph_staleness_s

    @graph_staleness_s.setter
    def graph_staleness_s(self, value):
        if __debug__:
            assert \
                isinstance(value, float), \
                "The 'graph_staleness_s' field must be of type 'float'"
            assert not (value < -1.7976931348623157e+308 or value > 1.7976931348623157e+308) or math.isinf(value), \
                "The 'graph_staleness_s' field must be a double in [-1.7976931348623157e+308, 1.7976931348623157e+308]"
        self._graph_staleness_s = value

    @builtins.property
    def current_confidence(self):
        """Message field 'current_confidence'."""
        return self._current_confidence

    @current_confidence.setter
    def current_confidence(self, value):
        if __debug__:
            assert \
                isinstance(value, float), \
                "The 'current_confidence' field must be of type 'float'"
            assert not (value < -1.7976931348623157e+308 or value > 1.7976931348623157e+308) or math.isinf(value), \
                "The 'current_confidence' field must be a double in [-1.7976931348623157e+308, 1.7976931348623157e+308]"
        self._current_confidence = value

    @builtins.property
    def current_reason(self):
        """Message field 'current_reason'."""
        return self._current_reason

    @current_reason.setter
    def current_reason(self, value):
        if __debug__:
            assert \
                isinstance(value, str), \
                "The 'current_reason' field must be of type 'str'"
        self._current_reason = value

    @builtins.property
    def current_waypoint(self):
        """Message field 'current_waypoint'."""
        return self._current_waypoint

    @current_waypoint.setter
    def current_waypoint(self, value):
        if __debug__:
            assert \
                isinstance(value, int), \
                "The 'current_waypoint' field must be of type 'int'"
            assert value >= -2147483648 and value < 2147483648, \
                "The 'current_waypoint' field must be an integer in [-2147483648, 2147483647]"
        self._current_waypoint = value

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
