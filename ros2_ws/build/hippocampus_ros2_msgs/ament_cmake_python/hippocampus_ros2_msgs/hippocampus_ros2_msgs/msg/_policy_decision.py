# generated from rosidl_generator_py/resource/_idl.py.em
# with input from hippocampus_ros2_msgs:msg/PolicyDecision.idl
# generated code does not contain a copyright notice


# Import statements for member types

import builtins  # noqa: E402, I100

import math  # noqa: E402, I100

import rosidl_parser.definition  # noqa: E402, I100


class Metaclass_PolicyDecision(type):
    """Metaclass of message 'PolicyDecision'."""

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
                'hippocampus_ros2_msgs.msg.PolicyDecision')
            logger.debug(
                'Failed to import needed modules for type support:\n' +
                traceback.format_exc())
        else:
            cls._CREATE_ROS_MESSAGE = module.create_ros_message_msg__msg__policy_decision
            cls._CONVERT_FROM_PY = module.convert_from_py_msg__msg__policy_decision
            cls._CONVERT_TO_PY = module.convert_to_py_msg__msg__policy_decision
            cls._TYPE_SUPPORT = module.type_support_msg__msg__policy_decision
            cls._DESTROY_ROS_MESSAGE = module.destroy_ros_message_msg__msg__policy_decision

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


class PolicyDecision(metaclass=Metaclass_PolicyDecision):
    """Message class 'PolicyDecision'."""

    __slots__ = [
        '_linear_x',
        '_angular_z',
        '_linear_z',
        '_confidence',
        '_reason',
        '_next_waypoint',
        '_stamp',
    ]

    _fields_and_field_types = {
        'linear_x': 'double',
        'angular_z': 'double',
        'linear_z': 'double',
        'confidence': 'double',
        'reason': 'string',
        'next_waypoint': 'int32',
        'stamp': 'builtin_interfaces/Time',
    }

    SLOT_TYPES = (
        rosidl_parser.definition.BasicType('double'),  # noqa: E501
        rosidl_parser.definition.BasicType('double'),  # noqa: E501
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
        self.linear_x = kwargs.get('linear_x', float())
        self.angular_z = kwargs.get('angular_z', float())
        self.linear_z = kwargs.get('linear_z', float())
        self.confidence = kwargs.get('confidence', float())
        self.reason = kwargs.get('reason', str())
        self.next_waypoint = kwargs.get('next_waypoint', int())
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
        if self.linear_x != other.linear_x:
            return False
        if self.angular_z != other.angular_z:
            return False
        if self.linear_z != other.linear_z:
            return False
        if self.confidence != other.confidence:
            return False
        if self.reason != other.reason:
            return False
        if self.next_waypoint != other.next_waypoint:
            return False
        if self.stamp != other.stamp:
            return False
        return True

    @classmethod
    def get_fields_and_field_types(cls):
        from copy import copy
        return copy(cls._fields_and_field_types)

    @builtins.property
    def linear_x(self):
        """Message field 'linear_x'."""
        return self._linear_x

    @linear_x.setter
    def linear_x(self, value):
        if __debug__:
            assert \
                isinstance(value, float), \
                "The 'linear_x' field must be of type 'float'"
            assert not (value < -1.7976931348623157e+308 or value > 1.7976931348623157e+308) or math.isinf(value), \
                "The 'linear_x' field must be a double in [-1.7976931348623157e+308, 1.7976931348623157e+308]"
        self._linear_x = value

    @builtins.property
    def angular_z(self):
        """Message field 'angular_z'."""
        return self._angular_z

    @angular_z.setter
    def angular_z(self, value):
        if __debug__:
            assert \
                isinstance(value, float), \
                "The 'angular_z' field must be of type 'float'"
            assert not (value < -1.7976931348623157e+308 or value > 1.7976931348623157e+308) or math.isinf(value), \
                "The 'angular_z' field must be a double in [-1.7976931348623157e+308, 1.7976931348623157e+308]"
        self._angular_z = value

    @builtins.property
    def linear_z(self):
        """Message field 'linear_z'."""
        return self._linear_z

    @linear_z.setter
    def linear_z(self, value):
        if __debug__:
            assert \
                isinstance(value, float), \
                "The 'linear_z' field must be of type 'float'"
            assert not (value < -1.7976931348623157e+308 or value > 1.7976931348623157e+308) or math.isinf(value), \
                "The 'linear_z' field must be a double in [-1.7976931348623157e+308, 1.7976931348623157e+308]"
        self._linear_z = value

    @builtins.property
    def confidence(self):
        """Message field 'confidence'."""
        return self._confidence

    @confidence.setter
    def confidence(self, value):
        if __debug__:
            assert \
                isinstance(value, float), \
                "The 'confidence' field must be of type 'float'"
            assert not (value < -1.7976931348623157e+308 or value > 1.7976931348623157e+308) or math.isinf(value), \
                "The 'confidence' field must be a double in [-1.7976931348623157e+308, 1.7976931348623157e+308]"
        self._confidence = value

    @builtins.property
    def reason(self):
        """Message field 'reason'."""
        return self._reason

    @reason.setter
    def reason(self, value):
        if __debug__:
            assert \
                isinstance(value, str), \
                "The 'reason' field must be of type 'str'"
        self._reason = value

    @builtins.property
    def next_waypoint(self):
        """Message field 'next_waypoint'."""
        return self._next_waypoint

    @next_waypoint.setter
    def next_waypoint(self, value):
        if __debug__:
            assert \
                isinstance(value, int), \
                "The 'next_waypoint' field must be of type 'int'"
            assert value >= -2147483648 and value < 2147483648, \
                "The 'next_waypoint' field must be an integer in [-2147483648, 2147483647]"
        self._next_waypoint = value

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
