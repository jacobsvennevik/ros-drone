# generated from rosidl_generator_py/resource/_idl.py.em
# with input from hippocampus_ros2_msgs:msg/MissionGoal.idl
# generated code does not contain a copyright notice


# Import statements for member types

import builtins  # noqa: E402, I100

import math  # noqa: E402, I100

import rosidl_parser.definition  # noqa: E402, I100


class Metaclass_MissionGoal(type):
    """Metaclass of message 'MissionGoal'."""

    _CREATE_ROS_MESSAGE = None
    _CONVERT_FROM_PY = None
    _CONVERT_TO_PY = None
    _DESTROY_ROS_MESSAGE = None
    _TYPE_SUPPORT = None

    __constants = {
        'GOAL_TYPE_POINT': 0,
        'GOAL_TYPE_NODE': 1,
        'GOAL_TYPE_REGION': 2,
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
                'hippocampus_ros2_msgs.msg.MissionGoal')
            logger.debug(
                'Failed to import needed modules for type support:\n' +
                traceback.format_exc())
        else:
            cls._CREATE_ROS_MESSAGE = module.create_ros_message_msg__msg__mission_goal
            cls._CONVERT_FROM_PY = module.convert_from_py_msg__msg__mission_goal
            cls._CONVERT_TO_PY = module.convert_to_py_msg__msg__mission_goal
            cls._TYPE_SUPPORT = module.type_support_msg__msg__mission_goal
            cls._DESTROY_ROS_MESSAGE = module.destroy_ros_message_msg__msg__mission_goal

            from builtin_interfaces.msg import Time
            if Time.__class__._TYPE_SUPPORT is None:
                Time.__class__.__import_type_support__()

            from geometry_msgs.msg import Point
            if Point.__class__._TYPE_SUPPORT is None:
                Point.__class__.__import_type_support__()

    @classmethod
    def __prepare__(cls, name, bases, **kwargs):
        # list constant names here so that they appear in the help text of
        # the message class under "Data and other attributes defined here:"
        # as well as populate each message instance
        return {
            'GOAL_TYPE_POINT': cls.__constants['GOAL_TYPE_POINT'],
            'GOAL_TYPE_NODE': cls.__constants['GOAL_TYPE_NODE'],
            'GOAL_TYPE_REGION': cls.__constants['GOAL_TYPE_REGION'],
        }

    @property
    def GOAL_TYPE_POINT(self):
        """Message constant 'GOAL_TYPE_POINT'."""
        return Metaclass_MissionGoal.__constants['GOAL_TYPE_POINT']

    @property
    def GOAL_TYPE_NODE(self):
        """Message constant 'GOAL_TYPE_NODE'."""
        return Metaclass_MissionGoal.__constants['GOAL_TYPE_NODE']

    @property
    def GOAL_TYPE_REGION(self):
        """Message constant 'GOAL_TYPE_REGION'."""
        return Metaclass_MissionGoal.__constants['GOAL_TYPE_REGION']


class MissionGoal(metaclass=Metaclass_MissionGoal):
    """
    Message class 'MissionGoal'.

    Constants:
      GOAL_TYPE_POINT
      GOAL_TYPE_NODE
      GOAL_TYPE_REGION
    """

    __slots__ = [
        '_goal_type',
        '_point_position',
        '_point_tolerance',
        '_node_id',
        '_region_center',
        '_region_radius',
        '_frame_id',
        '_timeout',
        '_is_reached',
        '_stamp',
    ]

    _fields_and_field_types = {
        'goal_type': 'uint8',
        'point_position': 'geometry_msgs/Point',
        'point_tolerance': 'double',
        'node_id': 'int32',
        'region_center': 'geometry_msgs/Point',
        'region_radius': 'double',
        'frame_id': 'string',
        'timeout': 'double',
        'is_reached': 'boolean',
        'stamp': 'builtin_interfaces/Time',
    }

    SLOT_TYPES = (
        rosidl_parser.definition.BasicType('uint8'),  # noqa: E501
        rosidl_parser.definition.NamespacedType(['geometry_msgs', 'msg'], 'Point'),  # noqa: E501
        rosidl_parser.definition.BasicType('double'),  # noqa: E501
        rosidl_parser.definition.BasicType('int32'),  # noqa: E501
        rosidl_parser.definition.NamespacedType(['geometry_msgs', 'msg'], 'Point'),  # noqa: E501
        rosidl_parser.definition.BasicType('double'),  # noqa: E501
        rosidl_parser.definition.UnboundedString(),  # noqa: E501
        rosidl_parser.definition.BasicType('double'),  # noqa: E501
        rosidl_parser.definition.BasicType('boolean'),  # noqa: E501
        rosidl_parser.definition.NamespacedType(['builtin_interfaces', 'msg'], 'Time'),  # noqa: E501
    )

    def __init__(self, **kwargs):
        assert all('_' + key in self.__slots__ for key in kwargs.keys()), \
            'Invalid arguments passed to constructor: %s' % \
            ', '.join(sorted(k for k in kwargs.keys() if '_' + k not in self.__slots__))
        self.goal_type = kwargs.get('goal_type', int())
        from geometry_msgs.msg import Point
        self.point_position = kwargs.get('point_position', Point())
        self.point_tolerance = kwargs.get('point_tolerance', float())
        self.node_id = kwargs.get('node_id', int())
        from geometry_msgs.msg import Point
        self.region_center = kwargs.get('region_center', Point())
        self.region_radius = kwargs.get('region_radius', float())
        self.frame_id = kwargs.get('frame_id', str())
        self.timeout = kwargs.get('timeout', float())
        self.is_reached = kwargs.get('is_reached', bool())
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
        if self.goal_type != other.goal_type:
            return False
        if self.point_position != other.point_position:
            return False
        if self.point_tolerance != other.point_tolerance:
            return False
        if self.node_id != other.node_id:
            return False
        if self.region_center != other.region_center:
            return False
        if self.region_radius != other.region_radius:
            return False
        if self.frame_id != other.frame_id:
            return False
        if self.timeout != other.timeout:
            return False
        if self.is_reached != other.is_reached:
            return False
        if self.stamp != other.stamp:
            return False
        return True

    @classmethod
    def get_fields_and_field_types(cls):
        from copy import copy
        return copy(cls._fields_and_field_types)

    @builtins.property
    def goal_type(self):
        """Message field 'goal_type'."""
        return self._goal_type

    @goal_type.setter
    def goal_type(self, value):
        if __debug__:
            assert \
                isinstance(value, int), \
                "The 'goal_type' field must be of type 'int'"
            assert value >= 0 and value < 256, \
                "The 'goal_type' field must be an unsigned integer in [0, 255]"
        self._goal_type = value

    @builtins.property
    def point_position(self):
        """Message field 'point_position'."""
        return self._point_position

    @point_position.setter
    def point_position(self, value):
        if __debug__:
            from geometry_msgs.msg import Point
            assert \
                isinstance(value, Point), \
                "The 'point_position' field must be a sub message of type 'Point'"
        self._point_position = value

    @builtins.property
    def point_tolerance(self):
        """Message field 'point_tolerance'."""
        return self._point_tolerance

    @point_tolerance.setter
    def point_tolerance(self, value):
        if __debug__:
            assert \
                isinstance(value, float), \
                "The 'point_tolerance' field must be of type 'float'"
            assert not (value < -1.7976931348623157e+308 or value > 1.7976931348623157e+308) or math.isinf(value), \
                "The 'point_tolerance' field must be a double in [-1.7976931348623157e+308, 1.7976931348623157e+308]"
        self._point_tolerance = value

    @builtins.property
    def node_id(self):
        """Message field 'node_id'."""
        return self._node_id

    @node_id.setter
    def node_id(self, value):
        if __debug__:
            assert \
                isinstance(value, int), \
                "The 'node_id' field must be of type 'int'"
            assert value >= -2147483648 and value < 2147483648, \
                "The 'node_id' field must be an integer in [-2147483648, 2147483647]"
        self._node_id = value

    @builtins.property
    def region_center(self):
        """Message field 'region_center'."""
        return self._region_center

    @region_center.setter
    def region_center(self, value):
        if __debug__:
            from geometry_msgs.msg import Point
            assert \
                isinstance(value, Point), \
                "The 'region_center' field must be a sub message of type 'Point'"
        self._region_center = value

    @builtins.property
    def region_radius(self):
        """Message field 'region_radius'."""
        return self._region_radius

    @region_radius.setter
    def region_radius(self, value):
        if __debug__:
            assert \
                isinstance(value, float), \
                "The 'region_radius' field must be of type 'float'"
            assert not (value < -1.7976931348623157e+308 or value > 1.7976931348623157e+308) or math.isinf(value), \
                "The 'region_radius' field must be a double in [-1.7976931348623157e+308, 1.7976931348623157e+308]"
        self._region_radius = value

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
    def timeout(self):
        """Message field 'timeout'."""
        return self._timeout

    @timeout.setter
    def timeout(self, value):
        if __debug__:
            assert \
                isinstance(value, float), \
                "The 'timeout' field must be of type 'float'"
            assert not (value < -1.7976931348623157e+308 or value > 1.7976931348623157e+308) or math.isinf(value), \
                "The 'timeout' field must be a double in [-1.7976931348623157e+308, 1.7976931348623157e+308]"
        self._timeout = value

    @builtins.property
    def is_reached(self):
        """Message field 'is_reached'."""
        return self._is_reached

    @is_reached.setter
    def is_reached(self, value):
        if __debug__:
            assert \
                isinstance(value, bool), \
                "The 'is_reached' field must be of type 'bool'"
        self._is_reached = value

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
