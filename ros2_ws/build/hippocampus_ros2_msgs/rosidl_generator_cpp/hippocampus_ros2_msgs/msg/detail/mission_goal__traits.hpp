// generated from rosidl_generator_cpp/resource/idl__traits.hpp.em
// with input from hippocampus_ros2_msgs:msg/MissionGoal.idl
// generated code does not contain a copyright notice

#ifndef HIPPOCAMPUS_ROS2_MSGS__MSG__DETAIL__MISSION_GOAL__TRAITS_HPP_
#define HIPPOCAMPUS_ROS2_MSGS__MSG__DETAIL__MISSION_GOAL__TRAITS_HPP_

#include <stdint.h>

#include <sstream>
#include <string>
#include <type_traits>

#include "hippocampus_ros2_msgs/msg/detail/mission_goal__struct.hpp"
#include "rosidl_runtime_cpp/traits.hpp"

// Include directives for member types
// Member 'point_position'
// Member 'region_center'
#include "geometry_msgs/msg/detail/point__traits.hpp"
// Member 'stamp'
#include "builtin_interfaces/msg/detail/time__traits.hpp"

namespace hippocampus_ros2_msgs
{

namespace msg
{

inline void to_flow_style_yaml(
  const MissionGoal & msg,
  std::ostream & out)
{
  out << "{";
  // member: goal_type
  {
    out << "goal_type: ";
    rosidl_generator_traits::value_to_yaml(msg.goal_type, out);
    out << ", ";
  }

  // member: point_position
  {
    out << "point_position: ";
    to_flow_style_yaml(msg.point_position, out);
    out << ", ";
  }

  // member: point_tolerance
  {
    out << "point_tolerance: ";
    rosidl_generator_traits::value_to_yaml(msg.point_tolerance, out);
    out << ", ";
  }

  // member: node_id
  {
    out << "node_id: ";
    rosidl_generator_traits::value_to_yaml(msg.node_id, out);
    out << ", ";
  }

  // member: region_center
  {
    out << "region_center: ";
    to_flow_style_yaml(msg.region_center, out);
    out << ", ";
  }

  // member: region_radius
  {
    out << "region_radius: ";
    rosidl_generator_traits::value_to_yaml(msg.region_radius, out);
    out << ", ";
  }

  // member: frame_id
  {
    out << "frame_id: ";
    rosidl_generator_traits::value_to_yaml(msg.frame_id, out);
    out << ", ";
  }

  // member: timeout
  {
    out << "timeout: ";
    rosidl_generator_traits::value_to_yaml(msg.timeout, out);
    out << ", ";
  }

  // member: is_reached
  {
    out << "is_reached: ";
    rosidl_generator_traits::value_to_yaml(msg.is_reached, out);
    out << ", ";
  }

  // member: stamp
  {
    out << "stamp: ";
    to_flow_style_yaml(msg.stamp, out);
  }
  out << "}";
}  // NOLINT(readability/fn_size)

inline void to_block_style_yaml(
  const MissionGoal & msg,
  std::ostream & out, size_t indentation = 0)
{
  // member: goal_type
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "goal_type: ";
    rosidl_generator_traits::value_to_yaml(msg.goal_type, out);
    out << "\n";
  }

  // member: point_position
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "point_position:\n";
    to_block_style_yaml(msg.point_position, out, indentation + 2);
  }

  // member: point_tolerance
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "point_tolerance: ";
    rosidl_generator_traits::value_to_yaml(msg.point_tolerance, out);
    out << "\n";
  }

  // member: node_id
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "node_id: ";
    rosidl_generator_traits::value_to_yaml(msg.node_id, out);
    out << "\n";
  }

  // member: region_center
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "region_center:\n";
    to_block_style_yaml(msg.region_center, out, indentation + 2);
  }

  // member: region_radius
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "region_radius: ";
    rosidl_generator_traits::value_to_yaml(msg.region_radius, out);
    out << "\n";
  }

  // member: frame_id
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "frame_id: ";
    rosidl_generator_traits::value_to_yaml(msg.frame_id, out);
    out << "\n";
  }

  // member: timeout
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "timeout: ";
    rosidl_generator_traits::value_to_yaml(msg.timeout, out);
    out << "\n";
  }

  // member: is_reached
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "is_reached: ";
    rosidl_generator_traits::value_to_yaml(msg.is_reached, out);
    out << "\n";
  }

  // member: stamp
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "stamp:\n";
    to_block_style_yaml(msg.stamp, out, indentation + 2);
  }
}  // NOLINT(readability/fn_size)

inline std::string to_yaml(const MissionGoal & msg, bool use_flow_style = false)
{
  std::ostringstream out;
  if (use_flow_style) {
    to_flow_style_yaml(msg, out);
  } else {
    to_block_style_yaml(msg, out);
  }
  return out.str();
}

}  // namespace msg

}  // namespace hippocampus_ros2_msgs

namespace rosidl_generator_traits
{

[[deprecated("use hippocampus_ros2_msgs::msg::to_block_style_yaml() instead")]]
inline void to_yaml(
  const hippocampus_ros2_msgs::msg::MissionGoal & msg,
  std::ostream & out, size_t indentation = 0)
{
  hippocampus_ros2_msgs::msg::to_block_style_yaml(msg, out, indentation);
}

[[deprecated("use hippocampus_ros2_msgs::msg::to_yaml() instead")]]
inline std::string to_yaml(const hippocampus_ros2_msgs::msg::MissionGoal & msg)
{
  return hippocampus_ros2_msgs::msg::to_yaml(msg);
}

template<>
inline const char * data_type<hippocampus_ros2_msgs::msg::MissionGoal>()
{
  return "hippocampus_ros2_msgs::msg::MissionGoal";
}

template<>
inline const char * name<hippocampus_ros2_msgs::msg::MissionGoal>()
{
  return "hippocampus_ros2_msgs/msg/MissionGoal";
}

template<>
struct has_fixed_size<hippocampus_ros2_msgs::msg::MissionGoal>
  : std::integral_constant<bool, false> {};

template<>
struct has_bounded_size<hippocampus_ros2_msgs::msg::MissionGoal>
  : std::integral_constant<bool, false> {};

template<>
struct is_message<hippocampus_ros2_msgs::msg::MissionGoal>
  : std::true_type {};

}  // namespace rosidl_generator_traits

#endif  // HIPPOCAMPUS_ROS2_MSGS__MSG__DETAIL__MISSION_GOAL__TRAITS_HPP_
