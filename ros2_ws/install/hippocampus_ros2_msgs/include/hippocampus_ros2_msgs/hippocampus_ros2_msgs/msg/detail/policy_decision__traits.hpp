// generated from rosidl_generator_cpp/resource/idl__traits.hpp.em
// with input from hippocampus_ros2_msgs:msg/PolicyDecision.idl
// generated code does not contain a copyright notice

#ifndef HIPPOCAMPUS_ROS2_MSGS__MSG__DETAIL__POLICY_DECISION__TRAITS_HPP_
#define HIPPOCAMPUS_ROS2_MSGS__MSG__DETAIL__POLICY_DECISION__TRAITS_HPP_

#include <stdint.h>

#include <sstream>
#include <string>
#include <type_traits>

#include "hippocampus_ros2_msgs/msg/detail/policy_decision__struct.hpp"
#include "rosidl_runtime_cpp/traits.hpp"

// Include directives for member types
// Member 'stamp'
#include "builtin_interfaces/msg/detail/time__traits.hpp"

namespace hippocampus_ros2_msgs
{

namespace msg
{

inline void to_flow_style_yaml(
  const PolicyDecision & msg,
  std::ostream & out)
{
  out << "{";
  // member: linear_x
  {
    out << "linear_x: ";
    rosidl_generator_traits::value_to_yaml(msg.linear_x, out);
    out << ", ";
  }

  // member: angular_z
  {
    out << "angular_z: ";
    rosidl_generator_traits::value_to_yaml(msg.angular_z, out);
    out << ", ";
  }

  // member: linear_z
  {
    out << "linear_z: ";
    rosidl_generator_traits::value_to_yaml(msg.linear_z, out);
    out << ", ";
  }

  // member: confidence
  {
    out << "confidence: ";
    rosidl_generator_traits::value_to_yaml(msg.confidence, out);
    out << ", ";
  }

  // member: reason
  {
    out << "reason: ";
    rosidl_generator_traits::value_to_yaml(msg.reason, out);
    out << ", ";
  }

  // member: next_waypoint
  {
    out << "next_waypoint: ";
    rosidl_generator_traits::value_to_yaml(msg.next_waypoint, out);
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
  const PolicyDecision & msg,
  std::ostream & out, size_t indentation = 0)
{
  // member: linear_x
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "linear_x: ";
    rosidl_generator_traits::value_to_yaml(msg.linear_x, out);
    out << "\n";
  }

  // member: angular_z
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "angular_z: ";
    rosidl_generator_traits::value_to_yaml(msg.angular_z, out);
    out << "\n";
  }

  // member: linear_z
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "linear_z: ";
    rosidl_generator_traits::value_to_yaml(msg.linear_z, out);
    out << "\n";
  }

  // member: confidence
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "confidence: ";
    rosidl_generator_traits::value_to_yaml(msg.confidence, out);
    out << "\n";
  }

  // member: reason
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "reason: ";
    rosidl_generator_traits::value_to_yaml(msg.reason, out);
    out << "\n";
  }

  // member: next_waypoint
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "next_waypoint: ";
    rosidl_generator_traits::value_to_yaml(msg.next_waypoint, out);
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

inline std::string to_yaml(const PolicyDecision & msg, bool use_flow_style = false)
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
  const hippocampus_ros2_msgs::msg::PolicyDecision & msg,
  std::ostream & out, size_t indentation = 0)
{
  hippocampus_ros2_msgs::msg::to_block_style_yaml(msg, out, indentation);
}

[[deprecated("use hippocampus_ros2_msgs::msg::to_yaml() instead")]]
inline std::string to_yaml(const hippocampus_ros2_msgs::msg::PolicyDecision & msg)
{
  return hippocampus_ros2_msgs::msg::to_yaml(msg);
}

template<>
inline const char * data_type<hippocampus_ros2_msgs::msg::PolicyDecision>()
{
  return "hippocampus_ros2_msgs::msg::PolicyDecision";
}

template<>
inline const char * name<hippocampus_ros2_msgs::msg::PolicyDecision>()
{
  return "hippocampus_ros2_msgs/msg/PolicyDecision";
}

template<>
struct has_fixed_size<hippocampus_ros2_msgs::msg::PolicyDecision>
  : std::integral_constant<bool, false> {};

template<>
struct has_bounded_size<hippocampus_ros2_msgs::msg::PolicyDecision>
  : std::integral_constant<bool, false> {};

template<>
struct is_message<hippocampus_ros2_msgs::msg::PolicyDecision>
  : std::true_type {};

}  // namespace rosidl_generator_traits

#endif  // HIPPOCAMPUS_ROS2_MSGS__MSG__DETAIL__POLICY_DECISION__TRAITS_HPP_
