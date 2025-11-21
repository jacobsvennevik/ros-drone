// generated from rosidl_generator_cpp/resource/idl__traits.hpp.em
// with input from hippocampus_ros2_msgs:msg/GraphEdge.idl
// generated code does not contain a copyright notice

#ifndef HIPPOCAMPUS_ROS2_MSGS__MSG__DETAIL__GRAPH_EDGE__TRAITS_HPP_
#define HIPPOCAMPUS_ROS2_MSGS__MSG__DETAIL__GRAPH_EDGE__TRAITS_HPP_

#include <stdint.h>

#include <sstream>
#include <string>
#include <type_traits>

#include "hippocampus_ros2_msgs/msg/detail/graph_edge__struct.hpp"
#include "rosidl_runtime_cpp/traits.hpp"

namespace hippocampus_ros2_msgs
{

namespace msg
{

inline void to_flow_style_yaml(
  const GraphEdge & msg,
  std::ostream & out)
{
  out << "{";
  // member: u
  {
    out << "u: ";
    rosidl_generator_traits::value_to_yaml(msg.u, out);
    out << ", ";
  }

  // member: v
  {
    out << "v: ";
    rosidl_generator_traits::value_to_yaml(msg.v, out);
    out << ", ";
  }

  // member: length
  {
    out << "length: ";
    rosidl_generator_traits::value_to_yaml(msg.length, out);
    out << ", ";
  }

  // member: traversable
  {
    out << "traversable: ";
    rosidl_generator_traits::value_to_yaml(msg.traversable, out);
  }
  out << "}";
}  // NOLINT(readability/fn_size)

inline void to_block_style_yaml(
  const GraphEdge & msg,
  std::ostream & out, size_t indentation = 0)
{
  // member: u
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "u: ";
    rosidl_generator_traits::value_to_yaml(msg.u, out);
    out << "\n";
  }

  // member: v
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "v: ";
    rosidl_generator_traits::value_to_yaml(msg.v, out);
    out << "\n";
  }

  // member: length
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "length: ";
    rosidl_generator_traits::value_to_yaml(msg.length, out);
    out << "\n";
  }

  // member: traversable
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "traversable: ";
    rosidl_generator_traits::value_to_yaml(msg.traversable, out);
    out << "\n";
  }
}  // NOLINT(readability/fn_size)

inline std::string to_yaml(const GraphEdge & msg, bool use_flow_style = false)
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
  const hippocampus_ros2_msgs::msg::GraphEdge & msg,
  std::ostream & out, size_t indentation = 0)
{
  hippocampus_ros2_msgs::msg::to_block_style_yaml(msg, out, indentation);
}

[[deprecated("use hippocampus_ros2_msgs::msg::to_yaml() instead")]]
inline std::string to_yaml(const hippocampus_ros2_msgs::msg::GraphEdge & msg)
{
  return hippocampus_ros2_msgs::msg::to_yaml(msg);
}

template<>
inline const char * data_type<hippocampus_ros2_msgs::msg::GraphEdge>()
{
  return "hippocampus_ros2_msgs::msg::GraphEdge";
}

template<>
inline const char * name<hippocampus_ros2_msgs::msg::GraphEdge>()
{
  return "hippocampus_ros2_msgs/msg/GraphEdge";
}

template<>
struct has_fixed_size<hippocampus_ros2_msgs::msg::GraphEdge>
  : std::integral_constant<bool, true> {};

template<>
struct has_bounded_size<hippocampus_ros2_msgs::msg::GraphEdge>
  : std::integral_constant<bool, true> {};

template<>
struct is_message<hippocampus_ros2_msgs::msg::GraphEdge>
  : std::true_type {};

}  // namespace rosidl_generator_traits

#endif  // HIPPOCAMPUS_ROS2_MSGS__MSG__DETAIL__GRAPH_EDGE__TRAITS_HPP_
