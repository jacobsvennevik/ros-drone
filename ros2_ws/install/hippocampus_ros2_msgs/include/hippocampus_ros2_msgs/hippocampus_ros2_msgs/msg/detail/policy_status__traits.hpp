// generated from rosidl_generator_cpp/resource/idl__traits.hpp.em
// with input from hippocampus_ros2_msgs:msg/PolicyStatus.idl
// generated code does not contain a copyright notice

#ifndef HIPPOCAMPUS_ROS2_MSGS__MSG__DETAIL__POLICY_STATUS__TRAITS_HPP_
#define HIPPOCAMPUS_ROS2_MSGS__MSG__DETAIL__POLICY_STATUS__TRAITS_HPP_

#include <stdint.h>

#include <sstream>
#include <string>
#include <type_traits>

#include "hippocampus_ros2_msgs/msg/detail/policy_status__struct.hpp"
#include "rosidl_runtime_cpp/traits.hpp"

// Include directives for member types
// Member 'stamp'
#include "builtin_interfaces/msg/detail/time__traits.hpp"

namespace hippocampus_ros2_msgs
{

namespace msg
{

inline void to_flow_style_yaml(
  const PolicyStatus & msg,
  std::ostream & out)
{
  out << "{";
  // member: is_active
  {
    out << "is_active: ";
    rosidl_generator_traits::value_to_yaml(msg.is_active, out);
    out << ", ";
  }

  // member: graph_stale
  {
    out << "graph_stale: ";
    rosidl_generator_traits::value_to_yaml(msg.graph_stale, out);
    out << ", ";
  }

  // member: using_snn
  {
    out << "using_snn: ";
    rosidl_generator_traits::value_to_yaml(msg.using_snn, out);
    out << ", ";
  }

  // member: hierarchical_enabled
  {
    out << "hierarchical_enabled: ";
    rosidl_generator_traits::value_to_yaml(msg.hierarchical_enabled, out);
    out << ", ";
  }

  // member: feature_compute_time_ms
  {
    out << "feature_compute_time_ms: ";
    rosidl_generator_traits::value_to_yaml(msg.feature_compute_time_ms, out);
    out << ", ";
  }

  // member: policy_decision_time_ms
  {
    out << "policy_decision_time_ms: ";
    rosidl_generator_traits::value_to_yaml(msg.policy_decision_time_ms, out);
    out << ", ";
  }

  // member: safety_filter_time_ms
  {
    out << "safety_filter_time_ms: ";
    rosidl_generator_traits::value_to_yaml(msg.safety_filter_time_ms, out);
    out << ", ";
  }

  // member: total_latency_ms
  {
    out << "total_latency_ms: ";
    rosidl_generator_traits::value_to_yaml(msg.total_latency_ms, out);
    out << ", ";
  }

  // member: graph_nodes
  {
    out << "graph_nodes: ";
    rosidl_generator_traits::value_to_yaml(msg.graph_nodes, out);
    out << ", ";
  }

  // member: graph_edges
  {
    out << "graph_edges: ";
    rosidl_generator_traits::value_to_yaml(msg.graph_edges, out);
    out << ", ";
  }

  // member: graph_staleness_s
  {
    out << "graph_staleness_s: ";
    rosidl_generator_traits::value_to_yaml(msg.graph_staleness_s, out);
    out << ", ";
  }

  // member: current_confidence
  {
    out << "current_confidence: ";
    rosidl_generator_traits::value_to_yaml(msg.current_confidence, out);
    out << ", ";
  }

  // member: current_reason
  {
    out << "current_reason: ";
    rosidl_generator_traits::value_to_yaml(msg.current_reason, out);
    out << ", ";
  }

  // member: current_waypoint
  {
    out << "current_waypoint: ";
    rosidl_generator_traits::value_to_yaml(msg.current_waypoint, out);
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
  const PolicyStatus & msg,
  std::ostream & out, size_t indentation = 0)
{
  // member: is_active
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "is_active: ";
    rosidl_generator_traits::value_to_yaml(msg.is_active, out);
    out << "\n";
  }

  // member: graph_stale
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "graph_stale: ";
    rosidl_generator_traits::value_to_yaml(msg.graph_stale, out);
    out << "\n";
  }

  // member: using_snn
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "using_snn: ";
    rosidl_generator_traits::value_to_yaml(msg.using_snn, out);
    out << "\n";
  }

  // member: hierarchical_enabled
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "hierarchical_enabled: ";
    rosidl_generator_traits::value_to_yaml(msg.hierarchical_enabled, out);
    out << "\n";
  }

  // member: feature_compute_time_ms
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "feature_compute_time_ms: ";
    rosidl_generator_traits::value_to_yaml(msg.feature_compute_time_ms, out);
    out << "\n";
  }

  // member: policy_decision_time_ms
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "policy_decision_time_ms: ";
    rosidl_generator_traits::value_to_yaml(msg.policy_decision_time_ms, out);
    out << "\n";
  }

  // member: safety_filter_time_ms
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "safety_filter_time_ms: ";
    rosidl_generator_traits::value_to_yaml(msg.safety_filter_time_ms, out);
    out << "\n";
  }

  // member: total_latency_ms
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "total_latency_ms: ";
    rosidl_generator_traits::value_to_yaml(msg.total_latency_ms, out);
    out << "\n";
  }

  // member: graph_nodes
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "graph_nodes: ";
    rosidl_generator_traits::value_to_yaml(msg.graph_nodes, out);
    out << "\n";
  }

  // member: graph_edges
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "graph_edges: ";
    rosidl_generator_traits::value_to_yaml(msg.graph_edges, out);
    out << "\n";
  }

  // member: graph_staleness_s
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "graph_staleness_s: ";
    rosidl_generator_traits::value_to_yaml(msg.graph_staleness_s, out);
    out << "\n";
  }

  // member: current_confidence
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "current_confidence: ";
    rosidl_generator_traits::value_to_yaml(msg.current_confidence, out);
    out << "\n";
  }

  // member: current_reason
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "current_reason: ";
    rosidl_generator_traits::value_to_yaml(msg.current_reason, out);
    out << "\n";
  }

  // member: current_waypoint
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "current_waypoint: ";
    rosidl_generator_traits::value_to_yaml(msg.current_waypoint, out);
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

inline std::string to_yaml(const PolicyStatus & msg, bool use_flow_style = false)
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
  const hippocampus_ros2_msgs::msg::PolicyStatus & msg,
  std::ostream & out, size_t indentation = 0)
{
  hippocampus_ros2_msgs::msg::to_block_style_yaml(msg, out, indentation);
}

[[deprecated("use hippocampus_ros2_msgs::msg::to_yaml() instead")]]
inline std::string to_yaml(const hippocampus_ros2_msgs::msg::PolicyStatus & msg)
{
  return hippocampus_ros2_msgs::msg::to_yaml(msg);
}

template<>
inline const char * data_type<hippocampus_ros2_msgs::msg::PolicyStatus>()
{
  return "hippocampus_ros2_msgs::msg::PolicyStatus";
}

template<>
inline const char * name<hippocampus_ros2_msgs::msg::PolicyStatus>()
{
  return "hippocampus_ros2_msgs/msg/PolicyStatus";
}

template<>
struct has_fixed_size<hippocampus_ros2_msgs::msg::PolicyStatus>
  : std::integral_constant<bool, false> {};

template<>
struct has_bounded_size<hippocampus_ros2_msgs::msg::PolicyStatus>
  : std::integral_constant<bool, false> {};

template<>
struct is_message<hippocampus_ros2_msgs::msg::PolicyStatus>
  : std::true_type {};

}  // namespace rosidl_generator_traits

#endif  // HIPPOCAMPUS_ROS2_MSGS__MSG__DETAIL__POLICY_STATUS__TRAITS_HPP_
