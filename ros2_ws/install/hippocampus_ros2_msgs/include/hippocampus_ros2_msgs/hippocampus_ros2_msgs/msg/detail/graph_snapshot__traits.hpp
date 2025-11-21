// generated from rosidl_generator_cpp/resource/idl__traits.hpp.em
// with input from hippocampus_ros2_msgs:msg/GraphSnapshot.idl
// generated code does not contain a copyright notice

#ifndef HIPPOCAMPUS_ROS2_MSGS__MSG__DETAIL__GRAPH_SNAPSHOT__TRAITS_HPP_
#define HIPPOCAMPUS_ROS2_MSGS__MSG__DETAIL__GRAPH_SNAPSHOT__TRAITS_HPP_

#include <stdint.h>

#include <sstream>
#include <string>
#include <type_traits>

#include "hippocampus_ros2_msgs/msg/detail/graph_snapshot__struct.hpp"
#include "rosidl_runtime_cpp/traits.hpp"

// Include directives for member types
// Member 'stamp'
// Member 'last_updated'
#include "builtin_interfaces/msg/detail/time__traits.hpp"
// Member 'nodes'
#include "hippocampus_ros2_msgs/msg/detail/graph_node__traits.hpp"
// Member 'edges'
#include "hippocampus_ros2_msgs/msg/detail/graph_edge__traits.hpp"

namespace hippocampus_ros2_msgs
{

namespace msg
{

inline void to_flow_style_yaml(
  const GraphSnapshot & msg,
  std::ostream & out)
{
  out << "{";
  // member: epoch_id
  {
    out << "epoch_id: ";
    rosidl_generator_traits::value_to_yaml(msg.epoch_id, out);
    out << ", ";
  }

  // member: frame_id
  {
    out << "frame_id: ";
    rosidl_generator_traits::value_to_yaml(msg.frame_id, out);
    out << ", ";
  }

  // member: stamp
  {
    out << "stamp: ";
    to_flow_style_yaml(msg.stamp, out);
    out << ", ";
  }

  // member: last_updated
  {
    out << "last_updated: ";
    to_flow_style_yaml(msg.last_updated, out);
    out << ", ";
  }

  // member: update_rate
  {
    out << "update_rate: ";
    rosidl_generator_traits::value_to_yaml(msg.update_rate, out);
    out << ", ";
  }

  // member: staleness_warning
  {
    out << "staleness_warning: ";
    rosidl_generator_traits::value_to_yaml(msg.staleness_warning, out);
    out << ", ";
  }

  // member: nodes
  {
    if (msg.nodes.size() == 0) {
      out << "nodes: []";
    } else {
      out << "nodes: [";
      size_t pending_items = msg.nodes.size();
      for (auto item : msg.nodes) {
        to_flow_style_yaml(item, out);
        if (--pending_items > 0) {
          out << ", ";
        }
      }
      out << "]";
    }
    out << ", ";
  }

  // member: edges
  {
    if (msg.edges.size() == 0) {
      out << "edges: []";
    } else {
      out << "edges: [";
      size_t pending_items = msg.edges.size();
      for (auto item : msg.edges) {
        to_flow_style_yaml(item, out);
        if (--pending_items > 0) {
          out << ", ";
        }
      }
      out << "]";
    }
  }
  out << "}";
}  // NOLINT(readability/fn_size)

inline void to_block_style_yaml(
  const GraphSnapshot & msg,
  std::ostream & out, size_t indentation = 0)
{
  // member: epoch_id
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "epoch_id: ";
    rosidl_generator_traits::value_to_yaml(msg.epoch_id, out);
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

  // member: stamp
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "stamp:\n";
    to_block_style_yaml(msg.stamp, out, indentation + 2);
  }

  // member: last_updated
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "last_updated:\n";
    to_block_style_yaml(msg.last_updated, out, indentation + 2);
  }

  // member: update_rate
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "update_rate: ";
    rosidl_generator_traits::value_to_yaml(msg.update_rate, out);
    out << "\n";
  }

  // member: staleness_warning
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "staleness_warning: ";
    rosidl_generator_traits::value_to_yaml(msg.staleness_warning, out);
    out << "\n";
  }

  // member: nodes
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    if (msg.nodes.size() == 0) {
      out << "nodes: []\n";
    } else {
      out << "nodes:\n";
      for (auto item : msg.nodes) {
        if (indentation > 0) {
          out << std::string(indentation, ' ');
        }
        out << "-\n";
        to_block_style_yaml(item, out, indentation + 2);
      }
    }
  }

  // member: edges
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    if (msg.edges.size() == 0) {
      out << "edges: []\n";
    } else {
      out << "edges:\n";
      for (auto item : msg.edges) {
        if (indentation > 0) {
          out << std::string(indentation, ' ');
        }
        out << "-\n";
        to_block_style_yaml(item, out, indentation + 2);
      }
    }
  }
}  // NOLINT(readability/fn_size)

inline std::string to_yaml(const GraphSnapshot & msg, bool use_flow_style = false)
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
  const hippocampus_ros2_msgs::msg::GraphSnapshot & msg,
  std::ostream & out, size_t indentation = 0)
{
  hippocampus_ros2_msgs::msg::to_block_style_yaml(msg, out, indentation);
}

[[deprecated("use hippocampus_ros2_msgs::msg::to_yaml() instead")]]
inline std::string to_yaml(const hippocampus_ros2_msgs::msg::GraphSnapshot & msg)
{
  return hippocampus_ros2_msgs::msg::to_yaml(msg);
}

template<>
inline const char * data_type<hippocampus_ros2_msgs::msg::GraphSnapshot>()
{
  return "hippocampus_ros2_msgs::msg::GraphSnapshot";
}

template<>
inline const char * name<hippocampus_ros2_msgs::msg::GraphSnapshot>()
{
  return "hippocampus_ros2_msgs/msg/GraphSnapshot";
}

template<>
struct has_fixed_size<hippocampus_ros2_msgs::msg::GraphSnapshot>
  : std::integral_constant<bool, false> {};

template<>
struct has_bounded_size<hippocampus_ros2_msgs::msg::GraphSnapshot>
  : std::integral_constant<bool, false> {};

template<>
struct is_message<hippocampus_ros2_msgs::msg::GraphSnapshot>
  : std::true_type {};

}  // namespace rosidl_generator_traits

#endif  // HIPPOCAMPUS_ROS2_MSGS__MSG__DETAIL__GRAPH_SNAPSHOT__TRAITS_HPP_
