// generated from rosidl_generator_cpp/resource/idl__builder.hpp.em
// with input from hippocampus_ros2_msgs:msg/GraphEdge.idl
// generated code does not contain a copyright notice

#ifndef HIPPOCAMPUS_ROS2_MSGS__MSG__DETAIL__GRAPH_EDGE__BUILDER_HPP_
#define HIPPOCAMPUS_ROS2_MSGS__MSG__DETAIL__GRAPH_EDGE__BUILDER_HPP_

#include <algorithm>
#include <utility>

#include "hippocampus_ros2_msgs/msg/detail/graph_edge__struct.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


namespace hippocampus_ros2_msgs
{

namespace msg
{

namespace builder
{

class Init_GraphEdge_traversable
{
public:
  explicit Init_GraphEdge_traversable(::hippocampus_ros2_msgs::msg::GraphEdge & msg)
  : msg_(msg)
  {}
  ::hippocampus_ros2_msgs::msg::GraphEdge traversable(::hippocampus_ros2_msgs::msg::GraphEdge::_traversable_type arg)
  {
    msg_.traversable = std::move(arg);
    return std::move(msg_);
  }

private:
  ::hippocampus_ros2_msgs::msg::GraphEdge msg_;
};

class Init_GraphEdge_length
{
public:
  explicit Init_GraphEdge_length(::hippocampus_ros2_msgs::msg::GraphEdge & msg)
  : msg_(msg)
  {}
  Init_GraphEdge_traversable length(::hippocampus_ros2_msgs::msg::GraphEdge::_length_type arg)
  {
    msg_.length = std::move(arg);
    return Init_GraphEdge_traversable(msg_);
  }

private:
  ::hippocampus_ros2_msgs::msg::GraphEdge msg_;
};

class Init_GraphEdge_v
{
public:
  explicit Init_GraphEdge_v(::hippocampus_ros2_msgs::msg::GraphEdge & msg)
  : msg_(msg)
  {}
  Init_GraphEdge_length v(::hippocampus_ros2_msgs::msg::GraphEdge::_v_type arg)
  {
    msg_.v = std::move(arg);
    return Init_GraphEdge_length(msg_);
  }

private:
  ::hippocampus_ros2_msgs::msg::GraphEdge msg_;
};

class Init_GraphEdge_u
{
public:
  Init_GraphEdge_u()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  Init_GraphEdge_v u(::hippocampus_ros2_msgs::msg::GraphEdge::_u_type arg)
  {
    msg_.u = std::move(arg);
    return Init_GraphEdge_v(msg_);
  }

private:
  ::hippocampus_ros2_msgs::msg::GraphEdge msg_;
};

}  // namespace builder

}  // namespace msg

template<typename MessageType>
auto build();

template<>
inline
auto build<::hippocampus_ros2_msgs::msg::GraphEdge>()
{
  return hippocampus_ros2_msgs::msg::builder::Init_GraphEdge_u();
}

}  // namespace hippocampus_ros2_msgs

#endif  // HIPPOCAMPUS_ROS2_MSGS__MSG__DETAIL__GRAPH_EDGE__BUILDER_HPP_
