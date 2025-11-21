// generated from rosidl_generator_cpp/resource/idl__builder.hpp.em
// with input from hippocampus_ros2_msgs:msg/GraphNode.idl
// generated code does not contain a copyright notice

#ifndef HIPPOCAMPUS_ROS2_MSGS__MSG__DETAIL__GRAPH_NODE__BUILDER_HPP_
#define HIPPOCAMPUS_ROS2_MSGS__MSG__DETAIL__GRAPH_NODE__BUILDER_HPP_

#include <algorithm>
#include <utility>

#include "hippocampus_ros2_msgs/msg/detail/graph_node__struct.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


namespace hippocampus_ros2_msgs
{

namespace msg
{

namespace builder
{

class Init_GraphNode_tags
{
public:
  explicit Init_GraphNode_tags(::hippocampus_ros2_msgs::msg::GraphNode & msg)
  : msg_(msg)
  {}
  ::hippocampus_ros2_msgs::msg::GraphNode tags(::hippocampus_ros2_msgs::msg::GraphNode::_tags_type arg)
  {
    msg_.tags = std::move(arg);
    return std::move(msg_);
  }

private:
  ::hippocampus_ros2_msgs::msg::GraphNode msg_;
};

class Init_GraphNode_degree
{
public:
  explicit Init_GraphNode_degree(::hippocampus_ros2_msgs::msg::GraphNode & msg)
  : msg_(msg)
  {}
  Init_GraphNode_tags degree(::hippocampus_ros2_msgs::msg::GraphNode::_degree_type arg)
  {
    msg_.degree = std::move(arg);
    return Init_GraphNode_tags(msg_);
  }

private:
  ::hippocampus_ros2_msgs::msg::GraphNode msg_;
};

class Init_GraphNode_normal
{
public:
  explicit Init_GraphNode_normal(::hippocampus_ros2_msgs::msg::GraphNode & msg)
  : msg_(msg)
  {}
  Init_GraphNode_degree normal(::hippocampus_ros2_msgs::msg::GraphNode::_normal_type arg)
  {
    msg_.normal = std::move(arg);
    return Init_GraphNode_degree(msg_);
  }

private:
  ::hippocampus_ros2_msgs::msg::GraphNode msg_;
};

class Init_GraphNode_position
{
public:
  explicit Init_GraphNode_position(::hippocampus_ros2_msgs::msg::GraphNode & msg)
  : msg_(msg)
  {}
  Init_GraphNode_normal position(::hippocampus_ros2_msgs::msg::GraphNode::_position_type arg)
  {
    msg_.position = std::move(arg);
    return Init_GraphNode_normal(msg_);
  }

private:
  ::hippocampus_ros2_msgs::msg::GraphNode msg_;
};

class Init_GraphNode_node_id
{
public:
  Init_GraphNode_node_id()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  Init_GraphNode_position node_id(::hippocampus_ros2_msgs::msg::GraphNode::_node_id_type arg)
  {
    msg_.node_id = std::move(arg);
    return Init_GraphNode_position(msg_);
  }

private:
  ::hippocampus_ros2_msgs::msg::GraphNode msg_;
};

}  // namespace builder

}  // namespace msg

template<typename MessageType>
auto build();

template<>
inline
auto build<::hippocampus_ros2_msgs::msg::GraphNode>()
{
  return hippocampus_ros2_msgs::msg::builder::Init_GraphNode_node_id();
}

}  // namespace hippocampus_ros2_msgs

#endif  // HIPPOCAMPUS_ROS2_MSGS__MSG__DETAIL__GRAPH_NODE__BUILDER_HPP_
