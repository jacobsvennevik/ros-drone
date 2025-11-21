// generated from rosidl_generator_cpp/resource/idl__builder.hpp.em
// with input from hippocampus_ros2_msgs:msg/GraphSnapshot.idl
// generated code does not contain a copyright notice

#ifndef HIPPOCAMPUS_ROS2_MSGS__MSG__DETAIL__GRAPH_SNAPSHOT__BUILDER_HPP_
#define HIPPOCAMPUS_ROS2_MSGS__MSG__DETAIL__GRAPH_SNAPSHOT__BUILDER_HPP_

#include <algorithm>
#include <utility>

#include "hippocampus_ros2_msgs/msg/detail/graph_snapshot__struct.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


namespace hippocampus_ros2_msgs
{

namespace msg
{

namespace builder
{

class Init_GraphSnapshot_edges
{
public:
  explicit Init_GraphSnapshot_edges(::hippocampus_ros2_msgs::msg::GraphSnapshot & msg)
  : msg_(msg)
  {}
  ::hippocampus_ros2_msgs::msg::GraphSnapshot edges(::hippocampus_ros2_msgs::msg::GraphSnapshot::_edges_type arg)
  {
    msg_.edges = std::move(arg);
    return std::move(msg_);
  }

private:
  ::hippocampus_ros2_msgs::msg::GraphSnapshot msg_;
};

class Init_GraphSnapshot_nodes
{
public:
  explicit Init_GraphSnapshot_nodes(::hippocampus_ros2_msgs::msg::GraphSnapshot & msg)
  : msg_(msg)
  {}
  Init_GraphSnapshot_edges nodes(::hippocampus_ros2_msgs::msg::GraphSnapshot::_nodes_type arg)
  {
    msg_.nodes = std::move(arg);
    return Init_GraphSnapshot_edges(msg_);
  }

private:
  ::hippocampus_ros2_msgs::msg::GraphSnapshot msg_;
};

class Init_GraphSnapshot_staleness_warning
{
public:
  explicit Init_GraphSnapshot_staleness_warning(::hippocampus_ros2_msgs::msg::GraphSnapshot & msg)
  : msg_(msg)
  {}
  Init_GraphSnapshot_nodes staleness_warning(::hippocampus_ros2_msgs::msg::GraphSnapshot::_staleness_warning_type arg)
  {
    msg_.staleness_warning = std::move(arg);
    return Init_GraphSnapshot_nodes(msg_);
  }

private:
  ::hippocampus_ros2_msgs::msg::GraphSnapshot msg_;
};

class Init_GraphSnapshot_update_rate
{
public:
  explicit Init_GraphSnapshot_update_rate(::hippocampus_ros2_msgs::msg::GraphSnapshot & msg)
  : msg_(msg)
  {}
  Init_GraphSnapshot_staleness_warning update_rate(::hippocampus_ros2_msgs::msg::GraphSnapshot::_update_rate_type arg)
  {
    msg_.update_rate = std::move(arg);
    return Init_GraphSnapshot_staleness_warning(msg_);
  }

private:
  ::hippocampus_ros2_msgs::msg::GraphSnapshot msg_;
};

class Init_GraphSnapshot_last_updated
{
public:
  explicit Init_GraphSnapshot_last_updated(::hippocampus_ros2_msgs::msg::GraphSnapshot & msg)
  : msg_(msg)
  {}
  Init_GraphSnapshot_update_rate last_updated(::hippocampus_ros2_msgs::msg::GraphSnapshot::_last_updated_type arg)
  {
    msg_.last_updated = std::move(arg);
    return Init_GraphSnapshot_update_rate(msg_);
  }

private:
  ::hippocampus_ros2_msgs::msg::GraphSnapshot msg_;
};

class Init_GraphSnapshot_stamp
{
public:
  explicit Init_GraphSnapshot_stamp(::hippocampus_ros2_msgs::msg::GraphSnapshot & msg)
  : msg_(msg)
  {}
  Init_GraphSnapshot_last_updated stamp(::hippocampus_ros2_msgs::msg::GraphSnapshot::_stamp_type arg)
  {
    msg_.stamp = std::move(arg);
    return Init_GraphSnapshot_last_updated(msg_);
  }

private:
  ::hippocampus_ros2_msgs::msg::GraphSnapshot msg_;
};

class Init_GraphSnapshot_frame_id
{
public:
  explicit Init_GraphSnapshot_frame_id(::hippocampus_ros2_msgs::msg::GraphSnapshot & msg)
  : msg_(msg)
  {}
  Init_GraphSnapshot_stamp frame_id(::hippocampus_ros2_msgs::msg::GraphSnapshot::_frame_id_type arg)
  {
    msg_.frame_id = std::move(arg);
    return Init_GraphSnapshot_stamp(msg_);
  }

private:
  ::hippocampus_ros2_msgs::msg::GraphSnapshot msg_;
};

class Init_GraphSnapshot_epoch_id
{
public:
  Init_GraphSnapshot_epoch_id()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  Init_GraphSnapshot_frame_id epoch_id(::hippocampus_ros2_msgs::msg::GraphSnapshot::_epoch_id_type arg)
  {
    msg_.epoch_id = std::move(arg);
    return Init_GraphSnapshot_frame_id(msg_);
  }

private:
  ::hippocampus_ros2_msgs::msg::GraphSnapshot msg_;
};

}  // namespace builder

}  // namespace msg

template<typename MessageType>
auto build();

template<>
inline
auto build<::hippocampus_ros2_msgs::msg::GraphSnapshot>()
{
  return hippocampus_ros2_msgs::msg::builder::Init_GraphSnapshot_epoch_id();
}

}  // namespace hippocampus_ros2_msgs

#endif  // HIPPOCAMPUS_ROS2_MSGS__MSG__DETAIL__GRAPH_SNAPSHOT__BUILDER_HPP_
