// generated from rosidl_generator_cpp/resource/idl__builder.hpp.em
// with input from hippocampus_ros2_msgs:msg/PolicyStatus.idl
// generated code does not contain a copyright notice

#ifndef HIPPOCAMPUS_ROS2_MSGS__MSG__DETAIL__POLICY_STATUS__BUILDER_HPP_
#define HIPPOCAMPUS_ROS2_MSGS__MSG__DETAIL__POLICY_STATUS__BUILDER_HPP_

#include <algorithm>
#include <utility>

#include "hippocampus_ros2_msgs/msg/detail/policy_status__struct.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


namespace hippocampus_ros2_msgs
{

namespace msg
{

namespace builder
{

class Init_PolicyStatus_stamp
{
public:
  explicit Init_PolicyStatus_stamp(::hippocampus_ros2_msgs::msg::PolicyStatus & msg)
  : msg_(msg)
  {}
  ::hippocampus_ros2_msgs::msg::PolicyStatus stamp(::hippocampus_ros2_msgs::msg::PolicyStatus::_stamp_type arg)
  {
    msg_.stamp = std::move(arg);
    return std::move(msg_);
  }

private:
  ::hippocampus_ros2_msgs::msg::PolicyStatus msg_;
};

class Init_PolicyStatus_current_waypoint
{
public:
  explicit Init_PolicyStatus_current_waypoint(::hippocampus_ros2_msgs::msg::PolicyStatus & msg)
  : msg_(msg)
  {}
  Init_PolicyStatus_stamp current_waypoint(::hippocampus_ros2_msgs::msg::PolicyStatus::_current_waypoint_type arg)
  {
    msg_.current_waypoint = std::move(arg);
    return Init_PolicyStatus_stamp(msg_);
  }

private:
  ::hippocampus_ros2_msgs::msg::PolicyStatus msg_;
};

class Init_PolicyStatus_current_reason
{
public:
  explicit Init_PolicyStatus_current_reason(::hippocampus_ros2_msgs::msg::PolicyStatus & msg)
  : msg_(msg)
  {}
  Init_PolicyStatus_current_waypoint current_reason(::hippocampus_ros2_msgs::msg::PolicyStatus::_current_reason_type arg)
  {
    msg_.current_reason = std::move(arg);
    return Init_PolicyStatus_current_waypoint(msg_);
  }

private:
  ::hippocampus_ros2_msgs::msg::PolicyStatus msg_;
};

class Init_PolicyStatus_current_confidence
{
public:
  explicit Init_PolicyStatus_current_confidence(::hippocampus_ros2_msgs::msg::PolicyStatus & msg)
  : msg_(msg)
  {}
  Init_PolicyStatus_current_reason current_confidence(::hippocampus_ros2_msgs::msg::PolicyStatus::_current_confidence_type arg)
  {
    msg_.current_confidence = std::move(arg);
    return Init_PolicyStatus_current_reason(msg_);
  }

private:
  ::hippocampus_ros2_msgs::msg::PolicyStatus msg_;
};

class Init_PolicyStatus_graph_staleness_s
{
public:
  explicit Init_PolicyStatus_graph_staleness_s(::hippocampus_ros2_msgs::msg::PolicyStatus & msg)
  : msg_(msg)
  {}
  Init_PolicyStatus_current_confidence graph_staleness_s(::hippocampus_ros2_msgs::msg::PolicyStatus::_graph_staleness_s_type arg)
  {
    msg_.graph_staleness_s = std::move(arg);
    return Init_PolicyStatus_current_confidence(msg_);
  }

private:
  ::hippocampus_ros2_msgs::msg::PolicyStatus msg_;
};

class Init_PolicyStatus_graph_edges
{
public:
  explicit Init_PolicyStatus_graph_edges(::hippocampus_ros2_msgs::msg::PolicyStatus & msg)
  : msg_(msg)
  {}
  Init_PolicyStatus_graph_staleness_s graph_edges(::hippocampus_ros2_msgs::msg::PolicyStatus::_graph_edges_type arg)
  {
    msg_.graph_edges = std::move(arg);
    return Init_PolicyStatus_graph_staleness_s(msg_);
  }

private:
  ::hippocampus_ros2_msgs::msg::PolicyStatus msg_;
};

class Init_PolicyStatus_graph_nodes
{
public:
  explicit Init_PolicyStatus_graph_nodes(::hippocampus_ros2_msgs::msg::PolicyStatus & msg)
  : msg_(msg)
  {}
  Init_PolicyStatus_graph_edges graph_nodes(::hippocampus_ros2_msgs::msg::PolicyStatus::_graph_nodes_type arg)
  {
    msg_.graph_nodes = std::move(arg);
    return Init_PolicyStatus_graph_edges(msg_);
  }

private:
  ::hippocampus_ros2_msgs::msg::PolicyStatus msg_;
};

class Init_PolicyStatus_total_latency_ms
{
public:
  explicit Init_PolicyStatus_total_latency_ms(::hippocampus_ros2_msgs::msg::PolicyStatus & msg)
  : msg_(msg)
  {}
  Init_PolicyStatus_graph_nodes total_latency_ms(::hippocampus_ros2_msgs::msg::PolicyStatus::_total_latency_ms_type arg)
  {
    msg_.total_latency_ms = std::move(arg);
    return Init_PolicyStatus_graph_nodes(msg_);
  }

private:
  ::hippocampus_ros2_msgs::msg::PolicyStatus msg_;
};

class Init_PolicyStatus_safety_filter_time_ms
{
public:
  explicit Init_PolicyStatus_safety_filter_time_ms(::hippocampus_ros2_msgs::msg::PolicyStatus & msg)
  : msg_(msg)
  {}
  Init_PolicyStatus_total_latency_ms safety_filter_time_ms(::hippocampus_ros2_msgs::msg::PolicyStatus::_safety_filter_time_ms_type arg)
  {
    msg_.safety_filter_time_ms = std::move(arg);
    return Init_PolicyStatus_total_latency_ms(msg_);
  }

private:
  ::hippocampus_ros2_msgs::msg::PolicyStatus msg_;
};

class Init_PolicyStatus_policy_decision_time_ms
{
public:
  explicit Init_PolicyStatus_policy_decision_time_ms(::hippocampus_ros2_msgs::msg::PolicyStatus & msg)
  : msg_(msg)
  {}
  Init_PolicyStatus_safety_filter_time_ms policy_decision_time_ms(::hippocampus_ros2_msgs::msg::PolicyStatus::_policy_decision_time_ms_type arg)
  {
    msg_.policy_decision_time_ms = std::move(arg);
    return Init_PolicyStatus_safety_filter_time_ms(msg_);
  }

private:
  ::hippocampus_ros2_msgs::msg::PolicyStatus msg_;
};

class Init_PolicyStatus_feature_compute_time_ms
{
public:
  explicit Init_PolicyStatus_feature_compute_time_ms(::hippocampus_ros2_msgs::msg::PolicyStatus & msg)
  : msg_(msg)
  {}
  Init_PolicyStatus_policy_decision_time_ms feature_compute_time_ms(::hippocampus_ros2_msgs::msg::PolicyStatus::_feature_compute_time_ms_type arg)
  {
    msg_.feature_compute_time_ms = std::move(arg);
    return Init_PolicyStatus_policy_decision_time_ms(msg_);
  }

private:
  ::hippocampus_ros2_msgs::msg::PolicyStatus msg_;
};

class Init_PolicyStatus_hierarchical_enabled
{
public:
  explicit Init_PolicyStatus_hierarchical_enabled(::hippocampus_ros2_msgs::msg::PolicyStatus & msg)
  : msg_(msg)
  {}
  Init_PolicyStatus_feature_compute_time_ms hierarchical_enabled(::hippocampus_ros2_msgs::msg::PolicyStatus::_hierarchical_enabled_type arg)
  {
    msg_.hierarchical_enabled = std::move(arg);
    return Init_PolicyStatus_feature_compute_time_ms(msg_);
  }

private:
  ::hippocampus_ros2_msgs::msg::PolicyStatus msg_;
};

class Init_PolicyStatus_using_snn
{
public:
  explicit Init_PolicyStatus_using_snn(::hippocampus_ros2_msgs::msg::PolicyStatus & msg)
  : msg_(msg)
  {}
  Init_PolicyStatus_hierarchical_enabled using_snn(::hippocampus_ros2_msgs::msg::PolicyStatus::_using_snn_type arg)
  {
    msg_.using_snn = std::move(arg);
    return Init_PolicyStatus_hierarchical_enabled(msg_);
  }

private:
  ::hippocampus_ros2_msgs::msg::PolicyStatus msg_;
};

class Init_PolicyStatus_graph_stale
{
public:
  explicit Init_PolicyStatus_graph_stale(::hippocampus_ros2_msgs::msg::PolicyStatus & msg)
  : msg_(msg)
  {}
  Init_PolicyStatus_using_snn graph_stale(::hippocampus_ros2_msgs::msg::PolicyStatus::_graph_stale_type arg)
  {
    msg_.graph_stale = std::move(arg);
    return Init_PolicyStatus_using_snn(msg_);
  }

private:
  ::hippocampus_ros2_msgs::msg::PolicyStatus msg_;
};

class Init_PolicyStatus_is_active
{
public:
  Init_PolicyStatus_is_active()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  Init_PolicyStatus_graph_stale is_active(::hippocampus_ros2_msgs::msg::PolicyStatus::_is_active_type arg)
  {
    msg_.is_active = std::move(arg);
    return Init_PolicyStatus_graph_stale(msg_);
  }

private:
  ::hippocampus_ros2_msgs::msg::PolicyStatus msg_;
};

}  // namespace builder

}  // namespace msg

template<typename MessageType>
auto build();

template<>
inline
auto build<::hippocampus_ros2_msgs::msg::PolicyStatus>()
{
  return hippocampus_ros2_msgs::msg::builder::Init_PolicyStatus_is_active();
}

}  // namespace hippocampus_ros2_msgs

#endif  // HIPPOCAMPUS_ROS2_MSGS__MSG__DETAIL__POLICY_STATUS__BUILDER_HPP_
