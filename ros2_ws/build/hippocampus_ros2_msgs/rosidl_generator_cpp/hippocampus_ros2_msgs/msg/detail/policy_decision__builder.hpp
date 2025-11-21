// generated from rosidl_generator_cpp/resource/idl__builder.hpp.em
// with input from hippocampus_ros2_msgs:msg/PolicyDecision.idl
// generated code does not contain a copyright notice

#ifndef HIPPOCAMPUS_ROS2_MSGS__MSG__DETAIL__POLICY_DECISION__BUILDER_HPP_
#define HIPPOCAMPUS_ROS2_MSGS__MSG__DETAIL__POLICY_DECISION__BUILDER_HPP_

#include <algorithm>
#include <utility>

#include "hippocampus_ros2_msgs/msg/detail/policy_decision__struct.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


namespace hippocampus_ros2_msgs
{

namespace msg
{

namespace builder
{

class Init_PolicyDecision_stamp
{
public:
  explicit Init_PolicyDecision_stamp(::hippocampus_ros2_msgs::msg::PolicyDecision & msg)
  : msg_(msg)
  {}
  ::hippocampus_ros2_msgs::msg::PolicyDecision stamp(::hippocampus_ros2_msgs::msg::PolicyDecision::_stamp_type arg)
  {
    msg_.stamp = std::move(arg);
    return std::move(msg_);
  }

private:
  ::hippocampus_ros2_msgs::msg::PolicyDecision msg_;
};

class Init_PolicyDecision_next_waypoint
{
public:
  explicit Init_PolicyDecision_next_waypoint(::hippocampus_ros2_msgs::msg::PolicyDecision & msg)
  : msg_(msg)
  {}
  Init_PolicyDecision_stamp next_waypoint(::hippocampus_ros2_msgs::msg::PolicyDecision::_next_waypoint_type arg)
  {
    msg_.next_waypoint = std::move(arg);
    return Init_PolicyDecision_stamp(msg_);
  }

private:
  ::hippocampus_ros2_msgs::msg::PolicyDecision msg_;
};

class Init_PolicyDecision_reason
{
public:
  explicit Init_PolicyDecision_reason(::hippocampus_ros2_msgs::msg::PolicyDecision & msg)
  : msg_(msg)
  {}
  Init_PolicyDecision_next_waypoint reason(::hippocampus_ros2_msgs::msg::PolicyDecision::_reason_type arg)
  {
    msg_.reason = std::move(arg);
    return Init_PolicyDecision_next_waypoint(msg_);
  }

private:
  ::hippocampus_ros2_msgs::msg::PolicyDecision msg_;
};

class Init_PolicyDecision_confidence
{
public:
  explicit Init_PolicyDecision_confidence(::hippocampus_ros2_msgs::msg::PolicyDecision & msg)
  : msg_(msg)
  {}
  Init_PolicyDecision_reason confidence(::hippocampus_ros2_msgs::msg::PolicyDecision::_confidence_type arg)
  {
    msg_.confidence = std::move(arg);
    return Init_PolicyDecision_reason(msg_);
  }

private:
  ::hippocampus_ros2_msgs::msg::PolicyDecision msg_;
};

class Init_PolicyDecision_linear_z
{
public:
  explicit Init_PolicyDecision_linear_z(::hippocampus_ros2_msgs::msg::PolicyDecision & msg)
  : msg_(msg)
  {}
  Init_PolicyDecision_confidence linear_z(::hippocampus_ros2_msgs::msg::PolicyDecision::_linear_z_type arg)
  {
    msg_.linear_z = std::move(arg);
    return Init_PolicyDecision_confidence(msg_);
  }

private:
  ::hippocampus_ros2_msgs::msg::PolicyDecision msg_;
};

class Init_PolicyDecision_angular_z
{
public:
  explicit Init_PolicyDecision_angular_z(::hippocampus_ros2_msgs::msg::PolicyDecision & msg)
  : msg_(msg)
  {}
  Init_PolicyDecision_linear_z angular_z(::hippocampus_ros2_msgs::msg::PolicyDecision::_angular_z_type arg)
  {
    msg_.angular_z = std::move(arg);
    return Init_PolicyDecision_linear_z(msg_);
  }

private:
  ::hippocampus_ros2_msgs::msg::PolicyDecision msg_;
};

class Init_PolicyDecision_linear_x
{
public:
  Init_PolicyDecision_linear_x()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  Init_PolicyDecision_angular_z linear_x(::hippocampus_ros2_msgs::msg::PolicyDecision::_linear_x_type arg)
  {
    msg_.linear_x = std::move(arg);
    return Init_PolicyDecision_angular_z(msg_);
  }

private:
  ::hippocampus_ros2_msgs::msg::PolicyDecision msg_;
};

}  // namespace builder

}  // namespace msg

template<typename MessageType>
auto build();

template<>
inline
auto build<::hippocampus_ros2_msgs::msg::PolicyDecision>()
{
  return hippocampus_ros2_msgs::msg::builder::Init_PolicyDecision_linear_x();
}

}  // namespace hippocampus_ros2_msgs

#endif  // HIPPOCAMPUS_ROS2_MSGS__MSG__DETAIL__POLICY_DECISION__BUILDER_HPP_
