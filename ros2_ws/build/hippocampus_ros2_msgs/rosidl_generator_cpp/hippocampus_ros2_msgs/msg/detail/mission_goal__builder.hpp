// generated from rosidl_generator_cpp/resource/idl__builder.hpp.em
// with input from hippocampus_ros2_msgs:msg/MissionGoal.idl
// generated code does not contain a copyright notice

#ifndef HIPPOCAMPUS_ROS2_MSGS__MSG__DETAIL__MISSION_GOAL__BUILDER_HPP_
#define HIPPOCAMPUS_ROS2_MSGS__MSG__DETAIL__MISSION_GOAL__BUILDER_HPP_

#include <algorithm>
#include <utility>

#include "hippocampus_ros2_msgs/msg/detail/mission_goal__struct.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


namespace hippocampus_ros2_msgs
{

namespace msg
{

namespace builder
{

class Init_MissionGoal_stamp
{
public:
  explicit Init_MissionGoal_stamp(::hippocampus_ros2_msgs::msg::MissionGoal & msg)
  : msg_(msg)
  {}
  ::hippocampus_ros2_msgs::msg::MissionGoal stamp(::hippocampus_ros2_msgs::msg::MissionGoal::_stamp_type arg)
  {
    msg_.stamp = std::move(arg);
    return std::move(msg_);
  }

private:
  ::hippocampus_ros2_msgs::msg::MissionGoal msg_;
};

class Init_MissionGoal_is_reached
{
public:
  explicit Init_MissionGoal_is_reached(::hippocampus_ros2_msgs::msg::MissionGoal & msg)
  : msg_(msg)
  {}
  Init_MissionGoal_stamp is_reached(::hippocampus_ros2_msgs::msg::MissionGoal::_is_reached_type arg)
  {
    msg_.is_reached = std::move(arg);
    return Init_MissionGoal_stamp(msg_);
  }

private:
  ::hippocampus_ros2_msgs::msg::MissionGoal msg_;
};

class Init_MissionGoal_timeout
{
public:
  explicit Init_MissionGoal_timeout(::hippocampus_ros2_msgs::msg::MissionGoal & msg)
  : msg_(msg)
  {}
  Init_MissionGoal_is_reached timeout(::hippocampus_ros2_msgs::msg::MissionGoal::_timeout_type arg)
  {
    msg_.timeout = std::move(arg);
    return Init_MissionGoal_is_reached(msg_);
  }

private:
  ::hippocampus_ros2_msgs::msg::MissionGoal msg_;
};

class Init_MissionGoal_frame_id
{
public:
  explicit Init_MissionGoal_frame_id(::hippocampus_ros2_msgs::msg::MissionGoal & msg)
  : msg_(msg)
  {}
  Init_MissionGoal_timeout frame_id(::hippocampus_ros2_msgs::msg::MissionGoal::_frame_id_type arg)
  {
    msg_.frame_id = std::move(arg);
    return Init_MissionGoal_timeout(msg_);
  }

private:
  ::hippocampus_ros2_msgs::msg::MissionGoal msg_;
};

class Init_MissionGoal_region_radius
{
public:
  explicit Init_MissionGoal_region_radius(::hippocampus_ros2_msgs::msg::MissionGoal & msg)
  : msg_(msg)
  {}
  Init_MissionGoal_frame_id region_radius(::hippocampus_ros2_msgs::msg::MissionGoal::_region_radius_type arg)
  {
    msg_.region_radius = std::move(arg);
    return Init_MissionGoal_frame_id(msg_);
  }

private:
  ::hippocampus_ros2_msgs::msg::MissionGoal msg_;
};

class Init_MissionGoal_region_center
{
public:
  explicit Init_MissionGoal_region_center(::hippocampus_ros2_msgs::msg::MissionGoal & msg)
  : msg_(msg)
  {}
  Init_MissionGoal_region_radius region_center(::hippocampus_ros2_msgs::msg::MissionGoal::_region_center_type arg)
  {
    msg_.region_center = std::move(arg);
    return Init_MissionGoal_region_radius(msg_);
  }

private:
  ::hippocampus_ros2_msgs::msg::MissionGoal msg_;
};

class Init_MissionGoal_node_id
{
public:
  explicit Init_MissionGoal_node_id(::hippocampus_ros2_msgs::msg::MissionGoal & msg)
  : msg_(msg)
  {}
  Init_MissionGoal_region_center node_id(::hippocampus_ros2_msgs::msg::MissionGoal::_node_id_type arg)
  {
    msg_.node_id = std::move(arg);
    return Init_MissionGoal_region_center(msg_);
  }

private:
  ::hippocampus_ros2_msgs::msg::MissionGoal msg_;
};

class Init_MissionGoal_point_tolerance
{
public:
  explicit Init_MissionGoal_point_tolerance(::hippocampus_ros2_msgs::msg::MissionGoal & msg)
  : msg_(msg)
  {}
  Init_MissionGoal_node_id point_tolerance(::hippocampus_ros2_msgs::msg::MissionGoal::_point_tolerance_type arg)
  {
    msg_.point_tolerance = std::move(arg);
    return Init_MissionGoal_node_id(msg_);
  }

private:
  ::hippocampus_ros2_msgs::msg::MissionGoal msg_;
};

class Init_MissionGoal_point_position
{
public:
  explicit Init_MissionGoal_point_position(::hippocampus_ros2_msgs::msg::MissionGoal & msg)
  : msg_(msg)
  {}
  Init_MissionGoal_point_tolerance point_position(::hippocampus_ros2_msgs::msg::MissionGoal::_point_position_type arg)
  {
    msg_.point_position = std::move(arg);
    return Init_MissionGoal_point_tolerance(msg_);
  }

private:
  ::hippocampus_ros2_msgs::msg::MissionGoal msg_;
};

class Init_MissionGoal_goal_type
{
public:
  Init_MissionGoal_goal_type()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  Init_MissionGoal_point_position goal_type(::hippocampus_ros2_msgs::msg::MissionGoal::_goal_type_type arg)
  {
    msg_.goal_type = std::move(arg);
    return Init_MissionGoal_point_position(msg_);
  }

private:
  ::hippocampus_ros2_msgs::msg::MissionGoal msg_;
};

}  // namespace builder

}  // namespace msg

template<typename MessageType>
auto build();

template<>
inline
auto build<::hippocampus_ros2_msgs::msg::MissionGoal>()
{
  return hippocampus_ros2_msgs::msg::builder::Init_MissionGoal_goal_type();
}

}  // namespace hippocampus_ros2_msgs

#endif  // HIPPOCAMPUS_ROS2_MSGS__MSG__DETAIL__MISSION_GOAL__BUILDER_HPP_
