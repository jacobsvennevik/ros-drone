// generated from rosidl_generator_cpp/resource/idl__struct.hpp.em
// with input from hippocampus_ros2_msgs:msg/MissionGoal.idl
// generated code does not contain a copyright notice

#ifndef HIPPOCAMPUS_ROS2_MSGS__MSG__DETAIL__MISSION_GOAL__STRUCT_HPP_
#define HIPPOCAMPUS_ROS2_MSGS__MSG__DETAIL__MISSION_GOAL__STRUCT_HPP_

#include <algorithm>
#include <array>
#include <memory>
#include <string>
#include <vector>

#include "rosidl_runtime_cpp/bounded_vector.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


// Include directives for member types
// Member 'point_position'
// Member 'region_center'
#include "geometry_msgs/msg/detail/point__struct.hpp"
// Member 'stamp'
#include "builtin_interfaces/msg/detail/time__struct.hpp"

#ifndef _WIN32
# define DEPRECATED__hippocampus_ros2_msgs__msg__MissionGoal __attribute__((deprecated))
#else
# define DEPRECATED__hippocampus_ros2_msgs__msg__MissionGoal __declspec(deprecated)
#endif

namespace hippocampus_ros2_msgs
{

namespace msg
{

// message struct
template<class ContainerAllocator>
struct MissionGoal_
{
  using Type = MissionGoal_<ContainerAllocator>;

  explicit MissionGoal_(rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  : point_position(_init),
    region_center(_init),
    stamp(_init)
  {
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      this->goal_type = 0;
      this->point_tolerance = 0.0;
      this->node_id = 0l;
      this->region_radius = 0.0;
      this->frame_id = "";
      this->timeout = 0.0;
      this->is_reached = false;
    }
  }

  explicit MissionGoal_(const ContainerAllocator & _alloc, rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  : point_position(_alloc, _init),
    region_center(_alloc, _init),
    frame_id(_alloc),
    stamp(_alloc, _init)
  {
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      this->goal_type = 0;
      this->point_tolerance = 0.0;
      this->node_id = 0l;
      this->region_radius = 0.0;
      this->frame_id = "";
      this->timeout = 0.0;
      this->is_reached = false;
    }
  }

  // field types and members
  using _goal_type_type =
    uint8_t;
  _goal_type_type goal_type;
  using _point_position_type =
    geometry_msgs::msg::Point_<ContainerAllocator>;
  _point_position_type point_position;
  using _point_tolerance_type =
    double;
  _point_tolerance_type point_tolerance;
  using _node_id_type =
    int32_t;
  _node_id_type node_id;
  using _region_center_type =
    geometry_msgs::msg::Point_<ContainerAllocator>;
  _region_center_type region_center;
  using _region_radius_type =
    double;
  _region_radius_type region_radius;
  using _frame_id_type =
    std::basic_string<char, std::char_traits<char>, typename std::allocator_traits<ContainerAllocator>::template rebind_alloc<char>>;
  _frame_id_type frame_id;
  using _timeout_type =
    double;
  _timeout_type timeout;
  using _is_reached_type =
    bool;
  _is_reached_type is_reached;
  using _stamp_type =
    builtin_interfaces::msg::Time_<ContainerAllocator>;
  _stamp_type stamp;

  // setters for named parameter idiom
  Type & set__goal_type(
    const uint8_t & _arg)
  {
    this->goal_type = _arg;
    return *this;
  }
  Type & set__point_position(
    const geometry_msgs::msg::Point_<ContainerAllocator> & _arg)
  {
    this->point_position = _arg;
    return *this;
  }
  Type & set__point_tolerance(
    const double & _arg)
  {
    this->point_tolerance = _arg;
    return *this;
  }
  Type & set__node_id(
    const int32_t & _arg)
  {
    this->node_id = _arg;
    return *this;
  }
  Type & set__region_center(
    const geometry_msgs::msg::Point_<ContainerAllocator> & _arg)
  {
    this->region_center = _arg;
    return *this;
  }
  Type & set__region_radius(
    const double & _arg)
  {
    this->region_radius = _arg;
    return *this;
  }
  Type & set__frame_id(
    const std::basic_string<char, std::char_traits<char>, typename std::allocator_traits<ContainerAllocator>::template rebind_alloc<char>> & _arg)
  {
    this->frame_id = _arg;
    return *this;
  }
  Type & set__timeout(
    const double & _arg)
  {
    this->timeout = _arg;
    return *this;
  }
  Type & set__is_reached(
    const bool & _arg)
  {
    this->is_reached = _arg;
    return *this;
  }
  Type & set__stamp(
    const builtin_interfaces::msg::Time_<ContainerAllocator> & _arg)
  {
    this->stamp = _arg;
    return *this;
  }

  // constant declarations
  static constexpr uint8_t GOAL_TYPE_POINT =
    0u;
  static constexpr uint8_t GOAL_TYPE_NODE =
    1u;
  static constexpr uint8_t GOAL_TYPE_REGION =
    2u;

  // pointer types
  using RawPtr =
    hippocampus_ros2_msgs::msg::MissionGoal_<ContainerAllocator> *;
  using ConstRawPtr =
    const hippocampus_ros2_msgs::msg::MissionGoal_<ContainerAllocator> *;
  using SharedPtr =
    std::shared_ptr<hippocampus_ros2_msgs::msg::MissionGoal_<ContainerAllocator>>;
  using ConstSharedPtr =
    std::shared_ptr<hippocampus_ros2_msgs::msg::MissionGoal_<ContainerAllocator> const>;

  template<typename Deleter = std::default_delete<
      hippocampus_ros2_msgs::msg::MissionGoal_<ContainerAllocator>>>
  using UniquePtrWithDeleter =
    std::unique_ptr<hippocampus_ros2_msgs::msg::MissionGoal_<ContainerAllocator>, Deleter>;

  using UniquePtr = UniquePtrWithDeleter<>;

  template<typename Deleter = std::default_delete<
      hippocampus_ros2_msgs::msg::MissionGoal_<ContainerAllocator>>>
  using ConstUniquePtrWithDeleter =
    std::unique_ptr<hippocampus_ros2_msgs::msg::MissionGoal_<ContainerAllocator> const, Deleter>;
  using ConstUniquePtr = ConstUniquePtrWithDeleter<>;

  using WeakPtr =
    std::weak_ptr<hippocampus_ros2_msgs::msg::MissionGoal_<ContainerAllocator>>;
  using ConstWeakPtr =
    std::weak_ptr<hippocampus_ros2_msgs::msg::MissionGoal_<ContainerAllocator> const>;

  // pointer types similar to ROS 1, use SharedPtr / ConstSharedPtr instead
  // NOTE: Can't use 'using' here because GNU C++ can't parse attributes properly
  typedef DEPRECATED__hippocampus_ros2_msgs__msg__MissionGoal
    std::shared_ptr<hippocampus_ros2_msgs::msg::MissionGoal_<ContainerAllocator>>
    Ptr;
  typedef DEPRECATED__hippocampus_ros2_msgs__msg__MissionGoal
    std::shared_ptr<hippocampus_ros2_msgs::msg::MissionGoal_<ContainerAllocator> const>
    ConstPtr;

  // comparison operators
  bool operator==(const MissionGoal_ & other) const
  {
    if (this->goal_type != other.goal_type) {
      return false;
    }
    if (this->point_position != other.point_position) {
      return false;
    }
    if (this->point_tolerance != other.point_tolerance) {
      return false;
    }
    if (this->node_id != other.node_id) {
      return false;
    }
    if (this->region_center != other.region_center) {
      return false;
    }
    if (this->region_radius != other.region_radius) {
      return false;
    }
    if (this->frame_id != other.frame_id) {
      return false;
    }
    if (this->timeout != other.timeout) {
      return false;
    }
    if (this->is_reached != other.is_reached) {
      return false;
    }
    if (this->stamp != other.stamp) {
      return false;
    }
    return true;
  }
  bool operator!=(const MissionGoal_ & other) const
  {
    return !this->operator==(other);
  }
};  // struct MissionGoal_

// alias to use template instance with default allocator
using MissionGoal =
  hippocampus_ros2_msgs::msg::MissionGoal_<std::allocator<void>>;

// constant definitions
#if __cplusplus < 201703L
// static constexpr member variable definitions are only needed in C++14 and below, deprecated in C++17
template<typename ContainerAllocator>
constexpr uint8_t MissionGoal_<ContainerAllocator>::GOAL_TYPE_POINT;
#endif  // __cplusplus < 201703L
#if __cplusplus < 201703L
// static constexpr member variable definitions are only needed in C++14 and below, deprecated in C++17
template<typename ContainerAllocator>
constexpr uint8_t MissionGoal_<ContainerAllocator>::GOAL_TYPE_NODE;
#endif  // __cplusplus < 201703L
#if __cplusplus < 201703L
// static constexpr member variable definitions are only needed in C++14 and below, deprecated in C++17
template<typename ContainerAllocator>
constexpr uint8_t MissionGoal_<ContainerAllocator>::GOAL_TYPE_REGION;
#endif  // __cplusplus < 201703L

}  // namespace msg

}  // namespace hippocampus_ros2_msgs

#endif  // HIPPOCAMPUS_ROS2_MSGS__MSG__DETAIL__MISSION_GOAL__STRUCT_HPP_
