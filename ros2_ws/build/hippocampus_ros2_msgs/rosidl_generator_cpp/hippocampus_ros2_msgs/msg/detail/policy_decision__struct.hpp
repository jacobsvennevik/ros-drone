// generated from rosidl_generator_cpp/resource/idl__struct.hpp.em
// with input from hippocampus_ros2_msgs:msg/PolicyDecision.idl
// generated code does not contain a copyright notice

#ifndef HIPPOCAMPUS_ROS2_MSGS__MSG__DETAIL__POLICY_DECISION__STRUCT_HPP_
#define HIPPOCAMPUS_ROS2_MSGS__MSG__DETAIL__POLICY_DECISION__STRUCT_HPP_

#include <algorithm>
#include <array>
#include <memory>
#include <string>
#include <vector>

#include "rosidl_runtime_cpp/bounded_vector.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


// Include directives for member types
// Member 'stamp'
#include "builtin_interfaces/msg/detail/time__struct.hpp"

#ifndef _WIN32
# define DEPRECATED__hippocampus_ros2_msgs__msg__PolicyDecision __attribute__((deprecated))
#else
# define DEPRECATED__hippocampus_ros2_msgs__msg__PolicyDecision __declspec(deprecated)
#endif

namespace hippocampus_ros2_msgs
{

namespace msg
{

// message struct
template<class ContainerAllocator>
struct PolicyDecision_
{
  using Type = PolicyDecision_<ContainerAllocator>;

  explicit PolicyDecision_(rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  : stamp(_init)
  {
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      this->linear_x = 0.0;
      this->angular_z = 0.0;
      this->linear_z = 0.0;
      this->confidence = 0.0;
      this->reason = "";
      this->next_waypoint = 0l;
    }
  }

  explicit PolicyDecision_(const ContainerAllocator & _alloc, rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  : reason(_alloc),
    stamp(_alloc, _init)
  {
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      this->linear_x = 0.0;
      this->angular_z = 0.0;
      this->linear_z = 0.0;
      this->confidence = 0.0;
      this->reason = "";
      this->next_waypoint = 0l;
    }
  }

  // field types and members
  using _linear_x_type =
    double;
  _linear_x_type linear_x;
  using _angular_z_type =
    double;
  _angular_z_type angular_z;
  using _linear_z_type =
    double;
  _linear_z_type linear_z;
  using _confidence_type =
    double;
  _confidence_type confidence;
  using _reason_type =
    std::basic_string<char, std::char_traits<char>, typename std::allocator_traits<ContainerAllocator>::template rebind_alloc<char>>;
  _reason_type reason;
  using _next_waypoint_type =
    int32_t;
  _next_waypoint_type next_waypoint;
  using _stamp_type =
    builtin_interfaces::msg::Time_<ContainerAllocator>;
  _stamp_type stamp;

  // setters for named parameter idiom
  Type & set__linear_x(
    const double & _arg)
  {
    this->linear_x = _arg;
    return *this;
  }
  Type & set__angular_z(
    const double & _arg)
  {
    this->angular_z = _arg;
    return *this;
  }
  Type & set__linear_z(
    const double & _arg)
  {
    this->linear_z = _arg;
    return *this;
  }
  Type & set__confidence(
    const double & _arg)
  {
    this->confidence = _arg;
    return *this;
  }
  Type & set__reason(
    const std::basic_string<char, std::char_traits<char>, typename std::allocator_traits<ContainerAllocator>::template rebind_alloc<char>> & _arg)
  {
    this->reason = _arg;
    return *this;
  }
  Type & set__next_waypoint(
    const int32_t & _arg)
  {
    this->next_waypoint = _arg;
    return *this;
  }
  Type & set__stamp(
    const builtin_interfaces::msg::Time_<ContainerAllocator> & _arg)
  {
    this->stamp = _arg;
    return *this;
  }

  // constant declarations

  // pointer types
  using RawPtr =
    hippocampus_ros2_msgs::msg::PolicyDecision_<ContainerAllocator> *;
  using ConstRawPtr =
    const hippocampus_ros2_msgs::msg::PolicyDecision_<ContainerAllocator> *;
  using SharedPtr =
    std::shared_ptr<hippocampus_ros2_msgs::msg::PolicyDecision_<ContainerAllocator>>;
  using ConstSharedPtr =
    std::shared_ptr<hippocampus_ros2_msgs::msg::PolicyDecision_<ContainerAllocator> const>;

  template<typename Deleter = std::default_delete<
      hippocampus_ros2_msgs::msg::PolicyDecision_<ContainerAllocator>>>
  using UniquePtrWithDeleter =
    std::unique_ptr<hippocampus_ros2_msgs::msg::PolicyDecision_<ContainerAllocator>, Deleter>;

  using UniquePtr = UniquePtrWithDeleter<>;

  template<typename Deleter = std::default_delete<
      hippocampus_ros2_msgs::msg::PolicyDecision_<ContainerAllocator>>>
  using ConstUniquePtrWithDeleter =
    std::unique_ptr<hippocampus_ros2_msgs::msg::PolicyDecision_<ContainerAllocator> const, Deleter>;
  using ConstUniquePtr = ConstUniquePtrWithDeleter<>;

  using WeakPtr =
    std::weak_ptr<hippocampus_ros2_msgs::msg::PolicyDecision_<ContainerAllocator>>;
  using ConstWeakPtr =
    std::weak_ptr<hippocampus_ros2_msgs::msg::PolicyDecision_<ContainerAllocator> const>;

  // pointer types similar to ROS 1, use SharedPtr / ConstSharedPtr instead
  // NOTE: Can't use 'using' here because GNU C++ can't parse attributes properly
  typedef DEPRECATED__hippocampus_ros2_msgs__msg__PolicyDecision
    std::shared_ptr<hippocampus_ros2_msgs::msg::PolicyDecision_<ContainerAllocator>>
    Ptr;
  typedef DEPRECATED__hippocampus_ros2_msgs__msg__PolicyDecision
    std::shared_ptr<hippocampus_ros2_msgs::msg::PolicyDecision_<ContainerAllocator> const>
    ConstPtr;

  // comparison operators
  bool operator==(const PolicyDecision_ & other) const
  {
    if (this->linear_x != other.linear_x) {
      return false;
    }
    if (this->angular_z != other.angular_z) {
      return false;
    }
    if (this->linear_z != other.linear_z) {
      return false;
    }
    if (this->confidence != other.confidence) {
      return false;
    }
    if (this->reason != other.reason) {
      return false;
    }
    if (this->next_waypoint != other.next_waypoint) {
      return false;
    }
    if (this->stamp != other.stamp) {
      return false;
    }
    return true;
  }
  bool operator!=(const PolicyDecision_ & other) const
  {
    return !this->operator==(other);
  }
};  // struct PolicyDecision_

// alias to use template instance with default allocator
using PolicyDecision =
  hippocampus_ros2_msgs::msg::PolicyDecision_<std::allocator<void>>;

// constant definitions

}  // namespace msg

}  // namespace hippocampus_ros2_msgs

#endif  // HIPPOCAMPUS_ROS2_MSGS__MSG__DETAIL__POLICY_DECISION__STRUCT_HPP_
