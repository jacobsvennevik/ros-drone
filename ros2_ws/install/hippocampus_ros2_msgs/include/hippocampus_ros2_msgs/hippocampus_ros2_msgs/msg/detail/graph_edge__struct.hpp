// generated from rosidl_generator_cpp/resource/idl__struct.hpp.em
// with input from hippocampus_ros2_msgs:msg/GraphEdge.idl
// generated code does not contain a copyright notice

#ifndef HIPPOCAMPUS_ROS2_MSGS__MSG__DETAIL__GRAPH_EDGE__STRUCT_HPP_
#define HIPPOCAMPUS_ROS2_MSGS__MSG__DETAIL__GRAPH_EDGE__STRUCT_HPP_

#include <algorithm>
#include <array>
#include <memory>
#include <string>
#include <vector>

#include "rosidl_runtime_cpp/bounded_vector.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


#ifndef _WIN32
# define DEPRECATED__hippocampus_ros2_msgs__msg__GraphEdge __attribute__((deprecated))
#else
# define DEPRECATED__hippocampus_ros2_msgs__msg__GraphEdge __declspec(deprecated)
#endif

namespace hippocampus_ros2_msgs
{

namespace msg
{

// message struct
template<class ContainerAllocator>
struct GraphEdge_
{
  using Type = GraphEdge_<ContainerAllocator>;

  explicit GraphEdge_(rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  {
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      this->u = 0l;
      this->v = 0l;
      this->length = 0.0;
      this->traversable = false;
    }
  }

  explicit GraphEdge_(const ContainerAllocator & _alloc, rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  {
    (void)_alloc;
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      this->u = 0l;
      this->v = 0l;
      this->length = 0.0;
      this->traversable = false;
    }
  }

  // field types and members
  using _u_type =
    int32_t;
  _u_type u;
  using _v_type =
    int32_t;
  _v_type v;
  using _length_type =
    double;
  _length_type length;
  using _traversable_type =
    bool;
  _traversable_type traversable;

  // setters for named parameter idiom
  Type & set__u(
    const int32_t & _arg)
  {
    this->u = _arg;
    return *this;
  }
  Type & set__v(
    const int32_t & _arg)
  {
    this->v = _arg;
    return *this;
  }
  Type & set__length(
    const double & _arg)
  {
    this->length = _arg;
    return *this;
  }
  Type & set__traversable(
    const bool & _arg)
  {
    this->traversable = _arg;
    return *this;
  }

  // constant declarations

  // pointer types
  using RawPtr =
    hippocampus_ros2_msgs::msg::GraphEdge_<ContainerAllocator> *;
  using ConstRawPtr =
    const hippocampus_ros2_msgs::msg::GraphEdge_<ContainerAllocator> *;
  using SharedPtr =
    std::shared_ptr<hippocampus_ros2_msgs::msg::GraphEdge_<ContainerAllocator>>;
  using ConstSharedPtr =
    std::shared_ptr<hippocampus_ros2_msgs::msg::GraphEdge_<ContainerAllocator> const>;

  template<typename Deleter = std::default_delete<
      hippocampus_ros2_msgs::msg::GraphEdge_<ContainerAllocator>>>
  using UniquePtrWithDeleter =
    std::unique_ptr<hippocampus_ros2_msgs::msg::GraphEdge_<ContainerAllocator>, Deleter>;

  using UniquePtr = UniquePtrWithDeleter<>;

  template<typename Deleter = std::default_delete<
      hippocampus_ros2_msgs::msg::GraphEdge_<ContainerAllocator>>>
  using ConstUniquePtrWithDeleter =
    std::unique_ptr<hippocampus_ros2_msgs::msg::GraphEdge_<ContainerAllocator> const, Deleter>;
  using ConstUniquePtr = ConstUniquePtrWithDeleter<>;

  using WeakPtr =
    std::weak_ptr<hippocampus_ros2_msgs::msg::GraphEdge_<ContainerAllocator>>;
  using ConstWeakPtr =
    std::weak_ptr<hippocampus_ros2_msgs::msg::GraphEdge_<ContainerAllocator> const>;

  // pointer types similar to ROS 1, use SharedPtr / ConstSharedPtr instead
  // NOTE: Can't use 'using' here because GNU C++ can't parse attributes properly
  typedef DEPRECATED__hippocampus_ros2_msgs__msg__GraphEdge
    std::shared_ptr<hippocampus_ros2_msgs::msg::GraphEdge_<ContainerAllocator>>
    Ptr;
  typedef DEPRECATED__hippocampus_ros2_msgs__msg__GraphEdge
    std::shared_ptr<hippocampus_ros2_msgs::msg::GraphEdge_<ContainerAllocator> const>
    ConstPtr;

  // comparison operators
  bool operator==(const GraphEdge_ & other) const
  {
    if (this->u != other.u) {
      return false;
    }
    if (this->v != other.v) {
      return false;
    }
    if (this->length != other.length) {
      return false;
    }
    if (this->traversable != other.traversable) {
      return false;
    }
    return true;
  }
  bool operator!=(const GraphEdge_ & other) const
  {
    return !this->operator==(other);
  }
};  // struct GraphEdge_

// alias to use template instance with default allocator
using GraphEdge =
  hippocampus_ros2_msgs::msg::GraphEdge_<std::allocator<void>>;

// constant definitions

}  // namespace msg

}  // namespace hippocampus_ros2_msgs

#endif  // HIPPOCAMPUS_ROS2_MSGS__MSG__DETAIL__GRAPH_EDGE__STRUCT_HPP_
