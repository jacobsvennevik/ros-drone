// generated from rosidl_generator_cpp/resource/idl__struct.hpp.em
// with input from hippocampus_ros2_msgs:msg/GraphNode.idl
// generated code does not contain a copyright notice

#ifndef HIPPOCAMPUS_ROS2_MSGS__MSG__DETAIL__GRAPH_NODE__STRUCT_HPP_
#define HIPPOCAMPUS_ROS2_MSGS__MSG__DETAIL__GRAPH_NODE__STRUCT_HPP_

#include <algorithm>
#include <array>
#include <memory>
#include <string>
#include <vector>

#include "rosidl_runtime_cpp/bounded_vector.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


// Include directives for member types
// Member 'position'
#include "geometry_msgs/msg/detail/point__struct.hpp"
// Member 'normal'
#include "geometry_msgs/msg/detail/vector3__struct.hpp"

#ifndef _WIN32
# define DEPRECATED__hippocampus_ros2_msgs__msg__GraphNode __attribute__((deprecated))
#else
# define DEPRECATED__hippocampus_ros2_msgs__msg__GraphNode __declspec(deprecated)
#endif

namespace hippocampus_ros2_msgs
{

namespace msg
{

// message struct
template<class ContainerAllocator>
struct GraphNode_
{
  using Type = GraphNode_<ContainerAllocator>;

  explicit GraphNode_(rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  : position(_init),
    normal(_init)
  {
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      this->node_id = 0l;
      this->degree = 0ul;
    }
  }

  explicit GraphNode_(const ContainerAllocator & _alloc, rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  : position(_alloc, _init),
    normal(_alloc, _init)
  {
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      this->node_id = 0l;
      this->degree = 0ul;
    }
  }

  // field types and members
  using _node_id_type =
    int32_t;
  _node_id_type node_id;
  using _position_type =
    geometry_msgs::msg::Point_<ContainerAllocator>;
  _position_type position;
  using _normal_type =
    geometry_msgs::msg::Vector3_<ContainerAllocator>;
  _normal_type normal;
  using _degree_type =
    uint32_t;
  _degree_type degree;
  using _tags_type =
    std::vector<std::basic_string<char, std::char_traits<char>, typename std::allocator_traits<ContainerAllocator>::template rebind_alloc<char>>, typename std::allocator_traits<ContainerAllocator>::template rebind_alloc<std::basic_string<char, std::char_traits<char>, typename std::allocator_traits<ContainerAllocator>::template rebind_alloc<char>>>>;
  _tags_type tags;

  // setters for named parameter idiom
  Type & set__node_id(
    const int32_t & _arg)
  {
    this->node_id = _arg;
    return *this;
  }
  Type & set__position(
    const geometry_msgs::msg::Point_<ContainerAllocator> & _arg)
  {
    this->position = _arg;
    return *this;
  }
  Type & set__normal(
    const geometry_msgs::msg::Vector3_<ContainerAllocator> & _arg)
  {
    this->normal = _arg;
    return *this;
  }
  Type & set__degree(
    const uint32_t & _arg)
  {
    this->degree = _arg;
    return *this;
  }
  Type & set__tags(
    const std::vector<std::basic_string<char, std::char_traits<char>, typename std::allocator_traits<ContainerAllocator>::template rebind_alloc<char>>, typename std::allocator_traits<ContainerAllocator>::template rebind_alloc<std::basic_string<char, std::char_traits<char>, typename std::allocator_traits<ContainerAllocator>::template rebind_alloc<char>>>> & _arg)
  {
    this->tags = _arg;
    return *this;
  }

  // constant declarations

  // pointer types
  using RawPtr =
    hippocampus_ros2_msgs::msg::GraphNode_<ContainerAllocator> *;
  using ConstRawPtr =
    const hippocampus_ros2_msgs::msg::GraphNode_<ContainerAllocator> *;
  using SharedPtr =
    std::shared_ptr<hippocampus_ros2_msgs::msg::GraphNode_<ContainerAllocator>>;
  using ConstSharedPtr =
    std::shared_ptr<hippocampus_ros2_msgs::msg::GraphNode_<ContainerAllocator> const>;

  template<typename Deleter = std::default_delete<
      hippocampus_ros2_msgs::msg::GraphNode_<ContainerAllocator>>>
  using UniquePtrWithDeleter =
    std::unique_ptr<hippocampus_ros2_msgs::msg::GraphNode_<ContainerAllocator>, Deleter>;

  using UniquePtr = UniquePtrWithDeleter<>;

  template<typename Deleter = std::default_delete<
      hippocampus_ros2_msgs::msg::GraphNode_<ContainerAllocator>>>
  using ConstUniquePtrWithDeleter =
    std::unique_ptr<hippocampus_ros2_msgs::msg::GraphNode_<ContainerAllocator> const, Deleter>;
  using ConstUniquePtr = ConstUniquePtrWithDeleter<>;

  using WeakPtr =
    std::weak_ptr<hippocampus_ros2_msgs::msg::GraphNode_<ContainerAllocator>>;
  using ConstWeakPtr =
    std::weak_ptr<hippocampus_ros2_msgs::msg::GraphNode_<ContainerAllocator> const>;

  // pointer types similar to ROS 1, use SharedPtr / ConstSharedPtr instead
  // NOTE: Can't use 'using' here because GNU C++ can't parse attributes properly
  typedef DEPRECATED__hippocampus_ros2_msgs__msg__GraphNode
    std::shared_ptr<hippocampus_ros2_msgs::msg::GraphNode_<ContainerAllocator>>
    Ptr;
  typedef DEPRECATED__hippocampus_ros2_msgs__msg__GraphNode
    std::shared_ptr<hippocampus_ros2_msgs::msg::GraphNode_<ContainerAllocator> const>
    ConstPtr;

  // comparison operators
  bool operator==(const GraphNode_ & other) const
  {
    if (this->node_id != other.node_id) {
      return false;
    }
    if (this->position != other.position) {
      return false;
    }
    if (this->normal != other.normal) {
      return false;
    }
    if (this->degree != other.degree) {
      return false;
    }
    if (this->tags != other.tags) {
      return false;
    }
    return true;
  }
  bool operator!=(const GraphNode_ & other) const
  {
    return !this->operator==(other);
  }
};  // struct GraphNode_

// alias to use template instance with default allocator
using GraphNode =
  hippocampus_ros2_msgs::msg::GraphNode_<std::allocator<void>>;

// constant definitions

}  // namespace msg

}  // namespace hippocampus_ros2_msgs

#endif  // HIPPOCAMPUS_ROS2_MSGS__MSG__DETAIL__GRAPH_NODE__STRUCT_HPP_
