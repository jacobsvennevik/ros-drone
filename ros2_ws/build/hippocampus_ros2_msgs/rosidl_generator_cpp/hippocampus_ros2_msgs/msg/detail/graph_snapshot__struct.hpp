// generated from rosidl_generator_cpp/resource/idl__struct.hpp.em
// with input from hippocampus_ros2_msgs:msg/GraphSnapshot.idl
// generated code does not contain a copyright notice

#ifndef HIPPOCAMPUS_ROS2_MSGS__MSG__DETAIL__GRAPH_SNAPSHOT__STRUCT_HPP_
#define HIPPOCAMPUS_ROS2_MSGS__MSG__DETAIL__GRAPH_SNAPSHOT__STRUCT_HPP_

#include <algorithm>
#include <array>
#include <memory>
#include <string>
#include <vector>

#include "rosidl_runtime_cpp/bounded_vector.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


// Include directives for member types
// Member 'stamp'
// Member 'last_updated'
#include "builtin_interfaces/msg/detail/time__struct.hpp"
// Member 'nodes'
#include "hippocampus_ros2_msgs/msg/detail/graph_node__struct.hpp"
// Member 'edges'
#include "hippocampus_ros2_msgs/msg/detail/graph_edge__struct.hpp"

#ifndef _WIN32
# define DEPRECATED__hippocampus_ros2_msgs__msg__GraphSnapshot __attribute__((deprecated))
#else
# define DEPRECATED__hippocampus_ros2_msgs__msg__GraphSnapshot __declspec(deprecated)
#endif

namespace hippocampus_ros2_msgs
{

namespace msg
{

// message struct
template<class ContainerAllocator>
struct GraphSnapshot_
{
  using Type = GraphSnapshot_<ContainerAllocator>;

  explicit GraphSnapshot_(rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  : stamp(_init),
    last_updated(_init)
  {
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      this->epoch_id = 0ul;
      this->frame_id = "";
      this->update_rate = 0.0;
      this->staleness_warning = false;
    }
  }

  explicit GraphSnapshot_(const ContainerAllocator & _alloc, rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  : frame_id(_alloc),
    stamp(_alloc, _init),
    last_updated(_alloc, _init)
  {
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      this->epoch_id = 0ul;
      this->frame_id = "";
      this->update_rate = 0.0;
      this->staleness_warning = false;
    }
  }

  // field types and members
  using _epoch_id_type =
    uint32_t;
  _epoch_id_type epoch_id;
  using _frame_id_type =
    std::basic_string<char, std::char_traits<char>, typename std::allocator_traits<ContainerAllocator>::template rebind_alloc<char>>;
  _frame_id_type frame_id;
  using _stamp_type =
    builtin_interfaces::msg::Time_<ContainerAllocator>;
  _stamp_type stamp;
  using _last_updated_type =
    builtin_interfaces::msg::Time_<ContainerAllocator>;
  _last_updated_type last_updated;
  using _update_rate_type =
    double;
  _update_rate_type update_rate;
  using _staleness_warning_type =
    bool;
  _staleness_warning_type staleness_warning;
  using _nodes_type =
    std::vector<hippocampus_ros2_msgs::msg::GraphNode_<ContainerAllocator>, typename std::allocator_traits<ContainerAllocator>::template rebind_alloc<hippocampus_ros2_msgs::msg::GraphNode_<ContainerAllocator>>>;
  _nodes_type nodes;
  using _edges_type =
    std::vector<hippocampus_ros2_msgs::msg::GraphEdge_<ContainerAllocator>, typename std::allocator_traits<ContainerAllocator>::template rebind_alloc<hippocampus_ros2_msgs::msg::GraphEdge_<ContainerAllocator>>>;
  _edges_type edges;

  // setters for named parameter idiom
  Type & set__epoch_id(
    const uint32_t & _arg)
  {
    this->epoch_id = _arg;
    return *this;
  }
  Type & set__frame_id(
    const std::basic_string<char, std::char_traits<char>, typename std::allocator_traits<ContainerAllocator>::template rebind_alloc<char>> & _arg)
  {
    this->frame_id = _arg;
    return *this;
  }
  Type & set__stamp(
    const builtin_interfaces::msg::Time_<ContainerAllocator> & _arg)
  {
    this->stamp = _arg;
    return *this;
  }
  Type & set__last_updated(
    const builtin_interfaces::msg::Time_<ContainerAllocator> & _arg)
  {
    this->last_updated = _arg;
    return *this;
  }
  Type & set__update_rate(
    const double & _arg)
  {
    this->update_rate = _arg;
    return *this;
  }
  Type & set__staleness_warning(
    const bool & _arg)
  {
    this->staleness_warning = _arg;
    return *this;
  }
  Type & set__nodes(
    const std::vector<hippocampus_ros2_msgs::msg::GraphNode_<ContainerAllocator>, typename std::allocator_traits<ContainerAllocator>::template rebind_alloc<hippocampus_ros2_msgs::msg::GraphNode_<ContainerAllocator>>> & _arg)
  {
    this->nodes = _arg;
    return *this;
  }
  Type & set__edges(
    const std::vector<hippocampus_ros2_msgs::msg::GraphEdge_<ContainerAllocator>, typename std::allocator_traits<ContainerAllocator>::template rebind_alloc<hippocampus_ros2_msgs::msg::GraphEdge_<ContainerAllocator>>> & _arg)
  {
    this->edges = _arg;
    return *this;
  }

  // constant declarations

  // pointer types
  using RawPtr =
    hippocampus_ros2_msgs::msg::GraphSnapshot_<ContainerAllocator> *;
  using ConstRawPtr =
    const hippocampus_ros2_msgs::msg::GraphSnapshot_<ContainerAllocator> *;
  using SharedPtr =
    std::shared_ptr<hippocampus_ros2_msgs::msg::GraphSnapshot_<ContainerAllocator>>;
  using ConstSharedPtr =
    std::shared_ptr<hippocampus_ros2_msgs::msg::GraphSnapshot_<ContainerAllocator> const>;

  template<typename Deleter = std::default_delete<
      hippocampus_ros2_msgs::msg::GraphSnapshot_<ContainerAllocator>>>
  using UniquePtrWithDeleter =
    std::unique_ptr<hippocampus_ros2_msgs::msg::GraphSnapshot_<ContainerAllocator>, Deleter>;

  using UniquePtr = UniquePtrWithDeleter<>;

  template<typename Deleter = std::default_delete<
      hippocampus_ros2_msgs::msg::GraphSnapshot_<ContainerAllocator>>>
  using ConstUniquePtrWithDeleter =
    std::unique_ptr<hippocampus_ros2_msgs::msg::GraphSnapshot_<ContainerAllocator> const, Deleter>;
  using ConstUniquePtr = ConstUniquePtrWithDeleter<>;

  using WeakPtr =
    std::weak_ptr<hippocampus_ros2_msgs::msg::GraphSnapshot_<ContainerAllocator>>;
  using ConstWeakPtr =
    std::weak_ptr<hippocampus_ros2_msgs::msg::GraphSnapshot_<ContainerAllocator> const>;

  // pointer types similar to ROS 1, use SharedPtr / ConstSharedPtr instead
  // NOTE: Can't use 'using' here because GNU C++ can't parse attributes properly
  typedef DEPRECATED__hippocampus_ros2_msgs__msg__GraphSnapshot
    std::shared_ptr<hippocampus_ros2_msgs::msg::GraphSnapshot_<ContainerAllocator>>
    Ptr;
  typedef DEPRECATED__hippocampus_ros2_msgs__msg__GraphSnapshot
    std::shared_ptr<hippocampus_ros2_msgs::msg::GraphSnapshot_<ContainerAllocator> const>
    ConstPtr;

  // comparison operators
  bool operator==(const GraphSnapshot_ & other) const
  {
    if (this->epoch_id != other.epoch_id) {
      return false;
    }
    if (this->frame_id != other.frame_id) {
      return false;
    }
    if (this->stamp != other.stamp) {
      return false;
    }
    if (this->last_updated != other.last_updated) {
      return false;
    }
    if (this->update_rate != other.update_rate) {
      return false;
    }
    if (this->staleness_warning != other.staleness_warning) {
      return false;
    }
    if (this->nodes != other.nodes) {
      return false;
    }
    if (this->edges != other.edges) {
      return false;
    }
    return true;
  }
  bool operator!=(const GraphSnapshot_ & other) const
  {
    return !this->operator==(other);
  }
};  // struct GraphSnapshot_

// alias to use template instance with default allocator
using GraphSnapshot =
  hippocampus_ros2_msgs::msg::GraphSnapshot_<std::allocator<void>>;

// constant definitions

}  // namespace msg

}  // namespace hippocampus_ros2_msgs

#endif  // HIPPOCAMPUS_ROS2_MSGS__MSG__DETAIL__GRAPH_SNAPSHOT__STRUCT_HPP_
