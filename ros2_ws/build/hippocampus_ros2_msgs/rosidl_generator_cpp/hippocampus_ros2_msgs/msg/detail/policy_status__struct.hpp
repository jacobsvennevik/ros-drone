// generated from rosidl_generator_cpp/resource/idl__struct.hpp.em
// with input from hippocampus_ros2_msgs:msg/PolicyStatus.idl
// generated code does not contain a copyright notice

#ifndef HIPPOCAMPUS_ROS2_MSGS__MSG__DETAIL__POLICY_STATUS__STRUCT_HPP_
#define HIPPOCAMPUS_ROS2_MSGS__MSG__DETAIL__POLICY_STATUS__STRUCT_HPP_

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
# define DEPRECATED__hippocampus_ros2_msgs__msg__PolicyStatus __attribute__((deprecated))
#else
# define DEPRECATED__hippocampus_ros2_msgs__msg__PolicyStatus __declspec(deprecated)
#endif

namespace hippocampus_ros2_msgs
{

namespace msg
{

// message struct
template<class ContainerAllocator>
struct PolicyStatus_
{
  using Type = PolicyStatus_<ContainerAllocator>;

  explicit PolicyStatus_(rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  : stamp(_init)
  {
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      this->is_active = false;
      this->graph_stale = false;
      this->using_snn = false;
      this->hierarchical_enabled = false;
      this->feature_compute_time_ms = 0.0;
      this->policy_decision_time_ms = 0.0;
      this->safety_filter_time_ms = 0.0;
      this->total_latency_ms = 0.0;
      this->graph_nodes = 0ul;
      this->graph_edges = 0ul;
      this->graph_staleness_s = 0.0;
      this->current_confidence = 0.0;
      this->current_reason = "";
      this->current_waypoint = 0l;
    }
  }

  explicit PolicyStatus_(const ContainerAllocator & _alloc, rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  : current_reason(_alloc),
    stamp(_alloc, _init)
  {
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      this->is_active = false;
      this->graph_stale = false;
      this->using_snn = false;
      this->hierarchical_enabled = false;
      this->feature_compute_time_ms = 0.0;
      this->policy_decision_time_ms = 0.0;
      this->safety_filter_time_ms = 0.0;
      this->total_latency_ms = 0.0;
      this->graph_nodes = 0ul;
      this->graph_edges = 0ul;
      this->graph_staleness_s = 0.0;
      this->current_confidence = 0.0;
      this->current_reason = "";
      this->current_waypoint = 0l;
    }
  }

  // field types and members
  using _is_active_type =
    bool;
  _is_active_type is_active;
  using _graph_stale_type =
    bool;
  _graph_stale_type graph_stale;
  using _using_snn_type =
    bool;
  _using_snn_type using_snn;
  using _hierarchical_enabled_type =
    bool;
  _hierarchical_enabled_type hierarchical_enabled;
  using _feature_compute_time_ms_type =
    double;
  _feature_compute_time_ms_type feature_compute_time_ms;
  using _policy_decision_time_ms_type =
    double;
  _policy_decision_time_ms_type policy_decision_time_ms;
  using _safety_filter_time_ms_type =
    double;
  _safety_filter_time_ms_type safety_filter_time_ms;
  using _total_latency_ms_type =
    double;
  _total_latency_ms_type total_latency_ms;
  using _graph_nodes_type =
    uint32_t;
  _graph_nodes_type graph_nodes;
  using _graph_edges_type =
    uint32_t;
  _graph_edges_type graph_edges;
  using _graph_staleness_s_type =
    double;
  _graph_staleness_s_type graph_staleness_s;
  using _current_confidence_type =
    double;
  _current_confidence_type current_confidence;
  using _current_reason_type =
    std::basic_string<char, std::char_traits<char>, typename std::allocator_traits<ContainerAllocator>::template rebind_alloc<char>>;
  _current_reason_type current_reason;
  using _current_waypoint_type =
    int32_t;
  _current_waypoint_type current_waypoint;
  using _stamp_type =
    builtin_interfaces::msg::Time_<ContainerAllocator>;
  _stamp_type stamp;

  // setters for named parameter idiom
  Type & set__is_active(
    const bool & _arg)
  {
    this->is_active = _arg;
    return *this;
  }
  Type & set__graph_stale(
    const bool & _arg)
  {
    this->graph_stale = _arg;
    return *this;
  }
  Type & set__using_snn(
    const bool & _arg)
  {
    this->using_snn = _arg;
    return *this;
  }
  Type & set__hierarchical_enabled(
    const bool & _arg)
  {
    this->hierarchical_enabled = _arg;
    return *this;
  }
  Type & set__feature_compute_time_ms(
    const double & _arg)
  {
    this->feature_compute_time_ms = _arg;
    return *this;
  }
  Type & set__policy_decision_time_ms(
    const double & _arg)
  {
    this->policy_decision_time_ms = _arg;
    return *this;
  }
  Type & set__safety_filter_time_ms(
    const double & _arg)
  {
    this->safety_filter_time_ms = _arg;
    return *this;
  }
  Type & set__total_latency_ms(
    const double & _arg)
  {
    this->total_latency_ms = _arg;
    return *this;
  }
  Type & set__graph_nodes(
    const uint32_t & _arg)
  {
    this->graph_nodes = _arg;
    return *this;
  }
  Type & set__graph_edges(
    const uint32_t & _arg)
  {
    this->graph_edges = _arg;
    return *this;
  }
  Type & set__graph_staleness_s(
    const double & _arg)
  {
    this->graph_staleness_s = _arg;
    return *this;
  }
  Type & set__current_confidence(
    const double & _arg)
  {
    this->current_confidence = _arg;
    return *this;
  }
  Type & set__current_reason(
    const std::basic_string<char, std::char_traits<char>, typename std::allocator_traits<ContainerAllocator>::template rebind_alloc<char>> & _arg)
  {
    this->current_reason = _arg;
    return *this;
  }
  Type & set__current_waypoint(
    const int32_t & _arg)
  {
    this->current_waypoint = _arg;
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
    hippocampus_ros2_msgs::msg::PolicyStatus_<ContainerAllocator> *;
  using ConstRawPtr =
    const hippocampus_ros2_msgs::msg::PolicyStatus_<ContainerAllocator> *;
  using SharedPtr =
    std::shared_ptr<hippocampus_ros2_msgs::msg::PolicyStatus_<ContainerAllocator>>;
  using ConstSharedPtr =
    std::shared_ptr<hippocampus_ros2_msgs::msg::PolicyStatus_<ContainerAllocator> const>;

  template<typename Deleter = std::default_delete<
      hippocampus_ros2_msgs::msg::PolicyStatus_<ContainerAllocator>>>
  using UniquePtrWithDeleter =
    std::unique_ptr<hippocampus_ros2_msgs::msg::PolicyStatus_<ContainerAllocator>, Deleter>;

  using UniquePtr = UniquePtrWithDeleter<>;

  template<typename Deleter = std::default_delete<
      hippocampus_ros2_msgs::msg::PolicyStatus_<ContainerAllocator>>>
  using ConstUniquePtrWithDeleter =
    std::unique_ptr<hippocampus_ros2_msgs::msg::PolicyStatus_<ContainerAllocator> const, Deleter>;
  using ConstUniquePtr = ConstUniquePtrWithDeleter<>;

  using WeakPtr =
    std::weak_ptr<hippocampus_ros2_msgs::msg::PolicyStatus_<ContainerAllocator>>;
  using ConstWeakPtr =
    std::weak_ptr<hippocampus_ros2_msgs::msg::PolicyStatus_<ContainerAllocator> const>;

  // pointer types similar to ROS 1, use SharedPtr / ConstSharedPtr instead
  // NOTE: Can't use 'using' here because GNU C++ can't parse attributes properly
  typedef DEPRECATED__hippocampus_ros2_msgs__msg__PolicyStatus
    std::shared_ptr<hippocampus_ros2_msgs::msg::PolicyStatus_<ContainerAllocator>>
    Ptr;
  typedef DEPRECATED__hippocampus_ros2_msgs__msg__PolicyStatus
    std::shared_ptr<hippocampus_ros2_msgs::msg::PolicyStatus_<ContainerAllocator> const>
    ConstPtr;

  // comparison operators
  bool operator==(const PolicyStatus_ & other) const
  {
    if (this->is_active != other.is_active) {
      return false;
    }
    if (this->graph_stale != other.graph_stale) {
      return false;
    }
    if (this->using_snn != other.using_snn) {
      return false;
    }
    if (this->hierarchical_enabled != other.hierarchical_enabled) {
      return false;
    }
    if (this->feature_compute_time_ms != other.feature_compute_time_ms) {
      return false;
    }
    if (this->policy_decision_time_ms != other.policy_decision_time_ms) {
      return false;
    }
    if (this->safety_filter_time_ms != other.safety_filter_time_ms) {
      return false;
    }
    if (this->total_latency_ms != other.total_latency_ms) {
      return false;
    }
    if (this->graph_nodes != other.graph_nodes) {
      return false;
    }
    if (this->graph_edges != other.graph_edges) {
      return false;
    }
    if (this->graph_staleness_s != other.graph_staleness_s) {
      return false;
    }
    if (this->current_confidence != other.current_confidence) {
      return false;
    }
    if (this->current_reason != other.current_reason) {
      return false;
    }
    if (this->current_waypoint != other.current_waypoint) {
      return false;
    }
    if (this->stamp != other.stamp) {
      return false;
    }
    return true;
  }
  bool operator!=(const PolicyStatus_ & other) const
  {
    return !this->operator==(other);
  }
};  // struct PolicyStatus_

// alias to use template instance with default allocator
using PolicyStatus =
  hippocampus_ros2_msgs::msg::PolicyStatus_<std::allocator<void>>;

// constant definitions

}  // namespace msg

}  // namespace hippocampus_ros2_msgs

#endif  // HIPPOCAMPUS_ROS2_MSGS__MSG__DETAIL__POLICY_STATUS__STRUCT_HPP_
