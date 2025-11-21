# Next Development Steps

## Overview

This document outlines potential next steps for the SNN Policy System, organized by priority and effort.

## High Priority (Immediate Value)

### 1. ROS 2 Message Types ⭐
**Status**: Not Started  
**Effort**: 2-3 hours  
**Priority**: High

Create custom ROS 2 message types for better integration:
- `PolicyDecision.msg` - Policy decision messages
- `PolicyStatus.msg` - Policy diagnostics/status
- `MissionGoal.msg` - Mission goal representation
- `GraphSnapshot.msg` - Graph snapshot for visualization

**Benefits**:
- Better type safety
- Easier debugging with `ros2 topic echo`
- Standard ROS 2 patterns
- Enables visualization tools

**Files to create**:
- `ros2_ws/src/hippocampus_ros2_msgs/msg/PolicyDecision.msg`
- `ros2_ws/src/hippocampus_ros2_msgs/msg/PolicyStatus.msg`
- `ros2_ws/src/hippocampus_ros2_msgs/msg/MissionGoal.msg`

### 2. Mission Goal Publisher/Subscriber ⭐
**Status**: Not Started  
**Effort**: 1-2 hours  
**Priority**: High

Create ROS 2 node to publish/subscribe mission goals:
- Subscribe to `/mission/goal` (or publish from external planner)
- Convert between ROS messages and `Mission` data structures
- Support dynamic goal updates

**Benefits**:
- Enables integration with mission planners
- Dynamic goal updates during flight
- Standard ROS 2 communication

### 3. Graph Visualization Topics ⭐
**Status**: Not Started  
**Effort**: 2-3 hours  
**Priority**: Medium-High

Publish graph visualization for RViz:
- `/topo_graph` - Graph structure (nodes, edges)
- `/policy/waypoint` - Current waypoint visualization
- `/policy/path` - Planned path visualization

**Benefits**:
- Visual debugging
- Real-time graph inspection
- Path planning visualization

## Medium Priority (Enhanced Functionality)

### 4. Sensor Integration
**Status**: Not Started  
**Effort**: 4-6 hours  
**Priority**: Medium

Integrate sensor data into safety features:
- Subscribe to `/scan` (LiDAR) or `/depth` (depth camera)
- Update `compute_safety_features()` with real obstacle data
- Obstacle detection and avoidance

**Benefits**:
- Real obstacle avoidance
- Better safety features
- Real-world applicability

### 5. Training Pipeline Integration
**Status**: Not Started  
**Effort**: 6-8 hours  
**Priority**: Medium

Create utilities for training data collection:
- Record policy decisions and features
- Export training datasets
- Integration with existing training scripts

**Benefits**:
- Easier model training
- Data collection automation
- Reproducible experiments

### 6. Performance Monitoring & Diagnostics
**Status**: Not Started  
**Effort**: 3-4 hours  
**Priority**: Medium

Add performance monitoring:
- Latency measurements per component
- Feature computation time
- Policy decision time
- Safety filtering time
- Publish diagnostics topic

**Benefits**:
- Performance optimization insights
- Real-time monitoring
- Bottleneck identification

## Lower Priority (Nice to Have)

### 7. Advanced Mission Types
**Status**: Framework Ready  
**Effort**: 4-6 hours  
**Priority**: Low

Implement additional goal types:
- `RegionGoal` - Navigate to region
- `SequentialGoal` - Multiple waypoints
- `ExploreGoal` - Exploration behavior

**Benefits**:
- More flexible mission planning
- Complex navigation scenarios

### 8. Advanced Safety Features
**Status**: Framework Ready  
**Effort**: 3-4 hours  
**Priority**: Low

Enhance safety system:
- Geofence enforcement
- No-fly zone checking
- Altitude limits (3D)
- Emergency stop triggers

**Benefits**:
- Enhanced safety
- Regulatory compliance
- Real-world deployment readiness

### 9. Multi-Robot Support
**Status**: Not Started  
**Effort**: 8-10 hours  
**Priority**: Low

Support multiple robots:
- Namespace-based topics
- Shared graph updates
- Collision avoidance between robots

**Benefits**:
- Swarm capabilities
- Multi-robot coordination

### 10. ROS 2 Services
**Status**: Not Started  
**Effort**: 2-3 hours  
**Priority**: Low

Add ROS 2 services:
- `~/set_goal` - Set mission goal
- `~/get_status` - Get policy status
- `~/reset` - Reset policy state

**Benefits**:
- Request-response patterns
- Better integration with planners

## Quick Wins (1-2 hours each)

### 11. Enhanced Logging
- Structured logging with policy decisions
- Feature vector logging
- Performance metrics logging

### 12. Configuration Validation
- Validate YAML configs at startup
- Parameter range checking
- Helpful error messages

### 13. Example Scripts
- Simple goal publisher script
- Graph visualization script
- Performance profiling script

### 14. Documentation Improvements
- API reference
- Tutorial notebooks
- Video demonstrations

## Recommended Order

1. **ROS 2 Message Types** (High value, low effort)
2. **Mission Goal Publisher** (Enables dynamic goals)
3. **Graph Visualization** (Debugging aid)
4. **Sensor Integration** (Real-world applicability)
5. **Performance Monitoring** (Optimization insights)

## Implementation Notes

### ROS 2 Message Types
```bash
# Create message package
cd ros2_ws/src
ros2 pkg create --build-type ament_cmake hippocampus_ros2_msgs
# Add .msg files
# Update CMakeLists.txt and package.xml
```

### Mission Goal Publisher
- Simple node that publishes goals
- Can be triggered by external planner
- Supports all goal types

### Graph Visualization
- Use `visualization_msgs/MarkerArray`
- Similar to existing `BrainNode` visualization
- Add path markers

## Estimated Total Effort

- **High Priority**: ~8-12 hours
- **Medium Priority**: ~13-18 hours
- **Lower Priority**: ~17-23 hours
- **Quick Wins**: ~6-8 hours

**Total**: ~44-61 hours of development

---

**Last Updated**: 2025-01-27  
**Status**: Ready for implementation

