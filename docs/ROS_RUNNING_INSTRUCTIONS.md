# ROS 2 Running Instructions

## ✅ Integration Status

All code is ready and validated:
- ✅ Message files created and validated
- ✅ Node syntax validated
- ✅ Launch files created
- ✅ Package structure complete
- ✅ All sanity checks passing

## Prerequisites

To actually run the ROS 2 nodes, you need:

1. **ROS 2 installed** (Humble, Jazzy, or later)
2. **ROS 2 sourced** in your terminal:
   ```bash
   source /opt/ros/<distro>/setup.bash
   ```
3. **Dependencies installed**:
   ```bash
   pip install numpy
   # ROS 2 dependencies should be installed with ROS 2
   ```

## Building

```bash
cd ros2_ws

# Build message package first
colcon build --packages-select hippocampus_ros2_msgs

# Source the workspace
source install/setup.bash

# Build the main package
colcon build --packages-select hippocampus_ros2

# Source again (to get both packages)
source install/setup.bash
```

## Running

### 1. Policy Node

```bash
# Terminal 1: Launch policy node
ros2 launch hippocampus_ros2 policy.launch.py

# Terminal 2: Publish fake odometry
ros2 topic pub /odom nav_msgs/msg/Odometry \
  "{pose: {pose: {position: {x: 0.5, y: 0.5}, orientation: {w: 1.0}}}}" -r 10

# Terminal 3: Check outputs
ros2 topic echo /cmd_vel
ros2 topic echo /policy/decision
ros2 topic echo /policy/status
```

### 2. Mission Publisher

```bash
# Publish a point goal
ros2 launch hippocampus_ros2 mission_publisher.launch.py \
  goal_type:=point \
  goal_x:=10.0 \
  goal_y:=10.0

# Or publish a node goal
ros2 launch hippocampus_ros2 mission_publisher.launch.py \
  goal_type:=node \
  node_id:=5
```

### 3. With Visualization

```bash
# Launch policy node with visualization
ros2 launch hippocampus_ros2 policy.launch.py \
  enable_viz:=true \
  viz_rate_hz:=2.0

# In RViz:
# - Add MarkerArray display for /policy/graph
# - Add MarkerArray display for /policy/waypoint
# - Set Fixed Frame to "map"
```

## Verification

### Check Topics

```bash
# List all topics
ros2 topic list

# Should see:
# /cmd_vel
# /odom
# /policy/decision
# /policy/status
# /policy/graph (if viz enabled)
# /policy/waypoint (if viz enabled)
# /mission/goal (if mission publisher running)
```

### Check Messages

```bash
# View policy decision
ros2 topic echo /policy/decision

# View policy status
ros2 topic echo /policy/status

# Check message types
ros2 interface show hippocampus_ros2_msgs/msg/PolicyDecision
```

### Check Node Status

```bash
# List nodes
ros2 node list

# Should see:
# /policy_node
# /mission_publisher (if running)
```

## Troubleshooting

### "Package not found"
```bash
# Make sure you've built and sourced:
cd ros2_ws
colcon build
source install/setup.bash
```

### "Message type not found"
```bash
# Build message package first:
colcon build --packages-select hippocampus_ros2_msgs
source install/setup.bash
```

### "rclpy not found"
```bash
# Source ROS 2:
source /opt/ros/<distro>/setup.bash
```

### "No module named numpy"
```bash
# Install dependencies:
pip install numpy
```

## Expected Behavior

1. **Policy Node**:
   - Subscribes to `/odom`
   - Publishes `/cmd_vel` with velocity commands
   - Publishes `/policy/decision` with policy decisions
   - Publishes `/policy/status` with diagnostics
   - Publishes `/policy/graph` if visualization enabled

2. **Mission Publisher**:
   - Publishes `/mission/goal` with goal information
   - Can publish once or periodically

3. **Integration**:
   - Policy node receives goals from `/mission/goal`
   - Policy makes decisions based on current pose and goal
   - Commands are published to `/cmd_vel`

## Next Steps

Once ROS 2 is properly set up:
1. Build the packages
2. Run the nodes
3. Verify topics are publishing
4. Test with a simulator or real robot

---

**Status**: Code Ready ✅  
**Requires**: ROS 2 installation and setup

