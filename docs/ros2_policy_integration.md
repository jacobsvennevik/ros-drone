# ROS 2 Policy Integration

This document describes the ROS 2 integration of the SNN Policy Service.

## Overview

The policy system is integrated into ROS 2 through the `PolicyNode`, which:
- Subscribes to `/odom` (robot pose)
- Publishes `/cmd_vel` (velocity commands)
- Uses the `SpikingPolicyService` for decision-making
- Supports hierarchical planning with `GraphNavigationService`
- Supports both 2D and 3D operation

## Architecture

```
ROS Topics
    ↓
PolicyNode (ROS 2 Node)
    ↓
SpikingPolicyService (SNNController interface)
    ↓
SpatialFeatureService → TopologyService → PlaceCellController
    ↓
ActionArbitrationSafety
    ↓
/cmd_vel (Twist)
```

## Usage

### Launch the Policy Node

```bash
cd ros2_ws
colcon build --packages-select hippocampus_ros2
source install/setup.bash
ros2 launch hippocampus_ros2 policy.launch.py
```

### With Custom Parameters

```bash
ros2 launch hippocampus_ros2 policy.launch.py \
    pose_topic:=/sim/odom \
    cmd_vel_topic:=/sim/cmd_vel \
    enable_hierarchical:=true \
    use_snn:=true \
    is_3d:=true
```

### Configuration File

Edit `config/policy.yaml` to customize:
- Control rates
- Velocity limits
- SNN model paths
- Navigation algorithms
- 3D mode

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `pose_topic` | string | `/odom` | Input odometry topic |
| `cmd_vel_topic` | string | `/cmd_vel` | Output velocity command topic |
| `mission_topic` | string | `/mission/goal` | Mission goal topic (optional) |
| `control_rate` | double | `10.0` | Control loop frequency (Hz) |
| `max_linear` | double | `0.3` | Max linear velocity (m/s) |
| `max_angular` | double | `1.0` | Max angular velocity (rad/s) |
| `max_vertical` | double | `0.2` | Max vertical velocity (m/s) for 3D |
| `enable_hierarchical` | bool | `false` | Enable graph navigation |
| `navigation_algorithm` | string | `dijkstra` | Path planning algorithm |
| `use_snn` | bool | `false` | Use SNN model (requires PyTorch) |
| `snn_model_path` | string | `""` | Path to SNN checkpoint |
| `is_3d` | bool | `false` | Enable 3D mode |

## Topics

### Subscribed Topics

- `/odom` (`nav_msgs/msg/Odometry`) - Robot pose/odometry
- `/mission/goal` (optional) - Mission goal updates

### Published Topics

- `/cmd_vel` (`geometry_msgs/msg/Twist`) - Velocity commands
- `/policy_action` (`std_msgs/msg/Float32MultiArray`) - Raw policy actions

## Integration with Existing BrainNode

The `PolicyNode` is separate from `BrainNode` but follows the same pattern:
- Both use `SNNController` interface
- Both subscribe to `/odom` and publish `/cmd_vel`
- `PolicyNode` adds policy-specific features (goals, navigation, safety)

You can run either:
- `BrainNode` - Simple controller (PlaceCellController or SnnTorchController)
- `PolicyNode` - Full policy system with features, navigation, safety

## Testing

### Smoke Test

1. Launch the node:
   ```bash
   ros2 launch hippocampus_ros2 policy.launch.py
   ```

2. Publish fake odometry (in another terminal):
   ```bash
   ros2 topic pub /odom nav_msgs/msg/Odometry "{pose: {pose: {position: {x: 0.5, y: 0.5}}}}" -r 10
   ```

3. Check outputs:
   ```bash
   ros2 topic echo /cmd_vel
   ros2 topic echo /policy_action
   ```

### With Simulator

```bash
# Terminal 1: Launch simulator
ros2 launch <simulator_package> <simulator>.launch.py

# Terminal 2: Launch policy node
ros2 launch hippocampus_ros2 policy.launch.py

# Terminal 3: Monitor topics
ros2 topic hz /cmd_vel
```

## Next Steps

- [ ] Add ROS 2 message types for `PolicyDecision` and `PolicyStatus`
- [ ] Add mission goal publisher/subscriber
- [ ] Add graph visualization topics
- [ ] Add diagnostics topics
- [ ] Integrate with sensor topics (LiDAR, depth cameras)

## See Also

- `ros2_ws/src/hippocampus_ros2/hippocampus_ros2/nodes/policy_node.py` - Node implementation
- `ros2_ws/src/hippocampus_ros2/launch/policy.launch.py` - Launch file
- `ros2_ws/src/hippocampus_ros2/config/policy.yaml` - Configuration
- `docs/snn_policy_architecture/` - Policy architecture documentation

