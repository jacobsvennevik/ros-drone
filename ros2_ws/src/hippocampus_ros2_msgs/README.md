# hippocampus_ros2_msgs

ROS 2 message definitions for the hippocampus_ros2 policy system.

## Messages

### PolicyDecision.msg
Policy decision made by the SNN Policy Service.
- Action proposal (linear_x, angular_z, linear_z)
- Confidence score
- Decision reason
- Next waypoint ID

### PolicyStatus.msg
Diagnostics and status information for the policy system.
- Status flags (active, stale, SNN mode, hierarchical)
- Performance metrics (latency breakdown)
- Graph information
- Current state

### MissionGoal.msg
Navigation goal for the policy system.
- Goal types: POINT, NODE, REGION
- Position/tolerance for point goals
- Node ID for node goals
- Region center/radius for region goals

### GraphSnapshot.msg
Snapshot of the topological graph.
- Graph metadata (epoch, frame, timestamps)
- Node list (with positions, normals, degrees)
- Edge list (with lengths, traversability)

## Building

```bash
cd ros2_ws
colcon build --packages-select hippocampus_ros2_msgs
source install/setup.bash
```

## Usage

After building, import in Python:
```python
from hippocampus_ros2_msgs.msg import PolicyDecision, PolicyStatus, MissionGoal, GraphSnapshot
```

