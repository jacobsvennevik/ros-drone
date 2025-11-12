# hippocampus_ros2 ROS 2 Package

Python (`ament_python`) package that exposes hippocampal controllers as ROS 2 nodes. The package lives inside a standard overlay workspace:

```
ros2_ws/
  src/
    hippocampus_ros2/
      package.xml
      setup.py
      hippocampus_ros2/
        nodes/
          brain_node.py
```

The nodes follow the [“Writing a simple publisher and subscriber” rclpy tutorial](https://docs.ros.org/en/humble/Tutorials/Intermediate/Writing-A-Simple-Py-Publisher-And-Subscriber.html) structure: create a `Node` subclass, declare parameters, and spin with an executor.

## Build & Run

```bash
cd ros2_ws
colcon build --packages-select hippocampus_ros2
source install/setup.bash
ros2 run hippocampus_ros2 snn_brain_node
```

Make sure the accompanying core library is available in the same environment, e.g. from the repository root:

```bash
pip install -e .[dev]
```

The node imports `hippocampus_core` and steps a selected `SNNController` when odometry messages arrive.

## Launch Workflow

Launch the node with its default parameter file:

```bash
ros2 launch hippocampus_ros2 brain.launch.py
```

Override the parameter file or individual topics without editing source:

```bash
ros2 launch hippocampus_ros2 brain.launch.py params_file:=/path/to/custom.yaml
ros2 launch hippocampus_ros2 brain.launch.py pose_topic:=/sim/odom cmd_vel_topic:=/sim/cmd_vel
ros2 launch hippocampus_ros2 brain.launch.py use_bag_replay:=true
```

`brain.launch.py` declares `use_sim_time`, `use_bag_replay`, `params_file`, `pose_topic`, and `cmd_vel_topic` launch arguments, making it easy to swap configurations for system tests, Gazebo runs, or CI bring-up without touching code. The default configuration lives in `config/brain.yaml` and documents each runtime parameter.

## Parameters

| Name | Type | Default | Description |
| --- | --- | --- | --- |
| `pose_topic` | `string` | `/odom` | Input `nav_msgs/msg/Odometry` topic. |
| `cmd_vel_topic` | `string` | `/cmd_vel` | Output `geometry_msgs/msg/Twist` topic. |
| `control_rate` | `double` | `10.0` | Timer frequency in Hz. |
| `max_linear` | `double` | `0.3` | Clamp for `Twist.linear.x` (m/s). |
| `max_angular` | `double` | `1.0` | Clamp for `Twist.angular.z` (rad/s). |
| `log_every_n_cycles` | `int` | `10` | Diagnostics logging interval. |
| `controller_backend` | `string` | `place_cells` | Controller implementation (`place_cells`, `snntorch`, etc.). |
| `model_path` | `string` | `""` | Optional path to a trained model for neural backends. |
| `enable_viz` | `bool` | `false` | Toggle publication of RViz `MarkerArray` overlays. |
| `viz_rate_hz` | `double` | `2.0` | Marker publication frequency. |
| `viz_frame_id` | `string` | `map` | Frame used for RViz markers (set to your global frame). |
| `viz_trail_length` | `int` | `200` | Number of recent poses to retain in the trajectory line strip. |

The node publishes `geometry_msgs/msg/Twist` messages whose meaningful fields are `linear.x` (forward velocity) and `angular.z` (yaw rate). Other components remain zeroed.

## Topic Inspection Cheat-Sheet

Inspect and debug topics with the ROS 2 CLI ([Understanding ROS 2 topics](https://docs.ros.org/en/humble/Tutorials/Beginner-CLI-Tools/Understanding-ROS2-Topics/Understanding-ROS2-Topics.html)):

```bash
ros2 topic list
ros2 topic echo /snn_action
ros2 topic echo /cmd_vel
```

## Developer Smoke Test

1. Build and source the workspace (as above).
2. Run the node in one terminal:
   ```bash
   ros2 run hippocampus_ros2 snn_brain_node
   ```
3. In a second terminal, publish fake odometry using a small rclpy script patterned after the tutorial linked earlier. Minimal example sketch:
   ```python
   import math
   import rclpy
   from rclpy.node import Node
   from nav_msgs.msg import Odometry

   class FakeOdomPublisher(Node):
       def __init__(self):
           super().__init__("fake_odom_publisher")
           self.pub = self.create_publisher(Odometry, "/odom", 10)
           self.timer = self.create_timer(0.1, self.tick)
           self.t = 0.0

       def tick(self):
           msg = Odometry()
           msg.pose.pose.position.x = 0.5 * math.cos(self.t)
           msg.pose.pose.position.y = 0.5 * math.sin(self.t)
           self.pub.publish(msg)
           self.t += 0.1

   rclpy.init()
   node = FakeOdomPublisher()
   rclpy.spin(node)
   ```
   Save it as `fake_odom.py`, add execution permissions, and run with `ros2 run` or `python fake_odom.py` inside the sourced environment.
4. Watch the outputs:
   ```bash
   ros2 topic echo /snn_action
   ros2 topic echo /cmd_vel
   ```
   Confirm that `/snn_action` and `/cmd_vel` messages appear while odometry is being published.

Optional helper scripts or launch files can live under `hippocampus_ros2/scripts/` for automating this smoke test; keep them dependency-light (`rclpy` only).

## Visualizing in RViz

1. Launch RViz and set **Global Options → Fixed Frame** to match `viz_frame_id` (default `map`).
2. Add a **Marker** display targeting `/brain_markers`.
3. Enable the `pc_centers`, `pc_fields`, and `graph_edges` namespaces to view place-cell positions, field radii, and the current topological graph. If `enable_viz` is true, a red `agent_trail` line strip traces recent poses.

The marker layout follows the [RViz marker display types tutorial](https://docs.ros.org/en/kilted/Tutorials/Intermediate/RViz/Marker-Display-types/Marker-Display-types.html) and the [`visualization_msgs/msg/Marker`](https://docs.ros.org/en/noetic/api/visualization_msgs/html/msg/Marker.html) / [`MarkerArray`](http://docs.ros.org/en/noetic/api/visualization_msgs/html/msg/MarkerArray.html) message specifications. For extra context on common marker usage patterns, see the community discussion on MarkerArray techniques ([Robotics Stack Exchange thread](https://robotics.stackexchange.com/questions/98648/methods-of-visualization-msgs-markerarray)).

### rosbag2 Replay (for future deterministic tests)

To capture and replay deterministic streams during debugging or CI, record topics into a bag:

```bash
ros2 bag record /odom /snn_action /cmd_vel /brain_markers
```

and play them back later with:

```bash
ros2 bag play <bag_directory>
```

See the [rosbag2 tutorials](https://docs.ros.org/en/humble/Tutorials/Intermediate/Rosbag2/rosbag2_cli.html) for more detail on recording, replay, and filtering. Helper scripts under `scripts/` provide quick wrappers:

- `record_brain_topics.sh [output_prefix] [topics...]`
- `replay_odom.sh <bag_directory> [ros2 bag play args...]`

For an end-to-end walk-through (record, inspect with `ros2 bag info`, replay with launch integration), refer to `docs/rosbag_workflow.md`.

