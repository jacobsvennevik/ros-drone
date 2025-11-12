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

