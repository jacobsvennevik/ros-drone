# Minimal Gazebo System Test

## Why this exists

This setup mirrors the [Nav2 system tests](https://github.com/ros-navigation/navigation2/tree/main/nav2_system_tests) philosophy: boot a headless Gazebo world, bring up the control node under test, and watch for basic topic activity. It gives a fast smoke test that the `snn_brain_node` still produces bounded `Twist` outputs when fed a plausible pose stream, without spinning up the larger Nav2 bringup stack.

For more on how the Nav2 team keeps these tests running in CI, see the [continuous simulation article on docs.nav2.org](https://docs.nav2.org/tutorials/docs/continuous_simulation/continuous_simulation.html).

## How to run

- Build your workspace (`colcon build --merge-install`) and source it.
- Launch the test:

  ```
  ros2 launch hippocampus_ros2 brain_system_smoke.launch.py
  ```

- Optional arguments mirror the assertions (e.g. `timeout:=15.0`, `min_messages:=3`, `max_linear:=0.3`, `max_angular:=1.0`). Pass `gui:=true` locally if you want to open a Gazebo window.

The launch file starts Gazebo with `system_tests/worlds/minimal.world`, a deterministic `/odom` publisher (`system_tests/scripts/pose_publisher.py`), the standard `brain.launch.py`, and the assertion helper described below.

## Reading `assert_topics.py`

`system_tests/scripts/assert_topics.py` subscribes to `/snn_action` (`Float32MultiArray`) and `/cmd_vel` (`Twist`). Within the configured timeout (default 20 s) it needs to see at least `min_messages` on each topic, and every `Twist` must respect the configured `max_linear` and `max_angular` clamps. On success it exits with code 0 and logs:

```
[INFO] Topic assertions satisfied.
```

Failures exit with code 1, print the first offending condition, and the launch shuts down. You can run the checker by itself (after starting the simulation) with:

```
python3 $(ros2 pkg prefix hippocampus_ros2)/share/hippocampus_ros2/system_tests/scripts/assert_topics.py
```

## QoS & stability notes

All publishers and subscriptions currently use the ROS 2 default QoS (`reliability=reliable`, `durability=volatile`, depth = 10, keep-last). This keeps the smoke test deterministic and aligned with the node’s defaults. If you start testing high-rate or lossy transports, consider relaxing to best-effort (see the [ROS 2 QoS overview](https://docs.ros.org/en/humble/Concepts/About-Quality-of-Service-Settings.html)) but keep the same depth so the assertions remain predictable.



