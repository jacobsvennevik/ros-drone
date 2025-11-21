# ROS Smoke Test Guide

This document summarizes quick manual checks for the hippocampus ROS 2 node. Perform these after building and sourcing `ros2_ws`.

## Live Odometry Feed

1. Launch the brain node:
   ```bash
   ros2 launch hippocampus_ros2 brain.launch.py
   ```
2. Provide odometry (e.g., simulator, `fake_odom.py`, or hardware).
3. Verify outputs:
   ```bash
   ros2 topic echo /snn_action
   ros2 topic echo /cmd_vel
   ```

## Replay Mode (rosbag2)

1. Start the brain node in replay mode:
   ```bash
   ros2 launch hippocampus_ros2 brain.launch.py use_bag_replay:=true use_sim_time:=true
   ```
2. In another terminal, replay an odometry bag:
   ```bash
   ros2 bag play <bag_directory>
   ```
   - The bag should publish `/odom`; the node will regenerate `/snn_action`, `/cmd_vel`, and `/brain_markers`.
3. QA checkpoints:
   - `ros2 topic echo /snn_action` updates while the bag is running.
   - `ros2 topic echo /cmd_vel` shows bounded Twist commands.
   - (Optional) Inspect `/brain_markers` in RViz if visualization is enabled.

See `docs/rosbag_workflow.md` for details on recording and inspecting bags.



