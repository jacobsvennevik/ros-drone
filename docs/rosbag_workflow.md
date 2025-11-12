# ROS 2 rosbag2 Workflow

This guide captures the minimal record/replay workflow for the hippocampus brain node. It uses the ROS 2 [`ros2 bag`](https://docs.ros.org/en/humble/Tutorials/Intermediate/Rosbag2/ros2bag.html) CLI and assumes your workspace has been built and sourced.

## Record a Run

1. Source your workspace:
   ```bash
   source ros2_ws/install/setup.bash
   ```
2. Start the brain node (via `ros2 launch hippocampus_ros2 brain.launch.py`, simulator, etc.).
3. In a second terminal, record the core topics:
   ```bash
   ros2 bag record -o bags/brain_run /odom /snn_action /cmd_vel /brain_markers
   ```
   - `-o bags/brain_run` creates a timestamped bag inside `bags/brain_run/`.
   - The topic list matches the pose input, brain output, velocity command, and RViz markers.

Stop recording with `Ctrl+C` when the experiment is finished.

## Inspect a Bag

```bash
ros2 bag info bags/brain_run
```

This reports duration, storage type, message counts, and QoS profiles. Use this to verify that all expected topics were captured.

## Replay Odometry

Replay publishes the recorded topics exactly as they were captured. To drive the brain node deterministically:

1. Launch the brain node (optionally with `use_bag_replay:=true` and `use_sim_time:=true`):
   ```bash
   ros2 launch hippocampus_ros2 brain.launch.py use_bag_replay:=true use_sim_time:=true
   ```
2. In another terminal, play the bag:
   ```bash
   ros2 bag play bags/brain_run
   ```

While replay is running, `/odom` is published from the bag; the brain node processes those messages and republishes `/snn_action`, `/cmd_vel`, and `/brain_markers`.

## Notes

- Bags can be replayed multiple times; use `--loop` to run continuously.
- To remap topics during replay, append `--remap /odom:=/new/odom/topic`.
- Keep bag directories under version control ignore rules if they are large.

