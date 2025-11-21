# Build Instructions

## 🍎 macOS Users: Use Docker!

Since you're on macOS, ROS 2 doesn't run natively. Use the Docker helper:

```bash
# From project root
./scripts/ros2_docker.sh build
```

Or open an interactive shell:
```bash
./scripts/ros2_docker.sh shell
```

See `docs/ROS2_MACOS_INSTALL.md` for full instructions.

---

## Linux Users: Native Build

You're already in `ros2_ws`, so just run:

```bash
# 1. Source ROS 2 (if not already done)
source /opt/ros/<your-distro>/setup.bash

# 2. Build message package first
colcon build --packages-select hippocampus_ros2_msgs

# 3. Source the workspace
source install/setup.bash

# 4. Build the main package
colcon build --packages-select hippocampus_ros2

# 5. Source again (to get both packages)
source install/setup.bash
```

## One-liner (if ROS 2 is already sourced)

```bash
colcon build --packages-select hippocampus_ros2_msgs && source install/setup.bash && colcon build --packages-select hippocampus_ros2 && source install/setup.bash
```

## Verify Build

```bash
# Check that packages were built
ls install/hippocampus_ros2_msgs
ls install/hippocampus_ros2

# Check message types
ros2 interface list | grep hippocampus
```

## Common Issues

### "colcon: command not found"
- You need to source ROS 2 first:
  ```bash
  source /opt/ros/humble/setup.bash  # or your distro
  ```

### "Package not found"
- Make sure you're in `ros2_ws` directory
- Check that `src/hippocampus_ros2_msgs` and `src/hippocampus_ros2` exist

### Build errors
- Check that all dependencies are installed
- Make sure ROS 2 is properly sourced

