# Quick Start: ROS 2 on macOS

## Step 1: Install ROS 2 (Docker)

Since you're on macOS, we'll use Docker:

```bash
# Pull ROS 2 image (first time only, ~2GB download)
docker pull osrf/ros:humble-desktop-full
```

## Step 2: Build Your Packages

```bash
# From project root
./scripts/ros2_docker.sh build
```

This will:
- Create a Docker container with ROS 2
- Mount your workspace
- Build your packages

## Step 3: Open ROS 2 Shell

```bash
./scripts/ros2_docker.sh shell
```

You'll be in a bash shell inside the container with:
- ROS 2 sourced automatically
- Your workspace at `/home/ros/ros2_ws`
- All ROS 2 commands available

## Step 4: Launch Nodes

```bash
# In the ROS 2 shell, launch policy node
ros2 launch hippocampus_ros2 policy.launch.py

# Or use the helper script
./scripts/ros2_docker.sh launch policy.launch.py
```

## Available Commands

```bash
./scripts/ros2_docker.sh shell      # Open interactive shell
./scripts/ros2_docker.sh build      # Build packages
./scripts/ros2_docker.sh launch <file>  # Launch a node
./scripts/ros2_docker.sh stop       # Stop container
./scripts/ros2_docker.sh remove     # Remove container
```

## Troubleshooting

### Docker not running
```bash
# Start Docker Desktop, then try again
```

### Container already exists
```bash
# Remove old container
./scripts/ros2_docker.sh remove
```

### Need to rebuild
```bash
./scripts/ros2_docker.sh build
```

---

**Next**: See `docs/ROS2_MACOS_INSTALL.md` for detailed instructions.

