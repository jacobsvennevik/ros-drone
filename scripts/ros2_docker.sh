#!/bin/bash
# ROS 2 Docker Helper Script for macOS
# This script runs ROS 2 in a Docker container with your workspace mounted

set -e

# Configuration
ROS_DISTRO=${ROS_DISTRO:-humble}
IMAGE_NAME="osrf/ros:${ROS_DISTRO}-desktop-full"
CONTAINER_NAME="ros2_${ROS_DISTRO}"

# Get the project root (parent of scripts/)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
ROS2_WS="$PROJECT_ROOT/ros2_ws"

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}ROS 2 Docker Helper${NC}"
echo "===================="
echo "ROS Distro: $ROS_DISTRO"
echo "Image: $IMAGE_NAME"
echo "Project: $PROJECT_ROOT"
echo ""

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo "❌ Docker is not running. Please start Docker Desktop."
    exit 1
fi

# Check if image exists
if ! docker images | grep -q "osrf/ros.*${ROS_DISTRO}"; then
    echo "📦 Pulling ROS 2 image (this may take a while)..."
    docker pull "$IMAGE_NAME"
fi

# Create ros2_ws if it doesn't exist
mkdir -p "$ROS2_WS/src"

# Check if container exists
if docker ps -a | grep -q "$CONTAINER_NAME"; then
    echo "🔄 Starting existing container..."
    docker start "$CONTAINER_NAME" > /dev/null
else
    echo "🆕 Creating new container..."
    docker run -d \
        --name "$CONTAINER_NAME" \
        --network host \
        -v "$PROJECT_ROOT:/workspace" \
        -v "$ROS2_WS:/home/ros/ros2_ws" \
        -e DISPLAY=${DISPLAY:-:0} \
        "$IMAGE_NAME" \
        tail -f /dev/null
fi

# Function to run commands in container
run_in_container() {
    # Use -i only (not -it) for non-interactive commands
    if [ -t 0 ]; then
        # Interactive terminal available
        docker exec -it \
            -w /home/ros/ros2_ws \
            "$CONTAINER_NAME" \
            bash -c "source /opt/ros/$ROS_DISTRO/setup.bash && $*"
    else
        # No TTY available (script mode)
        docker exec -i \
            -w /home/ros/ros2_ws \
            "$CONTAINER_NAME" \
            bash -c "source /opt/ros/$ROS_DISTRO/setup.bash && $*"
    fi
}

# Parse command
case "${1:-shell}" in
    shell|bash)
        echo "🐚 Opening ROS 2 shell..."
        echo "   Workspace: /home/ros/ros2_ws"
        echo "   Your code: /workspace"
        echo ""
        echo "💡 Tips:"
        echo "   - Your ros2_ws is at /home/ros/ros2_ws"
        echo "   - Project root is at /workspace"
        echo "   - ROS 2 is already sourced"
        echo ""
        docker exec -it \
            -w /home/ros/ros2_ws \
            "$CONTAINER_NAME" \
            bash -c "source /opt/ros/$ROS_DISTRO/setup.bash && exec bash"
        ;;
    build)
        echo "🔨 Building packages..."
        run_in_container "colcon build --packages-select hippocampus_ros2_msgs hippocampus_ros2"
        ;;
    build-all)
        echo "🔨 Building all packages..."
        run_in_container "colcon build"
        ;;
    source)
        echo "📦 Sourcing workspace..."
        run_in_container "source install/setup.bash && exec bash"
        ;;
    test)
        echo "🧪 Running tests..."
        run_in_container "colcon test --packages-select hippocampus_ros2_msgs hippocampus_ros2"
        ;;
    launch)
        if [ -z "$2" ]; then
            echo "Usage: $0 launch <launch_file> [args...]"
            echo "Example: $0 launch policy.launch.py"
            exit 1
        fi
        shift
        run_in_container "source install/setup.bash && ros2 launch hippocampus_ros2 $*"
        ;;
    stop)
        echo "🛑 Stopping container..."
        docker stop "$CONTAINER_NAME"
        ;;
    remove)
        echo "🗑️  Removing container..."
        docker stop "$CONTAINER_NAME" 2>/dev/null || true
        docker rm "$CONTAINER_NAME"
        ;;
    logs)
        docker logs -f "$CONTAINER_NAME"
        ;;
    *)
        echo "Usage: $0 [command]"
        echo ""
        echo "Commands:"
        echo "  shell       Open interactive ROS 2 shell (default)"
        echo "  build       Build hippocampus packages"
        echo "  build-all   Build all packages"
        echo "  source      Source workspace and open shell"
        echo "  test        Run tests"
        echo "  launch      Launch a node (e.g., 'launch policy.launch.py')"
        echo "  stop        Stop container"
        echo "  remove      Remove container"
        echo "  logs        Show container logs"
        echo ""
        echo "Examples:"
        echo "  $0                    # Open shell"
        echo "  $0 build              # Build packages"
        echo "  $0 launch policy.launch.py  # Launch policy node"
        ;;
esac

