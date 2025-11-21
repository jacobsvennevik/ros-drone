# ROS 2 Installation on macOS

## Overview

ROS 2 doesn't have native macOS support, so we'll use **Docker** (recommended) or build from source.

## Option 1: Docker (Recommended) ✅

### Prerequisites
- Docker Desktop installed (you already have it!)
- Docker Desktop running

### Quick Start

1. **Pull ROS 2 Docker image:**
   ```bash
   docker pull osrf/ros:humble-desktop-full
   ```

2. **Create a helper script** (see below)

3. **Run ROS 2 in Docker:**
   ```bash
   ./scripts/ros2_docker.sh
   ```

### Docker Helper Script

I'll create a script that:
- Mounts your workspace
- Shares network with host
- Keeps ROS 2 environment active

## Option 2: Build from Source (Advanced)

This is more complex but gives you native performance.

### Steps:

1. **Install dependencies:**
   ```bash
   brew install cmake python@3.11 git wget
   brew install tinyxml2
   brew install cppcheck
   ```

2. **Clone ROS 2:**
   ```bash
   mkdir -p ~/ros2_ws/src
   cd ~/ros2_ws
   wget https://raw.githubusercontent.com/ros2/ros2/humble/ros2.repos
   vcs import src < ros2.repos
   ```

3. **Build:**
   ```bash
   cd ~/ros2_ws
   colcon build --symlink-install
   ```

⚠️ **Note**: Building from source on macOS can take hours and may have compatibility issues.

## Option 3: Linux VM (Alternative)

- Use VirtualBox/VMware with Ubuntu
- Install ROS 2 natively in the VM
- Share files via shared folders

---

## Recommended: Docker Setup

Let me create a Docker helper script for you.

