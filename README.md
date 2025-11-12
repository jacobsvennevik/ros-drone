# ros-drone

[![ROS CI](https://github.com/jacobhornsvennevik/ros-drone/actions/workflows/ros-ci.yml/badge.svg)](https://github.com/jacobhornsvennevik/ros-drone/actions/workflows/ros-ci.yml)

Hippocampal-inspired navigation experiments written in Python with optional ROS 2 integration. The core package (`hippocampus_core`) simulates a 2D arena, Gaussian place cells, coactivity tracking, and a topological graph builder. ROS 2 nodes in `hippocampus_ros2` wrap these controllers for robot experiments.

## Getting Started

1. Create and activate a virtual environment (recommended):
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```
2. Install the project in editable mode with dev extras:
   ```bash
   pip install -e .[dev]
   ```
   This installs `hippocampus_core` from the `src/` layout along with core dependencies (`numpy`, `matplotlib`, `networkx`) and dev tooling (`pytest`, `nox`). Optional extras such as `torch`/`snntorch` can be added as needed.

## Running the Core Simulation

Execute the standalone simulator:

```bash
python main.py
```

### Plot Cheat-Sheet

- *Trajectory:* agent path through the rectangular arena.
- *Place-cell centres:* sampled Gaussian receptive fields laid over the arena.
- *Topological graph:* nodes are place-cell centres; edges appear when coactivity exceeds the configured threshold and the cells lie within the allowed spatial distance.

## Testing
## Continuous Integration

- The `ROS Package CI` workflow builds `ros2_ws` with ROS 2 Humble on Ubuntu 22.04 runners, invoking `colcon build` and `colcon test` against `hippocampus_ros2`.
- Reproduce locally by sourcing a Humble installation, then:

```bash
cd ros2_ws
colcon build --packages-select hippocampus_ros2 --symlink-install
colcon test --packages-select hippocampus_ros2
```

Fast unit tests cover firing rates, coactivity, topology construction, and the place-cell controller. After the editable install, run:

```bash
pytest
```

or, for a clean-room run:

```bash
nox -s tests
```

> **Note:** Some shells export both `NO_COLOR` and `FORCE_COLOR`, which confuses `nox`. If that happens, run `env -u NO_COLOR -u FORCE_COLOR nox -s tests`.

## Experiments and Models

- Spiking-controller experiments (e.g. online R-STDP) live under `experiments/`.
- Store trained snnTorch or PyTorch checkpoints under `experiments/checkpoints/` (create the folder if needed) and point controller configs to those paths.

## ROS 2 Integration

The `ros2_ws/src/hippocampus_ros2` package provides an `ament_python` node (`snn_brain_node`) that subscribes to odometry and publishes velocity commands. See `ros2_ws/src/hippocampus_ros2/README.md` for build instructions, parameters, topic inspection tips, and a developer smoke-test walkthrough.
