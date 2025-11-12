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
- Offline policy training with synthetic experts runs through `python -m experiments.train_snntorch_policy --output-dir models`; the script now exports both `snn_controller.pt` (state_dict) and `snn_controller.ts` (TorchScript) alongside normalisation metadata.
- ROS deployments accept either format. Set `controller_backend:=snntorch`, `model_kind:=state_dict|torchscript`, and point `model_path`/`torchscript_path` in `brain.yaml` (or launch overrides) to the exported files.
- Keep trained checkpoints in `models/` (default) or your preferred directory. The `.pt` bundle includes observation statistics, action scaling, rollout length, and HPO metadata for provenance.

### HPO Quickstart

- Run `python -m experiments.hpo_snntorch --trials 30` to launch an Optuna study that tunes hidden size, β, learning rate, unroll horizon, surrogate gradient, and batch size with a median pruner (short epochs for rapid iteration).
- The script retrains the best configuration, saves refreshed artifacts, and writes `models/best_trial.json` summarising trial metrics, params, and artifact locations.
- Use `--refit-epochs` for longer retraining once you have a promising region, and adjust `--episodes/--steps` for larger expert datasets when you move beyond quick sweeps.

## ROS 2 Integration

The `ros2_ws/src/hippocampus_ros2` package provides an `ament_python` node (`snn_brain_node`) that subscribes to odometry and publishes velocity commands. See `ros2_ws/src/hippocampus_ros2/README.md` for build instructions, parameters, topic inspection tips, and a developer smoke-test walkthrough.

## Profiling & Tracing

- Use `ros2 launch hippocampus_ros2 tracing.launch.py trace_session:=hippocampus_profile trace_duration:=15.0` to capture a 15-second trace with `ros2_tracing`. Traces land under `ros2_ws/traces/` by default (Linux-only; ensure `ros-${ROS_DISTRO}-ros2-tracing` is installed).
- Analyse captured traces with `ros2 trace list`, `ros2 trace info`, or GUI tools such as Tracy or LTTng viewer to inspect callback latency and executor timing while the snnTorch controller runs.

## Verification Checklist

Follow this sequence to validate the training → deployment pipeline (see the [PyTorch save/load state_dict guide](https://pytorch.org/tutorials/beginner/saving_loading_models.html) for background on the checkpoint format):

1. **Train & export** – run `python -m experiments.train_snntorch_policy --output-dir models` (or the dataset-driven controller trainer) to create `models/snn_controller.pt` and `models/snn_controller.ts`. Inspect the checkpoint keys (`version`, `model_state`, `obs_mean`, `obs_std`, `action_scale`, `time_steps`) with small `python` snippets and verify the TorchScript module loads with `torch.jit.load`.
2. **Local inference sanity** – load the checkpoint with `SnnTorchController.from_checkpoint(...)`, call `step(...)` on a synthetic observation, and verify the 2-D action is clamped by the stored `action_scale`.
3. **ROS bring-up** – `ros2 launch hippocampus_ros2 brain.launch.py params_file:=/path/to/brain.yaml`, then monitor `ros2 topic hz /cmd_vel` and `ros2 topic echo /cmd_vel` to confirm timing and clamps. Switch between state_dict and TorchScript deployments by setting `model_kind`/`torchscript_path` in the YAML (or via launch overrides).
4. **Integration test** – `colcon test --packages-select hippocampus_ros2`; the launch-testing suite asserts messages on `/snn_action` and `/cmd_vel` within the configured bounds.
5. **Deterministic replay** – `ros2 bag record /odom /snn_action /cmd_vel`, stop recording, then `ros2 bag play <bag> --loop --remap /cmd_vel:=/unused` to drive the controller with stored odometry; expect minor timing jitter only.
6. **CI** – push to GitHub and verify the `action-ros-ci` workflow (with `target-ros2-distro: humble`) and the fast pytest job both pass.
