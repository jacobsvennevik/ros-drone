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

   For Betti number computation (persistent homology), install the optional dependency:
   ```bash
   pip install ripser
   # or
   pip install gudhi
   ```
   Or install with the `ph` extra:
   ```bash
   pip install -e .[dev,ph]
   ```

## Running the Core Simulation

Execute the standalone simulator:

```bash
python main.py
```

### Plot Cheat-Sheet

- *Trajectory:* agent path through the rectangular arena.
- *Place-cell centres:* sampled Gaussian receptive fields laid over the arena.
- *Topological graph:* nodes are place-cell centres; edges appear when coactivity exceeds the configured threshold and the cells lie within the allowed spatial distance.

## Topological Mapping Features

### Integration Window (ϖ)

The integration window gates edge admission in the topological graph, preventing transient coactivity from creating spurious connections. This implements the "integrator" mechanism from Hoffman et al. (2016) for stable map learning.

**Two windows:**
- **Coincidence window (w)**: Short window (≈ hundreds of ms) to detect pairwise coactivity events. Configured via `coactivity_window` in `PlaceCellControllerConfig`.
- **Integration window (ϖ)**: Long window (minutes) that gates edge admission. A pair must exceed the coactivity threshold for at least this duration before an edge is added. Configured via `integration_window` in `PlaceCellControllerConfig`.

**Usage:**
```python
from hippocampus_core.controllers.place_cell_controller import (
    PlaceCellController,
    PlaceCellControllerConfig,
)

config = PlaceCellControllerConfig(
    num_place_cells=120,
    coactivity_window=0.2,      # w: coincidence window (200ms)
    coactivity_threshold=5.0,
    integration_window=60.0,     # ϖ: integration window (60 seconds)
)

controller = PlaceCellController(environment=env, config=config)
```

**Benefits:**
- Reduces false edges from transient coactivity
- More stable topological maps
- Matches biological hippocampal dynamics

Set `integration_window=None` (default) to disable this feature and maintain backward compatibility.

### Betti Number Computation

Betti numbers quantify the topological structure of the learned graph, enabling verification that the topology matches the physical environment (e.g., identifying holes, obstacles, or connected components).

**What are Betti numbers?**
- **b₀**: Number of connected components
- **b₁**: Number of 1D holes (loops)
- **b₂**: Number of 2D holes (voids)
- etc.

**Installation:**
```bash
pip install ripser
# or
pip install gudhi
```

**Usage:**
```python
graph = controller.get_graph()

# Compute Betti numbers (requires ripser or gudhi)
betti = graph.compute_betti_numbers(max_dim=2)

print(f"Connected components (b_0): {betti[0]}")
print(f"Loops/holes (b_1): {betti[1]}")
print(f"Voids (b_2): {betti[2]}")

# Verify topology matches environment
assert betti[0] == graph.num_components()  # b_0 should equal component count
```

**Backend selection:**
```python
# Auto-select (prefers ripser, falls back to gudhi)
betti = graph.compute_betti_numbers(backend="auto")

# Explicit backend
betti = graph.compute_betti_numbers(backend="ripser")
betti = graph.compute_betti_numbers(backend="gudhi")
```

**Example validation:**
- For a connected arena: expect `b_0 = 1`, `b_1 = 0`
- For an arena with one hole: expect `b_0 = 1`, `b_1 = 1`
- For disconnected regions: `b_0` equals the number of components

See `examples/integration_window_demo.py` and `examples/betti_numbers_demo.py` for basic examples.

**For comprehensive documentation:**
- Usage guide: `docs/topological_mapping_usage.md`
- Paper analysis: `docs/hoffman_2016_analysis.md`
- Visualization: `examples/topology_learning_visualization.py` (Betti number tracking over time)
- Validation: `experiments/validate_hoffman_2016.py` (comprehensive validation experiment)

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
