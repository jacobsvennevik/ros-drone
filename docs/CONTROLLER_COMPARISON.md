# Controller Comparison Guide

This guide helps you choose the right hippocampal navigation controller for your use case.

📖 **See `docs/LEGACY_CODE.md` for clarification on what's "legacy" vs "current" - nothing is deprecated, both are valid choices.**

## Quick Decision Tree

```
Do you need HD/grid attractors for biological fidelity?
├─ YES → BatNavigationController
│        (requires heading data)
│
└─ NO → Do you have a trained SNN model?
         ├─ YES → SnnTorchController
         │        (requires model checkpoint)
         │
         └─ NO → PlaceCellController
                  (default, simplest)
```

## Controller Overview

### 1. PlaceCellController (Default)

**What it does:**
- Place cell population with Gaussian receptive fields
- Coactivity tracking and topological graph building
- Basic hippocampal-inspired mapping

**Observation format:** `[x, y]` (2D position)

**When to use:**
- ✅ Getting started with hippocampal navigation
- ✅ Need lightweight, deterministic behavior
- ✅ Don't need heading information
- ✅ Focused on topology learning only
- ✅ Quick prototyping

**Example:**
```python
from hippocampus_core.controllers.place_cell_controller import (
    PlaceCellController,
    PlaceCellControllerConfig,
)

config = PlaceCellControllerConfig(
    num_place_cells=120,
    integration_window=480.0,  # 8 minutes
)
controller = PlaceCellController(environment=env, config=config)

# Step with [x, y] observation
obs = np.array([0.5, 0.5])
action = controller.step(obs, dt=0.05)
```

**Pros:**
- Simple and fast
- Well-tested and stable
- No heading data required
- Good for topology validation

**Cons:**
- No directional awareness
- Less biologically realistic
- Limited to place cell behavior

---

### 2. BatNavigationController (Recommended for Biological Fidelity)

**What it does:**
- HD attractor network for head-direction tracking
- Grid cell attractor with path integration
- Conjunctive place cells combining HD + grid
- Periodic calibration to correct drift
- Full bat hippocampal navigation model

**Observation format:** `[x, y, θ]` (2D position + heading in radians)

**When to use:**
- ✅ Need biological fidelity (bat navigation studies)
- ✅ Have heading/odometry data available
- ✅ Want HD and grid cell diagnostics
- ✅ Studying directional navigation
- ✅ Validating against Rubin/Yartsev papers

**Example:**
```python
from hippocampus_core.controllers.bat_navigation_controller import (
    BatNavigationController,
    BatNavigationControllerConfig,
)

config = BatNavigationControllerConfig(
    num_place_cells=80,
    hd_num_neurons=72,
    grid_size=(16, 16),
    calibration_interval=250,
    integration_window=480.0,
)
controller = BatNavigationController(environment=env, config=config)

# Step with [x, y, theta] observation
obs = np.array([0.5, 0.5, 0.0])  # x, y, heading
action = controller.step(obs, dt=0.05)

# Access HD/grid statistics
hd_estimate = controller.hd_attractor.estimate_heading()
grid_drift = controller.grid_attractor.drift_metric()
```

**Pros:**
- ✅ Biologically realistic (reproduces bat behavior)
- ✅ HD tracking for directional awareness
- ✅ Grid cell path integration
- ✅ Rich diagnostics (HD error, grid drift, etc.)
- ✅ Automatic drift correction via calibration

**Cons:**
- ⚠️ Requires heading data in observations
- ⚠️ More computational overhead
- ⚠️ More parameters to tune

**Validation notebooks:**
- `notebooks/rubin_hd_validation.ipynb` - Head-direction tuning validation
- `notebooks/yartsev_grid_without_theta.ipynb` - Grid stability without theta

---

### 3. SnnTorchController (For Trained Models)

**What it does:**
- Loads pre-trained SNN models (PyTorch/snnTorch)
- Stateful spiking neural network inference
- Customizable via training data

**Observation format:** Model-dependent (typically `[x, y, cos(θ), sin(θ)]` or similar)

**When to use:**
- ✅ Have a trained SNN model checkpoint
- ✅ Need learned navigation behavior
- ✅ Want to deploy trained policies
- ✅ Custom neural network architectures

**Example:**
```python
from hippocampus_core.controllers.snntorch_controller import (
    SnnTorchController,
)

controller = SnnTorchController.from_checkpoint(
    checkpoint_path="models/snn_controller.pt",
    device="cpu",
    model_kind="state_dict",
)

# Step with observation (format depends on training)
obs = np.array([0.5, 0.5, 0.8, 0.6])  # x, y, cos(theta), sin(theta)
action = controller.step(obs, dt=0.05)
```

**Pros:**
- ✅ Learned navigation policies
- ✅ Flexible architectures
- ✅ GPU acceleration support

**Cons:**
- ⚠️ Requires training data and process
- ⚠️ Model-dependent observation format
- ⚠️ Need PyTorch/snnTorch installed

---

## Feature Comparison

| Feature | PlaceCellController | BatNavigationController | SnnTorchController |
|---------|---------------------|-------------------------|-------------------|
| **Observation** | `[x, y]` | `[x, y, θ]` | Model-dependent |
| **Heading required** | ❌ No | ✅ Yes | Depends |
| **HD tracking** | ❌ No | ✅ Yes | Depends |
| **Grid cells** | ❌ No | ✅ Yes | Depends |
| **Topology building** | ✅ Yes | ✅ Yes | ❌ No |
| **Calibration** | ❌ No | ✅ Yes | ❌ No |
| **Biological fidelity** | ⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ |
| **Computational cost** | Low | Medium | Medium-High |
| **Training required** | ❌ No | ❌ No | ✅ Yes |
| **ROS integration** | ✅ Yes | ✅ Yes | ✅ Yes |

## ROS 2 Integration

All controllers are available in ROS 2 nodes:

### Brain Node
```bash
# Place cells (default)
ros2 launch hippocampus_ros2 brain.launch.py \
  controller_backend:=place_cells

# Bat navigation (requires heading in /odom)
ros2 launch hippocampus_ros2 brain.launch.py \
  controller_backend:=bat_navigation

# SNN model (requires model_path)
ros2 launch hippocampus_ros2 brain.launch.py \
  controller_backend:=snntorch \
  model_path:=/path/to/model.pt
```

### Policy Node
```bash
# Place cells (default)
ros2 launch hippocampus_ros2 policy.launch.py \
  controller_backend:=place_cells

# Bat navigation (recommended for biological studies)
ros2 launch hippocampus_ros2 policy.launch.py \
  controller_backend:=bat_navigation
```

See `docs/ROS_RUNNING_INSTRUCTIONS.md` for details.

## Migration Guide

### From PlaceCellController to BatNavigationController

1. **Update observations:**
   ```python
   # Before: [x, y]
   obs = np.array([position[0], position[1]])
   
   # After: [x, y, theta]
   obs = np.array([position[0], position[1], heading])
   ```

2. **Update imports:**
   ```python
   from hippocampus_core.controllers.bat_navigation_controller import (
       BatNavigationController,
       BatNavigationControllerConfig,
   )
   ```

3. **Update configuration:**
   ```python
   # Add HD/grid parameters
   config = BatNavigationControllerConfig(
       num_place_cells=80,
       hd_num_neurons=72,
       grid_size=(16, 16),
       calibration_interval=250,
       # ... other PlaceCellControllerConfig params still work
   )
   ```

4. **Access additional features:**
   ```python
   # HD/grid statistics
   hd_estimate = controller.hd_attractor.estimate_heading()
   grid_estimate = controller.grid_attractor.estimate_position()
   grid_drift = controller.grid_attractor.drift_metric()
   ```

## Troubleshooting

### "BatNavigationController requires observations containing (x, y, θ)"
**Solution:** Ensure your observations include heading data:
```python
obs = np.array([x, y, theta])  # theta in radians
```

### HD estimate drifting
**Solution:** Check calibration interval and heading update frequency:
```python
config = BatNavigationControllerConfig(
    calibration_interval=250,  # Calibrate every 250 steps
    # ...
)
```

### Grid activity unstable
**Solution:** See `notebooks/yartsev_grid_without_theta.ipynb` for grid drift diagnostics.

## Further Reading

- **Betti numbers verification**: `docs/BETTI_USAGE_GUIDE.md`
- **Topological mapping**: `docs/topological_mapping_usage.md`
- **ROS integration**: `docs/ROS_RUNNING_INSTRUCTIONS.md`
- **Bat validation**: `notebooks/rubin_hd_validation.ipynb`, `notebooks/yartsev_grid_without_theta.ipynb`
- **Paper analysis**: `docs/hoffman_2016_analysis.md`, `docs/rubin_2014_analysis.md`

## Summary

- **Start simple**: Use `PlaceCellController` for basic topology learning
- **Go biological**: Use `BatNavigationController` for bat navigation studies or when heading data is available
- **Deploy trained**: Use `SnnTorchController` when you have trained models

All controllers build topological graphs and integrate with the policy service stack.

