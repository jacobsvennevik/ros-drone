# Legacy vs Current Code

This document clarifies what code is considered **legacy** (older but still supported) versus **current** (actively recommended) in the codebase.

## Summary

**Important**: Nothing is truly deprecated or removed. "Legacy" means "older, simpler option" - it's still maintained and useful, but newer alternatives exist with more features.

## Controllers

### ✅ Current / Recommended

#### 1. **BatNavigationController** (Recommended for Biological Studies)
- **Status**: ✅ Current, actively developed
- **Use when**: You need biological fidelity, HD/grid attractors, or heading data
- **Location**: `src/hippocampus_core/controllers/bat_navigation_controller.py`
- **Observation format**: `[x, y, θ]` (requires heading)
- **Features**: 
  - HD attractor network
  - Grid cell attractor
  - Conjunctive place cells
  - Periodic calibration
  - HD/grid diagnostics

#### 2. **PlaceCellController** (Legacy, but Still Valid)
- **Status**: ⚠️ Legacy (simpler, original implementation), but still maintained and useful
- **Use when**: 
  - Getting started / learning
  - Don't have heading data
  - Need simple, deterministic behavior
  - Quick prototyping
- **Location**: `src/hippocampus_core/controllers/place_cell_controller.py`
- **Observation format**: `[x, y]` (no heading required)
- **Features**: 
  - Place cells only
  - Coactivity tracking
  - Topology building

**Note**: Despite being labeled "legacy" in some docs, `PlaceCellController` is still the **default** choice and remains a valid option. "Legacy" here means "original, simpler version" - not deprecated.

#### 3. **SnnTorchController** (For Trained Models)
- **Status**: ✅ Current, for specific use cases
- **Use when**: You have trained SNN models
- **Location**: `src/hippocampus_core/controllers/snntorch_controller.py`
- **Observation format**: Model-dependent

---

## Why "Legacy" for PlaceCellController?

The term "legacy" appears in documentation and comments, but it means:

- **Original implementation**: `PlaceCellController` was the first controller
- **Simpler option**: No HD/grid layers, just place cells → topology
- **Still recommended**: For use cases that don't need biological fidelity
- **Not deprecated**: Still maintained, tested, and supported

The newer `BatNavigationController` extends `PlaceCellController` and adds HD/grid layers, making it more biologically realistic. But `PlaceCellController` remains useful for:
- Simpler workflows
- When heading data isn't available
- Quick prototyping
- Teaching/learning

---

## ROS 2 Integration

### Controller Backends

```bash
# Current / Recommended (if you have heading data)
ros2 launch hippocampus_ros2 brain.launch.py \
  controller_backend:=bat_navigation

# Legacy (but still default and valid)
ros2 launch hippocampus_ros2 brain.launch.py \
  controller_backend:=place_cells  # or omit (default)

# For trained models
ros2 launch hippocampus_ros2 brain.launch.py \
  controller_backend:=snntorch \
  model_path:=/path/to/model.pt
```

**Status**:
- `bat_navigation`: ✅ Current, recommended
- `place_cells`: ⚠️ Legacy (default), still valid
- `snntorch`: ✅ Current, for trained models

---

## Examples / Demos

### Current Examples (Use BatNavigationController)

- `examples/policy_demo.py` - ✅ Updated to use BatNavigationController
- `examples/obstacle_environment_demo.py` - ✅ Supports both (via `--controller` flag)
- `examples/topology_learning_visualization.py` - ✅ Supports both (via `--controller` flag)

### Legacy Examples (Use PlaceCellController)

- `examples/integration_window_demo.py` - ⚠️ Still uses PlaceCellController (may update later)
- `examples/betti_numbers_demo.py` - ⚠️ Still uses PlaceCellController (may update later)

**Note**: These aren't "wrong" - they just use the simpler controller. They could be updated to support both, but it's not a priority.

---

## Validation Scripts

### Current / Recommended

- `experiments/sweep_rubin_hd_validation.py` - ✅ Uses BatNavigationController
- `experiments/sweep_yartsev_grid_validation.py` - ✅ Uses BatNavigationController
- `notebooks/rubin_hd_validation.ipynb` - ✅ Uses BatNavigationController
- `notebooks/yartsev_grid_without_theta.ipynb` - ✅ Uses BatNavigationController

### Legacy (Still Valid)

- `experiments/validate_hoffman_2016.py` - ⚠️ Uses PlaceCellController (by design - validates topology learning)
- `experiments/validate_hoffman_2016_with_stats.py` - ⚠️ Uses PlaceCellController (by design)

**Note**: These validation scripts use `PlaceCellController` by design - they're testing topology learning, not HD/grid functionality. This is intentional and correct.

---

## Code Patterns

### Legacy Pattern (Still Works)

```python
# PlaceCellController - still valid and supported
from hippocampus_core.controllers.place_cell_controller import (
    PlaceCellController,
    PlaceCellControllerConfig,
)

config = PlaceCellControllerConfig(num_place_cells=120)
controller = PlaceCellController(environment=env, config=config)

obs = np.array([x, y])  # Just position
controller.step(obs, dt=0.05)
```

### Current Pattern (Recommended if Heading Available)

```python
# BatNavigationController - recommended for biological studies
from hippocampus_core.controllers.bat_navigation_controller import (
    BatNavigationController,
    BatNavigationControllerConfig,
)

config = BatNavigationControllerConfig(
    num_place_cells=80,
    hd_num_neurons=72,
    grid_size=(16, 16),
    calibration_interval=250,
)
controller = BatNavigationController(environment=env, config=config)

obs = np.array([x, y, theta])  # Position + heading
controller.step(obs, dt=0.05)

# Access HD/grid diagnostics
hd_estimate = controller.hd_attractor.estimate_heading()
grid_drift = controller.grid_attractor.drift_metric()
```

---

## Migration Path

If you're using `PlaceCellController` and want to upgrade:

1. **Do you have heading data?** → If no, stay with `PlaceCellController`
2. **Do you need biological fidelity?** → If yes, migrate to `BatNavigationController`
3. **Are you just learning?** → Stay with `PlaceCellController` (simpler)

See `docs/CONTROLLER_COMPARISON.md` for detailed migration guide.

---

## What to Use When

| Use Case | Controller | Status |
|----------|-----------|--------|
| **Getting started** | PlaceCellController | ⚠️ Legacy, but recommended |
| **Learning the codebase** | PlaceCellController | ⚠️ Legacy, but recommended |
| **No heading data** | PlaceCellController | ⚠️ Legacy, but required |
| **Quick prototyping** | PlaceCellController | ⚠️ Legacy, but recommended |
| **Biological studies** | BatNavigationController | ✅ Current, recommended |
| **HD/grid diagnostics** | BatNavigationController | ✅ Current, required |
| **Rubin/Yartsev validation** | BatNavigationController | ✅ Current, required |
| **Trained SNN models** | SnnTorchController | ✅ Current, for models |

---

## Summary

**Nothing is deprecated.** "Legacy" means:
- ✅ Older, simpler implementation
- ✅ Still maintained and supported
- ✅ Still the default for simple use cases
- ✅ Valid choice when heading data isn't available

**"Current" means**:
- ✅ Newer implementation with more features
- ✅ Recommended when biological fidelity is needed
- ✅ Requires heading data

**Both are valid** - choose based on your needs:
- **Simple use case?** → `PlaceCellController` (legacy, but still recommended)
- **Biological fidelity?** → `BatNavigationController` (current, recommended)

---

## References

- **Controller Comparison**: `docs/CONTROLLER_COMPARISON.md`
- **Architecture**: `docs/ARCHITECTURE.md`
- **ROS Integration**: `docs/ROS_RUNNING_INSTRUCTIONS.md`
- **Troubleshooting**: `docs/troubleshooting.md`

