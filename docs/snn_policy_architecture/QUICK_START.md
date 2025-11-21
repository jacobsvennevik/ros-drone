# SNN Policy Service - Quick Start Guide

## Installation

The policy service is part of the `hippocampus_core` package. For SNN functionality, install PyTorch and snnTorch:

```bash
pip install torch snntorch
```

## Basic Usage

### 1. Heuristic Mode (No SNN Required)

```python
from hippocampus_core.env import Environment
from hippocampus_core.controllers.place_cell_controller import (
    PlaceCellController,
    PlaceCellControllerConfig,
)
from hippocampus_core.policy import (
    TopologyService,
    SpatialFeatureService,
    SpikingPolicyService,
    ActionArbitrationSafety,
    RobotState,
    Mission,
    MissionGoal,
    GoalType,
    PointGoal,
)

# Setup
env = Environment(width=1.0, height=1.0)
place_controller = PlaceCellController(env, PlaceCellControllerConfig(), rng)

# Policy services
ts = TopologyService()
sfs = SpatialFeatureService(ts)
sps = SpikingPolicyService(sfs)  # Heuristic mode
aas = ActionArbitrationSafety()

# Mission
mission = Mission(
    goal=MissionGoal(type=GoalType.POINT, value=PointGoal((0.9, 0.9)))
)

# Control loop
for step in range(100):
    # Update mapping
    position = agent.step(dt)
    place_controller.step(position, dt)
    if step % 10 == 0:
        ts.update_from_controller(place_controller)
    
    # Policy decision
    robot_state = RobotState(pose=(position[0], position[1], heading), time=step*dt)
    features, context = sfs.build_features(robot_state, mission)
    decision = sps.decide(features, context, dt)
    
    # Safety filter
    graph_snapshot = ts.get_graph_snapshot(robot_state.time)
    safe_cmd = aas.filter(decision, robot_state, graph_snapshot, mission)
    
    # Apply action
    v, omega = safe_cmd.cmd
    # ... apply to agent ...
```

### 2. SNN Mode (Requires PyTorch/snnTorch)

```python
from hippocampus_core.policy import PolicySNN

# Create SNN model
snn_model = PolicySNN(
    feature_dim=44,  # 2D: computed from features
    hidden_dim=64,
    output_dim=2,
    beta=0.9,
)

# Create policy service with SNN
sps = SpikingPolicyService(
    sfs,
    config={
        "encoding_scheme": "rate",
        "num_steps": 1,
    },
    snn_model=snn_model,
)

# Use same as above - will use SNN inference
decision = sps.decide(features, context, dt)
# decision.reason will be "snn" if SNN used, "heuristic" if fallback
```

## Running Examples

### Heuristic Demo
```bash
python3 examples/policy_demo.py
```

### SNN Demo (requires PyTorch/snnTorch)
```bash
python3 examples/snn_policy_demo.py
```

## Component Overview

- **TopologyService**: Wraps `TopologicalGraph`, provides snapshots
- **SpatialFeatureService**: Builds features from graph + robot state
- **SpikingPolicyService**: Makes decisions (heuristic or SNN)
- **ActionArbitrationSafety**: Filters decisions through safety constraints

## Feature Dimensions

**2D Configuration** (default):
- Goal: 3 features
- Neighbors (k=8): 32 features
- Topology: 3 features
- Safety: 4 features
- Dynamics: 2 features (optional)
- **Total**: ~44 features (without dynamics) or ~46 (with dynamics)

## Configuration

```python
config = {
    "max_linear": 0.3,        # m/s
    "max_angular": 1.0,       # rad/s
    "encoding_scheme": "rate", # "rate", "latency", "delta"
    "num_steps": 1,           # Time steps for encoding
    "history_length": 10,     # Temporal context history
    "device": "cpu",          # "cpu", "cuda", "mps"
}
```

## Next Steps

- See `examples/policy_demo.py` for complete heuristic example
- See `examples/snn_policy_demo.py` for SNN example
- See `docs/snn_policy_architecture/` for detailed architecture docs

