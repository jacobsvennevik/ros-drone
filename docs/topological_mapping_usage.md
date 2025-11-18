# Topological Mapping Usage Guide

This guide explains how to use the topological mapping features based on Hoffman et al. (2016), including integration windows and Betti number verification.

## Overview

The topological mapping system builds a graph representation of space from place cell coactivity. Key features:

1. **Pairwise coactivity detection** with short coincidence window `w` (~200ms)
2. **Integration window `ϖ`** (~8 minutes) that gates edge admission
3. **Spatial gating** via `max_edge_distance` to prevent spurious connections
4. **Betti number computation** for topological verification

See `docs/hoffman_2016_analysis.md` for theoretical background.

## Basic Usage

### Creating a Controller

```python
from hippocampus_core.controllers.place_cell_controller import (
    PlaceCellController,
    PlaceCellControllerConfig,
)
from hippocampus_core.env import Environment

# Create environment
env = Environment(width=1.0, height=1.0)

# Configure controller
config = PlaceCellControllerConfig(
    num_place_cells=120,
    sigma=0.15,                    # Place field size
    max_rate=20.0,                 # Peak firing rate (Hz)
    coactivity_window=0.2,         # w: coincidence window (200ms)
    coactivity_threshold=5.0,      # Minimum coactivity count for edge
    max_edge_distance=0.3,         # Spatial gating (place field overlap)
    integration_window=480.0,      # ϖ: integration window (8 minutes)
)

controller = PlaceCellController(environment=env, config=config)
```

### Running a Simulation

```python
import numpy as np
from hippocampus_core.env import Agent

# Create agent for navigation
agent = Agent(environment=env, random_state=np.random.default_rng(42))

# Run simulation
dt = 0.05  # Time step in seconds
for step in range(1000):
    position = agent.step(dt)
    action = controller.step(np.asarray(position), dt)
    
    # Access graph periodically
    if step % 100 == 0:
        graph = controller.get_graph()
        print(f"Time: {controller.current_time:.1f}s, "
              f"Edges: {graph.num_edges()}, "
              f"Components: {graph.num_components()}")
```

## Integration Window (`ϖ`)

The integration window gates edge admission: edges are only added after pairwise coactivity has exceeded the threshold for at least `ϖ` seconds.

### When to Use Integration Window

**Use integration window (`ϖ > 0`) when**:
- You need **stable, accurate maps** with fewer spurious connections
- The agent moves **fast** (high-speed scenarios)
- You want to **validate** against Hoffman et al. (2016) findings
- You're doing **research** and need topological verification

**Skip integration window (`ϖ = None`) when**:
- You need **quick map building** for real-time control
- The agent moves **slowly** (transient coactivity is less problematic)
- You're doing **exploratory** mapping without strict accuracy requirements

### Configuration Examples

**Without integration window (immediate edge admission)**:
```python
config = PlaceCellControllerConfig(
    coactivity_window=0.2,
    coactivity_threshold=5.0,
    integration_window=None,  # No gating
)
```

**With integration window (recommended for research)**:
```python
config = PlaceCellControllerConfig(
    coactivity_window=0.2,         # w: detect coactivity (200ms)
    coactivity_threshold=5.0,
    integration_window=480.0,      # ϖ: gate edges (8 minutes)
)
```

**Shorter integration window (faster learning, less stable)**:
```python
config = PlaceCellControllerConfig(
    integration_window=120.0,      # 2 minutes (less stable)
)
```

### Understanding the Two Windows

- **Coincidence window `w`** (~200ms): Detects if two place cells spike together
  - Short timescale for detecting pairwise coactivity events
  - Used by `CoactivityTracker` to increment coactivity counts

- **Integration window `ϖ`** (~8 minutes): Time over which evidence must persist
  - Long timescale for filtering transient coactivity
  - Gates edge admission in `TopologicalGraph.build_from_coactivity()`

**Example**: A pair might exceed the threshold at t=100s, but the edge is only admitted at t=580s (after 480s integration window).

## Betti Number Verification

Betti numbers verify that the learned graph topology matches the physical environment:

- **b₀**: Number of connected components (should be 1 for connected space)
- **b₁**: Number of 1D holes/loops (should match physical obstacles)
- **b₂**: Number of 2D holes/voids (usually 0 in 2D)

### Computing Betti Numbers

**Requirements**: Install `ripser` or `gudhi`:
```bash
pip install ripser
# or
pip install gudhi
```

**Usage**:
```python
graph = controller.get_graph()

try:
    betti = graph.compute_betti_numbers(max_dim=2)
    print(f"b₀ (components): {betti[0]}")
    print(f"b₁ (holes): {betti[1]}")
    print(f"b₂ (voids): {betti[2]}")
    
    # Verify topology
    assert betti[0] == 1, f"Space should be connected (b₀=1), got {betti[0]}"
    assert betti[1] == 0, f"No holes expected (b₁=0), got {betti[1]}"
except ImportError:
    print("Install ripser or gudhi for Betti number computation")
```

### Interpreting Results

**Expected for simple connected arena (no obstacles)**:
- b₀ = 1 (one connected component)
- b₁ = 0 (no holes)
- b₂ = 0 (no voids)

**Expected for arena with one obstacle (e.g., central column)**:
- b₀ = 1 (one connected component)
- b₁ = 1 (one hole encircling the obstacle)
- b₂ = 0 (no voids in 2D)

**Problem signs**:
- b₀ > 1 → Fragmented map (multiple disconnected components)
- b₁ >> 1 → Many spurious loops (too many false connections)
- High b₂ → Indicates 3D structure or over-connection

### Tracking Betti Numbers Over Time

```python
import numpy as np

betti_over_time = []

for step in range(1000):
    position = agent.step(dt)
    controller.step(np.asarray(position), dt)
    
    if step % 100 == 0:
        graph = controller.get_graph()
        try:
            betti = graph.compute_betti_numbers(max_dim=1)
            betti_over_time.append({
                'time': controller.current_time,
                'b0': betti[0],
                'b1': betti[1],
            })
        except ImportError:
            pass

# Plot evolution
import matplotlib.pyplot as plt
times = [b['time'] for b in betti_over_time]
b0 = [b['b0'] for b in betti_over_time]
b1 = [b['b1'] for b in betti_over_time]

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.plot(np.array(times) / 60, b0, 'b-', label='b₀ (components)')
plt.xlabel('Time (minutes)')
plt.ylabel('Betti number')
plt.title('Components over time')
plt.axhline(y=1, color='r', linestyle='--', label='Target')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(np.array(times) / 60, b1, 'r-', label='b₁ (holes)')
plt.xlabel('Time (minutes)')
plt.ylabel('Betti number')
plt.title('Holes over time')
plt.legend()
plt.show()
```

## Parameter Tuning

### Coactivity Threshold (`coactivity_threshold`)

- **Lower threshold** (e.g., 3.0): More edges, faster learning, more spurious connections
- **Higher threshold** (e.g., 10.0): Fewer edges, slower learning, more reliable connections

**Recommendation**: Start with 5.0, adjust based on place cell firing rates and navigation density.

### Place Field Size (`sigma`)

- **Smaller sigma** (e.g., 0.1): More precise, requires more cells for coverage
- **Larger sigma** (e.g., 0.2): Broader coverage, less precise localization

**Recommendation**: Set `sigma` such that `num_place_cells × (2*sigma)²` covers your environment.

### Max Edge Distance (`max_edge_distance`)

- **Default**: `2.0 * sigma` (ensures place field overlap)
- **Tighter**: `1.5 * sigma` (stricter spatial gating)
- **Looser**: `2.5 * sigma` (allows more connections)

**Recommendation**: Keep default unless you have specific requirements.

## Examples

### Example 1: Basic Usage

```python
from hippocampus_core.controllers.place_cell_controller import (
    PlaceCellController,
    PlaceCellControllerConfig,
)
from hippocampus_core.env import Agent, Environment
import numpy as np

env = Environment(width=1.0, height=1.0)
config = PlaceCellControllerConfig(
    num_place_cells=100,
    sigma=0.15,
    coactivity_window=0.2,
    integration_window=480.0,
)
controller = PlaceCellController(env, config)
agent = Agent(env, random_state=np.random.default_rng(42))

for _ in range(1000):
    position = agent.step(0.05)
    controller.step(np.asarray(position), 0.05)

graph = controller.get_graph()
print(f"Final graph: {graph.num_nodes()} nodes, {graph.num_edges()} edges")
```

### Example 2: With Betti Number Verification

```python
# ... (setup as above) ...

graph = controller.get_graph()

# Compute Betti numbers
try:
    betti = graph.compute_betti_numbers(max_dim=2)
    print(f"Topology: b₀={betti[0]}, b₁={betti[1]}, b₂={betti[2]}")
    
    if betti[0] == 1 and betti[1] <= 1:
        print("✓ Topology matches expected structure")
    else:
        print("⚠ Topology may have spurious features")
except ImportError:
    print("Install ripser for Betti number computation")
```

### Example 3: Comparing Integration Windows

See `examples/integration_window_demo.py` for a demonstration comparing maps with and without integration windows.

### Example 4: Full Validation Experiment

See `experiments/validate_hoffman_2016.py` for a comprehensive validation script that:
- Compares different integration window values
- Tracks Betti numbers over time
- Measures learning time T_min
- Generates comparison plots

### Example 5: Betti Number Tracking Over Time

See `examples/topology_learning_visualization.py` for visualization of:
- Barcode-style Betti number timelines
- Graph structure evolution
- Edge counts and components over time

## Troubleshooting

### Problem: Too many edges / spurious connections

**Solutions**:
1. Increase `coactivity_threshold`
2. Use integration window (`integration_window > 0`)
3. Decrease `max_edge_distance`
4. Increase `sigma` to reduce place field density

### Problem: Too few edges / fragmented map

**Solutions**:
1. Decrease `coactivity_threshold`
2. Increase navigation duration
3. Increase `num_place_cells` for better coverage
4. Decrease `max_edge_distance` (if too restrictive)

### Problem: Betti numbers don't match expected

**Check**:
1. Is the integration window long enough? (Try 480s / 8 minutes)
2. Has the agent explored enough? (Learning takes time)
3. Are place fields covering the space? (Check `num_place_cells` and `sigma`)
4. Is the spatial gating too restrictive? (Check `max_edge_distance`)

### Problem: ImportError for Betti numbers

**Solution**: Install persistent homology library:
```bash
pip install ripser
# or
pip install gudhi
```

## References

- **Paper**: Hoffman, K., Babichev, A., & Dabaghian, Y. (2016). Topological mapping of space in bat hippocampus. arXiv preprint arXiv:1601.04253.
- **Detailed analysis**: `docs/hoffman_2016_analysis.md`
- **Implementation comparison**: `.cursor/rules/topological-mapping-paper.mdc`
- **Principles guide**: `.cursor/rules/bat-hippocampus-principles.mdc`

