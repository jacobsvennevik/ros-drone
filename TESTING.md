# Testing Guide for Integration Window Feature

This document describes how to test the new integration window functionality.

## Quick Test Script

A standalone test script is available:

```bash
python3 test_integration_window.py
```

This script tests:
1. **CoactivityTracker integration tracking** - Verifies that pairs are tracked when they first exceed threshold
2. **Integration window edge gating** - Confirms that edges are properly gated by the integration window
3. **Backward compatibility** - Ensures `integration_window=None` works as before

## Running Existing Tests

### With pytest (if installed)

```bash
# Run all tests
python3 -m pytest tests/ -v

# Run specific test files
python3 -m pytest tests/test_coactivity.py -v
python3 -m pytest tests/test_placecell_controller.py -v

# Run new integration window tests
python3 -m pytest tests/test_coactivity.py::test_integration_window_tracking -v
python3 -m pytest tests/test_placecell_controller.py::test_integration_window_gates_edges -v
python3 -m pytest tests/test_placecell_controller.py::test_integration_window_allows_edges_after_duration -v

# Run topology/Betti number tests
python3 -m pytest tests/test_topology.py -v
python3 -m pytest tests/test_topology.py::test_get_maximal_cliques -v
python3 -m pytest tests/test_topology.py::test_compute_betti_numbers_simple_cycle -v
```

### Sanity Checks

The main.py file includes sanity checks that run automatically:

```bash
python3 main.py
```

The sanity check now tests both:
- Controller without integration window (backward compatibility)
- Controller with integration window (new feature)

It verifies that:
- Coactivity matrix is symmetric
- Graph structure is valid
- Integration window gates edges (fewer edges with window enabled)

## Test Coverage

### Unit Tests

**`tests/test_coactivity.py`**
- `test_integration_window_tracking()` - Tests threshold exceedance tracking

**`tests/test_placecell_controller.py`**
- `test_integration_window_gates_edges()` - Verifies edges are gated
- `test_integration_window_allows_edges_after_duration()` - Confirms edges appear after window

**`tests/test_topology.py`**
- `test_get_maximal_cliques()` - Tests clique extraction from graph
- `test_compute_betti_numbers_requires_dependency()` - Verifies graceful error handling
- `test_compute_betti_numbers_simple_cycle()` - Tests Betti computation on cycle graph (b_1=1)
- `test_compute_betti_numbers_disconnected()` - Tests Betti computation on disconnected graph

### Integration Tests

**`test_integration_window.py`**
- Comprehensive standalone test script
- Tests all aspects of integration window functionality
- Can be run without pytest

**`main.py` sanity checks**
- End-to-end simulation test
- Compares behavior with and without integration window

## Expected Behavior

### Without Integration Window (`integration_window=None`)
- Edges are admitted immediately when `coactivity[i, j] >= c_min`
- This is the original behavior (backward compatible)

### With Integration Window (`integration_window=2.0`)
- Edges are only admitted if:
  1. `coactivity[i, j] >= c_min` (threshold exceeded)
  2. `current_time - first_exceeded_time >= integration_window` (enough time elapsed)
- This filters out transient spurious connections
- Results in fewer edges, but more stable maps

## Example Usage

### Integration Window

```python
from hippocampus_core.controllers.place_cell_controller import (
    PlaceCellController,
    PlaceCellControllerConfig,
)

# Without integration window (original behavior)
config_old = PlaceCellControllerConfig(
    num_place_cells=50,
    coactivity_window=0.2,
    coactivity_threshold=5.0,
    integration_window=None,  # No gating
)

# With integration window (paper's approach)
config_new = PlaceCellControllerConfig(
    num_place_cells=50,
    coactivity_window=0.2,      # Coincidence window w (~200ms)
    coactivity_threshold=5.0,
    integration_window=480.0,   # Integration window ϖ (~8 minutes)
)
```

### Betti Number Computation

```python
from hippocampus_core.topology import TopologicalGraph

# Get graph from controller
graph = controller.get_graph()

# Extract maximal cliques
cliques = graph.get_maximal_cliques()
print(f"Found {len(cliques)} maximal cliques")

# Compute Betti numbers (requires ripser or gudhi)
try:
    betti = graph.compute_betti_numbers(max_dim=2)
    print(f"b_0 (connected components): {betti[0]}")
    print(f"b_1 (1D holes/loops): {betti[1]}")
    print(f"b_2 (2D holes/voids): {betti[2]}")
except ImportError:
    print("Install ripser: pip install ripser")
```

## Troubleshooting

### Import Errors
If you get import errors, make sure the `src/` directory is in your Python path:
```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))
```

### Missing Dependencies
Install required packages:
```bash
pip install numpy matplotlib networkx
```

For tests:
```bash
pip install pytest
```

Or install from project:
```bash
pip install -e .[dev]  # For development dependencies (pytest, etc.)
pip install -e .[ph]   # For persistent homology (ripser)
```

### Betti Number Computation

Betti number computation requires an optional dependency:

```bash
# Install ripser (recommended, lighter weight)
pip install ripser

# Or install gudhi (alternative)
pip install gudhi

# Or install via project optional dependencies
pip install -e .[ph]
```

If ripser/gudhi is not installed, `compute_betti_numbers()` will raise an informative `ImportError` with installation instructions.

