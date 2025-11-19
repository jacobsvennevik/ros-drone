# Example Gallery

This directory contains examples and descriptions of different topological mapping scenarios.

## Available Examples

### 1. Integration Window Comparison

**Script**: `examples/integration_window_demo.py`

**Description**: Demonstrates how the integration window (ϖ) gates edge admission, preventing transient coactivity from creating spurious connections.

**Key Features**:
- Compares maps with and without integration window
- Shows fewer spurious edges with integration window
- Validates Hoffman et al. (2016) finding

**Usage**:
```bash
python3 examples/integration_window_demo.py
```

**Expected Output**:
- Comparison of edge counts
- Demonstration that integration window reduces premature edge formation

---

### 2. Obstacle Environment with Holes

**Script**: `examples/obstacle_environment_demo.py`

**Description**: Shows how obstacles create topological holes (b₁ > 0) in the learned map. With a central obstacle, the system should correctly identify b₁ = 1.

**Key Features**:
- Environment with central circular obstacle
- Comparison with and without obstacle
- Visualization of how b₁ evolves over time
- Graph snapshots showing connections around obstacle

**Usage**:
```bash
python3 examples/obstacle_environment_demo.py
```

**Expected Output**:
- With obstacle: b₁ = 1 (one hole encircling obstacle)
- Without obstacle: b₁ = 0 (no holes)
- Comparison plots showing edge growth and b₁ evolution
- Final graph visualization with obstacle

**Screenshot locations**: `results/obstacle_comparison.png`

---

### 3. Topology Learning Visualization (Barcode Style)

**Script**: `examples/topology_learning_visualization.py`

**Description**: Visualizes how Betti numbers evolve over time in barcode style (like Figure 1A from Hoffman et al. 2016). Shows the persistence of topological features.

**Key Features**:
- Barcode timelines for b₀, b₁, b₂
- Time-series plots of graph metrics
- Graph snapshots at different learning stages
- Demonstrates integration window effects on learning time

**Usage**:
```bash
# With integration window (recommended)
python3 examples/topology_learning_visualization.py \
    --integration-window 480 \
    --duration 1200 \
    --output results/topology_480.png

# Without integration window (for comparison)
python3 examples/topology_learning_visualization.py \
    --integration-window 0 \
    --duration 600 \
    --output results/topology_none.png
```

**Expected Output**:
- Barcode showing b₀ dropping from many components → 1
- Barcode showing b₁ = 0 throughout (no spurious loops)
- Graph growth and component fragmentation plots
- Visual graph evolution snapshots

---

### 4. Validation Experiment

**Script**: `experiments/validate_hoffman_2016.py`

**Description**: Comprehensive validation experiment comparing different integration window values, verifying paper findings.

**Key Features**:
- Multiple integration window values
- Summary statistics table
- Comparison plots showing effects of integration window
- T_min (learning time) estimation

**Usage**:
```bash
# Basic validation
python3 experiments/validate_hoffman_2016.py

# With obstacle environment
python3 experiments/validate_hoffman_2016.py --obstacle --duration 1200

# Custom parameters
python3 experiments/validate_hoffman_2016.py \
    --integration-windows 0 120 240 480 \
    --duration 1200 \
    --num-cells 120 \
    --output results/validation.png
```

**Expected Output**:
- Summary table showing final edges, b₀, b₁, T_min for each integration window
- Six comparison plots:
  1. Graph growth (edges over time)
  2. Fragmentation (components over time)
  3. Spurious loops (b₁ over time)
  4. Final graph size vs. integration window
  5. Learning time vs. integration window
  6. Final topology vs. integration window

---

### 5. Betti Numbers Demo

**Script**: `examples/betti_numbers_demo.py`

**Description**: Demonstrates Betti number computation on known topologies (cycle, path, disconnected nodes).

**Key Features**:
- Tests on simple graph structures with known Betti numbers
- Validates persistent homology computation
- Shows correct topology identification

**Usage**:
```bash
python3 examples/betti_numbers_demo.py
```

**Expected Output**:
- Verification that cycle graphs have b₁ = 1
- Verification that path graphs have b₁ = 0
- Verification that disconnected nodes have b₀ = number of components

---

## Understanding the Outputs

### Barcode Plots

**What they show**: Persistence of topological features over time
- **X-axis**: Time (minutes)
- **Y-axis**: Betti number value
- **Horizontal bars**: Features that exist during that time period

**How to read**:
- Long bars = persistent features (stable topology)
- Short bars = transient features (learning artifacts)
- Multiple bars = multiple features of same type

### Time-Series Plots

**Graph Growth**: Shows how many edges form over time
- Should show delayed formation with integration window
- Should plateau at maximum once learning completes

**Fragmentation**: Shows component count (b₀) over time
- Starts at number of place cells (all isolated)
- Drops to 1 when graph becomes connected
- Target line shows desired state (b₀ = 1)

**Spurious Loops**: Shows b₁ over time
- Should stay at 0 for simple environments
- Should show peaks and then settle to correct value (e.g., b₁ = 1 with obstacle)
- Peaks indicate spurious loops that disappear during learning

---

## Example Scenarios

### Scenario 1: Simple Open Arena (No Obstacles)

**Expected**:
- b₀ → 1 (connected)
- b₁ = 0 (no holes)
- b₂ = 0 (no voids)

**Good for**: Testing basic functionality, integration window effects

---

### Scenario 2: Arena with Central Obstacle

**Expected**:
- b₀ → 1 (connected)
- b₁ → 1 (one hole around obstacle)
- b₂ = 0 (no voids)

**Good for**: Validating obstacle detection, testing topology learning with holes

---

### Scenario 3: Multiple Obstacles

**Expected**:
- b₀ → 1 (connected)
- b₁ → N (N holes, one per obstacle if well-separated)
- b₂ = 0 (no voids)

**Good for**: Testing complex topologies, validating hole counting

---

## Tips for Running Examples

1. **Start simple**: Run basic examples first before complex scenarios
2. **Check dependencies**: Ensure ripser/gudhi installed for Betti numbers
3. **Use sufficient duration**: For integration windows, ensure duration ≥ 2× window
4. **Save outputs**: Use `--output` flag to save plots for later inspection
5. **Compare results**: Run with and without integration window to see differences

---

## Troubleshooting Examples

**Example fails to run**:
- Check dependencies: `pip install -e .[dev,ph]`
- Verify Python version: Python 3.10+

**Example runs but results look wrong**:
- Check parameter values (duration, integration window, etc.)
- Verify environment setup (obstacle position, size)
- Review troubleshooting guide: `docs/troubleshooting.md`

**Example is too slow**:
- Reduce duration or number of place cells
- Use fewer integration window values
- Skip Betti number computation if not needed

---

## Contributing Examples

To add a new example:

1. Create script in `examples/` directory
2. Add description to this README
3. Include usage instructions
4. Document expected outputs
5. Add example output screenshots (optional but helpful)

