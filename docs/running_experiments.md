# Running Topological Mapping Experiments

This guide shows you how to run the validation experiment and visualization scripts for Hoffman et al. (2016) topological mapping.

## Prerequisites

### 1. Install Dependencies

Make sure you have the core dependencies installed:

```bash
# From project root
pip install -e .[dev]
```

### 2. Install Persistent Homology Library (Optional but Recommended)

For Betti number computation, install `ripser` (recommended) or `gudhi`:

```bash
pip install ripser
# or
pip install gudhi
```

Or install with the project extras:

```bash
pip install -e .[dev,ph]
```

**Note**: The scripts will still run without persistent homology libraries, but Betti number computation will be skipped.

## Running the Validation Experiment

The validation experiment compares different integration window values and shows how they affect map learning.

### Basic Usage

```bash
python3 experiments/validate_hoffman_2016.py
```

This runs with default parameters:
- Integration windows: `0 60 120 240 480` seconds (0 = no window, 480 = 8 minutes)
- Duration: 300 seconds (5 minutes) per experiment
- Number of place cells: 100
- Shows interactive plots at the end

### Custom Integration Windows

```bash
python3 experiments/validate_hoffman_2016.py --integration-windows 0 120 240 480 600
```

This compares 5 different integration window values (in seconds).

### Longer Simulation Duration

```bash
python3 experiments/validate_hoffman_2016.py --duration 600 --integration-windows 0 240 480
```

This runs each experiment for 10 minutes (600 seconds).

### Save Output Figure

```bash
python3 experiments/validate_hoffman_2016.py --output results/validation_comparison.png
```

This saves the comparison plots to a file instead of showing them interactively.

### All Options

```bash
python3 experiments/validate_hoffman_2016.py \
    --integration-windows 0 60 120 240 480 \
    --duration 300 \
    --num-cells 120 \
    --output results/validation.png \
    --seed 42
```

**Options:**
- `--integration-windows`: Integration window values in seconds (default: 0 60 120 240 480)
- `--duration`: Simulation duration in seconds per experiment (default: 300)
- `--num-cells`: Number of place cells (default: 100)
- `--output`: Path to save output figure (default: show interactively)
- `--seed`: Random seed for reproducibility (default: 42)

### What to Expect

The script will:
1. Run simulations for each integration window value
2. Track edges, components, and Betti numbers over time
3. Print summary statistics comparing the results
4. Generate 6 comparison plots:
   - **Graph Growth**: Number of edges over time
   - **Fragmentation**: Number of components (b₀) over time
   - **Spurious Loops**: Number of holes (b₁) over time
   - **Final Graph Size**: Final edge counts vs. integration window
   - **Learning Time**: Time to stable topology vs. integration window
   - **Final Topology**: Final b₁ vs. integration window

**Expected Output:**
```
======================================================================
Hoffman et al. (2016) Topological Mapping Validation
======================================================================

Integration windows: [0, 60, 120, 240, 480] seconds
Simulation duration: 300 seconds (5.0 minutes)
Number of place cells: 100

  Running with ϖ=0s... done
  Running with ϖ=60s... done
  Running with ϖ=120s... done
  Running with ϖ=240s... done
  Running with ϖ=480s... done

======================================================================
Summary Statistics
======================================================================

   ϖ (s) | Final Edges | Final b₀ | Final b₁ | T_min (min)
----------------------------------------------------------------------
   None |         152 |         2 |       18 |        2.1
     60 |         145 |         1 |        8 |        2.8
    120 |         138 |         1 |        5 |        3.2
    240 |         132 |         1 |        2 |        3.8
    480 |         128 |         1 |        1 |        4.2
```

## Running the Topology Learning Visualization

This script visualizes how the topological map evolves over time, with barcode-style Betti number timelines (like Figure 1A from the paper).

### Basic Usage

```bash
python3 examples/topology_learning_visualization.py
```

This runs with default parameters:
- Integration window: 480 seconds (8 minutes)
- Duration: 600 seconds (10 minutes)
- Number of place cells: 120
- Shows interactive plots

### Custom Integration Window

```bash
python3 examples/topology_learning_visualization.py --integration-window 240
```

Use a 4-minute integration window instead of 8 minutes.

### Longer Duration

```bash
python3 examples/topology_learning_visualization.py --duration 1200
```

Run for 20 minutes to see more complete learning.

### Disable Integration Window

```bash
python3 examples/topology_learning_visualization.py --integration-window 0
```

This disables the integration window (ϖ = None) to compare with enabled case.

### Save Output Figure

```bash
python3 examples/topology_learning_visualization.py --output results/topology_evolution.png
```

### All Options

```bash
python3 examples/topology_learning_visualization.py \
    --integration-window 480 \
    --duration 600 \
    --num-cells 120 \
    --output results/topology.png \
    --seed 42
```

**Options:**
- `--integration-window`: Integration window in seconds (default: 480, use 0 for None)
- `--duration`: Simulation duration in seconds (default: 600)
- `--num-cells`: Number of place cells (default: 120)
- `--output`: Path to save output figure (default: show interactively)
- `--seed`: Random seed for reproducibility (default: 42)

### What to Expect

The script will:
1. Run a single simulation tracking topology evolution
2. Sample the graph ~100 times during the simulation
3. Compute Betti numbers at each sample point
4. Generate a comprehensive 3×3 grid of plots:
   - **Row 1**: Barcode-style Betti number timelines (b₀, b₁, b₂)
   - **Row 2**: Edge counts, components, and Betti numbers over time (line plots)
   - **Row 3**: Graph structure snapshots (early, middle, final)

**Expected Output:**
```
======================================================================
Topology Learning Visualization
Inspired by Hoffman et al. (2016) Figure 1A
======================================================================

Running simulation...
  Duration: 10.0 minutes
  Integration window: 480.0s
  Sampling every 6.0 seconds

  Progress: 100%
  Simulation complete!

Generating visualizations...
Saved figure to: results/topology.png
```

## Quick Comparison: With vs. Without Integration Window

To quickly compare the effect of integration windows, run both scripts side-by-side:

### Terminal 1: Without Integration Window
```bash
python3 examples/topology_learning_visualization.py \
    --integration-window 0 \
    --duration 600 \
    --output results/no_integration_window.png
```

### Terminal 2: With Integration Window
```bash
python3 examples/topology_learning_visualization.py \
    --integration-window 480 \
    --duration 600 \
    --output results/with_integration_window.png
```

Then compare the two output images:
- **Without integration window**: More spurious loops (higher b₁), potentially fragmented (b₀ > 1)
- **With integration window**: Fewer spurious loops (lower b₁), connected (b₀ = 1)

## Troubleshooting

### ImportError: No module named 'ripser'

**Solution**: Install ripser:
```bash
pip install ripser
```

The scripts will still run without it, but Betti number computation will be skipped.

### Plots Don't Appear (Non-GUI Environment)

**Solution**: Save plots to files instead:
```bash
python3 experiments/validate_hoffman_2016.py --output results.png
python3 examples/topology_learning_visualization.py --output results.png
```

### Scripts Run Slowly

**Solutions**:
1. Reduce `--duration` for faster experiments
2. Reduce `--num-cells` (fewer place cells = faster computation)
3. Reduce number of integration windows in validation script

### Betti Numbers Show -1

**Meaning**: Persistent homology library (ripser/gudhi) not available.

**Solution**: Install ripser or gudhi (see Prerequisites above).

## Expected Results

Based on Hoffman et al. (2016), you should observe:

1. **Integration window reduces spurious loops**: b₁ should decrease as integration window increases
2. **Integration window reduces fragmentation**: b₀ should converge to 1 faster with integration window
3. **Learning time increases slightly**: T_min may increase with longer integration windows, but maps are more stable
4. **Final topology is more accurate**: With integration window, final b₁ should match expected topology

## Next Steps

After running these experiments:
1. Compare results with the paper findings (see `docs/hoffman_2016_analysis.md`)
2. Try different parameter values (place cell count, sigma, etc.)
3. Experiment with different environments (size, obstacles)
4. Use the validation results to tune your own experiments

## References

- **Paper**: Hoffman, K., Babichev, A., & Dabaghian, Y. (2016). Topological mapping of space in bat hippocampus. arXiv:1601.04253.
- **Usage Guide**: `docs/topological_mapping_usage.md`
- **Paper Analysis**: `docs/hoffman_2016_analysis.md`

