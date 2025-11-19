# Quick Start: Running Topological Mapping Experiments

This is a quick reference for running the validation experiment and visualization scripts.

## Step 1: Install Dependencies (if needed)

```bash
# Install core dependencies
pip install ripser

# Install persistent homology library for Betti number computation (recommended)
pip install ripser
```

## Step 2: Run the Validation Experiment

This compares different integration window values to validate Hoffman et al. (2016) findings.

### Quick Run (default settings):
```bash
python3 experiments/validate_hoffman_2016.py
```

**What it does:**
- Runs 5 experiments with integration windows: 0, 60, 120, 240, 480 seconds
- Each runs for 5 minutes (300 seconds)
- Shows comparison plots at the end

### Save to File:
```bash
python3 experiments/validate_hoffman_2016.py --output results/validation.png
```

### Custom Integration Windows:
```bash
python3 experiments/validate_hoffman_2016.py --integration-windows 0 120 240 480
```

### Longer Duration (more learning):
```bash
python3 experiments/validate_hoffman_2016.py --duration 600
```

## Step 3: Run the Topology Learning Visualization

This visualizes how Betti numbers evolve over time (like Figure 1A from the paper).

### Quick Run (default settings):
```bash
python3 examples/topology_learning_visualization.py
```

**What it does:**
- Runs a 10-minute simulation with 8-minute integration window
- Shows barcode-style Betti number timelines
- Displays graph structure evolution

### Save to File:
```bash
python3 examples/topology_learning_visualization.py --output results/topology_evolution.png
```

### Compare with vs. without Integration Window:

**Without integration window:**
```bash
python3 examples/topology_learning_visualization.py --integration-window 0 --output results/no_window.png
```

**With integration window:**
```bash
python3 examples/topology_learning_visualization.py --integration-window 480 --output results/with_window.png
```

## Step 4: Run Obstacle Environment Demo

Test topological learning with obstacles that create holes (b₁ > 0):

### Quick Run:
```bash
python3 examples/obstacle_environment_demo.py
```

**What it does:**
- Runs comparison with and without central obstacle
- Shows how obstacles create holes (b₁ = 1)
- Demonstrates correct topology learning

### With Validation Script:
```bash
python3 experiments/validate_hoffman_2016.py \
    --obstacle \
    --obstacle-radius 0.15 \
    --duration 1200 \
    --integration-windows 0 120 240 480
```

**Expected results:**
- With obstacle: b₁ = 1 (one hole around obstacle)
- Without obstacle: b₁ = 0 (no holes)

## What to Look For

### Validation Experiment Output:
- **Summary table** showing final edges, components (b₀), holes (b₁), and learning time
- **6 comparison plots** showing the effects of different integration windows

### Visualization Output:
- **Barcode timelines** (top row) showing when topological features persist
- **Graph metrics** (middle row) showing edges and components over time
- **Graph snapshots** (bottom row) showing structure at different times

## Expected Results

✅ **With integration window (ϖ = 480s)**:
- Fewer spurious loops (lower b₁)
- Connected map (b₀ = 1)
- More stable topology

❌ **Without integration window (ϖ = None)**:
- More spurious loops (higher b₁)
- Potentially fragmented (b₀ > 1)
- Less stable topology

## Troubleshooting

**"No module named 'numpy'"**: Install dependencies: `pip install -e .[dev]`

**"No module named 'ripser'"**: Install ripser: `pip install ripser` (Betti numbers will be skipped if not installed)

**Plots don't appear**: Use `--output filename.png` to save to file instead

**Scripts run slowly**: Reduce `--duration` or `--num-cells` for faster runs

## Full Documentation

For detailed documentation, see:
- `docs/running_experiments.md` - Complete guide with all options
- `docs/topological_mapping_usage.md` - Usage guide for integration windows and Betti numbers
- `docs/hoffman_2016_analysis.md` - Detailed paper analysis

